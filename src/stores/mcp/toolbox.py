"""MCP client side: borrow tools from other MCP servers.

MCP is async; the agent loop is sync (it runs in a worker thread so FastAPI's
event loop stays free). anyio's *blocking portal* bridges the two: it runs an
event loop on a background thread and `portal.call(...)` hands async work to
it and waits. The portal is thread-safe, so concurrent agent runs can share
one connection.

Servers are configured with the `MCP_SERVERS` env var (JSON):

    {
      "fetch":  {"command": "uvx", "args": ["mcp-server-fetch"]},
      "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
    }

Each borrowed tool is exposed to the LLM as `<server>__<tool>` so two servers
can both have a `search` tool without colliding.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from anyio.from_thread import start_blocking_portal
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
except ImportError:  # pragma: no cover - mcp is a required dependency
    Client = None

SEPARATOR = "__"


class McpToolbox:
    """One live connection to one MCP server, usable from sync code."""

    def __init__(self, name: str, config: Dict[str, Any], timeout: float = 30.0):
        self.name = name
        self.config = config
        self.timeout = timeout
        self.tools: List[Any] = []
        self._portal_cm = self._portal = None
        self._client_cm = self._client = None

    @asynccontextmanager
    async def _transport(self) -> AsyncIterator[Any]:
        if self.config.get("url"):
            http_client = create_mcp_http_client(headers=self.config.get("headers") or None)
            async with http_client:
                async with streamable_http_client(self.config["url"], http_client=http_client) as streams:
                    yield streams
        else:
            params = StdioServerParameters(
                command=self.config["command"],
                args=list(self.config.get("args", [])),
                env={**os.environ, **self.config.get("env", {})} if self.config.get("env") else None,
                cwd=self.config.get("cwd"),
            )
            with open(os.devnull, "w", encoding="utf-8") as sink:  # servers are chatty on stderr
                async with stdio_client(params, errlog=sink) as streams:
                    yield streams

    def open(self) -> "McpToolbox":
        if Client is None:
            raise RuntimeError("The 'mcp' package is not installed.")
        if self._client is not None:
            return self
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        try:
            self._client_cm = self._portal.wrap_async_context_manager(
                Client(self._transport(), read_timeout_seconds=self.timeout)
            )
            self._client = self._client_cm.__enter__()
            self.tools = self._portal.call(self._client.list_tools).tools
        except BaseException:
            self.close()  # never leave a half-open connection behind
            raise
        return self

    def close(self) -> None:
        if self._client_cm is not None:
            try:
                self._client_cm.__exit__(None, None, None)
            except BaseException:  # a server that already died must not mask the real error
                pass
            self._client_cm = self._client = None
        if self._portal_cm is not None:
            try:
                self._portal_cm.__exit__(None, None, None)
            except BaseException:
                pass
            self._portal_cm = self._portal = None

    def schemas(self) -> List[Dict[str, Any]]:
        """Describe the borrowed tools in the OpenAI-style function schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{self.name}{SEPARATOR}{tool.name}",
                    "description": (tool.description or "")[:1000],
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            }
            for tool in self.tools
        ]

    def call(self, tool: str, arguments: Dict[str, Any]) -> str:
        result = self._portal.call(self._client.call_tool, tool, arguments or {})
        text = "\n".join(getattr(c, "text", "") for c in (result.content or []) if getattr(c, "text", None))
        if result.is_error:
            return f"Error: {text or 'tool failed'}"
        if not text and getattr(result, "structured_content", None) is not None:
            return json.dumps(result.structured_content, ensure_ascii=False, default=str)
        return text


class McpRegistry:
    """Every configured MCP server, as one tool source for the agent.

    The agent only asks two things of it: `schemas()` and `call(name, args)`.
    A server that fails to start is logged and skipped — one broken optional
    integration must never take the medical assistant down.
    """

    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None):
        self.servers_config = servers or {}
        self.boxes: Dict[str, McpToolbox] = {}
        self.errors: Dict[str, str] = {}

    @classmethod
    def from_env(cls, raw: Optional[str]) -> "McpRegistry":
        if not raw or not raw.strip():
            return cls({})
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("MCP_SERVERS must be a JSON object")
            return cls(data)
        except Exception as e:
            logger.error("Invalid MCP_SERVERS config: %s", e)
            registry = cls({})
            registry.errors["config"] = str(e)
            return registry

    def open(self) -> "McpRegistry":
        for name, cfg in self.servers_config.items():
            try:
                self.boxes[name] = McpToolbox(name, cfg).open()
                logger.info("MCP server '%s' connected: %d tools", name, len(self.boxes[name].tools))
            except BaseException as e:  # noqa: BLE001 - see class docstring
                self.errors[name] = str(e).strip().splitlines()[0][:200] if str(e) else type(e).__name__
                logger.error("MCP server '%s' failed to start: %s", name, self.errors[name])
        return self

    def close(self) -> None:
        for box in self.boxes.values():
            box.close()
        self.boxes = {}

    def schemas(self) -> List[Dict[str, Any]]:
        return [schema for box in self.boxes.values() for schema in box.schemas()]

    def owns(self, name: str) -> bool:
        server = name.split(SEPARATOR, 1)[0]
        return SEPARATOR in name and server in self.boxes

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        server, tool = name.split(SEPARATOR, 1)
        try:
            return self.boxes[server].call(tool, arguments)
        except Exception as e:  # a broken server must not kill the loop
            return f"Error: {str(e).strip().splitlines()[0][:200] if str(e) else type(e).__name__}"

    def status(self) -> Dict[str, Any]:
        return {
            "connected": {n: [t.name for t in b.tools] for n, b in self.boxes.items()},
            "errors": self.errors,
        }
