"""Run Tabeeby as a local MCP server over stdio.

    cd src && python -m mcp_server            # stdio (Claude Desktop, Cursor, ...)
    cd src && python -m mcp_server --http     # streamable HTTP on :8001/mcp

stdout belongs to the protocol in stdio mode, so all logging goes to stderr.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
os.chdir(SRC)                      # .env and relative asset paths live here
sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from helpers.bootstrap import build_services  # noqa: E402
from mcp_server.server import ServiceContainer, build_mcp_server  # noqa: E402
from tools import web_search  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabeeby MCP server")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-agent", action="store_true", help="expose tools only, without ask_tabeeby")
    args = parser.parse_args()

    services = build_services(connect_mcp=False)  # never borrow tools from ourselves
    container = ServiceContainer(
        doctor_tools=services.doctor_tools,
        skills=services.skills,
        agent=None if args.no_agent else services.agent,
        web_search=web_search if services.config.TAVILY_API_KEY else None,
    )
    mcp = build_mcp_server(container)
    try:
        if args.http:
            import uvicorn

            uvicorn.run(mcp.streamable_http_app(), host="127.0.0.1", port=args.port)
        else:
            mcp.run()  # stdio
    finally:
        services.close()


if __name__ == "__main__":
    main()
