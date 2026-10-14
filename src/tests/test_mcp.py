"""MCP both ways: Tabeeby as a server, and Tabeeby borrowing tools as a client."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client

from agent import Agent
from mcp_server.server import ServiceContainer, build_mcp_server
from stores.mcp import McpRegistry
from tests.conftest import FakeLLM, FakeVectorDB, reply, tool_call
from tools.search_doctors import DoctorTools
from tools.skills import SkillLibrary

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_server.py")


def _server():
    llm = FakeLLM([])
    container = ServiceContainer(
        doctor_tools=DoctorTools(embedding_client=llm, vectordb_client=FakeVectorDB()),
        skills=SkillLibrary(),
    )
    return build_mcp_server(container)


def test_tabeeby_mcp_server_tools_resources_prompts():
    async def go():
        async with Client(_server()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"search_doctors", "check_emergency", "medical_web_search", "list_specialties",
                    "list_areas", "ask_tabeeby"} <= names

            res = await client.call_tool("search_doctors", {"query": "toothache", "area": "المعادي"})
            data = json.loads(res.content[0].text)
            assert data["doctors"][0]["name"] == "Dr. Test Molar"
            assert data["filters_applied"]["area"] == "El-Maadi"

            em = json.loads((await client.call_tool("check_emergency", {"text": "مش قادر اتنفس"})).content[0].text)
            assert em["is_emergency"] and "123" in em["action"]

            area = json.loads((await client.call_tool("list_areas", {"contains": "maadi"})).content[0].text)
            assert area["canonical_match"] == "El-Maadi"

            menu = json.loads((await client.read_resource("tabeeby://skills")).contents[0].text)
            assert any(s["name"] == "doctor-finder" for s in menu)
            body = (await client.read_resource("tabeeby://skills/emergency-triage")).contents[0].text
            assert "123" in body

            prompt = await client.get_prompt("triage", {"symptoms": "knee pain", "area": "Dokki"})
            assert "check_emergency" in prompt.messages[0].content.text

            not_ready = await client.call_tool("ask_tabeeby", {"question": "hi"})
            assert "not configured" in not_ready.content[0].text

    asyncio.run(go())


def test_agent_borrows_tools_from_external_mcp_server():
    registry = McpRegistry({"echo": {"command": sys.executable, "args": [FIXTURE]}}).open()
    try:
        assert registry.status()["connected"] == {"echo": ["shout", "fail"]}
        schemas = {s["function"]["name"] for s in registry.schemas()}
        assert schemas == {"echo__shout", "echo__fail"}

        llm = FakeLLM([
            reply(calls=[tool_call("echo__shout", {"text": "hello"}, "1"), tool_call("echo__fail", {"reason": "boom"}, "2")]),
            reply("done"),
        ])
        agent = Agent(client=llm, tools=[], mcp=registry)
        result = agent.run("say hello loudly")
        observations = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
        assert observations[0] == "HELLO"
        assert observations[1].startswith("Error")  # the server hides internals; the loop survives
        assert [t.source for t in result.tool_calls] == ["mcp", "mcp"]
        assert [t.ok for t in result.tool_calls] == [True, False]
        # The schemas (dicts) were offered to the LLM next to the callables.
        assert any(isinstance(t, dict) for t in llm.calls[0]["tools"])
    finally:
        registry.close()


def test_broken_mcp_server_is_skipped_not_fatal():
    registry = McpRegistry({"ghost": {"command": "definitely-not-a-real-binary-xyz"}}).open()
    assert "ghost" in registry.errors and registry.schemas() == []
    bad = McpRegistry.from_env("{not json")
    assert "config" in bad.errors
