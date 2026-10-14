import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent import Agent
from app import create_app
from helpers.bootstrap import Services
from helpers.session_store import SessionStore
from telegram_bot import TelegramChatHandler, markdown_to_telegram_html, split_message
from tests.conftest import DOCTOR, FakeLLM, FakeVectorDB, reply, tool_call
from tools.search_doctors import DoctorTools
from tools.skills import SkillLibrary


def _factory(settings, script):
    def build():
        llm = FakeLLM(script)
        tools = DoctorTools(embedding_client=llm, vectordb_client=FakeVectorDB())
        skills = SkillLibrary()
        agent = Agent(client=llm, tools=[tools.search_doctors], skills=skills)
        return Services(config=settings, llm=llm, embedding=llm, vector_db=FakeVectorDB(),
                        doctor_tools=tools, skills=skills, agent=agent)
    return build


def test_chat_session_health_skills_and_mcp_http(settings):
    settings.MCP_API_KEY = "secret-key"
    script = [
        reply(calls=[tool_call("search_doctors", {"query": "toothache"})]),
        reply(f"Dr. Test Molar — {DOCTOR['profile_url']}"),
        reply("You're welcome!"),
    ]
    app = create_app(_factory(settings, script), config=settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert any(s["name"] == "doctor-finder" for s in client.get("/skills").json())
        assert "123" in client.get("/skills/emergency-triage").json()["instructions"]
        assert "search_doctors" in client.get("/tools").json()["tools"]

        r = client.post("/chat", json={"prompt": "my tooth hurts", "session_id": "s1", "include_trace": True}).json()
        assert r["verified"] and r["stop_reason"] == "completed"
        assert r["trace"]["tool_calls"][0]["name"] == "search_doctors"
        r2 = client.post("/chat", json={"prompt": "thanks", "session_id": "s1"}).json()
        assert r2["response"] == "You're welcome!"
        assert len(app.state.sessions.get("api:s1")) == 4  # server remembered the conversation
        client.delete("/chat/s1")
        assert app.state.sessions.get("api:s1") == []

        init = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        assert client.post("/mcp/", json=init, headers=headers).status_code == 401
        ok = client.post("/mcp/", json=init, headers={**headers, "Authorization": "Bearer secret-key"})
        assert ok.status_code == 200, ok.text
        assert "search_doctors" in ok.text

        assert client.post("/telegram/webhook", json={}).status_code == 503  # no bot configured


def test_split_and_format_for_telegram():
    long = ("para " * 300 + "\n\n") * 6
    parts = split_message(long, limit=1000)
    assert all(len(p) <= 1000 for p in parts) and len(parts) > 3
    html = markdown_to_telegram_html("### Doctors\n**Dr. A** <x> [Book](https://vezeeta.com/a?b=1&c=2)\n* item `code`")
    assert "<b>Doctors</b>" in html and "<b>Dr. A</b>" in html and "&lt;x&gt;" in html
    assert '<a href="https://vezeeta.com/a?b=1&amp;c=2">Book</a>' in html
    assert "• item <code>code</code>" in html


def test_telegram_handler_commands_and_messages():
    sent = []

    class FakeBot:
        async def send_message(self, chat_id, text, **kw):
            sent.append(text)

        async def send_chat_action(self, **kw): ...

    agent = Agent(client=FakeLLM([reply("Please see a dentist. **Feel better!**")]), tools=[], verifiers=[])
    handler = TelegramChatHandler(FakeBot(), agent, SessionStore(), SkillLibrary())

    def update(text):
        return SimpleNamespace(message=SimpleNamespace(chat_id=7, text=text), edited_message=None)

    async def go():
        await handler.handle_update(update("/emergency"))
        await handler.handle_update(update("/skills"))
        await handler.handle_update(update("my tooth hurts"))
        await handler.handle_update(SimpleNamespace(message=SimpleNamespace(chat_id=7, text=None)))

    asyncio.run(go())
    assert "123" in sent[0]
    assert "doctor-finder" in sent[1]
    assert sent[2] == "Please see a dentist. <b>Feel better!</b>"
    assert "text messages" in sent[3]
    assert len(handler.sessions.get(7)) == 2
