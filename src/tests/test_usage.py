"""Usage & cost tracking: pricing by model_id, the store, the summary and the admin endpoint."""

import asyncio
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent import Agent
from app import create_app
from helpers.session_store import SessionStore
from helpers.usage_store import UsageStore, summarize, usage_event
from stores.llm.pricing import load_price_overrides, price_for, token_cost
from telegram_bot import TelegramChatHandler
from tests.conftest import FakeLLM, reply
from tests.test_api_and_telegram import _factory


def test_token_cost_by_model_id():
    # gpt-5.4-mini: $0.75 in, $0.075 cached, $4.50 out per 1M tokens.
    # 1,000 prompt tokens of which 400 cached, 200 output:
    # (600 × 0.75 + 400 × 0.075 + 200 × 4.50) / 1e6 = (450 + 30 + 900) / 1e6
    assert abs(token_cost("gpt-5.4-mini", 1000, 200, 400) - 0.00138) < 1e-12
    assert price_for("gpt-5.4-mini-2026-03-17") == price_for("gpt-5.4-mini")  # dated snapshot
    assert price_for("gpt-5-mini") != price_for("gpt-5")                      # no prefix confusion
    assert price_for("my-model") is None
    assert abs(token_cost("my-model", 1000, 100, fallback_input_per_token=1e-6, fallback_output_per_token=2e-6) - 0.0012) < 1e-12
    load_price_overrides('{"my-model": [1.0, 0.1, 2.0]}')
    try:
        assert abs(token_cost("my-model", 1000, 100) - (1000 * 1.0 + 100 * 2.0) / 1e6) < 1e-12
    finally:
        load_price_overrides(None)


def test_agent_reports_tokens_and_priced_cost():
    llm = FakeLLM([reply("Hello!", prompt_tokens=1000, completion_tokens=200)])
    llm.generation_model_id = "gpt-5.4-mini"
    result = Agent(client=llm, tools=[], verifiers=[]).run("hi")
    assert result.tokens == {"prompt": 1000, "completion": 200, "cached": 0, "llm_calls": 1, "model": "gpt-5.4-mini"}
    assert abs(result.run_cost - (1000 * 0.75 + 200 * 4.50) / 1e6) < 1e-12


def test_summary_groups_by_person_day_and_model():
    store = UsageStore()  # memory backend
    now = time.time()
    base = {"channel": "telegram", "model": "gpt-5.4-mini", "prompt_tokens": 1000, "cached_tokens": 0,
            "completion_tokens": 100, "seconds": 3.0, "stop_reason": "completed", "verified": True,
            "emergency": False, "skills": []}
    store.record({**base, "ts": now, "user_key": "tg:1", "user_name": "Amr", "cost_usd": 0.002})
    store.record({**base, "ts": now, "user_key": "tg:1", "user_name": "Amr", "cost_usd": 0.004})
    store.record({**base, "ts": now, "user_key": "tg:2", "user_name": "Sara", "cost_usd": 0.001, "emergency": True})
    store.record({**base, "ts": now - 40 * 86400, "user_key": "tg:3", "cost_usd": 9.0})  # outside 30 days
    s = summarize(store.events(now - 31 * 86400), days=30, now=now)
    assert s["totals"]["messages"] == 3 and s["totals"]["users"] == 2 and s["totals"]["emergencies"] == 1
    assert abs(s["totals"]["cost_usd"] - 0.007) < 1e-12
    assert s["users"][0]["user_key"] == "tg:1" and s["users"][0]["messages"] == 2
    assert len(s["daily"]) == 30 and abs(s["daily"][-1]["cost_usd"] - 0.007) < 1e-12
    assert s["models"][0]["model"] == "gpt-5.4-mini" and s["models"][0]["priced"]


def test_admin_dashboard_requires_key_and_chat_is_recorded(settings):
    settings.ADMIN_API_KEY = "admin-secret"
    app = create_app(_factory(settings, [reply("Hi there")]), config=settings)
    with TestClient(app) as client:
        assert "Tabeeby usage" in client.get("/admin/usage").text
        assert client.get("/admin/usage/data").status_code == 401
        client.post("/chat", json={"prompt": "hello", "session_id": "abc"})
        r = client.get("/admin/usage/data?days=7", headers={"Authorization": "Bearer admin-secret"})
        assert r.status_code == 200
        data = r.json()
        assert data["totals"]["messages"] == 1 and data["users"][0]["user_key"] == "api:abc"


def test_telegram_message_is_recorded_per_user():
    class FakeBot:
        async def send_message(self, chat_id, text, **kw): ...
        async def send_chat_action(self, **kw): ...

    usage = UsageStore()
    agent = Agent(client=FakeLLM([reply("See a dentist.")]), tools=[], verifiers=[])
    handler = TelegramChatHandler(FakeBot(), agent, SessionStore(), usage=usage)
    user = SimpleNamespace(id=42, first_name="Mona", last_name="Ali", username="mona_a")
    upd = SimpleNamespace(message=SimpleNamespace(chat_id=7, text="my tooth hurts", from_user=user), edited_message=None)
    asyncio.run(handler.handle_update(upd))
    [event] = usage.events(0)
    assert event["user_key"] == "tg:42" and event["user_name"] == "Mona Ali" and event["username"] == "mona_a"
    assert event["message_chars"] == len("my tooth hurts") and "text" not in event
