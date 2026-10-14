"""The public /chat API as a web frontend uses it."""

from fastapi.testclient import TestClient

from app import create_app
from tests.conftest import DOCTOR, reply, tool_call
from tests.test_api_and_telegram import _factory


def test_chat_creates_session_returns_doctors_and_remembers(settings):
    script = [
        reply(calls=[tool_call("search_doctors", {"query": "toothache"})]),
        reply(f"**Dr. Test Molar** — {DOCTOR['profile_url']}"),
        reply("You're welcome!"),
    ]
    app = create_app(_factory(settings, script), config=settings)
    with TestClient(app) as client:
        r = client.post("/chat", json={"prompt": "my tooth hurts"}).json()
        sid = r["session_id"]
        assert sid and r["direction"] == "ltr" and r["language"] == "en" and r["emergency"] is False
        assert r["doctors"][0]["profile_url"] == DOCTOR["profile_url"] and r["doctors"][0]["name"]
        assert r["history"] is None  # the server keeps it
        r2 = client.post("/chat", json={"prompt": "thanks", "session_id": sid}).json()
        assert r2["session_id"] == sid and r2["doctors"] == []
        msgs = client.get(f"/chat/{sid}").json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        client.delete(f"/chat/{sid}")
        assert client.get(f"/chat/{sid}").json()["messages"] == []
        assert client.post("/chat", json={"prompt": "x", "session_id": "bad id!"}).status_code == 422


def test_chat_allows_browsers_and_rate_limits(settings):
    settings.CORS_ALLOW_ORIGINS = "https://friend.example"
    settings.CHAT_RATE_LIMIT_MESSAGES = 1
    app = create_app(_factory(settings, [reply("Hi"), reply("Hi again")]), config=settings)
    with TestClient(app) as client:
        pre = client.options("/chat", headers={"Origin": "https://friend.example",
                                               "Access-Control-Request-Method": "POST",
                                               "Access-Control-Request-Headers": "content-type"})
        assert pre.status_code == 200
        assert pre.headers["access-control-allow-origin"] == "https://friend.example"
        assert client.post("/chat", json={"prompt": "hi"}).status_code == 200
        limited = client.post("/chat", json={"prompt": "hi"})
        assert limited.status_code == 429 and limited.json()["signal"] == "rate_limited"
        assert int(limited.headers["retry-after"]) > 0


def test_chat_emergency_and_arabic_fields(settings):
    app = create_app(_factory(settings, [reply("اتصل بالإسعاف فورًا على 123")]), config=settings)
    with TestClient(app) as client:
        r = client.post("/chat", json={"prompt": "جدتي فجأة وشها مايل"}).json()
        assert r["emergency"] and r["emergency_numbers"]["Ambulance (Egypt)"] == "123"
        assert r["language"] == "ar" and r["direction"] == "rtl"


def test_rate_limit_key_cannot_be_faked(settings):
    from types import SimpleNamespace
    from routes.chat import client_ip

    def req(xff, cf=None, hops=3):
        headers = {"x-forwarded-for": xff, **({"cf-connecting-ip": cf} if cf else {})}
        return SimpleNamespace(headers=headers, client=SimpleNamespace(host="10.0.0.1"),
                               app=SimpleNamespace(state=SimpleNamespace(config=SimpleNamespace(TRUSTED_PROXY_HOPS=hops))))

    real = "197.50.207.226,172.69.9.134, 10.130.22.68, 10.130.22.33"
    assert client_ip(req(real)) == "197.50.207.226"
    assert client_ip(req("1.2.3.4," + real)) == "197.50.207.226"  # a faked prefix is ignored
    assert client_ip(req("1.2.3.4," + real, cf="197.50.207.226")) == "197.50.207.226"
