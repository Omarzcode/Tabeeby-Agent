import time

from agent.budget import Budget
from agent.safety import detect_language, detect_red_flags
from tools.catalog import normalize_area, normalize_specialty


def test_budget_steps_actions_and_cost():
    b = Budget(max_steps=3, max_seconds=60, max_actions=2, max_cost_usd=0.01)
    assert not b.exhausted()
    b.tick()
    assert b.allow(5) == 2  # one turn asked for 5 calls; only 2 granted
    assert b.allow(1) == 0 and b.actions_left() == 0
    assert not b.can_use_tools()
    assert not b.exhausted()  # action limit is soft: the model may still answer
    b.spend(0.02)
    assert b.exhausted() and b.reason().startswith("cost budget")


def test_budget_step_and_time():
    b = Budget(max_steps=2, max_seconds=0.01, max_actions=None)
    b.tick()
    assert b.can_use_tools() is False  # last step is reserved for the final answer
    time.sleep(0.02)
    assert "time budget" in b.reason()
    b2 = Budget(max_steps=1, max_actions=None)
    b2.tick()
    assert "step budget" in b2.reason()


def test_red_flags_english_and_arabic():
    assert detect_red_flags("my dad has crushing chest pain").categories == ["cardiac"]
    assert "stroke" in detect_red_flags("جدتي وشها مايل ولسانها تقيل").categories or \
        detect_red_flags("وشه مايل").is_emergency
    assert detect_red_flags("مش قادر اتنفس").categories == ["breathing"]
    assert not detect_red_flags("I have a mild toothache").is_emergency
    assert not detect_red_flags("chest pain when I cough, mild").is_emergency


def test_language_detection():
    assert detect_language("ضرسي بيوجعني") == "ar"
    assert detect_language("my tooth hurts") == "en"
    assert detect_language("عندي pain في ضهري جامد") == "ar"


def test_area_normalisation():
    assert normalize_area("Maadi") == "El-Maadi"
    assert normalize_area("مدينة نصر") == "Nasr City"
    assert normalize_area("6 october") == "6Th Of October"
    assert normalize_area("nasr city") == "Nasr City"
    assert normalize_area("Heliopolis") == "Heliopolis"
    assert normalize_area("Zamalek") == "El-Zamalek"
    assert normalize_area("Atlantis") is None
    assert normalize_area(None) is None


def test_specialty_normalisation():
    assert normalize_specialty("Orthopedic Surgery") == "Orthopedics"
    assert normalize_specialty("dentist") == "Dentistry"
    assert normalize_specialty("cardiology") == "Cardiology"
    assert normalize_specialty("أطفال") == "Pediatrics"
    assert normalize_specialty("gynecology") == "Reproductive Medicine & Infertility"


def test_red_flags_feminine_egyptian_forms():
    assert "stroke" in detect_red_flags("جدتي فجأة وشها مايل ولسانها تقيل ومش قادرة تحرك ايدها").categories
    assert "breathing" in detect_red_flags("مراتي مش قادرة تتنفس").categories
    assert "consciousness" in detect_red_flags("أمي فاقدة الوعي").categories


def test_retry_helper_retries_transient_only():
    from stores.llm.retry import with_retries
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED Please retry in 0.01s.")
        return "ok"

    assert with_retries(flaky, max_delay=0.05) == "ok" and len(calls) == 2
    import pytest
    with pytest.raises(ValueError):
        with_retries(lambda: (_ for _ in ()).throw(ValueError("bad request")))


def test_ollama_provider_sends_api_key_for_cloud():
    from stores.llm.providers.OllamaProvider import OllamaProvider
    p = OllamaProvider(api_key="abc", api_url="https://ollama.com")
    assert p.client._client.headers["authorization"] == "Bearer abc"
    assert "authorization" not in OllamaProvider(api_key=None, api_url=None).client._client.headers
