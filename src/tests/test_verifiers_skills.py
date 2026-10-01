from agent.safety import detect_red_flags
from agent.verifiers import VerificationContext, repair_answer, verify_answer
from tools.skills import SkillLibrary

URL = "https://www.vezeeta.com/en/dr/test-molar"


def ctx(answer, prompt="my tooth hurts", urls=(URL,), hits=1):
    return VerificationContext(
        user_prompt=prompt, answer=answer, red_flags=detect_red_flags(prompt),
        language="ar" if any("؀" <= c <= "ۿ" for c in prompt) else "en",
        returned_urls=set(urls), doctor_search_hits=hits,
    )


def failed(report):
    return sorted(c["name"] for c in report.checks if not c["passed"])


def test_grounded_answer_passes():
    assert verify_answer(ctx(f"See **Dr. Test** — [Book]({URL}).")).passed


def test_fabricated_link_fails_with_specific_feedback():
    r = verify_answer(ctx(f"Book: {URL} or https://www.vezeeta.com/en/dr/made-up"))
    assert failed(r) == ["links_are_grounded"]
    assert "made-up" in r.feedback


def test_missing_booking_link_fails():
    assert failed(verify_answer(ctx("Go see Dr. Test, he is great."))) == ["booking_links_present"]


def test_emergency_and_language_checks():
    r = verify_answer(ctx("Please see a cardiologist soon.", prompt="crushing chest pain", urls=(), hits=0))
    assert failed(r) == ["emergency_protocol"]
    r = verify_answer(ctx("You should see a dentist.", prompt="ضرسي بيوجعني", urls=(), hits=0))
    assert failed(r) == ["language_matches"]


def test_repair_strips_fake_links_and_adds_banner():
    c = ctx("See https://www.vezeeta.com/en/dr/fake", prompt="crushing chest pain", urls=(), hits=0)
    fixed = repair_answer(c)
    assert "fake" not in fixed and "[link removed" in fixed
    assert fixed.startswith("🚨") and "123" in fixed


def test_skill_library_progressive_disclosure():
    lib = SkillLibrary()
    names = {s.name for s in lib.list()}
    assert {"emergency-triage", "doctor-finder", "egyptian-arabic", "medication-info",
            "lab-results", "pediatric-care", "mental-health-support"} <= names
    menu = lib.menu_prompt()
    assert "doctor-finder:" in menu and "## Instructions" not in menu  # level 1 only
    body = lib.load("emergency-triage")
    assert "123" in body and "first-aid.md" in body  # level 2
    assert "Cardiac" in lib.read_reference("emergency-triage", "first-aid.md")  # level 3
    assert lib.read_reference("emergency-triage", "../doctor-finder/SKILL.md").startswith("Blocked")
    assert lib.load("nope").startswith("No such skill")
    for skill in lib.list():
        assert "Use when" in skill.description, f"{skill.name} description lacks a trigger"


def test_homepage_link_is_not_treated_as_a_doctor_link():
    r = verify_answer(ctx("Search on https://www.vezeeta.com directly.", urls=(), hits=0))
    assert r.passed
