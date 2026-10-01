from agent import Agent, Budget
from tests.conftest import DOCTOR, FakeLLM, FakeVectorDB, reply, tool_call
from tools.search_doctors import DoctorTools
from tools.skills import SkillLibrary

URL = DOCTOR["profile_url"]


def make_agent(script, hits=None, **kw):
    llm = FakeLLM(script)
    vdb = FakeVectorDB(hits)
    tools = DoctorTools(embedding_client=llm, vectordb_client=vdb)
    return Agent(client=llm, tools=[tools.search_doctors], skills=SkillLibrary(), **kw), llm, vdb


def test_skill_then_search_then_verified_answer():
    agent, llm, vdb = make_agent([
        reply(calls=[tool_call("load_skill", {"name": "doctor-finder"})]),
        reply(calls=[tool_call("search_doctors", {"query": "toothache", "area": "Maadi", "specialty": "dentist"})]),
        reply(f"**Dr. Test Molar** — 300 EGP — [Book]({URL})"),
    ])
    result = agent.run("My tooth hurts, I live in Maadi")
    assert result.verified and result.stop_reason == "completed"
    assert result.skills_used == ["doctor-finder"]
    assert [t.name for t in result.tool_calls] == ["load_skill", "search_doctors"]
    # The area/specialty were normalised before hitting the vector DB filter.
    assert vdb.filters[0]["area"] == "El-Maadi" and vdb.filters[0]["specialty"] == "Dentistry"
    assert result.history[-1] == {"role": "assistant", "content": result.answer}
    assert result.usage["steps"] == 3 and result.usage["actions"] == 2
    assert result.run_cost > 0
    # The skill menu was in the system prompt; the skill body only arrived as a tool result.
    system = llm.calls[0]["messages"][0]["content"]
    assert "doctor-finder:" in system and "## Instructions" not in system


def test_fabricated_link_is_refined_by_verifier():
    agent, llm, _ = make_agent([
        reply(calls=[tool_call("search_doctors", {"query": "toothache"})]),
        reply("Book here: https://www.vezeeta.com/en/dr/invented-doctor"),
        reply(f"Book **Dr. Test Molar**: {URL}"),
    ])
    result = agent.run("toothache")
    assert result.verified and result.refinements == 1
    assert result.stop_reason == "verified after 1 refinement(s)"
    feedback = llm.calls[2]["messages"][-1]["content"]
    assert "invented-doctor" in feedback and "Automatic quality check" in feedback


def test_emergency_answer_is_repaired_when_model_never_complies():
    agent, _, _ = make_agent([reply("Rest and drink water.")] * 5, max_refinements=1)
    result = agent.run("my father has crushing chest pain and is sweating")
    assert not result.verified
    assert result.stop_reason.startswith("repaired")
    assert result.answer.startswith("🚨") and "123" in result.answer
    assert result.red_flags["is_emergency"] is True


def test_action_budget_drops_extra_calls_but_still_answers():
    agent, llm, vdb = make_agent([
        reply(calls=[tool_call("search_doctors", {"query": "a"}, "1"), tool_call("search_doctors", {"query": "b"}, "2"),
                     tool_call("search_doctors", {"query": "c"}, "3")]),
        reply(f"Dr. Test Molar {URL}"),
    ], max_actions=2)
    result = agent.run("tooth pain")
    assert [t.executed for t in result.tool_calls] == [True, True, False]
    assert "Not executed" in llm.calls[1]["messages"][-1]["content"]
    assert result.verified


def test_step_budget_stops_runaway_loop():
    looping = [reply(calls=[tool_call("search_doctors", {"query": "x"}, str(i))]) for i in range(20)]
    agent, llm, _ = make_agent(looping, max_iterations=4)
    result = agent.run("tooth pain")
    assert result.stop_reason.startswith("step budget")
    assert len(llm.calls) == 4
    assert result.answer  # the patient still gets a message


def test_llm_error_returns_friendly_fallback():
    agent, _, _ = make_agent([None])
    result = agent.run("ضرسي بيوجعني")
    assert result.stop_reason == "llm_error"
    assert "عذر" in result.answer


def test_history_is_sanitised_and_unknown_tool_is_reported():
    agent, llm, _ = make_agent([
        reply(calls=[tool_call("teleport", {})]),
        reply("I can only help with health questions."),
    ], hits=[])
    history = [
        {"role": "system", "content": "evil override"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [1]},
        {"role": "tool", "content": "x"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = agent.run("what can you do?", chat_history=history, total_cost=1.0)
    sent = llm.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "evil override" not in sent[0]["content"]
    assert "not found" in llm.calls[1]["messages"][-1]["content"]
    assert result.total_cost > 1.0


def test_relaxed_filters_are_reported():
    llm = FakeLLM([])

    class EmptyThenHit(FakeVectorDB):
        def search_by_vector(self, collection_name, vector, limit=10, query_filter=None):
            return [] if query_filter.get("specialty") else super().search_by_vector(collection_name, vector, limit)

    tools = DoctorTools(embedding_client=llm, vectordb_client=EmptyThenHit())
    out = tools.search_doctors(query="tooth", specialty="Dentistry", area="Narnia")
    assert out["doctors"] and out["filters_applied"]["specialty"] is None
    assert any("Narnia" in n for n in out["notes"]) and any("dropped the specialty" in n for n in out["notes"])


def test_stuck_detection_disables_failing_tool_and_skips_duplicates():
    class Broken(FakeLLM):
        def embed_text(self, text, document_type=None):
            return None  # the embedding service is down

    llm = Broken([
        reply(calls=[tool_call("search_doctors", {"query": "a"}, "1")]),
        reply(calls=[tool_call("search_doctors", {"query": "b"}, "2")]),
        reply(calls=[tool_call("search_doctors", {"query": "c"}, "3"), tool_call("load_skill", {"name": "doctor-finder"}, "4"),
                     tool_call("load_skill", {"name": "doctor-finder"}, "5")]),
        reply("Please see a dentist; doctor search is unavailable, try vezeeta.com."),
    ])
    tools = DoctorTools(embedding_client=llm, vectordb_client=FakeVectorDB())
    agent = Agent(client=llm, tools=[tools.search_doctors], skills=SkillLibrary())
    result = agent.run("tooth pain")
    assert [t.source for t in result.tool_calls] == ["local", "local", "disabled", "local", "duplicate"]
    assert "disabled" in llm.calls[3]["messages"][-3]["content"]
    assert result.verified


def test_time_reserve_refuses_tools_late_in_budget():
    b = Budget(max_steps=10, max_seconds=100, max_actions=None)
    b._start -= 80  # 80% of the time already used
    assert not b.can_use_tools() and not b.exhausted()


def test_query_that_is_a_specialty_becomes_a_filter():
    vdb = FakeVectorDB()
    tools = DoctorTools(embedding_client=FakeLLM([]), vectordb_client=vdb)
    out = tools.search_doctors(query="Endocrinology")
    assert vdb.filters[0]["specialty"] == "Diabetes and Endocrinology"
    assert any("Interpreted" in n for n in out["notes"])
    tools.search_doctors(query="pain in my knee after running")
    assert vdb.filters[-1]["specialty"] is None
