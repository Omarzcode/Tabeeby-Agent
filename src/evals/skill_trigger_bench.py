"""Does the RIGHT skill fire? (Workshop 5 trigger bench, for Tabeeby's skills.)

Asks the model for its FIRST move on each prompt and checks which skill it
loads. When a row fails, fix the skill's `description` — not Python.

    cd src && python -m evals.skill_trigger_bench
"""

from typing import List, Optional, Tuple

from helpers.bootstrap import build_services

CASES: List[Tuple[str, Optional[str]]] = [
    # It should just work
    ("I need a dentist in Maadi, my wisdom tooth hurts", "doctor-finder"),
    ("my father has crushing chest pain and is sweating", "emergency-triage"),
    # Arabic / Egyptian dialect
    ("ابني عنده سخونية من يومين", "pediatric-care"),
    ("عندي حموضة كل يوم بعد الأكل، أروح لمين؟", None),  # doctor-finder or egyptian-arabic are both fine
    # Two skills can claim it
    ("Can I take ibuprofen with my blood pressure medicine?", "medication-info"),
    ("My TSH came back 6.1, is that bad?", "lab-results"),
    ("I feel hopeless and can't sleep for weeks", "mental-health-support"),
    # Nothing should fire
    ("hi", "NONE"),
    ("what is the capital of Egypt?", "NONE"),
]


def first_skill(agent, prompt: str) -> Tuple[str, Optional[str]]:
    messages = [
        {"role": "system", "content": agent.build_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    response = agent.client.generate_response(prompt="", chat_history=messages, tools=agent._provider_tools(),
                                              temperature=0)
    calls = getattr(getattr(response, "message", None), "tool_calls", None) or []
    for call in calls:
        name, args = agent._extract_tool_call_info(call)
        if name == "load_skill":
            return "skill", str(args.get("name", "")).strip()
        return "tool", name
    return "answer", None


def main() -> None:
    services = build_services(connect_mcp=False)
    agent = services.agent
    print(f"Menu: {[s.name for s in services.skills.list()]}\n")
    ok_count = 0
    for prompt, expected in CASES:
        kind, got = first_skill(agent, prompt)
        if expected is None:
            ok = True  # informational row
        elif expected == "NONE":
            ok = kind != "skill"
        else:
            ok = got == expected
        ok_count += ok
        label = {"skill": str(got), "tool": f"{got} (tool)", "answer": "answered"}[kind]
        print(f"  {'OK' if ok else 'XX'}  {prompt[:55]:<57} expected={expected or 'any':<22} got={label}")
    print(f"\nScore: {ok_count}/{len(CASES)}")
    services.close()


if __name__ == "__main__":
    main()
