"""If you can't measure it, you can't ship it.

Runs the real agent (real LLM, real Qdrant) over `eval_set.json` and scores
every case with CODE, never with another LLM's opinion:

    cd src && python -m evals.run_eval                 # all cases
    cd src && python -m evals.run_eval --only cardiac  # cases whose id contains 'cardiac'

Writes a JSON report to evals/reports/. A FAIL is a question, not a verdict:
read the answer first — then fix either the agent (prompt/skill/tool) or the test.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agent.safety import detect_language
from helpers.bootstrap import build_services
from tools.catalog import normalize_area, normalize_specialty

HERE = Path(__file__).parent


def score_case(case: Dict[str, Any], result) -> List[Tuple[str, bool, str]]:
    exp = case["expect"]
    answer = result.answer or ""
    searches = [t for t in result.tool_calls if t.name.endswith("search_doctors") and t.executed]
    checks: List[Tuple[str, bool, str]] = []

    if "calls_search" in exp:
        checks.append(("calls_search", bool(searches) == exp["calls_search"], f"{len(searches)} search call(s)"))
    if "specialty" in exp:
        got = [normalize_specialty(s.arguments.get("specialty")) for s in searches]
        ok = exp["specialty"] in got or any(exp["specialty"].lower() in str(s.arguments).lower() for s in searches)
        checks.append(("specialty", ok, f"searched specialties={got}"))
    if "area" in exp:
        got = [normalize_area(s.arguments.get("area")) for s in searches]
        checks.append(("area", exp["area"] in got, f"searched areas={got}"))
    if "max_fee_at_most" in exp:
        fees = [s.arguments.get("max_fee") for s in searches if s.arguments.get("max_fee") is not None]
        ok = bool(fees) and all(float(f) <= exp["max_fee_at_most"] for f in fees)
        checks.append(("budget", ok, f"max_fee={fees}"))
    if "language" in exp:
        checks.append(("language", detect_language(answer) == exp["language"], f"answer language={detect_language(answer)}"))
    if "emergency" in exp:
        checks.append(("emergency_detected", bool(result.red_flags.get("is_emergency")) == exp["emergency"],
                       f"red_flags={result.red_flags.get('categories', [])}"))
    for needle in exp.get("must_contain", []):
        checks.append((f"contains:{needle}", needle in answer, ""))
    if exp.get("must_contain_any"):
        checks.append(("contains_any", any(n in answer for n in exp["must_contain_any"]), str(exp["must_contain_any"])))
    for needle in exp.get("must_not_contain", []):
        checks.append((f"not_contains:{needle}", needle not in answer, ""))
    if "skill" in exp:
        checks.append(("skill", exp["skill"] in result.skills_used, f"loaded={result.skills_used}"))
    checks.append(("self_verified", result.verified, result.stop_reason))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="run only cases whose id contains this text")
    parser.add_argument("--pause", type=float, default=8.0,
                        help="seconds to wait between cases (free-tier rate limits)")
    args = parser.parse_args()

    cases = [c for c in json.loads((HERE / "eval_set.json").read_text(encoding="utf-8")) if args.only in c["id"]]
    services = build_services()
    if services.agent is None:
        raise SystemExit(f"Agent not available: {services.errors}")

    report = []
    passed_cases = 0
    total_checks = passed_checks = 0
    total_cost = 0.0
    try:
        for i, case in enumerate(cases):
            if i:
                time.sleep(args.pause)
            t0 = time.time()
            result = services.agent.run(prompt=case["prompt"])
            checks = score_case(case, result)
            ok = all(c[1] for c in checks)
            passed_cases += ok
            total_checks += len(checks)
            passed_checks += sum(c[1] for c in checks)
            total_cost += result.run_cost
            print(f"[{'PASS' if ok else 'FAIL'}] {case['id']:<20} {time.time() - t0:5.1f}s  "
                  f"stop={result.stop_reason}  skills={result.skills_used}")
            for name, good, detail in checks:
                if not good:
                    print(f"        ✗ {name}: {detail}")
            report.append({
                "id": case["id"], "prompt": case["prompt"], "passed": ok,
                "checks": [{"name": n, "passed": g, "detail": d} for n, g, d in checks],
                "answer": result.answer, "metadata": result.metadata(),
            })
    finally:
        services.close()

    n = len(cases) or 1
    print(f"\nCases : {passed_cases}/{len(cases)} passed ({100 * passed_cases // n}%)")
    print(f"Checks: {passed_checks}/{total_checks} passed")
    print(f"Cost  : ${total_cost:.4f}")

    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({
        "cases_passed": passed_cases, "cases_total": len(cases),
        "checks_passed": passed_checks, "checks_total": total_checks,
        "cost_usd": total_cost, "results": report,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
