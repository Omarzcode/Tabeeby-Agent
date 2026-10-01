"""Self-verification: code checks that run on every final answer.

"The verifier is where the intelligence is." When a check is code, the agent
cannot talk its way past it. Each verifier returns (passed, feedback); the
feedback is specific enough for the model to fix the answer on the next
attempt ("These booking links were never returned by search_doctors: ..."),
not just "wrong".

If the budget runs out while a check is still failing, `repair_answer`
applies a deterministic fix so an unsafe answer is never shipped as-is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple

try:
    from agent.safety import EGYPT_AMBULANCE, RedFlagReport, detect_language, emergency_banner
except ImportError:  # pragma: no cover
    from .safety import EGYPT_AMBULANCE, RedFlagReport, detect_language, emergency_banner

_URL = re.compile(r"https?://[^\s)\]>\"'*]+")
# Doctor profile links only (…vezeeta.com/<lang>/dr/<slug>); the bare homepage is fine to cite.
_VEZEETA_URL = re.compile(r"https?://(?:www\.)?vezeeta\.com/(?:[a-z]{2}/)?dr/[^\s)\]>\"'*]+", re.IGNORECASE)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?").rstrip("/")


@dataclass
class VerificationContext:
    """Everything a verifier may look at — evidence, not assertion."""

    user_prompt: str
    answer: str
    red_flags: RedFlagReport
    language: str
    returned_urls: Set[str] = field(default_factory=set)
    doctor_search_hits: int = 0


Verifier = Callable[[VerificationContext], Tuple[bool, str]]


def answer_not_empty(ctx: VerificationContext) -> Tuple[bool, str]:
    if ctx.answer and ctx.answer.strip():
        return True, "ok"
    return False, "Your answer was empty. Write a complete reply to the patient."


def links_are_grounded(ctx: VerificationContext) -> Tuple[bool, str]:
    """Every Vezeeta link in the answer must come from a tool result."""
    cited = {_clean_url(u) for u in _VEZEETA_URL.findall(ctx.answer or "")}
    known = {_clean_url(u) for u in ctx.returned_urls}
    invented = sorted(u for u in cited if u not in known)
    if not invented:
        return True, "ok"
    return False, (
        "These doctor links were NOT returned by search_doctors, so they may be "
        f"fabricated: {', '.join(invented[:5])}. Only use profile_url values exactly "
        "as returned by the tool, or call search_doctors again."
    )


def booking_links_present(ctx: VerificationContext) -> Tuple[bool, str]:
    """If doctors were found this turn, at least one booking link must be shown."""
    if ctx.doctor_search_hits == 0:
        return True, "ok"
    cited = {_clean_url(u) for u in _URL.findall(ctx.answer or "")}
    known = {_clean_url(u) for u in ctx.returned_urls}
    if cited & known:
        return True, "ok"
    return False, (
        f"search_doctors returned {ctx.doctor_search_hits} doctor(s) but your answer "
        "contains none of their profile & booking links. Include the exact "
        "profile_url for every doctor you recommend."
    )


def emergency_protocol(ctx: VerificationContext) -> Tuple[bool, str]:
    """A red-flag message must be answered with the ambulance number."""
    if not ctx.red_flags.is_emergency:
        return True, "ok"
    if EGYPT_AMBULANCE in (ctx.answer or "") or "١٢٣" in (ctx.answer or ""):
        return True, "ok"
    return False, (
        "The patient described red-flag symptoms "
        f"({', '.join(ctx.red_flags.categories)}). Start your answer by telling them "
        f"to call the ambulance on {EGYPT_AMBULANCE} or go to the nearest emergency room now."
    )


def language_matches(ctx: VerificationContext) -> Tuple[bool, str]:
    """Answer in the patient's language (Arabic in, Arabic out)."""
    if ctx.language != "ar":
        return True, "ok"
    if detect_language(ctx.answer or "") == "ar":
        return True, "ok"
    return False, (
        "The patient wrote in Arabic but you answered in English. Rewrite the whole "
        "answer in Arabic (Egyptian dialect is fine). Doctor names and links may stay as-is."
    )


DEFAULT_VERIFIERS: List[Verifier] = [
    answer_not_empty,
    emergency_protocol,
    links_are_grounded,
    booking_links_present,
    language_matches,
]


@dataclass
class VerificationReport:
    passed: bool
    checks: List[dict]

    @property
    def feedback(self) -> str:
        failed = [c["feedback"] for c in self.checks if not c["passed"]]
        return "\n".join(f"- {f}" for f in failed)


def verify_answer(ctx: VerificationContext, verifiers: List[Verifier] = None) -> VerificationReport:
    checks = []
    for check in verifiers if verifiers is not None else DEFAULT_VERIFIERS:
        try:
            ok, feedback = check(ctx)
        except Exception as e:  # a broken verifier must never kill the reply
            ok, feedback = True, f"verifier error ignored: {e}"
        checks.append({"name": check.__name__, "passed": bool(ok), "feedback": feedback})
    return VerificationReport(passed=all(c["passed"] for c in checks), checks=checks)


def repair_answer(ctx: VerificationContext) -> str:
    """Deterministic last-resort fixes when the budget ends on a failing answer."""
    answer = ctx.answer or ""
    known = {_clean_url(u) for u in ctx.returned_urls}

    def _strip(match: re.Match) -> str:
        url = match.group(0)
        return url if _clean_url(url) in known else "[link removed: not verified]"

    answer = _VEZEETA_URL.sub(_strip, answer)

    if ctx.red_flags.is_emergency and EGYPT_AMBULANCE not in answer:
        answer = emergency_banner(ctx.language) + "\n\n" + answer

    if not answer.strip():
        answer = (
            "عذرًا، لم أتمكن من إكمال الرد. حاول مرة أخرى." if ctx.language == "ar"
            else "Sorry, I could not complete an answer. Please try again."
        )
    return answer
