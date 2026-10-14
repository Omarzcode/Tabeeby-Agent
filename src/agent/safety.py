"""Deterministic medical safety checks.

The LLM is good at triage but it is not a guarantee. These checks are plain
code: they run on every message, cost nothing, and cannot be talked out of
their answer. They feed both the agent (a pre-flight emergency note) and the
self-verification step (an emergency answer MUST contain the ambulance number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

EGYPT_AMBULANCE = "123"

EMERGENCY_NUMBERS: Dict[str, str] = {
    "Ambulance (Egypt)": "123",
    "Police": "122",
    "Fire": "180",
}

# Category -> phrases. Phrases are matched case-insensitively as substrings
# after whitespace normalisation, so keep them specific to avoid false alarms
# ("chest pain" alone is NOT a red flag; "crushing chest pain" is).
RED_FLAGS: Dict[str, List[str]] = {
    "cardiac": [
        "crushing chest pain", "severe chest pain", "chest pain spreading",
        "pain radiating to my arm", "heart attack", "chest pressure and sweating",
        "ألم شديد في الصدر", "وجع شديد في الصدر", "ذبحة", "جلطة في القلب",
        "أزمة قلبية", "ازمة قلبية",
    ],
    "stroke": [
        "face drooping", "slurred speech", "sudden weakness on one side",
        "can't move one side", "sudden numbness", "stroke",
        "جلطة في المخ", "جلطة دماغية", "سكتة دماغية", "شلل مفاجئ",
        # Egyptian dialect, masculine / feminine / first person
        "وشه مايل", "وشها مايل", "وشي مايل", "وجهه مائل", "وجهها مائل",
        "لسانه تقيل", "لسانها تقيل", "لساني تقيل", "كلامه مش مفهوم", "كلامها مش مفهوم",
        "مش قادر يحرك", "مش قادرة تحرك", "مش قادر احرك", "مش قادرة احرك",
    ],
    "breathing": [
        "can't breathe", "cannot breathe", "unable to breathe", "choking",
        "severe shortness of breath", "lips turning blue",
        "مش قادر اتنفس", "مش قادر أتنفس", "مش قادرة اتنفس", "مش قادرة أتنفس", "مش عارف اتنفس",
        "مش قادر يتنفس", "مش قادرة تتنفس", "اختناق", "بيتخنق", "بتتخنق",
        "ضيق شديد في التنفس",
    ],
    "consciousness": [
        "unconscious", "passed out and not waking", "not responding",
        "seizure", "convulsing",
        "فاقد الوعي", "فاقدة الوعي", "فقد الوعي", "فقدت الوعي", "مغمى عليه", "مغمى عليها",
        "مش بيفوق", "مش بتفوق", "تشنجات", "بيتشنج", "بتتشنج",
    ],
    "bleeding_trauma": [
        "heavy bleeding", "won't stop bleeding", "uncontrolled bleeding",
        "severe bleeding", "car accident", "head injury", "stab wound",
        "نزيف شديد", "نزيف مش بيقف", "حادثة عربية", "حادث سيارة", "اتخبط في دماغه",
    ],
    "anaphylaxis": [
        "anaphylaxis", "throat swelling", "throat is closing", "tongue swelling",
        "حساسية شديدة", "زوري بيقفل", "لساني ورم",
    ],
    "poisoning": [
        "overdose", "swallowed poison", "drank bleach", "took too many pills",
        "تسمم", "شرب كلور", "بلع حبوب كتير", "جرعة زائدة",
    ],
    "self_harm": [
        "kill myself", "suicide", "end my life", "want to die",
        "انتحار", "انتحر", "عايز اموت", "عاوز أموت", "أنهي حياتي",
    ],
}

_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_LATIN_CHAR = re.compile(r"[A-Za-z]")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


@dataclass
class RedFlagReport:
    """Result of scanning one message for life-threatening symptoms."""

    categories: List[str]
    matches: List[str]

    @property
    def is_emergency(self) -> bool:
        return bool(self.categories)

    def as_dict(self) -> dict:
        return {
            "is_emergency": self.is_emergency,
            "categories": self.categories,
            "matches": self.matches,
            "action": (
                f"Call the ambulance on {EGYPT_AMBULANCE} or go to the nearest "
                "emergency room now." if self.is_emergency else
                "No red-flag symptoms detected. Book a regular consultation."
            ),
            "emergency_numbers": EMERGENCY_NUMBERS,
        }


def detect_red_flags(text: str) -> RedFlagReport:
    """Scan text for emergency red-flag phrases (English, MSA and Egyptian Arabic)."""
    norm = _normalise(text)
    categories: List[str] = []
    matches: List[str] = []
    for category, phrases in RED_FLAGS.items():
        for phrase in phrases:
            if phrase.lower() in norm:
                matches.append(phrase)
                if category not in categories:
                    categories.append(category)
    return RedFlagReport(categories=categories, matches=matches)


def detect_language(text: str) -> str:
    """Return 'ar' when Arabic script dominates the letters in text, else 'en'."""
    arabic = len(_ARABIC_CHAR.findall(text or ""))
    latin = len(_LATIN_CHAR.findall(text or ""))
    if arabic == 0 and latin == 0:
        return "en"
    return "ar" if arabic >= latin else "en"


def emergency_banner(language: str = "en") -> str:
    """A fixed emergency notice used when the LLM failed to include one."""
    if language == "ar":
        return (
            f"🚨 تنبيه طوارئ: الأعراض دي ممكن تكون خطيرة. اتصل بالإسعاف فورًا على "
            f"{EGYPT_AMBULANCE} أو روح أقرب طوارئ حالًا."
        )
    return (
        f"🚨 EMERGENCY: these symptoms can be life-threatening. Call the ambulance "
        f"on {EGYPT_AMBULANCE} or go to the nearest emergency room right now."
    )
