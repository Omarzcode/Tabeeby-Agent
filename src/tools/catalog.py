"""Canonical Vezeeta areas and specialties, plus a normaliser for each.

Qdrant payload filters use exact keyword matching, so the value the LLM
passes as `area` or `specialty` must equal the stored payload byte-for-byte.
Patients (and the LLM) say "Maadi", "6 October", "مدينة نصر" or
"Orthopedic Surgery" — none of which match. This module maps free text to
the canonical value (alias table first, then fuzzy matching).

`data/catalog.json` is generated from `assets/files/vezeeta.csv` and holds
every distinct `address` / `Speciality` with its doctor count.
"""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

AREA_ALIASES: Dict[str, str] = {
    # Cairo
    "nasr city": "Nasr City", "madinet nasr": "Nasr City", "مدينة نصر": "Nasr City", "مدينه نصر": "Nasr City",
    "new cairo": "New Cairo", "fifth settlement": "New Cairo", "5th settlement": "New Cairo", "tagamoa": "New Cairo",
    "التجمع": "New Cairo", "التجمع الخامس": "New Cairo", "القاهرة الجديدة": "New Cairo",
    "heliopolis": "Heliopolis", "masr el gedida": "Heliopolis", "مصر الجديدة": "Heliopolis",
    "maadi": "El-Maadi", "el maadi": "El-Maadi", "المعادي": "El-Maadi",
    "mohandessin": "Dokki And Mohandessin", "mohandeseen": "Dokki And Mohandessin", "dokki": "Dokki And Mohandessin",
    "المهندسين": "Dokki And Mohandessin", "الدقي": "Dokki And Mohandessin",
    "zamalek": "El-Zamalek", "الزمالك": "El-Zamalek",
    "mokattam": "El-Mokattam", "المقطم": "El-Mokattam",
    "rehab": "El-Rehab", "الرحاب": "El-Rehab", "مدينتي": "Madinaty",
    "shorouk": "El-Shorouk", "الشروق": "El-Shorouk", "obour": "El-Obour City", "العبور": "El-Obour City",
    "shubra": "Shoubra", "شبرا": "Shoubra", "شبرا الخيمة": "Shoubra El-Kheima",
    "ain shams": "Ain Shams", "عين شمس": "Ain Shams", "helwan": "Helwan", "حلوان": "Helwan",
    "manyal": "El-Manyal", "المنيل": "El-Manyal", "abbasia": "El-Abbasia", "العباسية": "El-Abbasia",
    "downtown": "West El-Balad", "wust el balad": "West El-Balad", "وسط البلد": "West El-Balad",
    "kattameya": "El-Kattameya", "القطامية": "El-Kattameya", "matareya": "El-Matareya", "المطرية": "El-Matareya",
    "zaitoun": "El-Zaitoun", "الزيتون": "El-Zaitoun", "hadayek el kobba": "Hadayek El-Kobba", "حدائق القبة": "Hadayek El-Kobba",
    "sayeda zeinab": "El-Sayeda Zainab", "السيدة زينب": "El-Sayeda Zainab", "marg": "El-Marg", "المرج": "El-Marg",
    # Giza
    "6 october": "6Th Of October", "6th of october": "6Th Of October", "october": "6Th Of October",
    "اكتوبر": "6Th Of October", "أكتوبر": "6Th Of October", "6 اكتوبر": "6Th Of October", "٦ أكتوبر": "6Th Of October",
    "sheikh zayed": "El-Sheikh Zayed", "zayed": "El-Sheikh Zayed", "الشيخ زايد": "El-Sheikh Zayed", "زايد": "El-Sheikh Zayed",
    "faisal": "Faisal", "فيصل": "Faisal", "haram": "El-Haram", "الهرم": "El-Haram",
    "hadayek el ahram": "Hadayek El-Ahram", "حدائق الأهرام": "Hadayek El-Ahram", "حدائق الاهرام": "Hadayek El-Ahram",
    "giza": "El-Giza", "الجيزة": "El-Giza", "imbaba": "Imbaba", "إمبابة": "Imbaba", "امبابة": "Imbaba",
    "agouza": "El-Agouza", "العجوزة": "El-Agouza",
    # Alexandria & Delta & others
    "smouha": "Smouha", "سموحة": "Smouha", "sidi bishr": "Sidy Bishr", "سيدي بشر": "Sidy Bishr",
    "mansoura": "El-Mansoura", "المنصورة": "El-Mansoura", "tanta": "Tanta", "طنطا": "Tanta",
    "zagazig": "El-Zagazig", "الزقازيق": "El-Zagazig", "mahalla": "Mahalla", "المحلة": "Mahalla",
    "shebin el kom": "Shibin El-Kom", "شبين الكوم": "Shibin El-Kom", "assiut": "Assiut City", "أسيوط": "Assiut City",
    "sohag": "Sohag City", "سوهاج": "Sohag City", "hurghada": "Hurghada City", "الغردقة": "Hurghada City",
    "banha": "Banha", "بنها": "Banha", "10th of ramadan": "10Th Of Ramadan", "العاشر من رمضان": "10Th Of Ramadan",
    "damanhour": "Damanhur", "دمنهور": "Damanhur", "qena": "Qena City", "قنا": "Qena City",
    "minya": "El Minya City", "المنيا": "El Minya City", "suez": "Suez City", "السويس": "Suez City",
    "fayoum": "El-Fayoum City", "الفيوم": "El-Fayoum City", "ismailia": "El-Ismailia City", "الإسماعيلية": "El-Ismailia City",
    "port said": "Port Said City", "بورسعيد": "Port Said City", "luxor": "Luxor City", "الأقصر": "Luxor City",
    "aswan": "Aswan", "أسوان": "Aswan", "beni suef": "Beni Suef City", "بني سويف": "Beni Suef City",
    "agamy": "El-Agamy", "العجمي": "El-Agamy", "miami": "Miamy", "ميامي": "Miamy",
}

SPECIALTY_ALIASES: Dict[str, str] = {
    "dentist": "Dentistry", "dental": "Dentistry", "teeth": "Dentistry", "أسنان": "Dentistry", "اسنان": "Dentistry",
    "orthopedic": "Orthopedics", "orthopaedics": "Orthopedics", "orthopedic surgery": "Orthopedics",
    "bones": "Orthopedics", "عظام": "Orthopedics",
    "cardiologist": "Cardiology", "heart": "Cardiology", "قلب": "Cardiology",
    "pediatrician": "Pediatrics", "paediatrics": "Pediatrics", "children": "Pediatrics", "أطفال": "Pediatrics", "اطفال": "Pediatrics",
    "gynecology": "Reproductive Medicine & Infertility", "gynaecology": "Reproductive Medicine & Infertility",
    "obstetrics": "Reproductive Medicine & Infertility", "ob/gyn": "Reproductive Medicine & Infertility",
    "infertility": "Reproductive Medicine & Infertility", "نسا وتوليد": "Reproductive Medicine & Infertility",
    "نساء وتوليد": "Reproductive Medicine & Infertility", "حقن مجهري": "Reproductive Medicine & Infertility",
    "dermatologist": "Dermatology", "skin": "Dermatology", "جلدية": "Dermatology",
    "ear nose throat": "ENT", "otolaryngology": "ENT", "أنف وأذن": "ENT", "انف واذن": "ENT",
    "eye": "Ophthalmology", "eyes": "Ophthalmology", "ophthalmologist": "Ophthalmology", "رمد": "Ophthalmology", "عيون": "Ophthalmology",
    "psychiatrist": "Psychiatry", "mental health": "Psychiatry", "psychology": "Psychiatry", "نفسية": "Psychiatry", "نفسي": "Psychiatry",
    "internist": "Internal Medicine", "باطنة": "Internal Medicine", "باطنية": "Internal Medicine",
    "gastroenterologist": "Gastroenterology", "stomach": "Gastroenterology", "جهاز هضمي": "Gastroenterology",
    "neurologist": "Neurology", "مخ وأعصاب": "Neurology", "مخ واعصاب": "Neurology",
    "urologist": "Urology", "مسالك": "Urology", "مسالك بولية": "Urology",
    "pulmonology": "Chest and Respiratory", "chest": "Chest and Respiratory", "صدر": "Chest and Respiratory",
    "physiotherapy": "Physiotherapy and Sport Injuries", "physical therapy": "Physiotherapy and Sport Injuries",
    "علاج طبيعي": "Physiotherapy and Sport Injuries",
    "nutrition": "Dietitian and Nutrition", "dietitian": "Dietitian and Nutrition", "تغذية": "Dietitian and Nutrition",
    "endocrinology": "Diabetes and Endocrinology", "diabetes": "Diabetes and Endocrinology", "سكر": "Diabetes and Endocrinology",
    "غدد": "Diabetes and Endocrinology", "kidney": "Nephrology", "كلى": "Nephrology",
    "cancer": "Oncology", "أورام": "Oncology", "اورام": "Oncology", "surgeon": "General Surgery", "جراحة": "General Surgery",
    "جراحة عامة": "General Surgery", "plastic": "Plastic Surgery", "تجميل": "Plastic Surgery",
    "vascular": "Vascular Surgery", "أوعية دموية": "Vascular Surgery", "rheumatologist": "Rheumatology", "روماتيزم": "Rheumatology",
    "blood": "Hematology", "أمراض دم": "Hematology", "liver": "Hepatology", "كبد": "Hepatology",
    "gp": "General Practice", "general practitioner": "General Practice", "family doctor": "Family Medicine",
    "hearing": "Audiology", "سمعيات": "Audiology", "allergy": "Allergy and Immunology", "حساسية": "Allergy and Immunology",
    "spine": "Spinal Surgery", "عمود فقري": "Spinal Surgery", "elderly": "Geriatrics", "مسنين": "Geriatrics",
    "neurosurgeon": "Neurosurgery", "جراحة مخ وأعصاب": "Neurosurgery", "pain": "Pain Management", "علاج الألم": "Pain Management",
    "obesity": "Obesity and Laparoscopic Surgery", "سمنة": "Obesity and Laparoscopic Surgery",
}


@lru_cache(maxsize=1)
def _catalog() -> Dict[str, Dict[str, int]]:
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"areas": {}, "specialties": {}}


def all_areas() -> List[str]:
    return list(_catalog()["areas"].keys())


def all_specialties() -> List[str]:
    return list(_catalog()["specialties"].keys())


def _key(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\-_,.]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _resolve(value: Optional[str], canon: List[str], aliases: Dict[str, str], cutoff: float) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    raw = str(value).strip()
    if raw in canon:
        return raw
    key = _key(raw)
    by_key = {_key(c): c for c in canon}
    if key in by_key:
        return by_key[key]
    alias_keys = {_key(a): c for a, c in aliases.items()}
    if key in alias_keys:
        return alias_keys[key]
    for k in (key.removeprefix("el "), "el " + key):
        if k in by_key:
            return by_key[k]
    # Fuzzy: canonical names first, then aliases.
    match = difflib.get_close_matches(key, list(by_key) + list(alias_keys), n=1, cutoff=cutoff)
    if match:
        return by_key.get(match[0]) or alias_keys.get(match[0])
    return None


def normalize_area(area: Optional[str]) -> Optional[str]:
    """Map free-text area (English/Arabic, any spelling) to a canonical Vezeeta area."""
    return _resolve(area, all_areas(), AREA_ALIASES, cutoff=0.8)


def normalize_specialty(specialty: Optional[str]) -> Optional[str]:
    """Map a free-text specialty to the canonical Vezeeta specialty name."""
    return _resolve(specialty, all_specialties(), SPECIALTY_ALIASES, cutoff=0.75)
