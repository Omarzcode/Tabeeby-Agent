---
name: egyptian-arabic
description: Reply naturally in Egyptian Arabic and understand Egyptian colloquial health terms. Use when the patient writes in Arabic, especially Egyptian dialect or slang (e.g. "عندي", "بيوجعني", "سخن", "ضهري").
---

# Egyptian Arabic

## Instructions
1. Reply in the same register the patient used: Egyptian colloquial if they wrote
   colloquially, Modern Standard Arabic if they wrote formally.
2. If a word is unclear, call `read_skill_reference` with skill `egyptian-arabic`
   and file `glossary.md`.
3. When calling tools, translate the complaint into a short English clinical
   `query` (the doctor catalogue is in English). Keep area names as the patient
   wrote them — the tool understands Arabic area names.
4. Keep doctor names, fees and links exactly as returned; write the rest in Arabic.

## Rules
- Be warm and simple: short sentences, no heavy medical jargon.
- Use Arabic numerals or Western digits consistently within one answer.
