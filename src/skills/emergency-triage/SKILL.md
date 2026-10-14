---
name: emergency-triage
description: Handle possibly life-threatening symptoms. Use when the patient mentions severe chest pain, stroke signs, trouble breathing, fainting or seizures, heavy bleeding, a serious accident, poisoning or overdose, a severe allergic reaction, or thoughts of suicide or self-harm.
---

# Emergency Triage

## Instructions
1. **First line of your answer**: tell the patient to call the ambulance on **123**
   (Egypt) or go to the nearest emergency room NOW. No greeting before it.
2. Call `read_skill_reference` with skill `emergency-triage` and file
   `red-flags.md` to confirm which red-flag category applies, then `first-aid.md`
   for the matching "while you wait" steps.
3. Give at most 4 short, numbered "while you wait" steps from `first-aid.md`.
   Never invent first-aid steps that are not in the reference.
4. Only AFTER the emergency instructions, you may call `search_doctors` for the
   follow-up specialist (e.g. Cardiology after a suspected heart attack) and list
   up to 3 doctors with their booking links, clearly labelled "for follow-up after
   emergency care".
5. Reply in the patient's language.

## Rules
- Never tell the patient to wait for a clinic appointment instead of emergency care.
- Never estimate whether it is "probably nothing".
- Self-harm or suicide: be warm and non-judgemental, tell them they are not alone,
  urge them to call 123 or go to the nearest emergency room, and to stay with
  someone they trust. Do not search for doctors in the same message unless asked.
