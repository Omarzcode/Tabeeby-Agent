---
name: pediatric-care
description: Guide parents about a baby's or child's symptoms and find a pediatrician. Use when the patient is a child, infant or newborn (e.g. "my son", "my baby", "ابني", "بنتي", "البيبي").
---

# Pediatric Care

## Instructions
1. Check the warning signs in `warning-signs.md` (call `read_skill_reference`,
   skill `pediatric-care`). If any applies, start with the emergency instruction
   (call 123 or go to a pediatric emergency room) before anything else.
2. Ask nothing you can avoid; if the age is missing and it matters, search first
   and ask the age at the end.
3. Call `search_doctors` with specialty `Pediatrics` (or the matching pediatric
   subspecialty) and present the doctors with booking links.
4. Give 2–3 safe home-care tips that do not involve medicine doses.

## Rules
- Never suggest a medicine dose for a child.
- Babies under 3 months with fever are always urgent.
