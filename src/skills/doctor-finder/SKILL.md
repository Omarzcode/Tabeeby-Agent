---
name: doctor-finder
description: Find and recommend the right Vezeeta doctors for a complaint. Use when the patient describes a non-emergency symptom, pain, injury or condition, or asks for a doctor, clinic, specialist, cheap or nearby doctor, or a booking in an Egyptian area.
---

# Doctor Finder

## Instructions
1. Decide the specialty. If unsure, call `read_skill_reference` with skill
   `doctor-finder` and file `specialty-map.md`.
2. Pull the constraints out of the message: area (any spelling — the tool
   normalises "Maadi", "مدينة نصر", "6 October"), and budget (`max_fee` in EGP).
   Do NOT ask a follow-up question first if the complaint is clear; search, then
   offer to refine.
3. Call `search_doctors` with a short clinical `query` (the symptom in English),
   the `specialty`, and `area` / `max_fee` only if the patient gave them.
4. If the result has `notes` (e.g. filters were relaxed), tell the patient in one line.
5. Present up to 5 doctors using the card format in `card-format.md`
   (read it with `read_skill_reference` the first time in a conversation).
6. End with one line of practical next steps and a short disclaimer.

## Rules
- Every doctor, fee and link must come from the `search_doctors` result. Copy
  `profile_url` exactly; never build or guess a Vezeeta URL.
- Sort by relevance as returned; you may mention the lowest fee or most reviewed.
- No diagnosis and no prescriptions.
