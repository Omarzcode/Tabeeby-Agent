---
name: medication-info
description: Answer questions about a specific medicine, its side effects, or drug interactions using a web search with sources. Use when the patient names a drug or asks what a medicine is for, its side effects, or whether two medicines can be taken together.
---

# Medication Info

## Instructions
1. Call `web_search` with the medicine name plus the exact question (e.g.
   "metformin side effects", "ibuprofen and warfarin interaction").
2. Summarise in at most 5 bullet points, citing the source URL after each fact
   you took from the search results.
3. Then call `read_skill_reference` with skill `medication-info` and file
   `safety-rules.md` and apply every rule in it.
4. Recommend the relevant specialist and offer to find one with `search_doctors`.

## Rules
- Never give a personal dose, never tell the patient to start, stop or change a
  medicine — that is for their doctor or pharmacist.
- If the search returned nothing useful, say so; do not answer from memory alone.
