---
name: lab-results
description: Explain blood test or lab results in plain language and point to the right specialist. Use when the patient shares lab values or asks what a test result means (e.g. CBC, hemoglobin, HbA1c, cholesterol, TSH, liver or kidney function, vitamin D).
---

# Lab Results

## Instructions
1. Read the reference ranges with `read_skill_reference` (skill `lab-results`,
   file `reference-ranges.md`).
2. For each value the patient gave: say whether it is below, within or above the
   typical adult range, in one plain sentence. If the patient's lab printed its own
   range, that range wins.
3. Name the specialist who interprets that test and offer to find one with
   `search_doctors`.

## Rules
- Ranges differ by lab, age, sex and pregnancy. Always say so once.
- No diagnosis: "this can be seen in X, your doctor will confirm" at most.
