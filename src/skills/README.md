# Tabeeby Skills

> A tool is an appliance. A skill is a recipe.

Each folder is one skill. The agent sees only the `name` + `description`
(Level 1, always in the system prompt). It calls `load_skill(name)` to read the
body of `SKILL.md` (Level 2) and `read_skill_reference(skill, file)` for files in
`references/` (Level 3) only when the body tells it to.

**The description is the trigger.** If the wrong skill fires, fix the
description — then re-run `python -m evals.skill_trigger_bench`.

To add a skill: create `skills/<name>/SKILL.md` with frontmatter

```
---
name: <name>
description: <what it does>. Use when <the words a patient actually types>.
---
```

No Python changes are needed; it is discovered at startup.
