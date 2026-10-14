"""Skills: progressive disclosure of medical know-how.

Three levels, each read only when needed:

| Level | What                          | When it is read                        |
|-------|-------------------------------|----------------------------------------|
| 1     | `name` + `description`        | always — the menu in the system prompt |
| 2     | body of `SKILL.md`            | when the agent calls `load_skill`      |
| 3     | files in `references/`        | when the body says to read them        |

A tool gives the agent an ability; a skill gives it judgement (when, and how).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

logger = logging.getLogger(__name__)


def _split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Return (frontmatter dict, body) for a `--- ... ---` header."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            _, front, body = parts
            meta: Dict[str, str] = {}
            for line in front.strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            return meta, body.strip()
    return {}, text.strip()


@dataclass
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        _, body = _split_frontmatter((self.path / "SKILL.md").read_text(encoding="utf-8"))
        return body

    def references(self) -> List[str]:
        ref_dir = self.path / "references"
        if not ref_dir.is_dir():
            return []
        return sorted(p.name for p in ref_dir.iterdir() if p.is_file())

    def as_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "references": self.references()}


class SkillLibrary:
    """Discovers skills on disk and exposes them to the agent as two tools."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        """Level 1: read ONLY the frontmatter of every skill (the cheap menu)."""
        self.skills = {}
        if not self.skills_dir.is_dir():
            logger.warning("Skills directory not found: %s", self.skills_dir)
            return
        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            meta, _ = _split_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = meta.get("name") or skill_md.parent.name
            self.skills[name] = Skill(name=name, description=meta.get("description", ""), path=skill_md.parent)

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get((name or "").strip())

    def list(self) -> List[Skill]:
        return list(self.skills.values())

    def menu_prompt(self) -> str:
        """The Level-1 menu appended to the system prompt."""
        if not self.skills:
            return ""
        menu = "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())
        return (
            "\n\n### SKILLS (recipes you can load)\n"
            "You have skills available. Menu (name: when to use it):\n" + menu +
            "\n\nWhen the patient's message matches a skill, call `load_skill(name)` FIRST "
            "and follow the returned instructions. A skill may tell you to read a reference "
            "with `read_skill_reference(skill, file)`. A skill is NOT a callable tool; load it. "
            "Load at most two skills per message and do not reload a skill already loaded in "
            "this conversation. If no skill fits (e.g. a greeting), just answer."
        )

    def load(self, name: str) -> str:
        """Level 2: one skill's instructions, on demand."""
        skill = self.get(name)
        if not skill:
            return f"No such skill: {name!r}. Available: {', '.join(self.skills)}"
        refs = skill.references()
        suffix = f"\n\n(References available: {', '.join(refs)})" if refs else ""
        return skill.body() + suffix

    def read_reference(self, skill_name: str, file: str) -> str:
        """Level 3: a reference file, sandboxed to that skill's references/ folder."""
        skill = self.get(skill_name)
        if not skill:
            return f"No such skill: {skill_name!r}."
        ref_dir = (skill.path / "references").resolve()
        target = (ref_dir / (file or "")).resolve()
        if not target.is_relative_to(ref_dir):
            return f"Blocked: {file!r} is outside the skill's references folder."
        if not target.is_file():
            return f"No reference {file!r} in skill {skill_name!r}. Available: {', '.join(skill.references())}"
        return target.read_text(encoding="utf-8")

    def make_tools(self, on_load: Optional[Callable[[str], None]] = None) -> List[Callable[..., str]]:
        """Build the two callables the LLM sees (names and docstrings matter)."""
        library = self

        def load_skill(name: str) -> str:
            """Load a skill's full instructions by name (from the SKILLS menu in the system prompt).

            Args:
                name: The skill name exactly as listed in the menu, e.g. 'doctor-finder'.
            """
            if on_load:
                on_load(name)
            return library.load(name)

        def read_skill_reference(skill: str, file: str) -> str:
            """Read a reference file that a loaded skill told you to read.

            Args:
                skill: The skill name, e.g. 'emergency-triage'.
                file: The reference file name, e.g. 'first-aid.md'.
            """
            return library.read_reference(skill, file)

        return [load_skill, read_skill_reference]
