"""Tabeeby as an MCP server.

Any MCP client (Claude Desktop, Cursor, VS Code, another agent) can use
Tabeeby's capabilities without importing a line of our Python:

Tools
    search_doctors     hybrid vector + filter search over 17K Vezeeta doctors
    check_emergency    deterministic red-flag scan (English + Egyptian Arabic)
    medical_web_search Tavily-backed medical lookup
    list_areas / list_specialties   canonical filter values
    ask_tabeeby        the full agent loop (skills, budget, self-verification)

Resources
    tabeeby://emergency-numbers
    tabeeby://skills              the skill menu (Level 1)
    tabeeby://skills/{name}       one skill's instructions (Level 2)

Prompts
    triage(symptoms, area, max_fee)

Two transports are served from the same definition:
- stdio            `python -m mcp_server`   (local clients launch it)
- streamable HTTP  mounted by FastAPI at `/mcp` (remote clients connect to it)

Dependencies are resolved lazily through `ServiceContainer`, because the
FastAPI app mounts the server before its lifespan has built the clients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from agent.safety import EMERGENCY_NUMBERS, detect_red_flags
from tools.catalog import _catalog, normalize_area, normalize_specialty


@dataclass
class ServiceContainer:
    """Late-bound dependencies shared by the MCP server and the web app."""

    doctor_tools: Any = None
    skills: Any = None
    agent: Any = None
    web_search: Any = None


INSTRUCTIONS = (
    "Tabeeby is a medical navigation assistant for Egypt backed by the Vezeeta "
    "catalogue of ~17,000 doctors. Always run check_emergency on the patient's "
    "words first; if it reports an emergency, tell them to call 123. Use "
    "search_doctors to find doctors and only cite doctors and profile_url links "
    "it returns. Tabeeby does not diagnose or prescribe."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


def build_mcp_server(services: ServiceContainer) -> MCPServer:
    mcp = MCPServer(name="tabeeby", title="Tabeeby Medical Navigator", instructions=INSTRUCTIONS, version="2.0.0")

    @mcp.tool(annotations=READ_ONLY)
    def search_doctors(
        query: str,
        area: Optional[str] = None,
        min_fee: Optional[int] = None,
        max_fee: Optional[int] = None,
        specialty: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Find Vezeeta doctors in Egypt for a symptom or condition.

        Args:
            query: The complaint in a few English words, e.g. 'knee pain after football'.
            area: Optional area/city in English or Arabic, e.g. 'Maadi', 'مدينة نصر'.
            min_fee: Optional minimum consultation fee in EGP.
            max_fee: Optional maximum consultation fee in EGP.
            specialty: Optional specialty, e.g. 'Orthopedics'.
            limit: How many doctors to return (1-10).
        """
        if services.doctor_tools is None:
            return json.dumps({"error": "Doctor search is not configured on this server."})
        result = services.doctor_tools.search_doctors(
            query=query, area=area, min_fee=min_fee, max_fee=max_fee, specialty=specialty, limit=limit,
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def check_emergency(text: str) -> str:
        """Scan a patient's message for life-threatening red-flag symptoms (English and Arabic).

        Args:
            text: The patient's own words.
        """
        return json.dumps(detect_red_flags(text).as_dict(), ensure_ascii=False)

    @mcp.tool(annotations=READ_ONLY)
    def medical_web_search(query: str, max_results: int = 3) -> str:
        """Search the web for medical information (conditions, medicines, interactions) with source URLs.

        Args:
            query: Search keywords, e.g. 'metformin side effects'.
            max_results: Number of results (1-5).
        """
        if services.web_search is None:
            return json.dumps({"error": "Web search is not configured on this server."})
        return services.web_search(query=query, max_results=max(1, min(int(max_results), 5)))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def list_specialties() -> str:
        """List every Vezeeta specialty with its number of doctors."""
        return json.dumps(_catalog()["specialties"], ensure_ascii=False)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def list_areas(contains: Optional[str] = None) -> str:
        """List Vezeeta clinic areas with their number of doctors, or resolve a free-text area.

        Args:
            contains: Optional text; returns the canonical match and areas containing it.
        """
        areas = _catalog()["areas"]
        if not contains:
            return json.dumps(areas, ensure_ascii=False)
        needle = contains.strip().lower()
        return json.dumps({
            "canonical_match": normalize_area(contains),
            "areas": {a: n for a, n in areas.items() if needle in a.lower()},
        }, ensure_ascii=False)

    @mcp.tool(annotations=READ_ONLY)
    def ask_tabeeby(question: str) -> str:
        """Ask the full Tabeeby agent (triage + doctor search + safety checks) and get its final answer.

        Args:
            question: The patient's message in English or Arabic.
        """
        if services.agent is None:
            return "The Tabeeby agent is not configured on this server."
        result = services.agent.run(prompt=question)
        return result.answer

    @mcp.resource("tabeeby://emergency-numbers", mime_type="application/json")
    def emergency_numbers() -> str:
        """Emergency phone numbers in Egypt."""
        return json.dumps(EMERGENCY_NUMBERS, ensure_ascii=False)

    @mcp.resource("tabeeby://skills", mime_type="application/json")
    def skill_menu() -> str:
        """The Tabeeby skill menu: names, descriptions and reference files."""
        if services.skills is None:
            return "[]"
        return json.dumps([s.as_dict() for s in services.skills.list()], ensure_ascii=False)

    @mcp.resource("tabeeby://skills/{name}", mime_type="text/markdown")
    def skill_body(name: str) -> str:
        """The full instructions of one Tabeeby skill."""
        if services.skills is None:
            return "Skills are not configured."
        return services.skills.load(name)

    @mcp.prompt()
    def triage(symptoms: str, area: str = "", max_fee: str = "") -> str:
        """Triage a patient's symptoms and find suitable doctors with Tabeeby's tools."""
        constraints = []
        if area:
            constraints.append(f"area: {area}")
        if max_fee:
            constraints.append(f"max fee: {max_fee} EGP")
        spec_hint = normalize_specialty(symptoms)
        return (
            f"A patient in Egypt says: \"{symptoms}\".\n"
            + (f"Constraints: {', '.join(constraints)}.\n" if constraints else "")
            + (f"Likely specialty: {spec_hint}.\n" if spec_hint else "")
            + "1. Call check_emergency with their words; if it is an emergency, lead with 'call 123'.\n"
            "2. Decide the specialty and call search_doctors.\n"
            "3. Present each doctor with fee, area, reviews and the exact profile_url.\n"
            "4. Do not diagnose or prescribe; add a one-line disclaimer."
        )

    return mcp
