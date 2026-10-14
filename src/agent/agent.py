"""Tabeeby Medical Agent.

A goal-seeking tool loop with four reliability layers on top of plain
tool calling:

1. Skills   — a menu of medical recipes the model loads on demand
              (`load_skill` / `read_skill_reference`, progressive disclosure).
2. MCP      — tools borrowed from external MCP servers sit next to our own
              Python tools; the loop does not care where a tool lives.
3. Budget   — steps, seconds, tool actions and USD are all capped
              ("a loop without a budget is a bug").
4. Verify   — every final answer is checked by code (grounded links,
              emergency number, language, booking links). Failures are fed
              back as specific feedback and the model refines its answer;
              if the budget ends first, a deterministic repair is applied.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.budget import Budget
from agent.safety import detect_language, detect_red_flags
from agent.verifiers import (
    _VEZEETA_URL,
    DEFAULT_VERIFIERS,
    _clean_url,
    VerificationContext,
    Verifier,
    repair_answer,
    verify_answer,
)
from prompts.prompt_templatet import VEZEETA_SYSTEM_PROMPT
from stores.llm.LLMEnums import OllamaEnums
from stores.llm.LLMInterface import LLMInterface
from stores.llm.pricing import token_cost

MAX_OBSERVATION_CHARS = 6000

VERIFIER_FEEDBACK_TEMPLATE = (
    "[Automatic quality check — this message is from the system, not the patient]\n"
    "Your previous answer failed these checks:\n{feedback}\n"
    "Rewrite the COMPLETE answer for the patient, fixing every point. "
    "Do not mention this check."
)

TOOL_DISABLED_OBSERVATION = (
    "Not executed: '{name}' has failed {n} times in this conversation turn and is now "
    "disabled. Do not call it again. Answer with what you already have and tell the patient "
    "honestly what you could not do (e.g. suggest searching vezeeta.com directly)."
)

DUPLICATE_CALL_OBSERVATION = (
    "Not executed: you already called '{name}' with exactly these arguments; the result is "
    "above. Use it, change the arguments, or write your final answer."
)

MAX_TOOL_FAILURES = 2

BUDGET_SPENT_OBSERVATION = (
    "Not executed: the tool budget for this message is spent. Write your final "
    "answer now using only the information you already have."
)


@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict[str, Any]
    source: str
    executed: bool
    ok: bool


@dataclass
class AgentResult:
    """Everything one agent run produced — the answer plus the evidence."""

    answer: str
    history: List[Dict[str, str]]
    total_cost: float
    run_cost: float
    stop_reason: str
    verified: bool
    refinements: int
    checks: List[dict] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    red_flags: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    usage: Dict[str, Any] = field(default_factory=dict)
    tokens: Dict[str, Any] = field(default_factory=dict)
    doctors: List[Dict[str, Any]] = field(default_factory=list)

    def as_tuple(self) -> Tuple[str, List[Dict[str, str]], float]:
        """Backwards-compatible (answer, history, total_cost) shape."""
        return self.answer, self.history, self.total_cost

    def metadata(self) -> Dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "verified": self.verified,
            "refinements": self.refinements,
            "checks": self.checks,
            "skills_used": self.skills_used,
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "red_flags": self.red_flags,
            "language": self.language,
            "usage": self.usage,
            "tokens": self.tokens,
            "run_cost_usd": round(self.run_cost, 6),
        }


class Agent:
    """Tabeeby's reasoning loop. Stateless per request, safe to share across threads."""

    def __init__(
        self,
        client: LLMInterface,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        skills: Any = None,
        mcp: Any = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 8,
        max_seconds: float = 90.0,
        max_actions: Optional[int] = 12,
        max_cost_usd: Optional[float] = None,
        max_refinements: int = 2,
        verifiers: Optional[List[Verifier]] = None,
        total_cost: float = 0.0,
        input_token_cost: float = 0.0000014,
        output_token_cost: float = 0.0000044,
    ):
        """
        Args:
            client: Generation LLM provider implementing LLMInterface.
            tools: Python callables the model may call (name + docstring + type hints = schema).
            skills: Optional SkillLibrary; adds the skill menu and load_skill/read_skill_reference.
            mcp: Optional McpRegistry; its borrowed tools are offered next to ours.
            max_iterations: Max LLM calls per message (step budget).
            max_seconds: Wall-clock budget per message.
            max_actions: Max tool calls executed per message (None = unlimited).
            max_cost_usd: Optional USD budget per message.
            max_refinements: How many times a failed verification may be retried.
            verifiers: Code checks run on every final answer ([] disables verification).
            input_token_cost / output_token_cost: USD per token, used for cost tracking.
        """
        self.client = client
        self.system_prompt = system_prompt or VEZEETA_SYSTEM_PROMPT
        self.skills = skills
        self.mcp = mcp
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.max_actions = max_actions
        self.max_cost_usd = max_cost_usd
        self.max_refinements = max_refinements
        self.verifiers = DEFAULT_VERIFIERS if verifiers is None else verifiers
        self.total_cost = float(total_cost)
        self.input_token_cost = float(input_token_cost)
        self.output_token_cost = float(output_token_cost)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cost_lock = threading.Lock()

        self.tools: List[Callable[..., Any]] = []
        for t in tools or []:
            if hasattr(t, "search_doctors") and callable(getattr(t, "search_doctors")):
                self.tools.append(t.search_doctors)
            elif callable(t):
                self.tools.append(t)
        if skills is not None:
            self.tools.extend(skills.make_tools())

        self.tools_map: Dict[str, Callable[..., Any]] = {
            getattr(tool, "__name__", str(tool)): tool for tool in self.tools
        }

    # ── public helpers ───────────────────────────────────────────

    def new_budget(self) -> Budget:
        return Budget(
            max_steps=self.max_iterations,
            max_seconds=self.max_seconds,
            max_actions=self.max_actions,
            max_cost_usd=self.max_cost_usd,
        )

    def tool_names(self) -> List[str]:
        names = list(self.tools_map)
        if self.mcp is not None:
            names += [s["function"]["name"] for s in self.mcp.schemas()]
        return names

    def build_system_prompt(self, base: Optional[str] = None) -> str:
        prompt = base or self.system_prompt
        if self.skills is not None:
            prompt += self.skills.menu_prompt()
        return prompt

    def reset_total_cost(self, new_cost: float = 0.0) -> None:
        self.total_cost = float(new_cost)

    # ── internals ────────────────────────────────────────────────

    def _provider_tools(self) -> List[Any]:
        tools: List[Any] = list(self.tools)
        if self.mcp is not None:
            tools.extend(self.mcp.schemas())
        return tools

    def _construct_message(self, prompt: str, role: str, truncate: bool = False) -> Dict[str, Any]:
        if hasattr(self.client, "construct_prompt") and callable(self.client.construct_prompt):
            return self.client.construct_prompt(prompt=prompt, role=role, truncate=truncate)
        return {"role": role, "content": prompt}

    @property
    def model_id(self) -> Optional[str]:
        return getattr(self.client, "generation_model_id", None)

    @staticmethod
    def _response_tokens(response: Any) -> Tuple[int, int, int]:
        """(prompt, completion, cached) token counts from Ollama, OpenAI or Gemini response shapes."""
        if not response:
            return 0, 0, 0

        def _get(obj: Any, name: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        prompt_tokens = _get(response, "prompt_eval_count") or 0
        completion_tokens = _get(response, "eval_count") or 0
        cached_tokens = 0
        usage = _get(response, "usage")
        if usage:
            prompt_tokens = _get(usage, "prompt_tokens") or prompt_tokens
            completion_tokens = _get(usage, "completion_tokens") or completion_tokens
            cached_tokens = _get(usage, "cached_tokens") or 0
        try:
            return int(prompt_tokens), int(completion_tokens), int(cached_tokens)
        except (TypeError, ValueError):
            return 0, 0, 0

    def _calculate_response_cost(self, response: Any) -> float:
        """USD cost of one LLM response: the model's listed price if known, else the configured per-token prices."""
        prompt_tokens, completion_tokens, cached_tokens = self._response_tokens(response)
        return token_cost(
            self.model_id, prompt_tokens, completion_tokens, cached_tokens,
            fallback_input_per_token=self.input_token_cost,
            fallback_output_per_token=self.output_token_cost,
        )

    def _format_doctors_context(self, result: Any) -> str:
        """Format a search_doctors result into compact text for the model."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return result

        notes: List[str] = []
        doctors = result
        if isinstance(result, dict):
            if result.get("error"):
                return f"Error: {result['error']}"
            doctors = result.get("doctors", [])
            notes = result.get("notes") or []
            applied = {k: v for k, v in (result.get("filters_applied") or {}).items() if v is not None}
            if applied:
                notes = [f"Filters applied: {applied}"] + list(notes)

        if not isinstance(doctors, list) or not doctors:
            return "\n".join(notes + ["No matching doctors found for the specified search criteria."])

        formatted = []
        for idx, doc in enumerate(doctors, 1):
            if not isinstance(doc, dict):
                formatted.append(str(doc))
                continue
            lines = [f"Doctor {idx}:"]
            labels = [
                ("name", "Name"), ("specialty", "Specialty"), ("subspecialties_text", "Subspecialties"),
                ("address", "Clinic Location / Area"), ("fee", "Consultation Fee (EGP)"),
                ("reviews_count", "Patient Reviews"), ("waiting_time_min", "Average Waiting Time (min)"),
                ("profile_url", "Profile & Booking URL"),
            ]
            for key, label in labels:
                if doc.get(key) not in (None, ""):
                    lines.append(f"  - {label}: {doc[key]}")
            formatted.append("\n".join(lines))
        return "\n".join(notes + [""]) + "\n\n".join(formatted) if notes else "\n\n".join(formatted)

    @staticmethod
    def _extract_tool_call_info(tool_call: Any) -> Tuple[str, Dict[str, Any]]:
        """Safely extract (function name, argument dict) from any provider's tool call."""
        tool_name = ""
        raw_args: Any = {}
        if hasattr(tool_call, "function"):
            func = tool_call.function
            tool_name = getattr(func, "name", "")
            raw_args = getattr(func, "arguments", {})
        elif isinstance(tool_call, dict):
            func = tool_call.get("function", {})
            if isinstance(func, dict):
                tool_name = func.get("name", "")
                raw_args = func.get("arguments", {})
            else:
                tool_name = tool_call.get("name", "")
                raw_args = tool_call.get("arguments", {})
        else:
            tool_name = getattr(tool_call, "name", "")
            raw_args = getattr(tool_call, "arguments", {})

        if isinstance(raw_args, dict):
            return tool_name, dict(raw_args)
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return tool_name, parsed if isinstance(parsed, dict) else {"input": parsed}
            except (json.JSONDecodeError, TypeError):
                return tool_name, {}
        if raw_args is None:
            return tool_name, {}
        try:
            return tool_name, dict(raw_args)
        except Exception:
            return tool_name, {"input": raw_args}

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Any, str, str]:
        """Run one tool. Returns (raw_result, observation_text, source)."""
        if self.mcp is not None and self.mcp.owns(tool_name):
            self.logger.info("MCP tool %s(%s)", tool_name, tool_args)
            out = self.mcp.call(tool_name, tool_args)
            return out, out, "mcp"

        tool = self.tools_map.get(tool_name)
        if not tool:
            msg = f"Error: Tool '{tool_name}' not found. Available tools: {self.tool_names()}"
            self.logger.warning(msg)
            return {"error": msg}, msg, "unknown"

        try:
            self.logger.info("Tool %s(%s)", tool_name, tool_args)
            raw = tool(**tool_args)
        except Exception as e:  # failure is content: the loop must stay alive
            msg = f"Error executing tool '{tool_name}': {e}"
            self.logger.error(msg, exc_info=True)
            return {"error": msg}, msg, "local"

        if isinstance(raw, str):
            text = raw
        elif hasattr(raw, "model_dump"):
            text = json.dumps(raw.model_dump(), ensure_ascii=False, default=str)
        else:
            text = json.dumps(raw, ensure_ascii=False, default=str)
        return raw, text, "local"

    @staticmethod
    def _sanitize_history(chat_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Keep only plain user/assistant text turns (drop system, tool and tool-call entries)."""
        clean: List[Dict[str, str]] = []
        for msg in chat_history or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip() and not msg.get("tool_calls"):
                clean.append({"role": role, "content": content})
        return clean

    DOCTOR_CARD_FIELDS = ("name", "specialty", "subspecialties_text", "address", "fee",
                          "reviews_count", "waiting_time_min", "profile_url")

    @classmethod
    def _collect_doctors(cls, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            result = result.get("doctors", [])
        if not isinstance(result, list):
            return []
        return [{k: d[k] for k in cls.DOCTOR_CARD_FIELDS if d.get(k) not in (None, "")}
                for d in result if isinstance(d, dict) and d.get("profile_url")]

    @staticmethod
    def _cited_doctors(answer: str, found: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Doctors the final answer actually recommends, in the order it cites them."""
        cited: List[Dict[str, Any]] = []
        for url in _VEZEETA_URL.findall(answer or ""):
            doc = found.get(_clean_url(url))
            if doc and doc not in cited:
                cited.append(doc)
        return cited

    @staticmethod
    def _collect_urls(result: Any) -> List[str]:
        if isinstance(result, dict):
            result = result.get("doctors", [])
        if not isinstance(result, list):
            return []
        return [d["profile_url"] for d in result if isinstance(d, dict) and d.get("profile_url")]

    # ── the loop ─────────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        total_cost: Optional[float] = None,
        budget: Optional[Budget] = None,
    ) -> AgentResult:
        """Answer one patient message.

        Args:
            prompt: The patient's message.
            chat_history: Prior turns as [{"role": "user"|"assistant", "content": str}].
            system_prompt: Optional system prompt override (the skill menu is still appended).
            total_cost: Starting cumulative cost for this conversation (USD).
            budget: Optional Budget; defaults to the agent's configured limits.
        """
        budget = budget or self.new_budget()
        prompt = (prompt or "").strip()
        history = self._sanitize_history(chat_history)
        red_flags = detect_red_flags(prompt)
        language = detect_language(prompt)

        system_text = self.build_system_prompt(system_prompt)
        if red_flags.is_emergency:
            system_text += (
                "\n\n### SAFETY PRE-CHECK (automatic)\nThe latest patient message contains "
                f"emergency red flags ({', '.join(red_flags.categories)}). Load the "
                "`emergency-triage` skill if available, and START your answer by telling the "
                "patient to call the ambulance on 123 or go to the nearest emergency room."
            )

        messages: List[Dict[str, Any]] = [self._construct_message(system_text, OllamaEnums.SYSTEM.value)]
        messages.extend(dict(m) for m in history)
        messages.append(self._construct_message(prompt, OllamaEnums.USER.value, truncate=True))

        # URLs cited in earlier (already verified) answers count as grounded.
        returned_urls = set()
        for m in history:
            if m["role"] == "assistant":
                returned_urls.update(_VEZEETA_URL.findall(m["content"]))
        doctor_hits = 0
        found_doctors: Dict[str, Dict[str, Any]] = {}
        skills_used: List[str] = []
        records: List[ToolCallRecord] = []
        refinements = 0
        answer = ""
        checks: List[dict] = []
        verified = False
        stop_reason = ""
        run_cost = 0.0
        tokens = {"prompt": 0, "completion": 0, "cached": 0, "llm_calls": 0}
        tools_for_provider = self._provider_tools() or None
        failures: Dict[str, int] = {}
        seen_calls: set = set()

        while True:
            hard_stop = budget.reason()
            if hard_stop:
                stop_reason = hard_stop
                break

            budget.tick()
            response = self.client.generate_response(prompt="", chat_history=messages, tools=tools_for_provider)
            step_cost = self._calculate_response_cost(response)
            run_cost += step_cost
            p_tok, c_tok, cached_tok = self._response_tokens(response)
            tokens["prompt"] += p_tok
            tokens["completion"] += c_tok
            tokens["cached"] += cached_tok
            tokens["llm_calls"] += 1
            budget.spend(step_cost)

            message_obj = getattr(response, "message", None) if response else None
            if message_obj is None:
                self.logger.error("LLM returned an empty or invalid response.")
                stop_reason = "llm_error"
                break

            tool_calls = list(getattr(message_obj, "tool_calls", None) or [])
            if tool_calls:
                granted = budget.allow(len(tool_calls)) if budget.can_use_tools() else 0
                assistant_entry = self._construct_message(getattr(message_obj, "content", "") or "", OllamaEnums.ASSISTANT.value)
                assistant_entry["tool_calls"] = tool_calls
                messages.append(assistant_entry)

                for idx, tool_call in enumerate(tool_calls):
                    name, args = self._extract_tool_call_info(tool_call)
                    signature = (name, json.dumps(args, sort_keys=True, default=str))
                    if idx >= granted:
                        records.append(ToolCallRecord(name, args, "dropped", executed=False, ok=False))
                        observation = BUDGET_SPENT_OBSERVATION
                    elif failures.get(name, 0) >= MAX_TOOL_FAILURES:
                        # Stuck detection: a tool that keeps failing will not start working on try #3.
                        records.append(ToolCallRecord(name, args, "disabled", executed=False, ok=False))
                        observation = TOOL_DISABLED_OBSERVATION.format(name=name, n=failures[name])
                    elif signature in seen_calls:
                        records.append(ToolCallRecord(name, args, "duplicate", executed=False, ok=False))
                        observation = DUPLICATE_CALL_OBSERVATION.format(name=name)
                    else:
                        seen_calls.add(signature)
                        raw, text, source = self._execute_tool(name, args)
                        ok = not (isinstance(raw, dict) and raw.get("error")) and not str(text).startswith("Error")
                        records.append(ToolCallRecord(name, args, source, executed=True, ok=ok))
                        if not ok:
                            failures[name] = failures.get(name, 0) + 1
                        if name == "load_skill" and args.get("name") and args["name"] not in skills_used:
                            skills_used.append(str(args["name"]))
                        if name == "search_doctors" or name.endswith("__search_doctors"):
                            urls = self._collect_urls(raw)
                            returned_urls.update(urls)
                            doctor_hits += len(urls)
                            for doc in self._collect_doctors(raw):
                                found_doctors[_clean_url(doc["profile_url"])] = doc
                            observation = self._format_doctors_context(raw)
                        else:
                            observation = text
                    tool_entry = self._construct_message(str(observation)[:MAX_OBSERVATION_CHARS], OllamaEnums.TOOL.value)
                    tool_entry["name"] = name
                    tool_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else None)
                    if tool_id:
                        tool_entry["tool_call_id"] = tool_id
                    messages.append(tool_entry)
                continue

            # A final answer: verify it with code before it reaches the patient.
            answer = getattr(message_obj, "content", "") or ""
            ctx = VerificationContext(
                user_prompt=prompt, answer=answer, red_flags=red_flags, language=language,
                returned_urls=returned_urls, doctor_search_hits=doctor_hits,
            )
            report = verify_answer(ctx, self.verifiers)
            checks = report.checks
            if report.passed:
                verified = True
                stop_reason = "completed" if refinements == 0 else f"verified after {refinements} refinement(s)"
                break
            if refinements >= self.max_refinements or budget.exhausted():
                stop_reason = "repaired: verification still failing at end of budget"
                answer = repair_answer(ctx)
                break

            refinements += 1
            self.logger.info("Verification failed, refining (%d): %s", refinements, report.feedback)
            messages.append(self._construct_message(answer, OllamaEnums.ASSISTANT.value))
            messages.append(self._construct_message(
                VERIFIER_FEEDBACK_TEMPLATE.format(feedback=report.feedback), OllamaEnums.USER.value,
            ))

        if not verified:
            if not stop_reason.startswith("repaired"):
                # Budget or LLM failure before a good answer: never leave the patient empty-handed.
                ctx = VerificationContext(
                    user_prompt=prompt, answer=answer, red_flags=red_flags, language=language,
                    returned_urls=returned_urls, doctor_search_hits=doctor_hits,
                )
                if not answer.strip():
                    answer = (
                        "عذرًا، لم أتمكن من إكمال طلبك الآن. حاول تبسيط سؤالك أو أعد المحاولة."
                        if language == "ar" else
                        "I'm sorry, I couldn't complete your request this time. "
                        "Please try a simpler or more specific question."
                    )
                    ctx.answer = answer
                answer = repair_answer(ctx)

        with self._cost_lock:
            self.total_cost += run_cost
        start = float(total_cost) if total_cost is not None else 0.0

        new_history = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        return AgentResult(
            answer=answer,
            history=new_history,
            total_cost=start + run_cost,
            run_cost=run_cost,
            stop_reason=stop_reason,
            verified=verified,
            refinements=refinements,
            checks=checks,
            skills_used=skills_used,
            tool_calls=records,
            red_flags=red_flags.as_dict() if red_flags.is_emergency else {"is_emergency": False},
            language=language,
            usage=budget.usage(),
            tokens={**tokens, "model": self.model_id},
            doctors=self._cited_doctors(answer, found_doctors),
        )
