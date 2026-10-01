import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ..LLMEnums import GeminiEnums, OllamaEnums
from ..LLMInterface import LLMInterface
from ..retry import with_retries

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


@dataclass
class GeminiToolCallFunction:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeminiToolCall:
    id: Optional[str] = None
    type: str = "function"
    function: Optional[GeminiToolCallFunction] = None
    raw_part: Any = None


@dataclass
class GeminiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GeminiMessage:
    role: str = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[List[GeminiToolCall]] = None


@dataclass
class GeminiResponse:
    message: GeminiMessage
    prompt_eval_count: int = 0
    eval_count: int = 0
    usage: Optional[GeminiUsage] = None
    raw_response: Any = None


class GeminiProvider(LLMInterface):
    def __init__(
        self,
        api_key: Optional[str],
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = None
        self.logger = logging.getLogger(__name__)

        if genai is None:
            self.logger.error("google-genai package is not installed")
            return

        if not self.api_key:
            self.logger.warning("Gemini API key was not provided")

        self.client = genai.Client(api_key=self.api_key)

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        return text[:self.default_input_max_characters].strip()

    def construct_prompt(
        self, prompt: str, role: str, truncate: bool = False
    ) -> Dict[str, str]:
        if truncate:
            return {"role": role, "content": self.process_text(prompt)}

        return {"role": role, "content": prompt}

    def _build_response(self, raw_res: Any) -> GeminiResponse:
        content_text = ""
        tool_calls: List[GeminiToolCall] = []

        if raw_res and getattr(raw_res, "candidates", None) and len(raw_res.candidates) > 0:
            candidate = raw_res.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        content_text += part.text
                    if part.function_call:
                        fc = part.function_call
                        args = (
                            fc.args
                            if isinstance(fc.args, dict)
                            else (json.loads(fc.args) if fc.args else {})
                        )
                        tool_calls.append(
                            GeminiToolCall(
                                id=getattr(fc, "id", None),
                                type="function",
                                function=GeminiToolCallFunction(
                                    name=fc.name,
                                    arguments=args,
                                ),
                                raw_part=part,
                            )
                        )

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        usage = getattr(raw_res, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (
                prompt_tokens + completion_tokens
            )

        usage_obj = GeminiUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        message = GeminiMessage(
            role=GeminiEnums.ASSISTANT.value,
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
        )

        return GeminiResponse(
            message=message,
            prompt_eval_count=prompt_tokens,
            eval_count=completion_tokens,
            usage=usage_obj,
            raw_response=raw_res,
        )

    @staticmethod
    def _convert_tools(tools: Sequence[Any]) -> List[Any]:
        """Python callables pass through (the SDK reads their signature); JSON
        schemas — e.g. tools borrowed from MCP servers — become FunctionDeclarations."""
        callables: List[Any] = []
        declarations: List[Any] = []
        for tool in tools:
            if isinstance(tool, dict):
                spec = tool.get("function", tool)
                declarations.append(
                    types.FunctionDeclaration(
                        name=spec["name"],
                        description=spec.get("description", ""),
                        parameters_json_schema=spec.get("parameters") or {"type": "object", "properties": {}},
                    )
                )
            else:
                callables.append(tool)
        if declarations:
            callables.append(types.Tool(function_declarations=declarations))
        return callables

    def generate_response(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        truncate: bool = False,
    ) -> Any:
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.default_generation_max_output_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else self.default_generation_temperature
        )

        messages = list(chat_history) if chat_history else []
        if prompt and prompt.strip() and not (
            messages and messages[-1].get("content") == prompt.strip()
        ):
            messages.append(
                self.construct_prompt(
                    prompt=prompt,
                    role=GeminiEnums.USER.value,
                    truncate=truncate,
                )
            )

        system_instruction = None
        contents: List[Any] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content_str = msg.get("content", "")

            if role == GeminiEnums.SYSTEM.value or role == OllamaEnums.SYSTEM.value:
                system_instruction = content_str
                i += 1
                continue

            if role == GeminiEnums.USER.value or role == OllamaEnums.USER.value:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_str)],
                    )
                )
                i += 1
                continue

            if role in (
                GeminiEnums.ASSISTANT.value,
                GeminiEnums.MODEL.value,
                OllamaEnums.ASSISTANT.value,
            ):
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    model_parts = []
                    if content_str and content_str.strip():
                        model_parts.append(types.Part.from_text(text=content_str))
                    for tc in tool_calls:
                        if hasattr(tc, "raw_part") and tc.raw_part:
                            model_parts.append(tc.raw_part)
                        else:
                            tc_name = ""
                            tc_args = {}
                            if hasattr(tc, "function"):
                                tc_name = getattr(tc.function, "name", "")
                                tc_args = getattr(tc.function, "arguments", {})
                            elif isinstance(tc, dict):
                                func = tc.get("function", {})
                                tc_name = (
                                    func.get("name", "")
                                    if isinstance(func, dict)
                                    else tc.get("name", "")
                                )
                                tc_args = (
                                    func.get("arguments", {})
                                    if isinstance(func, dict)
                                    else tc.get("arguments", {})
                                )
                            model_parts.append(
                                types.Part.from_function_call(name=tc_name, args=tc_args)
                            )
                    contents.append(types.Content(role="model", parts=model_parts))

                    # Collect subsequent tool observations
                    i += 1
                    tool_response_parts = []
                    while i < len(messages) and messages[i].get("role") in (
                        GeminiEnums.TOOL.value,
                        GeminiEnums.FUNCTION.value,
                        OllamaEnums.TOOL.value,
                    ):
                        tool_msg = messages[i]
                        tool_content = tool_msg.get("content", "")
                        call_idx = len(tool_response_parts)
                        func_name = "tool_response"
                        if call_idx < len(tool_calls):
                            tc_item = tool_calls[call_idx]
                            func_name = (
                                getattr(getattr(tc_item, "function", None), "name", None)
                                or getattr(tc_item, "name", "tool_response")
                            )
                        tool_response_parts.append(
                            types.Part.from_function_response(
                                name=func_name,
                                response={"result": tool_content},
                            )
                        )
                        i += 1

                    if tool_response_parts:
                        contents.append(
                            types.Content(role="user", parts=tool_response_parts)
                        )
                    continue
                else:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=content_str or "")],
                        )
                    )
                    i += 1
                    continue

            if role in (
                GeminiEnums.TOOL.value,
                GeminiEnums.FUNCTION.value,
                OllamaEnums.TOOL.value,
            ):
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content_str or "")],
                    )
                )
                i += 1
                continue

            # Fallback
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content_str or "")],
                )
            )
            i += 1

        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if tools:
            config_params["tools"] = self._convert_tools(tools)
            config_params["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        config = types.GenerateContentConfig(**config_params)

        try:
            raw_res = with_retries(lambda: self.client.models.generate_content(
                model=self.generation_model_id,
                contents=contents if contents else (prompt or ""),
                config=config,
            ), label="Gemini generate")

            return self._build_response(raw_res)

        except Exception as e:
            self.logger.error(f"Error while generating text with Gemini: {e}")
            return None

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        if not text or not str(text).strip():
            self.logger.error("Error while embedding text with Gemini: empty input text")
            return None

        try:
            config_kwargs = {}
            if self.embedding_size:
                config_kwargs["output_dimensionality"] = self.embedding_size
            if document_type:
                config_kwargs["task_type"] = document_type

            config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

            response = with_retries(lambda: self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=str(text).strip(),
                config=config,
            ), label="Gemini embed")

            if not response or not response.embeddings or len(response.embeddings) == 0:
                self.logger.error("Error while embedding text with Gemini: empty response")
                return None

            return response.embeddings[0].values
        except Exception as e:
            self.logger.error(f"Error while embedding text with Gemini: {e}")
            return None
