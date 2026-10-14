import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, get_type_hints

from ..LLMEnums import OpenAIEnums
from ..LLMInterface import LLMInterface
from ..retry import with_retries

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class OpenAIToolCallFunction:
    name: str
    arguments: Union[Dict[str, Any], str] = field(default_factory=dict)


@dataclass
class OpenAIToolCall:
    id: Optional[str] = None
    type: str = "function"
    function: Optional[OpenAIToolCallFunction] = None


@dataclass
class OpenAIUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class OpenAIMessage:
    role: str = "assistant"
    content: Optional[str] = ""
    tool_calls: Optional[List[OpenAIToolCall]] = None


@dataclass
class OpenAIResponse:
    message: OpenAIMessage
    prompt_eval_count: int = 0
    eval_count: int = 0
    usage: Optional[OpenAIUsage] = None
    raw_response: Any = None


def function_to_tool_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Convert a Python callable or existing schema dict to OpenAI tool format."""
    if isinstance(func, dict):
        return func

    if hasattr(func, "schema") and isinstance(func.schema, dict):
        return func.schema

    name = getattr(func, "__name__", str(func))
    doc = inspect.getdoc(func) or ""

    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}

    properties = {}
    required = []
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "client_embedding", "client_vectorDB"):
            continue

        param_type = type_hints.get(param_name, str)
        origin = getattr(param_type, "__origin__", None)
        if origin is Union:
            args = getattr(param_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                param_type = non_none[0]

        json_type = type_mapping.get(param_type, "string")
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_url = api_url or os.getenv("OPENAI_API_URL")

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = None
        self.logger = logging.getLogger(__name__)

        if OpenAI is None:
            self.logger.error("openai package is not installed")
            return

        if not self.api_key:
            self.logger.warning("OpenAI API key was not provided")

        self.client = OpenAI(
            api_key=self.api_key or "placeholder_key",
            base_url=self.api_url if self.api_url and len(self.api_url) else None,
        )

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int = None) -> None:
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

    def _build_response(self, raw_res: Any) -> OpenAIResponse:
        content_text = ""
        tool_calls: List[OpenAIToolCall] = []

        if raw_res and raw_res.choices and len(raw_res.choices) > 0:
            msg = raw_res.choices[0].message
            content_text = getattr(msg, "content", "") or ""
            raw_tool_calls = getattr(msg, "tool_calls", None)
            if raw_tool_calls:
                for tc in raw_tool_calls:
                    func = getattr(tc, "function", None)
                    func_name = getattr(func, "name", "") if func else ""
                    raw_args = getattr(func, "arguments", {}) if func else {}
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = raw_args
                    else:
                        parsed_args = raw_args

                    tool_calls.append(
                        OpenAIToolCall(
                            id=getattr(tc, "id", None),
                            type="function",
                            function=OpenAIToolCallFunction(
                                name=func_name,
                                arguments=parsed_args,
                            ),
                        )
                    )

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0

        usage = getattr(raw_res, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (
                prompt_tokens + completion_tokens
            )
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", 0) or 0 if details else 0

        usage_obj = OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
        )

        message = OpenAIMessage(
            role=OpenAIEnums.ASSISTANT.value,
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
        )

        return OpenAIResponse(
            message=message,
            prompt_eval_count=prompt_tokens,
            eval_count=completion_tokens,
            usage=usage_obj,
            raw_response=raw_res,
        )

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
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
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
                    role=OpenAIEnums.USER.value,
                    truncate=truncate,
                )
            )

        formatted_messages: List[Dict[str, Any]] = []
        last_tool_calls = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append({"role": "system", "content": str(content or "")})

            elif role == "user":
                formatted_messages.append({"role": "user", "content": str(content or "")})

            elif role in ("assistant", "model"):
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": str(content) if content else None,
                }
                raw_tcs = msg.get("tool_calls")
                if raw_tcs:
                    last_tool_calls = []
                    formatted_tcs = []
                    for idx, tc in enumerate(raw_tcs):
                        tc_id = (
                            getattr(tc, "id", None)
                            or (tc.get("id") if isinstance(tc, dict) else None)
                            or f"call_{idx}"
                        )
                        func_name = ""
                        func_args = ""
                        if hasattr(tc, "function"):
                            func_name = getattr(tc.function, "name", "")
                            func_args = getattr(tc.function, "arguments", "")
                        elif isinstance(tc, dict):
                            func = tc.get("function", {})
                            func_name = (
                                func.get("name", "")
                                if isinstance(func, dict)
                                else tc.get("name", "")
                            )
                            func_args = (
                                func.get("arguments", "")
                                if isinstance(func, dict)
                                else tc.get("arguments", "")
                            )
                        if isinstance(func_args, dict):
                            func_args = json.dumps(func_args)

                        formatted_tc = {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": str(func_args),
                            },
                        }
                        formatted_tcs.append(formatted_tc)
                        last_tool_calls.append(tc_id)
                    assistant_msg["tool_calls"] = formatted_tcs
                formatted_messages.append(assistant_msg)

            elif role in ("tool", "function"):
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id and last_tool_calls:
                    tool_call_id = last_tool_calls.pop(0)
                if not tool_call_id:
                    tool_call_id = "call_default"

                formatted_messages.append({
                    "role": "tool",
                    "content": str(content or ""),
                    "tool_call_id": tool_call_id,
                })
            else:
                formatted_messages.append({"role": "user", "content": str(content or "")})

        formatted_tools = None
        if tools:
            formatted_tools = [function_to_tool_schema(t) for t in tools]

        try:
            kwargs: Dict[str, Any] = {
                "model": self.generation_model_id,
                "messages": formatted_messages,
                self._token_limit_param(): max_output_tokens,
                "temperature": temperature,
            }
            if formatted_tools:
                kwargs["tools"] = formatted_tools

            response = self._create_completion(kwargs)
            return self._build_response(response)

        except Exception as e:
            self.logger.error(f"Error while generating text with OpenAI: {e}")
            return None

    def _token_limit_param(self) -> str:
        """OpenAI's own API wants `max_completion_tokens` (newer models reject
        `max_tokens`); OpenAI-compatible hosts such as Groq still use `max_tokens`."""
        if not self.api_url or "api.openai.com" in self.api_url:
            return "max_completion_tokens"
        return "max_tokens"

    def _create_completion(self, kwargs: Dict[str, Any]) -> Any:
        """Call chat.completions, adapting once to models that reject a parameter.

        Some models only accept the default temperature, and hosts disagree on
        `max_tokens` vs `max_completion_tokens`. On such a 400 error the offending
        parameter is dropped or renamed and the call is retried.
        """
        call = lambda: with_retries(lambda: self.client.chat.completions.create(**kwargs), label="OpenAI-compatible")
        for _ in range(3):
            try:
                return call()
            except Exception as e:
                text = str(e)
                if "temperature" in text and "temperature" in kwargs:
                    self.logger.warning("Model rejected temperature; retrying with the default.")
                    kwargs.pop("temperature")
                elif "max_tokens" in text and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                elif "max_completion_tokens" in text and "max_completion_tokens" in kwargs:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                else:
                    raise
        return call()

    def embed_text(self, text: str, document_type: str = None) -> Optional[List[float]]:
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None

        if not text or not str(text).strip():
            self.logger.error("Error while embedding text with OpenAI: empty input text")
            return None

        try:
            kwargs: Dict[str, Any] = {
                "model": self.embedding_model_id,
                "input": str(text).strip(),
            }
            if self.embedding_size:
                kwargs["dimensions"] = self.embedding_size

            response = self.client.embeddings.create(**kwargs)

            if not response or not response.data or len(response.data) == 0:
                self.logger.error("Error while embedding text with OpenAI: empty response")
                return None

            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error while embedding text with OpenAI: {e}")
            return None
