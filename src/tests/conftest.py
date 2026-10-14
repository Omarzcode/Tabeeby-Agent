"""Offline test fixtures: a scripted fake LLM and fake vector search.

No API keys, network or Qdrant needed: `cd src && pytest`.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from helpers.config import Settings  # noqa: E402


def tool_call(name: str, arguments: Dict[str, Any], call_id: str = "c1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def reply(content: str = "", calls: List[Any] = None, prompt_tokens: int = 100, completion_tokens: int = 50):
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls or None),
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeLLM:
    """Returns scripted responses in order and records what it was sent."""

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []

    def construct_prompt(self, prompt, role, truncate=False):
        return {"role": role, "content": prompt}

    def generate_response(self, prompt, chat_history=None, max_output_tokens=None, temperature=None, tools=None):
        self.calls.append({"messages": [dict(m) for m in chat_history or []], "tools": tools})
        if not self.script:
            return reply("(script exhausted)")
        return self.script.pop(0)

    def set_generation_model(self, model_id): ...
    def set_embedding_model(self, model_id, embedding_size): ...
    def embed_text(self, text, document_type=None):
        return [0.1, 0.2, 0.3]


DOCTOR = {
    "name": "Dr. Test Molar", "specialty": "Dentistry", "address": "El-Maadi", "fee": 300,
    "reviews_count": 120, "profile_url": "https://www.vezeeta.com/en/dr/test-molar",
}


class FakeVectorDB:
    def __init__(self, hits=None):
        self.hits = [DOCTOR] if hits is None else hits
        self.filters = []
        self.client = object()

    def build_filter(self, **kw):
        self.filters.append(kw)
        return kw

    def search_by_vector(self, collection_name, vector, limit=10, query_filter=None):
        return [{"id": "1", "score": 0.9, "payload": h} for h in self.hits][:limit]

    def disconnect(self): ...


@pytest.fixture
def settings():
    return Settings(
        _env_file=None, GENERATION_BACKEND="FAKE", TELEGRAM_BOT_TOKEN=None, TAVILY_API_KEY=None,
        MCP_SERVERS=None, MCP_API_KEY=None,
    )
