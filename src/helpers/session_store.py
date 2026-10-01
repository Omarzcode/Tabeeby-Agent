"""In-memory conversation memory shared by the REST API and Telegram.

Only clean user/assistant text turns are stored (never tool calls), so a
trimmed history can never start with an orphaned tool response — which some
providers (Gemini) reject.

Note: memory lives in one process. If FastAPI Cloud scales to several
instances, a user may land on an instance without their history; swap this
class for a Redis-backed one with the same interface when that matters.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Hashable, List, Tuple


class SessionStore:
    def __init__(self, max_turns: int = 10, ttl_seconds: int = 6 * 3600, max_sessions: int = 10_000):
        self.max_messages = max(2, max_turns * 2)
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._data: Dict[Hashable, Tuple[float, List[dict]]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> List[dict]:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return []
            ts, history = item
            if time.time() - ts > self.ttl:
                self._data.pop(key, None)
                return []
            return list(history)

    def set(self, key: Hashable, history: List[dict]) -> None:
        clean = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ][-self.max_messages:]
        # Keep the window starting on a user turn.
        while clean and clean[0]["role"] != "user":
            clean.pop(0)
        with self._lock:
            if len(self._data) >= self.max_sessions and key not in self._data:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest, None)
            self._data[key] = (time.time(), clean)

    def clear(self, key: Hashable) -> None:
        with self._lock:
            self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)
