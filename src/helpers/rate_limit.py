"""A small in-memory sliding-window rate limiter (per process)."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Hashable


class RateLimiter:
    def __init__(self, max_events: int, window_seconds: float, max_keys: int = 50_000):
        self.max_events = max_events
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: Dict[Hashable, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: Hashable) -> float:
        """Record one event for `key`. Returns 0 if allowed, else seconds until the next slot frees up."""
        if self.max_events <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_events:
                return max(1.0, self.window - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > self.max_keys:
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
            return 0.0
