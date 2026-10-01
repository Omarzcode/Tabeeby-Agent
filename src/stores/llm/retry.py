"""Retry transient LLM API failures (rate limits, overload, flaky auth edges).

Free tiers are tight (Gemini free: 15 requests/minute) and one patient
message can take several LLM calls, so a 429 mid-loop is normal, not fatal.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)

_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit", "503", "UNAVAILABLE",
              "overloaded", "500 INTERNAL", "502", "504", "DEADLINE_EXCEEDED", "timed out")
_RETRY_IN = re.compile(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]?(\d+)s", re.IGNORECASE)


def is_transient(error: Exception) -> bool:
    text = str(error)
    return any(marker in text for marker in _TRANSIENT)


def suggested_delay(error: Exception, attempt: int, cap: float) -> float:
    match = _RETRY_IN.search(str(error))
    if match:
        delay = float(match.group(1) or match.group(2))
    else:
        delay = 1.5 * (2 ** attempt)
    return min(cap, delay + random.uniform(0, 0.5))


def with_retries(call: Callable[[], T], attempts: int = 3, max_delay: float = 12.0, label: str = "LLM") -> T:
    """Run `call`, retrying transient errors with server-suggested or exponential backoff."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:
            if attempt == attempts - 1 or not is_transient(e):
                raise
            delay = suggested_delay(e, attempt, max_delay)
            logger.warning("%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                           label, attempt + 1, attempts, delay, str(e)[:160])
            time.sleep(delay)
    raise RuntimeError("unreachable")
