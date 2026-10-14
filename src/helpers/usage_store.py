"""Usage and cost tracking: one event per answered message, per person.

Events go to their own Qdrant collection (`usage_events`, payload only), so they
survive redeploys and every app instance sees the same data. Without Qdrant they
are kept in memory. Recording never raises: a tracking failure must not cost a
patient their answer.

The patient's message text is NOT stored, only its length.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from stores.llm.pricing import price_for

logger = logging.getLogger(__name__)

try:
    from qdrant_client import models as qm
except ImportError:  # pragma: no cover
    qm = None


def usage_event(result: Any, channel: str, user_key: str, user_name: str = "",
                username: str = "", message_chars: int = 0) -> Dict[str, Any]:
    """Build one usage event from an AgentResult."""
    tokens = getattr(result, "tokens", {}) or {}
    usage = getattr(result, "usage", {}) or {}
    return {
        "ts": time.time(),
        "channel": channel,
        "user_key": user_key,
        "user_name": user_name or "",
        "username": username or "",
        "model": tokens.get("model") or "unknown",
        "prompt_tokens": int(tokens.get("prompt", 0)),
        "cached_tokens": int(tokens.get("cached", 0)),
        "completion_tokens": int(tokens.get("completion", 0)),
        "llm_calls": int(tokens.get("llm_calls", 0)),
        "cost_usd": float(getattr(result, "run_cost", 0.0) or 0.0),
        "seconds": float(usage.get("seconds", 0.0) or 0.0),
        "tool_actions": int(usage.get("actions", 0) or 0),
        "stop_reason": getattr(result, "stop_reason", "") or "",
        "verified": bool(getattr(result, "verified", False)),
        "emergency": bool((getattr(result, "red_flags", {}) or {}).get("is_emergency")),
        "skills": list(getattr(result, "skills_used", []) or []),
        "language": getattr(result, "language", "") or "",
        "message_chars": int(message_chars),
    }


class UsageStore:
    def __init__(self, qdrant_client: Any = None, collection: str = "usage_events", max_memory: int = 50_000):
        self.client = qdrant_client
        self.collection = collection
        self.max_memory = max_memory
        self._memory: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.backend = "memory"
        if self.client is not None and qm is not None:
            try:
                self._ensure_collection()
                self.backend = "qdrant"
            except Exception as e:
                logger.error("Usage tracking falls back to memory (Qdrant setup failed): %s", e)
                self.client = None

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            # Payload-only records: a constant 1-number vector satisfies Qdrant's schema.
            self.client.create_collection(
                self.collection, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT),
            )
        self.client.create_payload_index(self.collection, "ts", qm.PayloadSchemaType.FLOAT)
        self.client.create_payload_index(self.collection, "user_key", qm.PayloadSchemaType.KEYWORD)

    def record(self, event: Dict[str, Any]) -> None:
        try:
            if self.client is not None:
                self.client.upsert(self.collection, points=[
                    qm.PointStruct(id=str(uuid.uuid4()), vector=[1.0], payload=event),
                ])
            else:
                with self._lock:
                    self._memory.append(event)
                    del self._memory[:-self.max_memory]
        except Exception as e:
            logger.error("Could not record usage event: %s", e)

    def events(self, since_ts: float) -> List[Dict[str, Any]]:
        if self.client is None:
            with self._lock:
                return [e for e in self._memory if e.get("ts", 0) >= since_ts]
        out: List[Dict[str, Any]] = []
        flt = qm.Filter(must=[qm.FieldCondition(key="ts", range=qm.Range(gte=since_ts))])
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=1000, offset=offset,
                with_payload=True, with_vectors=False,
            )
            out.extend(p.payload for p in points)
            if offset is None:
                return out


def summarize(events: List[Dict[str, Any]], days: int, tz_name: str = "Africa/Cairo",
              now: Optional[float] = None) -> Dict[str, Any]:
    """Totals, a daily series, per-user, per-model and recent rows for the dashboard."""
    tz = ZoneInfo(tz_name)
    now = now if now is not None else time.time()
    today = datetime.fromtimestamp(now, tz).date()
    first_day = today - timedelta(days=days - 1)
    events = sorted((e for e in events if datetime.fromtimestamp(e["ts"], tz).date() >= first_day),
                    key=lambda e: e["ts"])

    def blank():
        return {"messages": 0, "cost_usd": 0.0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}

    def add(acc, e):
        acc["messages"] += 1
        acc["cost_usd"] += e.get("cost_usd", 0.0)
        for k in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            acc[k] += e.get(k, 0)

    totals, today_acc = blank(), blank()
    daily = {first_day + timedelta(days=i): blank() for i in range(days)}
    users: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Dict[str, Any]] = defaultdict(blank)
    stops: Dict[str, int] = defaultdict(int)
    emergencies = 0
    seconds: List[float] = []

    for e in events:
        day = datetime.fromtimestamp(e["ts"], tz).date()
        add(totals, e)
        add(daily[day], e)
        if day == today:
            add(today_acc, e)
        u = users.setdefault(e["user_key"], {**blank(), "user_key": e["user_key"], "channel": e.get("channel", ""),
                                             "first_seen": e["ts"]})
        add(u, e)
        u["user_name"] = e.get("user_name") or u.get("user_name", "")
        u["username"] = e.get("username") or u.get("username", "")
        u["last_seen"] = e["ts"]
        add(models[e.get("model", "unknown")], e)
        stops[(e.get("stop_reason") or "unknown").split(":")[0]] += 1
        emergencies += 1 if e.get("emergency") else 0
        seconds.append(e.get("seconds", 0.0))

    last7 = [daily[d]["cost_usd"] for d in sorted(daily)[-7:]]
    user_rows = sorted(users.values(), key=lambda u: u["cost_usd"], reverse=True)
    for u in user_rows:
        u["avg_cost_usd"] = u["cost_usd"] / u["messages"] if u["messages"] else 0.0
    seconds.sort()
    return {
        "days": days,
        "timezone": tz_name,
        "generated_at": now,
        "totals": {**totals, "users": len(users), "emergencies": emergencies,
                   "avg_cost_usd": totals["cost_usd"] / totals["messages"] if totals["messages"] else 0.0,
                   "median_seconds": seconds[len(seconds) // 2] if seconds else 0.0},
        "today": {**today_acc, "users": len({e["user_key"] for e in events
                                             if datetime.fromtimestamp(e["ts"], tz).date() == today})},
        "projected_30d_usd": (sum(last7) / len(last7)) * 30 if last7 else 0.0,
        "daily": [{"date": d.isoformat(), **v} for d, v in sorted(daily.items())],
        "users": user_rows,
        "models": [{"model": m, "priced": price_for(m) is not None, **v}
                   for m, v in sorted(models.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)],
        "stop_reasons": dict(stops),
        "recent": [{k: e.get(k) for k in ("ts", "user_key", "user_name", "username", "channel", "model",
                                            "cost_usd", "prompt_tokens", "cached_tokens", "completion_tokens",
                                            "seconds", "stop_reason", "emergency", "skills", "verified")}
                   for e in reversed(events[-50:])],
    }
