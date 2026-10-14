import asyncio
import logging
import secrets
from typing import Dict

from fastapi import APIRouter, Path, Request, status
from fastapi.responses import JSONResponse

from agent import Agent
from agent.safety import EMERGENCY_NUMBERS
from helpers.usage_store import usage_event
from models.enums.ResponseEnums import ResponseSignal
from routes.schemes.chat import ChatHistoryResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = r"^[A-Za-z0-9_\-:.]+$"
_session_locks: Dict[str, asyncio.Lock] = {}


def _error(code: int, signal: str, message: str, headers: Dict[str, str] = None) -> JSONResponse:
    return JSONResponse(status_code=code, content={"signal": signal, "message": message}, headers=headers)


def client_ip(req: Request) -> str:
    """The visitor's real IP, for rate limiting.

    The left part of X-Forwarded-For is whatever the caller sent, so it can be faked.
    Each trusted proxy (on FastAPI Cloud: Cloudflare, then two internal hops) appends
    the address it saw, so the real visitor is TRUSTED_PROXY_HOPS entries from the right.
    Cloudflare's own CF-Connecting-IP header is preferred when present.
    """
    cf = req.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    hops = max(0, int(getattr(req.app.state.config, "TRUSTED_PROXY_HOPS", 0) or 0))
    chain = [p.strip() for p in req.headers.get("x-forwarded-for", "").split(",") if p.strip()]
    if hops and chain:
        return chain[-(hops + 1)] if len(chain) > hops else chain[0]
    return req.client.host if req.client else "unknown"


def _session_lock(key: str) -> asyncio.Lock:
    if len(_session_locks) > 10_000:
        for k in [k for k, v in _session_locks.items() if not v.locked()]:
            del _session_locks[k]
    return _session_locks.setdefault(key, asyncio.Lock())


@router.post("", response_model=ChatResponse, responses={
    429: {"description": "Too many messages from this visitor; see the Retry-After header."},
    503: {"description": "The assistant is not ready."},
})
async def chat_endpoint(request: ChatRequest, req: Request):
    """Send one patient message and get Tabeeby's answer.

    First message: send only `prompt`. Keep the returned `session_id` and send it
    with every following message; the server remembers the last 10 turns for 6 hours.
    """
    agent: Agent = getattr(req.app.state, "agent", None)
    if not agent:
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, ResponseSignal.LLM_CLIENT_NOT_INITIALIZED.value,
                      "The assistant is starting or not configured. Try again shortly.")

    limiter = getattr(req.app.state, "chat_limiter", None)
    wait = limiter.check(client_ip(req)) if limiter else 0
    if wait:
        return _error(status.HTTP_429_TOO_MANY_REQUESTS, ResponseSignal.RATE_LIMITED.value,
                      f"Too many messages. Please wait {int(wait)} seconds and try again.",
                      headers={"Retry-After": str(int(wait))})

    stateless = request.session_id is None and request.chat_history is not None
    session_id = None if stateless else (request.session_id or secrets.token_hex(12))
    sessions = req.app.state.sessions
    key = f"api:{session_id}"

    lock = _session_lock(key) if session_id else asyncio.Lock()
    async with lock:  # one answer at a time per conversation
        history = request.chat_history if stateless else sessions.get(key)
        try:
            # The loop is sync (LLM + vector DB calls): run it in a worker thread so
            # the event loop keeps serving other requests.
            result = await asyncio.to_thread(agent.run, prompt=request.prompt, chat_history=history)
        except Exception:
            logger.exception("Agent run failed")
            return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, ResponseSignal.LLM_GENERATE_ERROR.value,
                          "Something went wrong while answering. Please try again.")
        if session_id:
            sessions.set(key, result.history)

    usage = getattr(req.app.state, "usage", None)
    if usage is not None:
        event = usage_event(result, channel="api", user_key=f"api:{session_id or 'stateless'}",
                            message_chars=len(request.prompt))
        await asyncio.to_thread(usage.record, event)

    emergency = bool(result.red_flags.get("is_emergency"))
    return ChatResponse(
        signal=ResponseSignal.LLM_GENERATE_SUCCESS.value,
        response=result.answer,
        session_id=session_id,
        language=result.language,
        direction="rtl" if result.language == "ar" else "ltr",
        emergency=emergency,
        emergency_numbers=EMERGENCY_NUMBERS if emergency else None,
        doctors=result.doctors,
        verified=result.verified,
        stop_reason=result.stop_reason,
        history=result.history if stateless else None,
        total_cost=round(result.run_cost, 6),
        trace=result.metadata() if request.include_trace else None,
    )


@router.get("/{session_id}", response_model=ChatHistoryResponse)
async def get_session(req: Request, session_id: str = Path(..., max_length=128, pattern=SESSION_ID_PATTERN)):
    """The conversation so far (e.g. to redraw the chat after a page reload). Empty if expired."""
    return ChatHistoryResponse(session_id=session_id, messages=req.app.state.sessions.get(f"api:{session_id}"))


@router.delete("/{session_id}")
async def reset_session(req: Request, session_id: str = Path(..., max_length=128, pattern=SESSION_ID_PATTERN)):
    """Forget a conversation ("New chat" button)."""
    req.app.state.sessions.clear(f"api:{session_id}")
    return {"signal": "session_cleared", "session_id": session_id}
