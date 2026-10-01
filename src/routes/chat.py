import asyncio
import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from agent import Agent
from models.enums.ResponseEnums import ResponseSignal
from routes.schemes.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    agent: Agent = getattr(req.app.state, "agent", None)
    if not agent:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"signal": ResponseSignal.LLM_CLIENT_NOT_INITIALIZED.value},
        )

    sessions = req.app.state.sessions
    history = sessions.get(f"api:{request.session_id}") if request.session_id else request.chat_history

    try:
        # The loop is sync (LLM + vector DB calls): run it in a worker thread so
        # the event loop keeps serving other requests.
        result = await asyncio.to_thread(
            agent.run, prompt=request.prompt, chat_history=history, total_cost=request.total_cost,
        )
    except Exception:
        logger.exception("Agent run failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.LLM_GENERATE_ERROR.value},
        )

    if request.session_id:
        sessions.set(f"api:{request.session_id}", result.history)

    return ChatResponse(
        signal=ResponseSignal.LLM_GENERATE_SUCCESS.value,
        response=result.answer,
        session_id=request.session_id,
        history=None if request.session_id else result.history,
        total_cost=round(result.total_cost, 6),
        stop_reason=result.stop_reason,
        verified=result.verified,
        emergency=bool(result.red_flags.get("is_emergency")),
        trace=result.metadata() if request.include_trace else None,
    )


@router.delete("/{session_id}")
async def reset_session(session_id: str, req: Request):
    req.app.state.sessions.clear(f"api:{session_id}")
    return {"signal": "session_cleared", "session_id": session_id}
