from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="The patient's message.")
    session_id: Optional[str] = Field(
        None, max_length=128,
        description="Optional id to let the server remember the conversation between calls.",
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional prior turns [{role: user|assistant, content: str}]; ignored when session_id is set.",
    )
    total_cost: float = 0.0
    include_trace: bool = Field(False, description="Return tool calls, skills, budget usage and verification checks.")


class ChatResponse(BaseModel):
    signal: str
    response: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    total_cost: Optional[float] = None
    stop_reason: Optional[str] = None
    verified: Optional[bool] = None
    emergency: Optional[bool] = None
    trace: Optional[Dict[str, Any]] = None
