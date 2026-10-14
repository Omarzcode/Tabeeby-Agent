from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="The patient's message.")
    session_id: Optional[str] = Field(
        None, max_length=128, pattern=r"^[A-Za-z0-9_\-:.]+$",
        description=(
            "Conversation id. Omit it on the first message: the server creates one and returns it. "
            "Send it back on every following message so the server remembers the conversation."
        ),
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Advanced: prior turns [{role: user|assistant, content: str}] for stateless use. "
                          "Ignored when session_id is set.",
    )
    total_cost: float = Field(0.0, description="Deprecated, ignored by the server's own accounting.")
    include_trace: bool = Field(False, description="Debugging: return tool calls, skills, budget usage and checks.")

    model_config = {"json_schema_extra": {"examples": [
        {"prompt": "ضرسي بيوجعني جامد وانا ساكن في المعادي"},
        {"prompt": "Any cheaper one under 300 EGP?", "session_id": "7f3c9a2e5b1d4c8e9a0b"},
    ]}}


class Doctor(BaseModel):
    name: Optional[str] = None
    specialty: Optional[str] = None
    subspecialties_text: Optional[str] = None
    address: Optional[str] = Field(None, description="Clinic area, e.g. 'El-Maadi'.")
    fee: Optional[float] = Field(None, description="Consultation fee in EGP.")
    reviews_count: Optional[int] = None
    waiting_time_min: Optional[float] = None
    profile_url: str = Field(..., description="Vezeeta profile & booking link.")


class ChatResponse(BaseModel):
    signal: str = Field(..., description="'llm_generate_success' on success.")
    response: Optional[str] = Field(None, description="The answer, in Markdown (bold, lists, links).")
    session_id: Optional[str] = Field(None, description="Send this back with the next message.")
    language: Optional[str] = Field(None, description="'ar' or 'en': the patient's language.")
    direction: Optional[str] = Field(None, description="'rtl' for Arabic, 'ltr' for English.")
    emergency: Optional[bool] = Field(None, description="True when the message has emergency red flags.")
    emergency_numbers: Optional[Dict[str, str]] = Field(None, description="Egypt emergency numbers, when emergency.")
    doctors: List[Doctor] = Field(default_factory=list, description="Doctors recommended in the answer, in order.")
    verified: Optional[bool] = None
    stop_reason: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = Field(None, description="Only in stateless mode (no session_id).")
    total_cost: Optional[float] = None
    trace: Optional[Dict[str, Any]] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(..., description="[{role: user|assistant, content: str}], oldest first.")
