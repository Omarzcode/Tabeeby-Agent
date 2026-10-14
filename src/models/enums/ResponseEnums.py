from enum import Enum

class ResponseSignal(Enum):

    # LLM Signals
    LLM_GENERATE_SUCCESS = "llm_generate_success"
    LLM_CLIENT_NOT_INITIALIZED = "llm_client_not_initialized"
    LLM_GENERATE_ERROR = "llm_generate_error"
    RATE_LIMITED = "rate_limited"

    # Embedding Signals
    EMBEDDING_SUCCESS = "embedding_success"
    EMBEDDING_FAILED = "embedding_failed"
    EMBEDDING_EMPTY_INPUT = "embedding_empty_input"
    