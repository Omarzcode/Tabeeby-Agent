from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file.

    This Pydantic settings container defines required and optional configuration
    values for the app, including API credentials, file handling limits, and
    backend selection.
    """

    APP_NAME: Optional[str] = None
    APP_VERSION: Optional[str] = None

    GENERATION_BACKEND: Optional[str] = None
    EMBEDDING_BACKEND: Optional[str] = None

    OLLAMA_API_KEY: Optional[str] = None
    OLLAMA_API_URL: Optional[str] = None

    GEMINI_API_KEY: Optional[str] = None

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None

    GROQ_API_KEY: Optional[str] = None
    GROQ_API_URL: Optional[str] = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None

    VECTOR_DB_BACKEND: Optional[str] = None
    VECTOR_DB_PATH: Optional[str] = None
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None


    VECTOR_DB_API_KEY:Optional[str] = None
    VECTOR_DB_URL:Optional[str] = None
    VECTOR_DB_COLLECTION_NAME: Optional[str] = "vezeeta_doctors"


    TAVILY_API_KEY: Optional[str] = None 

    TELEGRAM_BOT_TOKEN:Optional[str] = None 
    TELEGRAM_WEBHOOK_URL:Optional[str] = None 
    # Random string Telegram echoes back in the X-Telegram-Bot-Api-Secret-Token
    # header, so only Telegram can call /telegram/webhook.
    TELEGRAM_WEBHOOK_SECRET: Optional[str] = None

    # ── Agent loop budget (per patient message) ─────────────────
    AGENT_MAX_STEPS: int = 8
    AGENT_MAX_SECONDS: float = 90.0
    AGENT_MAX_ACTIONS: int = 12
    AGENT_MAX_COST_USD: Optional[float] = None
    AGENT_MAX_REFINEMENTS: int = 2
    AGENT_VERIFY: bool = True
    INPUT_TOKEN_COST_USD: float = 0.0000001
    OUTPUT_TOKEN_COST_USD: float = 0.0000004

    # ── Skills ──────────────────────────────────────────────────
    SKILLS_ENABLED: bool = True

    # ── MCP ─────────────────────────────────────────────────────
    # Serve Tabeeby as an MCP server at /mcp (streamable HTTP).
    MCP_SERVER_ENABLED: bool = True
    # Optional bearer token required by /mcp. Strongly recommended in production.
    MCP_API_KEY: Optional[str] = None
    # JSON object of external MCP servers whose tools the agent may borrow, e.g.
    # {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
    MCP_SERVERS: Optional[str] = None

    # ── Conversations ───────────────────────────────────────────
    SESSION_MAX_TURNS: int = 10
    SESSION_TTL_SECONDS: int = 60 * 60 * 6

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache(maxsize=1)
def get_settings()-> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings: The loaded application settings object.
    """
    return Settings()