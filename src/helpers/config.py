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
    # Extra/override prices by model_id, USD per 1M tokens: {"model": [input, cached_input, output]}
    MODEL_PRICES_JSON: Optional[str] = None

    # ── Usage & cost tracking ───────────────────────────────────
    USAGE_TRACKING_ENABLED: bool = True
    USAGE_COLLECTION_NAME: str = "usage_events"
    USAGE_TIMEZONE: str = "Africa/Cairo"
    # Required to open /admin/usage. Without it the dashboard stays locked.
    ADMIN_API_KEY: Optional[str] = None

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

    # ── Public /chat API (web frontends) ────────────────────────
    # Comma-separated origins allowed to call the API from a browser, e.g.
    # "https://tabeeby.app,http://localhost:5173". "*" allows any site.
    CORS_ALLOW_ORIGINS: str = "*"
    # Per visitor (IP address): at most this many messages per window. 0 disables.
    CHAT_RATE_LIMIT_MESSAGES: int = 20
    CHAT_RATE_LIMIT_WINDOW_SECONDS: int = 600
    # How many proxies append to X-Forwarded-For in front of the app. FastAPI Cloud
    # (checked 2026-10-14): Cloudflare + 2 internal hops = 3. 0 = trust the socket address.
    TRUSTED_PROXY_HOPS: int = 3

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