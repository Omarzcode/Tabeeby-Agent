"""Tabeeby Agent — FastAPI application.

Endpoints
    POST /chat               the agent (REST)
    POST /telegram/webhook   the agent (Telegram)
    /mcp                     Tabeeby as an MCP server (streamable HTTP)
    GET  /health             component status
    GET  /skills, /tools     what the agent can do
"""

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from telegram import Bot

from helpers.bootstrap import Services, build_services
from helpers.config import Settings, get_settings
from helpers.rate_limit import RateLimiter
from helpers.session_store import SessionStore
from helpers.usage_store import UsageStore
from mcp_server.server import ServiceContainer, build_mcp_server
from routes.admin import router as admin_router
from routes.chat import router as chat_router
from routes.info import router as info_router
from routes.telegram import router as telegram_router
from telegram_bot import TelegramChatHandler
from telegram_bot.handler import BOT_COMMANDS
from tools import web_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tabeeby")


class BearerAuth:
    """Tiny ASGI guard: require `Authorization: Bearer <key>` on the MCP endpoint."""

    def __init__(self, app, api_key: Optional[str]):
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.api_key:
            headers = dict(scope.get("headers") or [])
            given = headers.get(b"authorization", b"").decode()
            if not hmac.compare_digest(given, f"Bearer {self.api_key}"):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _setup_telegram(app: FastAPI, services: Services) -> None:
    config = services.config
    if not config.TELEGRAM_BOT_TOKEN or services.agent is None:
        return
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    app.state.tg_bot = bot
    app.state.tg_handler = TelegramChatHandler(
        bot, services.agent, app.state.sessions, services.skills, usage=app.state.usage,
    )
    try:
        await bot.set_my_commands(BOT_COMMANDS)
        if config.TELEGRAM_WEBHOOK_URL:
            await bot.set_webhook(
                url=config.TELEGRAM_WEBHOOK_URL.strip(),
                secret_token=config.TELEGRAM_WEBHOOK_SECRET or None,
                allowed_updates=["message", "edited_message"],
            )
            logger.info("Telegram webhook set to %s", config.TELEGRAM_WEBHOOK_URL.strip())
    except Exception as e:  # Telegram being unreachable must not stop the API
        logger.error("Telegram setup failed: %s", e)
        services.errors["telegram"] = str(e)


def create_app(
    services_factory: Callable[[], Services] = build_services,
    config: Optional[Settings] = None,
) -> FastAPI:
    config = config or get_settings()
    container = ServiceContainer()
    mcp = build_mcp_server(container) if config.MCP_SERVER_ENABLED else None
    mcp_http = None
    if mcp is not None:
        from mcp.server.transport_security import TransportSecuritySettings

        mcp_http = mcp.streamable_http_app(
            streamable_http_path="/",
            stateless_http=True,
            json_response=True,
            # Behind FastAPI Cloud's proxy the Host header is the public domain;
            # access is protected by MCP_API_KEY instead.
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = services_factory()
        app.state.services = services
        app.state.config = services.config
        app.state.agent = services.agent
        app.state.skills = services.skills
        app.state.sessions = SessionStore(
            max_turns=services.config.SESSION_MAX_TURNS, ttl_seconds=services.config.SESSION_TTL_SECONDS,
        )
        app.state.chat_limiter = RateLimiter(
            services.config.CHAT_RATE_LIMIT_MESSAGES, services.config.CHAT_RATE_LIMIT_WINDOW_SECONDS,
        )
        app.state.usage = None
        if services.config.USAGE_TRACKING_ENABLED:
            app.state.usage = UsageStore(
                getattr(services.vector_db, "client", None), collection=services.config.USAGE_COLLECTION_NAME,
            )
        container.doctor_tools = services.doctor_tools
        container.skills = services.skills
        container.agent = services.agent
        container.web_search = web_search if services.config.TAVILY_API_KEY else None

        await _setup_telegram(app, services)

        if mcp is not None:
            async with mcp.session_manager.run():
                yield
        else:
            yield

        services.close()

    app = FastAPI(
        title="Tabeeby Agent",
        version="2.0.0",
        description="AI medical navigator for Egypt: triage, Vezeeta doctor search, skills, MCP and self-verification.",
        lifespan=lifespan,
    )
    origins = [o.strip() for o in (config.CORS_ALLOW_ORIGINS or "").split(",") if o.strip()]
    if origins:
        # Lets browser frontends on other sites call /chat. No cookies are used, so no credentials.
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"], expose_headers=["Retry-After"], max_age=600,
        )
    app.include_router(chat_router)
    app.include_router(telegram_router)
    app.include_router(info_router)
    app.include_router(admin_router)
    if mcp_http is not None:
        app.mount("/mcp", BearerAuth(mcp_http, config.MCP_API_KEY))
    app.state.mcp_server = mcp

    @app.get("/")
    async def root():
        return {"message": "Server is running", "docs": "/docs", "health": "/health", "mcp": "/mcp/"}

    return app


app = create_app()
