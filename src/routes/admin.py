"""Admin: usage & cost dashboard.

GET /admin/usage        the dashboard page (asks for the admin key in the browser)
GET /admin/usage/data   the numbers, JSON; requires `Authorization: Bearer <ADMIN_API_KEY>`
"""

import asyncio
import hmac
import time
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from helpers.usage_store import summarize
from routes.chat import client_ip

router = APIRouter(prefix="/admin", tags=["Admin"], include_in_schema=False)

DASHBOARD = Path(__file__).parent / "templates" / "usage_dashboard.html"


def _authorized(req: Request) -> JSONResponse | None:
    key = req.app.state.config.ADMIN_API_KEY
    if not key:
        return JSONResponse({"error": "ADMIN_API_KEY is not set on the server."}, status_code=503)
    given = req.headers.get("authorization", "")
    if not hmac.compare_digest(given, f"Bearer {key}"):
        return JSONResponse({"error": "Wrong admin key."}, status_code=401)
    return None


@router.get("/whoami")
async def whoami(req: Request):
    """Diagnostics: how the platform proxy reports the caller's address (for the /chat rate limit)."""
    denied = _authorized(req)
    if denied:
        return denied
    return {"client_host": req.client.host if req.client else None,
            "x_forwarded_for": req.headers.get("x-forwarded-for"),
            "cf_connecting_ip": req.headers.get("cf-connecting-ip"),
            "rate_limit_key": client_ip(req)}


@router.get("/usage", response_class=HTMLResponse)
async def usage_dashboard():
    return HTMLResponse(DASHBOARD.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@router.get("/usage/data")
async def usage_data(req: Request, days: int = Query(30, ge=1, le=365)):
    denied = _authorized(req)
    if denied:
        return denied
    store = getattr(req.app.state, "usage", None)
    if store is None:
        return JSONResponse({"error": "Usage tracking is disabled (USAGE_TRACKING_ENABLED=false)."}, status_code=503)
    config = req.app.state.config
    since = time.time() - (days + 1) * 86400
    events = await asyncio.to_thread(store.events, since)
    data = summarize(events, days=days, tz_name=config.USAGE_TIMEZONE)
    data["backend"] = store.backend
    data["model_in_use"] = config.GENERATION_MODEL_ID
    return JSONResponse(data, headers={"Cache-Control": "no-store"})
