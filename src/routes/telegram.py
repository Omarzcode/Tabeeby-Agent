"""Telegram webhook endpoint.

Telegram retries a webhook that does not answer quickly, which used to cause
duplicate agent runs. The endpoint now verifies Telegram's secret header,
de-duplicates update ids, schedules the work in the background and returns
200 immediately.
"""

import asyncio
import hmac
import logging
from collections import OrderedDict
from typing import Set

from fastapi import APIRouter, Request, Response, status
from telegram import Update

router = APIRouter(prefix="/telegram", tags=["Telegram"])
logger = logging.getLogger(__name__)

_PROCESSED: "OrderedDict[int, None]" = OrderedDict()
_BACKGROUND: Set[asyncio.Task] = set()
_MAX_REMEMBERED = 5000


def _seen(update_id: int) -> bool:
    if update_id in _PROCESSED:
        return True
    _PROCESSED[update_id] = None
    while len(_PROCESSED) > _MAX_REMEMBERED:
        _PROCESSED.popitem(last=False)
    return False


@router.post("/webhook")
async def telegram_webhook(request: Request):
    handler = getattr(request.app.state, "tg_handler", None)
    if handler is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    secret = request.app.state.config.TELEGRAM_WEBHOOK_SECRET
    if secret:
        given = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(given, secret):
            return Response(status_code=status.HTTP_403_FORBIDDEN)

    try:
        data = await request.json()
        update = Update.de_json(data, handler.bot)
    except Exception:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    if update is None or _seen(update.update_id):
        return Response(status_code=status.HTTP_200_OK)

    task = asyncio.create_task(handler.handle_update(update))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    return Response(status_code=status.HTTP_200_OK)
