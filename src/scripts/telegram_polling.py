"""Run the Telegram bot locally with long polling — no public URL needed.

    cd src && python -m scripts.telegram_polling

It removes any webhook first (Telegram allows only one delivery mode at a
time). When you are done, start the deployed app again — its startup sets the
webhook back.
"""

import asyncio
import logging

from helpers.bootstrap import build_services
from helpers.session_store import SessionStore
from helpers.usage_store import UsageStore
from telegram import Bot
from telegram_bot import TelegramChatHandler
from telegram_bot.handler import BOT_COMMANDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("telegram-polling")


async def main() -> None:
    services = build_services()
    config = services.config
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")
    if services.agent is None:
        raise SystemExit(f"Agent could not be built: {services.errors}")

    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    sessions = SessionStore(max_turns=config.SESSION_MAX_TURNS, ttl_seconds=config.SESSION_TTL_SECONDS)
    usage = None
    if services.config.USAGE_TRACKING_ENABLED:
        usage = UsageStore(getattr(services.vector_db, "client", None), collection=services.config.USAGE_COLLECTION_NAME)
    handler = TelegramChatHandler(bot, services.agent, sessions, services.skills, usage=usage)
    tasks = set()

    async with bot:
        await bot.delete_webhook(drop_pending_updates=False)
        await bot.set_my_commands(BOT_COMMANDS)
        me = await bot.get_me()
        logger.info("Polling as @%s — press Ctrl+C to stop", me.username)
        offset = None
        try:
            while True:
                updates = await bot.get_updates(
                    offset=offset, timeout=30, allowed_updates=["message", "edited_message"],
                )
                for update in updates:
                    offset = update.update_id + 1
                    task = asyncio.create_task(handler.handle_update(update))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
        finally:
            services.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
