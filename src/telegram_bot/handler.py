"""Telegram conversation handler, shared by webhook mode and polling mode."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from telegram import Bot, LinkPreviewOptions, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest

from agent.safety import EMERGENCY_NUMBERS
from telegram_bot.formatting import markdown_to_telegram_html, split_message

logger = logging.getLogger(__name__)

NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

WELCOME = (
    "👋 Hello! I'm <b>Tabeeby (طبيبي)</b>, your medical navigator in Egypt.\n\n"
    "Tell me what you feel — in English or Arabic, even Egyptian slang — and I'll "
    "suggest the right specialty and real Vezeeta doctors with booking links. "
    "You can add your area and budget, e.g. <i>\"ضرسي بيوجعني في المعادي وميزانيتي 400\"</i>.\n\n"
    "🚨 In an emergency call <b>123</b> (ambulance).\n"
    "ℹ️ I don't diagnose or prescribe. /help for commands."
)

HELP = (
    "<b>Commands</b>\n"
    "/start — start over\n"
    "/reset — forget this conversation\n"
    "/emergency — emergency numbers in Egypt\n"
    "/skills — what I know how to do\n"
    "/help — this message\n\n"
    "Just type your symptoms, area and budget to find a doctor."
)

BOT_COMMANDS = [
    ("start", "Start over"),
    ("reset", "Forget this conversation"),
    ("emergency", "Emergency numbers in Egypt"),
    ("skills", "What Tabeeby can do"),
    ("help", "How to use Tabeeby"),
]


class TelegramChatHandler:
    def __init__(self, bot: Bot, agent: Any, sessions: Any, skills: Any = None):
        self.bot = bot
        self.agent = agent
        self.sessions = sessions
        self.skills = skills
        self._locks: Dict[int, asyncio.Lock] = {}

    def _lock(self, chat_id: int) -> asyncio.Lock:
        if len(self._locks) > 5000:
            self._locks = {k: v for k, v in self._locks.items() if v.locked()}
        return self._locks.setdefault(chat_id, asyncio.Lock())

    async def send(self, chat_id: int, text: str, markdown: bool = True) -> None:
        """Send a possibly long answer as several messages, HTML-formatted with plain-text fallback."""
        for chunk in split_message(text):
            if markdown:
                try:
                    await self.bot.send_message(
                        chat_id=chat_id, text=markdown_to_telegram_html(chunk),
                        parse_mode=ParseMode.HTML, link_preview_options=NO_PREVIEW,
                    )
                    continue
                except BadRequest as e:
                    logger.warning("HTML send failed (%s); falling back to plain text", e)
            await self.bot.send_message(chat_id=chat_id, text=chunk, link_preview_options=NO_PREVIEW)

    async def _keep_typing(self, chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass

    async def _command(self, chat_id: int, text: str) -> bool:
        cmd = text.split()[0].split("@")[0].lower()
        if cmd == "/start":
            self.sessions.clear(chat_id)
            await self.bot.send_message(chat_id=chat_id, text=WELCOME, parse_mode=ParseMode.HTML)
        elif cmd == "/help":
            await self.bot.send_message(chat_id=chat_id, text=HELP, parse_mode=ParseMode.HTML)
        elif cmd == "/reset":
            self.sessions.clear(chat_id)
            await self.bot.send_message(chat_id=chat_id, text="🧹 Conversation cleared. How can I help?")
        elif cmd == "/emergency":
            lines = [f"• {name}: <b>{num}</b>" for name, num in EMERGENCY_NUMBERS.items()]
            await self.bot.send_message(
                chat_id=chat_id, parse_mode=ParseMode.HTML,
                text="🚨 <b>Emergency numbers (Egypt)</b>\n" + "\n".join(lines)
                + "\n\nIf someone's life is in danger, call <b>123</b> now.",
            )
        elif cmd == "/skills":
            if not self.skills or not self.skills.list():
                await self.bot.send_message(chat_id=chat_id, text="No skills are loaded.")
            else:
                lines = [f"• <b>{s.name}</b> — {s.description.split('. Use when')[0]}" for s in self.skills.list()]
                await self.bot.send_message(chat_id=chat_id, parse_mode=ParseMode.HTML,
                                            text="🧠 <b>What I can do</b>\n" + "\n".join(lines))
        else:
            return False
        return True

    async def handle_update(self, update: Optional[Update]) -> None:
        """Process one Telegram update end-to-end. Never raises."""
        message = getattr(update, "message", None) or getattr(update, "edited_message", None)
        if not message:
            return
        chat_id = message.chat_id
        try:
            if not message.text:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text="I can only read text messages for now — please describe your symptoms in writing. ✍️",
                )
                return

            text = message.text.strip()
            if text.startswith("/") and await self._command(chat_id, text):
                return

            async with self._lock(chat_id):  # one answer at a time per chat
                stop = asyncio.Event()
                typing = asyncio.create_task(self._keep_typing(chat_id, stop))
                try:
                    history = self.sessions.get(chat_id)
                    result = await asyncio.to_thread(self.agent.run, prompt=text, chat_history=history)
                    self.sessions.set(chat_id, result.history)
                    logger.info(
                        "telegram chat=%s stop=%s verified=%s skills=%s usage=%s",
                        chat_id, result.stop_reason, result.verified, result.skills_used, result.usage,
                    )
                finally:
                    stop.set()
                    await typing
                await self.send(chat_id, result.answer)
        except Exception:
            logger.exception("Error handling Telegram update")
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text="Sorry, I ran into a problem answering that. Please try again. "
                         "If this is an emergency, call 123.",
                )
            except Exception:
                pass
