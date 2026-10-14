"""Turn the agent's Markdown into Telegram-safe HTML, in chunks Telegram accepts.

Telegram rejects messages over 4096 characters and its legacy Markdown parser
breaks on the `**bold**` and `[text](url)` the LLM writes, so we convert a
small, safe subset of Markdown to Telegram HTML and split long answers on
paragraph boundaries.
"""

from __future__ import annotations

import html
import re
from typing import List

TELEGRAM_LIMIT = 4096
SAFE_CHUNK = 3800  # leave room for HTML tags added after splitting


def split_message(text: str, limit: int = SAFE_CHUNK) -> List[str]:
    """Split text into chunks <= limit, preferring paragraph, then line, then word breaks."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 3:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 3:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
_CODE = re.compile(r"`([^`\n]+)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_BULLET = re.compile(r"^(\s*)[*-]\s+", re.MULTILINE)


def markdown_to_telegram_html(text: str) -> str:
    """Convert common Markdown to the HTML subset Telegram supports."""
    placeholders: List[str] = []

    def _stash(value: str) -> str:
        placeholders.append(value)
        return f"\x00{len(placeholders) - 1}\x00"

    # Protect code and links before escaping/other replacements touch them.
    text = _CODE.sub(lambda m: _stash(f"<code>{html.escape(m.group(1))}</code>"), text or "")
    text = _LINK.sub(
        lambda m: _stash(f'<a href="{html.escape(m.group(2), quote=True)}">{html.escape(m.group(1))}</a>'), text,
    )
    text = html.escape(text, quote=False)
    text = _HEADING.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _BULLET.sub(lambda m: f"{m.group(1)}• ", text)
    text = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _ITALIC.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], text)
    return text
