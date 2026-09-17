from __future__ import annotations

import asyncio
import logging
from typing import Any

from vkbottle.bot import Message

logger = logging.getLogger(__name__)

_PROGRESS_PREFIXES = (
    "🔍 AniKot анализирует кадр...",
    "🔍 AniKot выполняет поиск...",
)

_TERMINAL_PREFIXES = (
    "🎬",
    "🤔",
    "😿",
    "🚫",
    "⛔",
    "🔞",
    "💳",
    "⚠️",
)

_TERMINAL_MARKERS = (
    "Поиск отменён",
    "Для этого запроса требуется AniKot Pro+",
)


def _outgoing_message_id(result: Any) -> int | None:
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        for key in ("message_id", "id"):
            value = result.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return None
    for name in ("message_id", "id"):
        value = getattr(result, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _message_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if args:
        value = args[0]
    else:
        value = kwargs.get("message")
    return str(value or "")


def _is_progress(text: str) -> bool:
    return text.startswith(_PROGRESS_PREFIXES)


def _is_terminal(text: str) -> bool:
    return text.startswith(_TERMINAL_PREFIXES) or any(
        marker in text for marker in _TERMINAL_MARKERS
    )


def install_search_progress_cleanup() -> None:
    """Remove the temporary AniKot search-status message after a search finishes."""
    if getattr(Message, "_anikot_progress_cleanup_installed", False):
        return

    original_answer = Message.answer
    progress_messages: dict[tuple[int, int], int] = {}
    lock = asyncio.Lock()

    async def delete_progress(message: Message, message_id: int) -> None:
        try:
            await message.ctx_api.messages.delete(
                message_ids=[message_id],
                delete_for_all=True,
            )
        except Exception:
            logger.exception(
                "Failed to delete AniKot progress message id=%s peer=%s",
                message_id,
                message.peer_id,
            )

    async def patched_answer(self: Message, *args: Any, **kwargs: Any):
        text = _message_text(args, kwargs)
        key = (int(self.peer_id), int(self.from_id))

        if _is_progress(text):
            result = await original_answer(self, *args, **kwargs)
            message_id = _outgoing_message_id(result)
            if message_id:
                async with lock:
                    previous = progress_messages.get(key)
                    progress_messages[key] = message_id
                if previous and previous != message_id:
                    await delete_progress(self, previous)
            return result

        result = await original_answer(self, *args, **kwargs)

        if _is_terminal(text):
            async with lock:
                message_id = progress_messages.pop(key, None)
            if message_id:
                await delete_progress(self, message_id)

        return result

    Message.answer = patched_answer
    setattr(Message, "_anikot_progress_cleanup_installed", True)
    logger.info("AniKot search progress auto-delete enabled")
