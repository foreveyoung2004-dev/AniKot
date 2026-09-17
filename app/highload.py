from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import suppress
from typing import Any

from vkbottle import BaseMiddleware
from vkbottle.bot import Message

logger = logging.getLogger(__name__)


class HighLoadGuard:
    """In-process backpressure for bursts before handlers reach SQLite/VK/API."""

    def __init__(self, settings: Any) -> None:
        self.admin_ids = set(getattr(settings, "admin_ids", set()))
        self.window_seconds = max(1.0, float(os.getenv("MESSAGE_RATE_WINDOW_SECONDS", "10")))
        self.user_burst = max(2, min(int(os.getenv("MESSAGE_RATE_BURST", "12")), 100))
        self.max_inflight = max(10, min(int(os.getenv("MAX_INFLIGHT_MESSAGES", "120")), 1000))
        self.warning_interval = max(1.0, float(os.getenv("RATE_LIMIT_WARNING_INTERVAL", "8")))
        self.global_warning_per_second = max(
            0, min(int(os.getenv("OVERLOAD_WARNINGS_PER_SECOND", "5")), 50)
        )

        self._events: dict[int, deque[float]] = {}
        self._last_user_warning: dict[int, float] = {}
        self._global_warning_times: deque[float] = deque()
        self._lock = asyncio.Lock()

        self.active = 0
        self.peak_active = 0
        self.total = 0
        self.rejected_rate = 0
        self.rejected_overload = 0

    def _prune_users(self, now: float) -> None:
        cutoff = now - (self.window_seconds * 3)
        stale = [
            uid for uid, events in self._events.items()
            if not events or events[-1] < cutoff
        ]
        for uid in stale:
            self._events.pop(uid, None)
            self._last_user_warning.pop(uid, None)

    async def admit(self, vk_id: int) -> tuple[bool, bool, str]:
        if vk_id in self.admin_ids:
            async with self._lock:
                self.total += 1
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
            return True, False, "admin"

        now = time.monotonic()
        async with self._lock:
            self.total += 1
            if self.total % 1000 == 0:
                self._prune_users(now)

            # Shed global overload before allocating per-user tracking state.
            # This keeps a burst of many one-off users from growing the dicts.
            if self.active >= self.max_inflight:
                self.rejected_overload += 1
                one_second_ago = now - 1.0
                while self._global_warning_times and self._global_warning_times[0] < one_second_ago:
                    self._global_warning_times.popleft()
                notify = (
                    self.global_warning_per_second > 0
                    and len(self._global_warning_times) < self.global_warning_per_second
                )
                if notify:
                    self._global_warning_times.append(now)
                return False, notify, "overload"

            events = self._events.setdefault(vk_id, deque())
            cutoff = now - self.window_seconds
            while events and events[0] < cutoff:
                events.popleft()

            if len(events) >= self.user_burst:
                self.rejected_rate += 1
                last = self._last_user_warning.get(vk_id, 0.0)
                notify = now - last >= self.warning_interval
                if notify:
                    self._last_user_warning[vk_id] = now
                return False, notify, "rate_limit"

            events.append(now)
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            return True, False, "ok"

    async def release(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "active_messages": self.active,
            "peak_active_messages": self.peak_active,
            "total_messages": self.total,
            "rejected_rate_limit": self.rejected_rate,
            "rejected_overload": self.rejected_overload,
            "max_inflight_messages": self.max_inflight,
            "rate_burst": self.user_burst,
            "rate_window_seconds": self.window_seconds,
        }


def install_highload_guard(bot: Any, settings: Any) -> HighLoadGuard:
    guard = HighLoadGuard(settings)

    class AniKotLoadMiddleware(BaseMiddleware[Message]):
        async def pre(self) -> None:
            self._anikot_admitted = False
            vk_id = int(getattr(self.event, "from_id", 0) or 0)
            if vk_id <= 0:
                return

            admitted, notify, reason = await guard.admit(vk_id)
            if admitted:
                self._anikot_admitted = True
                return

            if notify:
                text = (
                    "⏳ Слишком много сообщений. Подождите несколько секунд."
                    if reason == "rate_limit"
                    else "⏳ AniKot сейчас перегружен. Повторите запрос через несколько секунд."
                )
                with suppress(Exception):
                    await self.event.answer(text)
            self.stop(reason)

        async def post(self) -> None:
            if getattr(self, "_anikot_admitted", False):
                await guard.release()
                self._anikot_admitted = False

    bot.labeler.message_view.register_middleware(AniKotLoadMiddleware)
    setattr(bot, "_anikot_load_guard", guard)
    logger.info(
        "High-load guard enabled max_inflight=%s burst=%s/%ss",
        guard.max_inflight,
        guard.user_burst,
        guard.window_seconds,
    )
    return guard
