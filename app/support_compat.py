from __future__ import annotations

import logging
from typing import Any

from vkbottle.bot import Message

logger = logging.getLogger(__name__)


def patch_support_rule_registration() -> None:
    """Fix vkbottle 4.11 support routing.

    ``coro=`` builds CoroutineRule, which invokes a coroutine function with no
    event arguments. SupportDesk.should_handle needs Message, so it must be
    registered through FuncRule (the ``func=`` shortcut).
    """
    from .support import SupportDesk

    if getattr(SupportDesk, "_anikot_rule_fix_applied", False):
        return

    def install(self: Any) -> None:
        before = len(self.bot.labeler.message_view.handlers)

        @self.bot.on.private_message(func=self.should_handle)
        async def support_handler(message: Message):
            await self.handle(message)

        handlers = self.bot.labeler.message_view.handlers
        if len(handlers) > before:
            handler = handlers.pop()
            handlers.insert(0, handler)

    SupportDesk.install = install
    SupportDesk._anikot_rule_fix_applied = True
    logger.info("Support routing compatibility fix applied (CoroutineRule -> FuncRule)")
