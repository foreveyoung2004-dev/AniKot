from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from vkbottle.bot import Bot, Message

from .config import Settings
from .db import Database

logger = logging.getLogger(__name__)

SUPPORT_COMMANDS = {"support", "/support", "поддержка", "/поддержка"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ids(value: str) -> set[int]:
    result: set[int] = set()
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                logger.warning("Invalid SUPPORT_ADMIN_IDS item: %r", part)
    return result


def _button(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": {
            "type": "text",
            "label": label,
            "payload": json.dumps(payload, ensure_ascii=False),
        }
    }


def _inline(rows: list[list[dict[str, Any]]]) -> str:
    return json.dumps(
        {"one_time": False, "inline": True, "buttons": rows},
        ensure_ascii=False,
    )


def reply_keyboard(ticket_id: int) -> str:
    return _inline(
        [[_button("Ответить", {"support_action": "reply", "ticket_id": ticket_id})]]
    )


def admin_controls_keyboard(ticket_id: int) -> str:
    return _inline(
        [[
            _button("Ожидать ответ", {"support_action": "wait", "ticket_id": ticket_id}),
            _button("Закрыть чат", {"support_action": "close", "ticket_id": ticket_id}),
        ]]
    )


class SupportDesk:
    def __init__(self, bot: Bot, settings: Settings, db: Database):
        self.bot = bot
        self.settings = settings
        self.db = db
        self.admin_ids = set(settings.admin_ids) | _parse_ids(
            os.getenv("SUPPORT_ADMIN_IDS", "")
        )
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        async with self.db.connection() as conn:
            await conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS support_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_vk_id INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    assigned_admin_vk_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    closed_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_support_active_user
                    ON support_tickets(user_vk_id)
                    WHERE status != 'closed';

                CREATE TABLE IF NOT EXISTS support_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id INTEGER NOT NULL,
                    sender_vk_id INTEGER NOT NULL,
                    sender_role TEXT NOT NULL,
                    text TEXT,
                    attachments TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(ticket_id) REFERENCES support_tickets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS support_admin_sessions (
                    admin_vk_id INTEGER PRIMARY KEY,
                    ticket_id INTEGER NOT NULL,
                    opened_at TEXT NOT NULL,
                    FOREIGN KEY(ticket_id) REFERENCES support_tickets(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_support_ticket_status
                    ON support_tickets(status, updated_at);
                """
            )
            await conn.commit()

    def _is_admin(self, vk_id: int) -> bool:
        return vk_id in self.admin_ids

    @staticmethod
    def _payload(message: Message) -> dict[str, Any]:
        try:
            payload = message.get_payload_json()
        except Exception:
            payload = None
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _ref(message: Message) -> str:
        value = getattr(message, "ref", None)
        return str(value or "").strip().lower()

    @staticmethod
    def _attachments(message: Message) -> list[str]:
        try:
            return list(message.get_attachment_strings() or [])
        except Exception:
            return []

    async def _active_ticket_for_user(self, vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT * FROM support_tickets
                    WHERE user_vk_id=? AND status!='closed'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _ticket_by_id(self, ticket_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM support_tickets WHERE id=?",
                    (ticket_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _active_admin_ticket(self, admin_vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT t.*
                    FROM support_admin_sessions s
                    JOIN support_tickets t ON t.id=s.ticket_id
                    WHERE s.admin_vk_id=? AND t.status!='closed'
                    """,
                    (admin_vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _create_ticket(self, user_vk_id: int) -> dict[str, Any]:
        now = _utc_now_iso()
        async with self._lock:
            existing = await self._active_ticket_for_user(user_vk_id)
            if existing:
                return existing
            async with self.db.connection() as conn:
                cursor = await conn.execute(
                    """
                    INSERT INTO support_tickets(
                        user_vk_id, status, created_at, updated_at
                    ) VALUES (?, 'draft', ?, ?)
                    """,
                    (user_vk_id, now, now),
                )
                await conn.commit()
                ticket_id = int(cursor.lastrowid)
            ticket = await self._ticket_by_id(ticket_id)
            if ticket is None:
                raise RuntimeError("support ticket was not created")
            return ticket

    async def _log_message(
        self,
        ticket_id: int,
        sender_vk_id: int,
        sender_role: str,
        text: str,
        attachments: list[str],
    ) -> None:
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO support_messages(
                    ticket_id, sender_vk_id, sender_role, text, attachments, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    sender_vk_id,
                    sender_role,
                    text or None,
                    json.dumps(attachments, ensure_ascii=False) if attachments else None,
                    _utc_now_iso(),
                ),
            )
            await conn.commit()

    async def _set_ticket(
        self,
        ticket_id: int,
        *,
        status: str | None = None,
        assigned_admin_vk_id: int | None = None,
        set_assignee: bool = False,
        closed: bool = False,
    ) -> None:
        fields = ["updated_at=?"]
        values: list[Any] = [_utc_now_iso()]
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if set_assignee:
            fields.append("assigned_admin_vk_id=?")
            values.append(assigned_admin_vk_id)
        if closed:
            fields.append("closed_at=?")
            values.append(_utc_now_iso())
        values.append(ticket_id)
        async with self.db.connection() as conn:
            await conn.execute(
                f"UPDATE support_tickets SET {', '.join(fields)} WHERE id=?",
                values,
            )
            await conn.commit()

    async def should_handle(self, message: Message) -> bool:
        if message.from_id <= 0:
            return False

        text = (message.text or "").strip().lower()
        payload = self._payload(message)
        action = str(payload.get("support_action") or "")
        ref = self._ref(message)

        if self._is_admin(message.from_id):
            if action in {"reply", "wait", "close"}:
                return True
            return await self._active_admin_ticket(message.from_id) is not None

        if text in SUPPORT_COMMANDS or ref == "support":
            return True
        return await self._active_ticket_for_user(message.from_id) is not None

    async def _notify_admins(
        self,
        ticket: dict[str, Any],
        text: str,
        attachments: list[str],
    ) -> None:
        ticket_id = int(ticket["id"])
        user_vk_id = int(ticket["user_vk_id"])
        assigned = ticket.get("assigned_admin_vk_id")
        recipients = [int(assigned)] if assigned else sorted(self.admin_ids)

        if not recipients:
            raise RuntimeError("SUPPORT_ADMIN_IDS/ADMIN_IDS are empty")

        notification = (
            f"🛠 Новое сообщение в поддержку #{ticket_id}\n"
            f"👤 [id{user_vk_id}|Пользователь]\n"
            f"🆔 ID: {user_vk_id}\n\n"
            f"💬 {text or 'Пользователь отправил вложение.'}"
        )
        kwargs: dict[str, Any] = {
            "random_id": 0,
            "message": notification,
            "keyboard": reply_keyboard(ticket_id),
        }
        if attachments:
            kwargs["attachment"] = ",".join(attachments)

        delivered = 0
        for admin_id in recipients:
            try:
                await self.bot.api.messages.send(peer_id=admin_id, **kwargs)
                delivered += 1
            except Exception:
                logger.exception(
                    "Support notification failed ticket=%s admin=%s",
                    ticket_id,
                    admin_id,
                )

        if delivered == 0:
            raise RuntimeError("support notification was not delivered")

    async def _handle_user(self, message: Message) -> None:
        text = (message.text or "").strip()
        lower = text.lower()
        ref = self._ref(message)

        if not self.admin_ids:
            await message.answer(
                "⚠️ Поддержка временно недоступна. Попробуйте позже."
            )
            return

        await self.db.ensure_user(message.from_id)
        ticket = await self._active_ticket_for_user(message.from_id)

        if lower in SUPPORT_COMMANDS or ref == "support":
            ticket = ticket or await self._create_ticket(message.from_id)
            if ticket["status"] == "draft":
                await message.answer(
                    f"🛠 Поддержка AniKot\n\n"
                    f"Обращение #{ticket['id']} создано.\n"
                    "Опишите проблему или вопрос одним сообщением. "
                    "Можно приложить скриншот."
                )
            else:
                await message.answer(
                    f"🛠 Обращение #{ticket['id']} уже открыто.\n\n"
                    "Напишите сообщение — оно будет передано поддержке."
                )
            return

        if ticket is None:
            ticket = await self._create_ticket(message.from_id)

        attachments = self._attachments(message)
        if not text and not attachments:
            await message.answer(
                "🛠 Отправьте текст сообщения или вложение для поддержки."
            )
            return

        await self._log_message(
            int(ticket["id"]),
            message.from_id,
            "user",
            text,
            attachments,
        )
        await self._set_ticket(int(ticket["id"]), status="open")
        ticket = await self._ticket_by_id(int(ticket["id"])) or ticket

        try:
            await self._notify_admins(ticket, text, attachments)
        except Exception:
            logger.exception("Support delivery failed ticket=%s", ticket["id"])
            await message.answer(
                "⚠️ Не удалось передать сообщение поддержке. Попробуйте чуть позже."
            )
            return

        await message.answer(
            f"✅ Сообщение отправлено в поддержку.\n"
            f"Обращение #{ticket['id']}. Ожидайте ответа."
        )

    async def _claim_ticket(self, admin_vk_id: int, ticket_id: int) -> tuple[bool, str]:
        async with self._lock:
            ticket = await self._ticket_by_id(ticket_id)
            if not ticket or ticket["status"] == "closed":
                return False, "Обращение уже закрыто."

            assigned = ticket.get("assigned_admin_vk_id")
            if assigned and int(assigned) != admin_vk_id:
                return False, "Это обращение уже взял другой оператор."

            async with self.db.connection() as conn:
                previous = await (
                    await conn.execute(
                        """
                        SELECT ticket_id FROM support_admin_sessions
                        WHERE admin_vk_id=?
                        """,
                        (admin_vk_id,),
                    )
                ).fetchone()
                if previous and int(previous["ticket_id"]) != ticket_id:
                    await conn.execute(
                        """
                        UPDATE support_tickets
                        SET status='waiting_user', updated_at=?
                        WHERE id=? AND status='in_progress'
                        """,
                        (_utc_now_iso(), int(previous["ticket_id"])),
                    )

                await conn.execute(
                    "DELETE FROM support_admin_sessions WHERE ticket_id=? AND admin_vk_id!=?",
                    (ticket_id, admin_vk_id),
                )
                await conn.execute(
                    """
                    INSERT INTO support_admin_sessions(admin_vk_id, ticket_id, opened_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(admin_vk_id) DO UPDATE SET
                        ticket_id=excluded.ticket_id,
                        opened_at=excluded.opened_at
                    """,
                    (admin_vk_id, ticket_id, _utc_now_iso()),
                )
                await conn.execute(
                    """
                    UPDATE support_tickets
                    SET status='in_progress',
                        assigned_admin_vk_id=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (admin_vk_id, _utc_now_iso(), ticket_id),
                )
                await conn.commit()

            return True, ""

    async def _handle_admin_action(
        self,
        message: Message,
        action: str,
        ticket_id: int,
    ) -> None:
        ticket = await self._ticket_by_id(ticket_id)
        if not ticket:
            await message.answer("⚠️ Обращение не найдено.")
            return

        if action == "reply":
            ok, reason = await self._claim_ticket(message.from_id, ticket_id)
            if not ok:
                await message.answer(f"⚠️ {reason}")
                return
            await message.answer(
                f"💬 Чат с [id{ticket['user_vk_id']}|пользователем] открыт.\n"
                f"Обращение #{ticket_id}.\n\n"
                "Теперь ваши обычные сообщения в этом диалоге "
                "отправляются только этому пользователю."
            )
            return

        assigned = ticket.get("assigned_admin_vk_id")
        if assigned and int(assigned) != message.from_id:
            await message.answer("⚠️ Это обращение ведёт другой оператор.")
            return

        if action == "wait":
            async with self.db.connection() as conn:
                await conn.execute(
                    "DELETE FROM support_admin_sessions WHERE admin_vk_id=? AND ticket_id=?",
                    (message.from_id, ticket_id),
                )
                await conn.execute(
                    """
                    UPDATE support_tickets
                    SET status='waiting_user', updated_at=?
                    WHERE id=? AND status!='closed'
                    """,
                    (_utc_now_iso(), ticket_id),
                )
                await conn.commit()
            await message.answer(
                f"⏳ Обращение #{ticket_id}: ожидаем ответ пользователя."
            )
            return

        if action == "close":
            async with self.db.connection() as conn:
                await conn.execute(
                    "DELETE FROM support_admin_sessions WHERE ticket_id=?",
                    (ticket_id,),
                )
                await conn.execute(
                    """
                    UPDATE support_tickets
                    SET status='closed', updated_at=?, closed_at=?
                    WHERE id=?
                    """,
                    (_utc_now_iso(), _utc_now_iso(), ticket_id),
                )
                await conn.commit()

            try:
                await self.bot.api.messages.send(
                    peer_id=int(ticket["user_vk_id"]),
                    random_id=0,
                    message=(
                        f"✅ Обращение #{ticket_id} закрыто.\n\n"
                        "Если понадобится помощь снова — напишите support."
                    ),
                )
            except Exception:
                logger.exception("Support close notification failed ticket=%s", ticket_id)

            await message.answer(f"✅ Чат #{ticket_id} закрыт.")
            return

    async def _handle_admin_message(self, message: Message) -> None:
        ticket = await self._active_admin_ticket(message.from_id)
        if ticket is None:
            return

        text = (message.text or "").strip()
        attachments = self._attachments(message)
        if not text and not attachments:
            await message.answer(
                "⚠️ Отправьте текст или вложение для пользователя.",
                keyboard=admin_controls_keyboard(int(ticket["id"])),
            )
            return

        kwargs: dict[str, Any] = {
            "peer_id": int(ticket["user_vk_id"]),
            "random_id": 0,
            "message": text or "🛠 Сообщение от поддержки AniKot",
        }
        if attachments:
            kwargs["attachment"] = ",".join(attachments)

        try:
            await self.bot.api.messages.send(**kwargs)
        except Exception:
            logger.exception(
                "Support admin reply failed ticket=%s admin=%s",
                ticket["id"],
                message.from_id,
            )
            await message.answer(
                "⚠️ Не удалось отправить сообщение пользователю.",
                keyboard=admin_controls_keyboard(int(ticket["id"])),
            )
            return

        await self._log_message(
            int(ticket["id"]),
            message.from_id,
            "admin",
            text,
            attachments,
        )
        await self._set_ticket(int(ticket["id"]), status="in_progress")
        await message.answer(
            "✅ Сообщение успешно отправлено.",
            keyboard=admin_controls_keyboard(int(ticket["id"])),
        )

    async def handle(self, message: Message) -> None:
        payload = self._payload(message)
        action = str(payload.get("support_action") or "")

        if self._is_admin(message.from_id):
            if action in {"reply", "wait", "close"}:
                try:
                    ticket_id = int(payload.get("ticket_id"))
                except (TypeError, ValueError):
                    await message.answer("⚠️ Некорректный номер обращения.")
                    return
                await self._handle_admin_action(message, action, ticket_id)
                return

            await self._handle_admin_message(message)
            return

        await self._handle_user(message)

    def install(self) -> None:
        before = len(self.bot.labeler.message_view.handlers)

        @self.bot.on.private_message(coro=self.should_handle)
        async def support_handler(message: Message):
            await self.handle(message)

        handlers = self.bot.labeler.message_view.handlers
        if len(handlers) > before:
            support_handler_obj = handlers.pop()
            handlers.insert(0, support_handler_obj)


async def install_support(bot: Bot, settings: Settings, db: Database) -> SupportDesk:
    desk = SupportDesk(bot, settings, db)
    await desk.init()
    desk.install()
    setattr(bot, "_anikot_support_desk", desk)
    logger.info("AniKot support desk enabled; admins=%s", sorted(desk.admin_ids))
    return desk
