from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from vkbottle.bot import Bot, Message

from .config import Settings
from .db import Database
from .support_ai import SupportAI

logger = logging.getLogger(__name__)

SUPPORT_COMMANDS = {"support", "/support", "поддержка", "/поддержка"}
AI_TOPICS = {
    "search": "Поиск и режимы",
    "account": "Аккаунт и запросы",
    "bonuses": "Бонусы и рефералы",
    "features": "Функции AniKot",
    "other": "Другое",
}
HUMAN_TOPICS = {
    "technical": "Техническая проблема",
    "payment": "Оплата",
}


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


def topics_keyboard() -> str:
    return _inline([
        [
            _button("Техническая проблема", {"support_topic": "technical"}),
            _button("Оплата", {"support_topic": "payment"}),
        ],
        [
            _button("Поиск и режимы", {"support_topic": "search"}),
            _button("Аккаунт и запросы", {"support_topic": "account"}),
        ],
        [
            _button("Бонусы и рефералы", {"support_topic": "bonuses"}),
            _button("Функции AniKot", {"support_topic": "features"}),
        ],
        [_button("Другое", {"support_topic": "other"})],
        [_button("Закрыть поддержку", {"support_action": "exit"})],
    ])


def ai_controls_keyboard() -> str:
    return _inline([
        [
            _button("Сменить тему", {"support_action": "topics"}),
            _button("Закрыть поддержку", {"support_action": "exit"}),
        ]
    ])


def human_wait_keyboard() -> str:
    return _inline([
        [_button("Сменить тему", {"support_action": "topics"})],
        [_button("Закрыть поддержку", {"support_action": "exit"})],
    ])


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
    def __init__(self, bot: Bot, settings: Settings, db: Database, ai: SupportAI):
        self.bot = bot
        self.settings = settings
        self.db = db
        self.ai = ai
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
                    topic TEXT,
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

                CREATE TABLE IF NOT EXISTS support_user_sessions (
                    user_vk_id INTEGER PRIMARY KEY,
                    mode TEXT NOT NULL,
                    topic TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS support_ai_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_vk_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_support_ticket_status
                    ON support_tickets(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_support_ai_user
                    ON support_ai_messages(user_vk_id, id);
                """
            )
            columns = {
                row["name"]
                for row in await (await conn.execute("PRAGMA table_info(support_tickets)")).fetchall()
            }
            if "topic" not in columns:
                await conn.execute("ALTER TABLE support_tickets ADD COLUMN topic TEXT")
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
        return str(getattr(message, "ref", None) or "").strip().lower()

    @staticmethod
    def _attachments(message: Message) -> list[str]:
        try:
            return list(message.get_attachment_strings() or [])
        except Exception:
            return []

    async def _session(self, vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM support_user_sessions WHERE user_vk_id=?",
                    (vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _set_session(self, vk_id: int, mode: str, topic: str | None = None) -> None:
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO support_user_sessions(user_vk_id, mode, topic, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_vk_id) DO UPDATE SET
                    mode=excluded.mode,
                    topic=excluded.topic,
                    updated_at=excluded.updated_at
                """,
                (vk_id, mode, topic, _utc_now_iso()),
            )
            await conn.commit()

    async def _clear_session(self, vk_id: int) -> None:
        async with self.db.connection() as conn:
            await conn.execute("DELETE FROM support_user_sessions WHERE user_vk_id=?", (vk_id,))
            await conn.commit()

    async def _clear_ai_history(self, vk_id: int) -> None:
        async with self.db.connection() as conn:
            await conn.execute("DELETE FROM support_ai_messages WHERE user_vk_id=?", (vk_id,))
            await conn.commit()

    async def _ai_history(self, vk_id: int) -> list[dict[str, str]]:
        async with self.db.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT role, content FROM support_ai_messages
                    WHERE user_vk_id=? ORDER BY id DESC LIMIT ?
                    """,
                    (vk_id, self.ai.history_limit),
                )
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    async def _save_ai_message(self, vk_id: int, role: str, content: str) -> None:
        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO support_ai_messages(user_vk_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (vk_id, role, content[:4000], _utc_now_iso()),
            )
            await conn.execute(
                """
                DELETE FROM support_ai_messages
                WHERE user_vk_id=? AND id NOT IN (
                    SELECT id FROM support_ai_messages
                    WHERE user_vk_id=? ORDER BY id DESC LIMIT 24
                )
                """,
                (vk_id, vk_id),
            )
            await conn.commit()

    async def _ai_rate_limited(self, vk_id: int) -> bool:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM support_ai_messages
                    WHERE user_vk_id=? AND role='user' AND created_at>=?
                    """,
                    (vk_id, cutoff),
                )
            ).fetchone()
            return int(row["n"] or 0) >= self.ai.hourly_limit

    async def _active_ticket_for_user(self, vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT * FROM support_tickets
                    WHERE user_vk_id=? AND status NOT IN ('closed', 'draft')
                    ORDER BY id DESC LIMIT 1
                    """,
                    (vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _draft_ticket_for_user(self, vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT * FROM support_tickets
                    WHERE user_vk_id=? AND status='draft'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _ticket_by_id(self, ticket_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute("SELECT * FROM support_tickets WHERE id=?", (ticket_id,))
            ).fetchone()
            return dict(row) if row else None

    async def _active_admin_ticket(self, admin_vk_id: int) -> dict[str, Any] | None:
        async with self.db.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT t.* FROM support_admin_sessions s
                    JOIN support_tickets t ON t.id=s.ticket_id
                    WHERE s.admin_vk_id=? AND t.status!='closed'
                    """,
                    (admin_vk_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    async def _create_or_reuse_ticket(self, user_vk_id: int, topic: str) -> dict[str, Any]:
        async with self._lock:
            active = await self._active_ticket_for_user(user_vk_id)
            if active:
                return active
            draft = await self._draft_ticket_for_user(user_vk_id)
            if draft:
                async with self.db.connection() as conn:
                    await conn.execute(
                        "UPDATE support_tickets SET topic=?, status='awaiting_user_issue', updated_at=? WHERE id=?",
                        (topic, _utc_now_iso(), int(draft["id"])),
                    )
                    await conn.commit()
                return await self._ticket_by_id(int(draft["id"])) or draft

            now = _utc_now_iso()
            async with self.db.connection() as conn:
                cursor = await conn.execute(
                    """
                    INSERT INTO support_tickets(user_vk_id, topic, status, created_at, updated_at)
                    VALUES (?, ?, 'awaiting_user_issue', ?, ?)
                    """,
                    (user_vk_id, topic, now, now),
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
                INSERT INTO support_messages(ticket_id, sender_vk_id, sender_role, text, attachments, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
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

    async def _set_ticket_status(self, ticket_id: int, status: str) -> None:
        async with self.db.connection() as conn:
            await conn.execute(
                "UPDATE support_tickets SET status=?, updated_at=? WHERE id=?",
                (status, _utc_now_iso(), ticket_id),
            )
            await conn.commit()

    async def should_handle(self, message: Message) -> bool:
        if message.from_id <= 0:
            return False
        payload = self._payload(message)
        action = str(payload.get("support_action") or "")
        topic = str(payload.get("support_topic") or "")
        text = (message.text or "").strip().lower()

        if self._is_admin(message.from_id):
            if action in {"reply", "wait", "close"}:
                return True
            return await self._active_admin_ticket(message.from_id) is not None

        if action in {"topics", "exit"} or topic in AI_TOPICS or topic in HUMAN_TOPICS:
            return True
        if text in SUPPORT_COMMANDS or self._ref(message) == "support":
            return True
        if await self._session(message.from_id):
            return True
        return await self._active_ticket_for_user(message.from_id) is not None

    async def _show_topics(self, message: Message) -> None:
        await self._set_session(message.from_id, "choose_topic")
        await message.answer(
            "🛠 Поддержка AniKot\n\n"
            "Выберите тему вопроса.\n"
            "Технические проблемы и вопросы оплаты сразу передаются администратору. "
            "По остальным темам сначала поможет AI-поддержка.",
            keyboard=topics_keyboard(),
        )

    async def _notify_admins(
        self,
        ticket: dict[str, Any],
        text: str,
        attachments: list[str],
        *,
        escalation_reason: str | None = None,
    ) -> None:
        ticket_id = int(ticket["id"])
        user_vk_id = int(ticket["user_vk_id"])
        assigned = ticket.get("assigned_admin_vk_id")
        recipients = [int(assigned)] if assigned else sorted(self.admin_ids)
        if not recipients:
            raise RuntimeError("SUPPORT_ADMIN_IDS/ADMIN_IDS are empty")

        topic_key = str(ticket.get("topic") or "other")
        topic_name = HUMAN_TOPICS.get(topic_key) or AI_TOPICS.get(topic_key) or topic_key
        extra = f"\n🤖 Причина передачи: {escalation_reason}" if escalation_reason else ""
        notification = (
            f"🛠 Новое сообщение в поддержку #{ticket_id}\n"
            f"📂 Тема: {topic_name}\n"
            f"👤 [id{user_vk_id}|Пользователь]\n"
            f"🆔 ID: {user_vk_id}\n\n"
            f"💬 {text or 'Пользователь отправил вложение.'}{extra}"
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
                logger.exception("Support notification failed ticket=%s admin=%s", ticket_id, admin_id)
        if delivered == 0:
            raise RuntimeError("support notification was not delivered")

    async def _forward_user_message(
        self,
        message: Message,
        ticket: dict[str, Any],
        *,
        escalation_reason: str | None = None,
    ) -> None:
        text = (message.text or "").strip()
        attachments = self._attachments(message)
        if not text and not attachments:
            await message.answer("🛠 Отправьте текст сообщения или вложение для поддержки.")
            return

        await self._log_message(int(ticket["id"]), message.from_id, "user", text, attachments)
        await self._set_ticket_status(int(ticket["id"]), "open")
        ticket = await self._ticket_by_id(int(ticket["id"])) or ticket
        try:
            await self._notify_admins(
                ticket,
                text,
                attachments,
                escalation_reason=escalation_reason,
            )
        except Exception:
            logger.exception("Support delivery failed ticket=%s", ticket["id"])
            await message.answer("⚠️ Не удалось передать сообщение поддержке. Попробуйте чуть позже.")
            return
        await self._set_session(message.from_id, "human", str(ticket.get("topic") or "other"))
        await message.answer(
            f"✅ Сообщение отправлено специалисту.\n"
            f"Обращение #{ticket['id']}. Ожидайте ответа.",
            keyboard=human_wait_keyboard(),
        )

    async def _select_topic(self, message: Message, topic: str) -> None:
        if topic in HUMAN_TOPICS:
            if not self.admin_ids:
                await message.answer("⚠️ Живая поддержка временно недоступна. Попробуйте позже.")
                return
            ticket = await self._create_or_reuse_ticket(message.from_id, topic)
            await self._set_session(message.from_id, "human_pending", topic)
            await message.answer(
                f"🛠 Тема: {HUMAN_TOPICS[topic]}\n\n"
                "Опишите вопрос или проблему одним сообщением. Можно приложить скриншот. "
                "Сообщение сразу получит администратор.",
                keyboard=human_wait_keyboard(),
            )
            return

        if topic in AI_TOPICS:
            await self._clear_ai_history(message.from_id)
            await self._set_session(message.from_id, "ai", topic)
            await message.answer(
                f"🤖 AI-поддержка AniKot\n"
                f"Тема: {AI_TOPICS[topic]}\n\n"
                "Напишите вопрос. Если он окажется техническим, связанным с оплатой или AI не будет знать точного ответа, я автоматически передам его администратору.",
                keyboard=ai_controls_keyboard(),
            )

    @staticmethod
    def _forced_human_topic(question: str) -> str | None:
        value = question.lower()
        payment_terms = (
            "оплат", "платёж", "платеж", "деньг", "возврат", "lava",
            "чек", "счёт", "счет", "банковск", "карта", "списал", "покупк",
            "не начислили после", "не начислился после",
        )
        technical_terms = (
            "ошибка", "баг", "сбой", "не работает", "не запуска", "завис",
            "вылет", "не отвечает", "сломал", "сломана", "сломано", "сломанный",
            "кнопка не", "поиск завис", "сервер", " api", "http ",
            "500", "502", "503", "таймаут", "timeout",
        )
        if any(term in value for term in payment_terms):
            return "payment"
        if any(term in value for term in technical_terms):
            return "technical"
        return None

    async def _handle_ai_question(self, message: Message, topic: str) -> None:
        question = (message.text or "").strip()
        if not question:
            await message.answer("🤖 Напишите вопрос текстом.", keyboard=ai_controls_keyboard())
            return
        forced_topic = self._forced_human_topic(question)
        if forced_topic:
            if not self.admin_ids:
                await message.answer(
                    "⚠️ Для этого вопроса нужен специалист, но живая поддержка сейчас недоступна. Попробуйте позже.",
                    keyboard=ai_controls_keyboard(),
                )
                return
            ticket = await self._create_or_reuse_ticket(message.from_id, forced_topic)
            await self._set_session(message.from_id, "human", forced_topic)
            await self._log_message(int(ticket["id"]), message.from_id, "user", question, [])
            await self._set_ticket_status(int(ticket["id"]), "open")
            ticket = await self._ticket_by_id(int(ticket["id"])) or ticket
            try:
                await self._notify_admins(
                    ticket,
                    question,
                    [],
                    escalation_reason="техническая или платёжная тема",
                )
            except Exception:
                logger.exception("Forced support escalation failed ticket=%s", ticket["id"])
                await message.answer("⚠️ Не удалось передать вопрос специалисту. Попробуйте позже.")
                return
            await message.answer(
                f"🛠 Этот вопрос требует специалиста.\n\n✅ Передал администратору. Обращение #{ticket['id']}.",
                keyboard=human_wait_keyboard(),
            )
            return

        if await self._ai_rate_limited(message.from_id):
            await message.answer(
                "⏳ Лимит вопросов AI-поддержке на этот час достигнут. Попробуйте позже или смените тему, если нужен администратор.",
                keyboard=ai_controls_keyboard(),
            )
            return

        user = await self.db.get_user(message.from_id) or {}
        history = await self._ai_history(message.from_id)
        await self._save_ai_message(message.from_id, "user", question)
        try:
            result = await self.ai.ask(topic, question, user, history)
        except Exception:
            logger.exception("Support AI failed user=%s topic=%s", message.from_id, topic)
            result = {
                "answer": "Не удалось получить надёжный ответ. Передаю вопрос специалисту.",
                "needs_admin": True,
                "reason": "ai_error",
            }

        answer = str(result.get("answer") or "").strip()
        needs_admin = bool(result.get("needs_admin"))
        reason = str(result.get("reason") or "ai_uncertain")

        if needs_admin:
            if not self.admin_ids:
                await message.answer(
                    "⚠️ Для этого вопроса нужен специалист, но живая поддержка сейчас недоступна. Попробуйте позже.",
                    keyboard=ai_controls_keyboard(),
                )
                return
            ticket = await self._create_or_reuse_ticket(message.from_id, topic)
            await self._set_session(message.from_id, "human", topic)
            await self._log_message(int(ticket["id"]), message.from_id, "user", question, [])
            await self._set_ticket_status(int(ticket["id"]), "open")
            ticket = await self._ticket_by_id(int(ticket["id"])) or ticket
            try:
                await self._notify_admins(ticket, question, [], escalation_reason=reason)
            except Exception:
                logger.exception("AI escalation delivery failed ticket=%s", ticket["id"])
                await message.answer("⚠️ Не удалось передать вопрос специалисту. Попробуйте позже.")
                return
            await message.answer(
                (answer or "Этот вопрос лучше проверить вручную.")
                + f"\n\n✅ Передал специалисту. Обращение #{ticket['id']}.",
                keyboard=human_wait_keyboard(),
            )
            return

        await self._save_ai_message(message.from_id, "assistant", answer)
        await message.answer(f"🤖 {answer}", keyboard=ai_controls_keyboard())

    async def _handle_user(self, message: Message) -> None:
        payload = self._payload(message)
        action = str(payload.get("support_action") or "")
        topic = str(payload.get("support_topic") or "")
        text = (message.text or "").strip()
        lower = text.lower()

        await self.db.ensure_user(message.from_id)

        if action == "exit":
            active = await self._active_ticket_for_user(message.from_id)
            if active and active.get("status") == "awaiting_user_issue":
                await self._set_ticket_status(int(active["id"]), "closed")
                active = None
            if active:
                await message.answer(
                    f"⚠️ Обращение #{active['id']} сейчас ведёт специалист. "
                    "Закрыть активный чат может оператор после завершения диалога.",
                    keyboard=human_wait_keyboard(),
                )
                return
            await self._clear_session(message.from_id)
            await self._clear_ai_history(message.from_id)
            await message.answer("✅ Поддержка закрыта. Чтобы открыть её снова, напишите support.")
            return

        if action == "topics":
            active = await self._active_ticket_for_user(message.from_id)
            if active and active.get("status") == "awaiting_user_issue":
                await self._set_ticket_status(int(active["id"]), "closed")
                active = None
            if active:
                await message.answer(
                    f"⚠️ У вас уже есть открытое обращение #{active['id']}. "
                    "Сначала дождитесь ответа или попросите администратора закрыть чат.",
                    keyboard=human_wait_keyboard(),
                )
                return
            await self._show_topics(message)
            return

        if topic in AI_TOPICS or topic in HUMAN_TOPICS:
            await self._select_topic(message, topic)
            return

        if lower in SUPPORT_COMMANDS or self._ref(message) == "support":
            active = await self._active_ticket_for_user(message.from_id)
            if active:
                await self._set_session(message.from_id, "human", str(active.get("topic") or "other"))
                await message.answer(
                    f"🛠 Обращение #{active['id']} уже открыто.\n\n"
                    "Напишите сообщение — оно будет передано специалисту.",
                    keyboard=human_wait_keyboard(),
                )
                return
            await self._show_topics(message)
            return

        active = await self._active_ticket_for_user(message.from_id)
        if active:
            await self._forward_user_message(message, active)
            return

        session = await self._session(message.from_id)
        if not session or session.get("mode") == "choose_topic":
            await self._show_topics(message)
            return

        mode = str(session.get("mode") or "")
        topic = str(session.get("topic") or "other")
        if mode == "human_pending":
            ticket = await self._create_or_reuse_ticket(message.from_id, topic)
            await self._forward_user_message(message, ticket)
            return
        if mode == "human":
            ticket = await self._active_ticket_for_user(message.from_id)
            if ticket:
                await self._forward_user_message(message, ticket)
            else:
                await self._show_topics(message)
            return
        if mode == "ai":
            await self._handle_ai_question(message, topic)
            return
        await self._show_topics(message)

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
                        "SELECT ticket_id FROM support_admin_sessions WHERE admin_vk_id=?",
                        (admin_vk_id,),
                    )
                ).fetchone()
                if previous and int(previous["ticket_id"]) != ticket_id:
                    await conn.execute(
                        """
                        UPDATE support_tickets SET status='waiting_user', updated_at=?
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
                        ticket_id=excluded.ticket_id, opened_at=excluded.opened_at
                    """,
                    (admin_vk_id, ticket_id, _utc_now_iso()),
                )
                await conn.execute(
                    """
                    UPDATE support_tickets
                    SET status='in_progress', assigned_admin_vk_id=?, updated_at=?
                    WHERE id=?
                    """,
                    (admin_vk_id, _utc_now_iso(), ticket_id),
                )
                await conn.commit()
            return True, ""

    async def _handle_admin_action(self, message: Message, action: str, ticket_id: int) -> None:
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
                "Теперь ваши обычные сообщения в этом диалоге отправляются только этому пользователю."
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
                    "UPDATE support_tickets SET status='waiting_user', updated_at=? WHERE id=? AND status!='closed'",
                    (_utc_now_iso(), ticket_id),
                )
                await conn.commit()
            await message.answer(f"⏳ Обращение #{ticket_id}: ожидаем ответ пользователя.")
            return

        if action == "close":
            async with self.db.connection() as conn:
                await conn.execute("DELETE FROM support_admin_sessions WHERE ticket_id=?", (ticket_id,))
                now = _utc_now_iso()
                await conn.execute(
                    "UPDATE support_tickets SET status='closed', updated_at=?, closed_at=? WHERE id=?",
                    (now, now, ticket_id),
                )
                await conn.execute(
                    "DELETE FROM support_user_sessions WHERE user_vk_id=?",
                    (int(ticket["user_vk_id"]),),
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
            logger.exception("Support admin reply failed ticket=%s admin=%s", ticket["id"], message.from_id)
            await message.answer(
                "⚠️ Не удалось отправить сообщение пользователю.",
                keyboard=admin_controls_keyboard(int(ticket["id"])),
            )
            return

        await self._log_message(int(ticket["id"]), message.from_id, "admin", text, attachments)
        await self._set_ticket_status(int(ticket["id"]), "in_progress")
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
            handler = handlers.pop()
            handlers.insert(0, handler)


async def install_support(
    bot: Bot,
    settings: Settings,
    db: Database,
    ai: SupportAI,
) -> SupportDesk:
    desk = SupportDesk(bot, settings, db, ai)
    await desk.init()
    desk.install()
    setattr(bot, "_anikot_support_desk", desk)
    logger.info(
        "AniKot hybrid support enabled; admins=%s ai_model=%s",
        sorted(desk.admin_ids),
        ai.preferred_model,
    )
    return desk
