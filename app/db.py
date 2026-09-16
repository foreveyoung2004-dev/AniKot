from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

from .config import Package, Settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _balance_column(balance_type: str) -> str:
    mapping = {
        "anikot": "requests_balance",
        "pro": "pro_balance",
        "proplus": "proplus_balance",
    }
    try:
        return mapping[balance_type]
    except KeyError:
        raise ValueError(f"unknown balance type: {balance_type}") from None


class Database:
    def __init__(self, settings: Settings):
        self.path = settings.database_path
        self.settings = settings
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def connection(self):
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA temp_store=FILE")
        await conn.execute("PRAGMA mmap_size=0")
        await conn.execute("PRAGMA journal_size_limit=4194304")
        await conn.execute(f"PRAGMA cache_size=-{int(self.settings.db_cache_kib)}")
        try:
            yield conn
        finally:
            await conn.close()

    async def _add_column_if_missing(self, db: aiosqlite.Connection, table: str, column: str, ddl: str) -> bool:
        rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        if column not in {r["name"] for r in rows}:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            return True
        return False

    async def init(self) -> None:
        async with self.connection() as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    vk_id INTEGER PRIMARY KEY,
                    requests_balance INTEGER NOT NULL DEFAULT 0,
                    pro_balance INTEGER NOT NULL DEFAULT 0,
                    proplus_balance INTEGER NOT NULL DEFAULT 0,
                    total_searches INTEGER NOT NULL DEFAULT 0,
                    total_pro_searches INTEGER NOT NULL DEFAULT 0,
                    total_proplus_searches INTEGER NOT NULL DEFAULT 0,
                    subscription_bonus_claimed INTEGER NOT NULL DEFAULT 0,
                    subscription_revoked INTEGER NOT NULL DEFAULT 0,
                    unsubscribe_count INTEGER NOT NULL DEFAULT 0,
                    unsubscribe_strikes INTEGER NOT NULL DEFAULT 0,
                    account_blocked INTEGER NOT NULL DEFAULT 0,
                    blocked_reason TEXT,
                    non_anime_warnings INTEGER NOT NULL DEFAULT 0,
                    temporary_block_until TEXT,
                    temporary_block_reason TEXT,
                    age_confirmed_at TEXT,
                    pending_package TEXT,
                    search_mode TEXT NOT NULL DEFAULT 'anikot',
                    activity_points_today INTEGER NOT NULL DEFAULT 0,
                    activity_rewards_today INTEGER NOT NULL DEFAULT 0,
                    activity_date TEXT,
                    last_activity_at REAL NOT NULL DEFAULT 0,
                    legal_accepted_at TEXT,
                    registration_completed_at TEXT,
                    onboarding_state TEXT NOT NULL DEFAULT 'legal',
                    referral_token TEXT UNIQUE,
                    referral_candidate_token TEXT,
                    referrer_vk_id INTEGER,
                    referral_frozen_until TEXT,
                    referral_reward_status TEXT NOT NULL DEFAULT 'none',
                    referral_rewarded_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vk_id INTEGER NOT NULL,
                    delta INTEGER NOT NULL,
                    balance_type TEXT NOT NULL DEFAULT 'anikot',
                    reason TEXT NOT NULL,
                    ref TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(reason, ref),
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS account_strikes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vk_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    event_ref TEXT NOT NULL UNIQUE,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS search_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vk_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    request_type TEXT NOT NULL DEFAULT 'anikot',
                    query TEXT,
                    engine TEXT,
                    success INTEGER NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS activity_events (
                    event_key TEXT PRIMARY KEY,
                    vk_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS payments (
                    local_id TEXT PRIMARY KEY,
                    external_id TEXT UNIQUE,
                    vk_id INTEGER NOT NULL,
                    package_key TEXT NOT NULL,
                    balance_type TEXT NOT NULL DEFAULT 'anikot',
                    requests_count INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    email TEXT NOT NULL,
                    payment_url TEXT,
                    status TEXT NOT NULL DEFAULT 'created',
                    raw_create_response TEXT,
                    raw_webhook TEXT,
                    created_at TEXT NOT NULL,
                    paid_at TEXT,
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS vision_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS checkout_sessions (
                    token TEXT PRIMARY KEY,
                    vk_id INTEGER NOT NULL,
                    package_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    local_payment_id TEXT,
                    payment_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(vk_id) REFERENCES users(vk_id) ON DELETE CASCADE
                );
                """
            )

            # Migration path from AniKot v1/v1.1.
            migrations = [
                ("users", "pro_balance", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "proplus_balance", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "total_pro_searches", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "total_proplus_searches", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "subscription_revoked", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "unsubscribe_count", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "unsubscribe_strikes", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "account_blocked", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "blocked_reason", "TEXT"),
                ("users", "non_anime_warnings", "INTEGER NOT NULL DEFAULT 0"),
                ("users", "temporary_block_until", "TEXT"),
                ("users", "temporary_block_reason", "TEXT"),
                ("users", "age_confirmed_at", "TEXT"),
                ("users", "search_mode", "TEXT NOT NULL DEFAULT 'anikot'"),
                ("users", "legal_accepted_at", "TEXT"),
                ("users", "registration_completed_at", "TEXT"),
                ("users", "onboarding_state", "TEXT NOT NULL DEFAULT 'legal'"),
                ("users", "referral_token", "TEXT"),
                ("users", "referral_candidate_token", "TEXT"),
                ("users", "referrer_vk_id", "INTEGER"),
                ("users", "referral_frozen_until", "TEXT"),
                ("users", "referral_reward_status", "TEXT NOT NULL DEFAULT 'none'"),
                ("users", "referral_rewarded_at", "TEXT"),
                ("ledger", "balance_type", "TEXT NOT NULL DEFAULT 'anikot'"),
                ("search_logs", "request_type", "TEXT NOT NULL DEFAULT 'anikot'"),
                ("payments", "balance_type", "TEXT NOT NULL DEFAULT 'anikot'"),
            ]
            registration_column_added = False
            for table, column, ddl in migrations:
                added = await self._add_column_if_missing(db, table, column, ddl)
                if table == "users" and column == "registration_completed_at" and added:
                    registration_column_added = True

            # Existing AniKot users are already registered. Only the one-time migration
            # marks them complete; new v1.5 users remain in the onboarding flow.
            if registration_column_added:
                await db.execute(
                    """UPDATE users
                    SET legal_accepted_at=COALESCE(legal_accepted_at, created_at),
                        registration_completed_at=COALESCE(registration_completed_at, created_at),
                        onboarding_state='complete',
                        referral_reward_status=COALESCE(referral_reward_status, 'none')"""
                )

            # Ensure every existing user has a stable individual referral token.
            rows = await (await db.execute("SELECT vk_id, referral_token FROM users")).fetchall()
            for row in rows:
                if not row["referral_token"]:
                    token = secrets.token_urlsafe(12)
                    while await (await db.execute("SELECT 1 FROM users WHERE referral_token=?", (token,))).fetchone():
                        token = secrets.token_urlsafe(12)
                    await db.execute("UPDATE users SET referral_token=? WHERE vk_id=?", (token, row["vk_id"]))

            try:
                await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_token ON users(referral_token)")
            except aiosqlite.OperationalError:
                pass
            await db.commit()

    async def ensure_user(self, vk_id: int) -> tuple[dict[str, Any], bool]:
        """Create a shell account. Bonuses are issued only after onboarding completes."""
        async with self._lock:
            async with self.connection() as db:
                now = utc_now_iso()
                await db.execute("BEGIN IMMEDIATE")
                existing = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                if existing:
                    await db.rollback()
                    return dict(existing), False

                token = secrets.token_urlsafe(12)
                while await (await db.execute("SELECT 1 FROM users WHERE referral_token=?", (token,))).fetchone():
                    token = secrets.token_urlsafe(12)
                await db.execute(
                    """INSERT INTO users
                    (vk_id, requests_balance, pro_balance, proplus_balance, onboarding_state,
                     referral_token, referral_reward_status, created_at, updated_at)
                    VALUES (?, 0, 0, 0, 'legal', ?, 'none', ?, ?)""",
                    (vk_id, token, now, now),
                )
                row = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                await db.commit()
                return dict(row), True

    async def set_referral_candidate(self, vk_id: int, token: str | None) -> None:
        async with self.connection() as db:
            await db.execute(
                "UPDATE users SET referral_candidate_token=?, updated_at=? WHERE vk_id=?",
                ((token or None), utc_now_iso(), vk_id),
            )
            await db.commit()

    async def accept_legal(self, vk_id: int) -> None:
        now = utc_now_iso()
        async with self.connection() as db:
            await db.execute(
                """UPDATE users SET legal_accepted_at=?, onboarding_state='referral_choice', updated_at=?
                WHERE vk_id=? AND registration_completed_at IS NULL""",
                (now, now, vk_id),
            )
            await db.commit()

    async def set_onboarding_state(self, vk_id: int, state: str) -> None:
        if state not in {"legal", "referral_choice", "referral_link", "complete"}:
            raise ValueError("unknown onboarding state")
        async with self.connection() as db:
            await db.execute(
                "UPDATE users SET onboarding_state=?, updated_at=? WHERE vk_id=?",
                (state, utc_now_iso(), vk_id),
            )
            await db.commit()

    async def apply_referral_token(self, vk_id: int, token: str) -> tuple[bool, str | None]:
        token = (token or "").strip()
        if not token:
            return False, "empty"
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                user = await (await db.execute(
                    "SELECT vk_id, referrer_vk_id, registration_completed_at FROM users WHERE vk_id=?",
                    (vk_id,),
                )).fetchone()
                referrer = await (await db.execute(
                    "SELECT vk_id, registration_completed_at FROM users WHERE referral_token=?",
                    (token,),
                )).fetchone()
                if not user or not referrer or not referrer["registration_completed_at"]:
                    await db.rollback()
                    return False, "not_found"
                if int(referrer["vk_id"]) == vk_id:
                    await db.rollback()
                    return False, "self"
                if user["registration_completed_at"] or user["referrer_vk_id"]:
                    await db.rollback()
                    return False, "locked"
                now = datetime.now(timezone.utc)
                frozen_until = now + timedelta(days=self.settings.referral_freeze_days)
                await db.execute(
                    """UPDATE users SET referrer_vk_id=?, referral_frozen_until=?,
                    referral_reward_status='pending', referral_candidate_token=NULL, updated_at=?
                    WHERE vk_id=?""",
                    (int(referrer["vk_id"]), frozen_until.isoformat(), now.isoformat(), vk_id),
                )
                await db.commit()
                return True, None

    async def complete_registration(self, vk_id: int) -> tuple[bool, dict[str, Any]]:
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                if not row:
                    await db.rollback()
                    raise RuntimeError("user missing")
                if row["registration_completed_at"]:
                    await db.rollback()
                    return False, dict(row)
                if not row["legal_accepted_at"]:
                    await db.rollback()
                    raise RuntimeError("legal acceptance missing")

                now = utc_now_iso()
                bonus = self.settings.registration_bonus_requests
                await db.execute(
                    """UPDATE users SET requests_balance=requests_balance+?,
                    registration_completed_at=?, onboarding_state='complete',
                    referral_candidate_token=NULL, updated_at=? WHERE vk_id=?""",
                    (bonus, now, now, vk_id),
                )
                if bonus:
                    await db.execute(
                        "INSERT OR IGNORE INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,'anikot','registration_bonus',?,?)",
                        (vk_id, bonus, f"registration:{vk_id}", now),
                    )
                updated = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                await db.commit()
                return True, dict(updated)

    async def release_mature_referrals(self, limit: int = 100) -> list[dict[str, Any]]:
        """Credit referral rewards after the 3-day hold. Idempotent through ledger refs."""
        now = datetime.now(timezone.utc)
        rewards: list[dict[str, Any]] = []
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                rows = await (await db.execute(
                    """SELECT vk_id, referrer_vk_id FROM users
                    WHERE referral_reward_status='pending'
                      AND registration_completed_at IS NOT NULL
                      AND referral_frozen_until IS NOT NULL
                      AND referral_frozen_until<=?
                    ORDER BY referral_frozen_until LIMIT ?""",
                    (now.isoformat(), limit),
                )).fetchall()
                for row in rows:
                    referred_id = int(row["vk_id"])
                    referrer_id = int(row["referrer_vk_id"] or 0)
                    if referrer_id <= 0:
                        await db.execute(
                            "UPDATE users SET referral_reward_status='invalid', updated_at=? WHERE vk_id=?",
                            (now.isoformat(), referred_id),
                        )
                        continue
                    ref = f"referral:{referred_id}"
                    try:
                        await db.execute(
                            "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,'pro','referral_reward',?,?)",
                            (referrer_id, self.settings.referral_reward_pro, ref, now.isoformat()),
                        )
                    except aiosqlite.IntegrityError:
                        await db.execute(
                            "UPDATE users SET referral_reward_status='credited', referral_rewarded_at=COALESCE(referral_rewarded_at,?), updated_at=? WHERE vk_id=?",
                            (now.isoformat(), now.isoformat(), referred_id),
                        )
                        continue
                    await db.execute(
                        "UPDATE users SET pro_balance=pro_balance+?, updated_at=? WHERE vk_id=?",
                        (self.settings.referral_reward_pro, now.isoformat(), referrer_id),
                    )
                    await db.execute(
                        "UPDATE users SET referral_reward_status='credited', referral_rewarded_at=?, updated_at=? WHERE vk_id=?",
                        (now.isoformat(), now.isoformat(), referred_id),
                    )
                    balance_row = await (await db.execute(
                        "SELECT pro_balance FROM users WHERE vk_id=?", (referrer_id,)
                    )).fetchone()
                    rewards.append({
                        "referrer_vk_id": referrer_id,
                        "referred_vk_id": referred_id,
                        "amount": self.settings.referral_reward_pro,
                        "balance": int(balance_row["pro_balance"]) if balance_row else 0,
                    })
                await db.commit()
        return rewards

    async def referral_stats(self, vk_id: int) -> dict[str, Any]:
        async with self.connection() as db:
            user = await (await db.execute(
                "SELECT referral_token FROM users WHERE vk_id=?", (vk_id,)
            )).fetchone()
            counts = await (await db.execute(
                """SELECT
                    SUM(CASE WHEN referral_reward_status='pending' THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN referral_reward_status='credited' THEN 1 ELSE 0 END) AS credited
                FROM users WHERE referrer_vk_id=?""",
                (vk_id,),
            )).fetchone()
            return {
                "token": str(user["referral_token"] or "") if user else "",
                "pending": int(counts["pending"] or 0),
                "credited": int(counts["credited"] or 0),
            }

    async def get_user(self, vk_id: int) -> dict[str, Any] | None:
        async with self.connection() as db:
            row = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
            return dict(row) if row else None

    async def is_account_blocked(self, vk_id: int) -> tuple[bool, int, str | None, str | None]:
        """Return permanent/temporary block status and auto-expire temporary abuse blocks."""
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                if not row:
                    await db.rollback()
                    return False, 0, None, None

                now = datetime.now(timezone.utc)
                temporary_until = row["temporary_block_until"]
                if temporary_until:
                    try:
                        until_dt = datetime.fromisoformat(str(temporary_until))
                        if until_dt.tzinfo is None:
                            until_dt = until_dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        until_dt = now

                    if until_dt <= now:
                        await db.execute(
                            "UPDATE users SET temporary_block_until=NULL, temporary_block_reason=NULL, non_anime_warnings=0, updated_at=? WHERE vk_id=?",
                            (now.isoformat(), vk_id),
                        )
                    else:
                        await db.commit()
                        return True, int(row["unsubscribe_strikes"] or 0), str(row["temporary_block_reason"] or "temporary"), until_dt.isoformat()

                permanent = bool(row["account_blocked"])
                reason = row["blocked_reason"]
                await db.commit()
                return permanent, int(row["unsubscribe_strikes"] or 0), reason, None

    async def register_non_anime_warning(self, vk_id: int) -> dict[str, Any]:
        """Add one warning for a clearly non-anime image and block at the configured limit."""
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute("SELECT non_anime_warnings FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                if not row:
                    await db.rollback()
                    return {"warnings": 0, "blocked": False, "blocked_until": None, "limit": self.settings.non_anime_warning_limit}

                now = datetime.now(timezone.utc)
                warnings = int(row["non_anime_warnings"] or 0) + 1
                blocked = warnings >= self.settings.non_anime_warning_limit
                blocked_until = None
                if blocked:
                    blocked_until = (now + timedelta(days=self.settings.non_anime_block_days)).isoformat()
                    await db.execute(
                        "UPDATE users SET non_anime_warnings=?, temporary_block_until=?, temporary_block_reason='non_anime_abuse', updated_at=? WHERE vk_id=?",
                        (warnings, blocked_until, now.isoformat(), vk_id),
                    )
                else:
                    await db.execute(
                        "UPDATE users SET non_anime_warnings=?, updated_at=? WHERE vk_id=?",
                        (warnings, now.isoformat(), vk_id),
                    )
                await db.commit()
                return {"warnings": warnings, "blocked": blocked, "blocked_until": blocked_until, "limit": self.settings.non_anime_warning_limit}

    async def set_pending_package(self, vk_id: int, package_key: str | None) -> None:
        async with self.connection() as db:
            await db.execute("UPDATE users SET pending_package=?, updated_at=? WHERE vk_id=?", (package_key, utc_now_iso(), vk_id))
            await db.commit()

    async def set_search_mode(self, vk_id: int, mode: str) -> None:
        if mode not in {"anikot", "pro", "proplus"}:
            raise ValueError("unknown search mode")
        async with self.connection() as db:
            await db.execute("UPDATE users SET search_mode=?, updated_at=? WHERE vk_id=?", (mode, utc_now_iso(), vk_id))
            await db.commit()

    async def confirm_adult_age(self, vk_id: int) -> None:
        async with self.connection() as db:
            await db.execute("UPDATE users SET age_confirmed_at=?, updated_at=? WHERE vk_id=?", (utc_now_iso(), utc_now_iso(), vk_id))
            await db.commit()

    async def consume_request(self, vk_id: int, balance_type: str = "anikot") -> int | None:
        col = _balance_column(balance_type)
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute(
                    f"SELECT {col}, account_blocked FROM users WHERE vk_id=?", (vk_id,)
                )).fetchone()
                if not row or int(row["account_blocked"] or 0) or int(row[col]) <= 0:
                    await db.rollback()
                    return None
                new_balance = int(row[col]) - 1
                now = utc_now_iso()
                pro_inc = 1 if balance_type == "pro" else 0
                proplus_inc = 1 if balance_type == "proplus" else 0
                await db.execute(
                    f"""UPDATE users
                    SET {col}=?, total_searches=total_searches+1,
                        total_pro_searches=total_pro_searches+?,
                        total_proplus_searches=total_proplus_searches+?,
                        updated_at=?
                    WHERE vk_id=?""",
                    (new_balance, pro_inc, proplus_inc, now, vk_id),
                )
                await db.execute(
                    "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,-1,?,'search',?,?)",
                    (vk_id, balance_type, str(uuid.uuid4()), now),
                )
                await db.commit()
                return new_balance

    async def add_requests(
        self,
        vk_id: int,
        amount: int,
        reason: str,
        ref: str | None = None,
        balance_type: str = "anikot",
    ) -> int:
        if amount <= 0:
            raise ValueError("amount must be positive")
        col = _balance_column(balance_type)
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                now = utc_now_iso()
                try:
                    await db.execute(
                        "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,?,?,?,?)",
                        (vk_id, amount, balance_type, reason, ref, now),
                    )
                except aiosqlite.IntegrityError:
                    row = await (await db.execute(f"SELECT {col} FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                    await db.rollback()
                    return int(row[col]) if row else 0
                await db.execute(f"UPDATE users SET {col}={col}+?, updated_at=? WHERE vk_id=?", (amount, now, vk_id))
                row = await (await db.execute(f"SELECT {col} FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                await db.commit()
                return int(row[col])

    async def refund_search_request(self, vk_id: int, balance_type: str, ref: str | None = None) -> int:
        return await self.add_requests(vk_id, 1, "search_refund", ref or str(uuid.uuid4()), balance_type)

    async def claim_subscription_bonus(self, vk_id: int) -> tuple[bool, int]:
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute(
                    "SELECT subscription_bonus_claimed, requests_balance, account_blocked FROM users WHERE vk_id=?",
                    (vk_id,),
                )).fetchone()
                if not row or int(row["account_blocked"] or 0):
                    await db.rollback()
                    return False, 0
                if row["subscription_bonus_claimed"]:
                    await db.rollback()
                    return False, int(row["requests_balance"])
                now = utc_now_iso()
                bonus = self.settings.subscription_bonus_requests
                await db.execute(
                    """UPDATE users
                    SET subscription_bonus_claimed=1, subscription_revoked=0,
                        requests_balance=requests_balance+?, updated_at=?
                    WHERE vk_id=?""",
                    (bonus, now, vk_id),
                )
                await db.execute(
                    "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,'anikot','subscription_bonus',?,?)",
                    (vk_id, bonus, str(vk_id), now),
                )
                updated = await (await db.execute("SELECT requests_balance FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                await db.commit()
                return True, int(updated["requests_balance"])

    async def register_unsubscribe_strike(self, vk_id: int, event_ref: str, details: str | None = None) -> dict[str, Any]:
        """Add one strike after a GROUP_LEAVE event.

        Strikes only apply to users who claimed the subscription bonus.
        Event refs are unique to avoid double-counting callback retries.
        """
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute(
                    """SELECT subscription_bonus_claimed, unsubscribe_strikes, account_blocked
                    FROM users WHERE vk_id=?""",
                    (vk_id,),
                )).fetchone()
                if not row or not int(row["subscription_bonus_claimed"] or 0):
                    await db.rollback()
                    return {
                        "applied": False,
                        "strikes": int(row["unsubscribe_strikes"] or 0) if row else 0,
                        "blocked": bool(row["account_blocked"]) if row else False,
                    }
                try:
                    await db.execute(
                        "INSERT INTO account_strikes(vk_id,kind,event_ref,details,created_at) VALUES (?,?,?,?,?)",
                        (vk_id, "unsubscribe", event_ref, details, utc_now_iso()),
                    )
                except aiosqlite.IntegrityError:
                    await db.rollback()
                    return {
                        "applied": False,
                        "strikes": int(row["unsubscribe_strikes"] or 0),
                        "blocked": bool(row["account_blocked"]),
                    }

                strikes = int(row["unsubscribe_strikes"] or 0) + 1
                blocked = strikes >= self.settings.unsubscribe_strike_limit
                now = utc_now_iso()
                reason = "3 strikes for unsubscribing after receiving the community bonus" if blocked else None
                await db.execute(
                    """UPDATE users
                    SET unsubscribe_strikes=?, unsubscribe_count=unsubscribe_count+1,
                        subscription_revoked=1, account_blocked=?,
                        blocked_reason=COALESCE(?, blocked_reason), updated_at=?
                    WHERE vk_id=?""",
                    (strikes, int(blocked), reason, now, vk_id),
                )
                await db.commit()
                return {"applied": True, "strikes": strikes, "blocked": blocked}

    async def unblock_account(self, vk_id: int, reset_strikes: bool = False) -> None:
        async with self.connection() as db:
            if reset_strikes:
                await db.execute(
                    """UPDATE users SET account_blocked=0, blocked_reason=NULL,
                    unsubscribe_strikes=0, non_anime_warnings=0,
                    temporary_block_until=NULL, temporary_block_reason=NULL,
                    updated_at=? WHERE vk_id=?""",
                    (utc_now_iso(), vk_id),
                )
            else:
                await db.execute(
                    """UPDATE users SET account_blocked=0, blocked_reason=NULL,
                    temporary_block_until=NULL, temporary_block_reason=NULL,
                    updated_at=? WHERE vk_id=?""",
                    (utc_now_iso(), vk_id),
                )
            await db.commit()

    async def record_activity(self, vk_id: int, event_key: str, kind: str, text: str) -> tuple[int, int] | None:
        if len((text or "").strip()) < self.settings.activity_min_text_length:
            return None
        today = datetime.now(timezone.utc).date().isoformat()
        now_ts = time.time()
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                try:
                    await db.execute(
                        "INSERT INTO activity_events(event_key,vk_id,kind,created_at) VALUES (?,?,?,?)",
                        (event_key, vk_id, kind, utc_now_iso()),
                    )
                except aiosqlite.IntegrityError:
                    await db.rollback()
                    return None
                row = await (await db.execute("SELECT * FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                if (
                    not row
                    or not row["registration_completed_at"]
                    or int(row["account_blocked"] or 0)
                    or now_ts - float(row["last_activity_at"] or 0) < self.settings.activity_cooldown_seconds
                ):
                    await db.rollback()
                    return None
                points = 0 if row["activity_date"] != today else int(row["activity_points_today"])
                rewards_today = 0 if row["activity_date"] != today else int(row["activity_rewards_today"])
                points += 1
                reward = 0
                if (
                    points % self.settings.activity_actions_per_reward == 0
                    and rewards_today < self.settings.activity_daily_reward_cap
                ):
                    reward = min(
                        self.settings.activity_reward_requests,
                        self.settings.activity_daily_reward_cap - rewards_today,
                    )
                    rewards_today += reward
                await db.execute(
                    """UPDATE users
                    SET activity_points_today=?, activity_rewards_today=?, activity_date=?,
                        last_activity_at=?, requests_balance=requests_balance+?, updated_at=?
                    WHERE vk_id=?""",
                    (points, rewards_today, today, now_ts, reward, utc_now_iso(), vk_id),
                )
                if reward:
                    await db.execute(
                        "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,'anikot','activity_bonus',?,?)",
                        (vk_id, reward, event_key, utc_now_iso()),
                    )
                updated = await (await db.execute("SELECT requests_balance FROM users WHERE vk_id=?", (vk_id,))).fetchone()
                await db.commit()
                return reward, int(updated["requests_balance"])

    async def get_vision_cache(self, cache_key: str) -> dict[str, Any] | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.settings.vision_cache_ttl_days)
        async with self.connection() as db:
            row = await (await db.execute(
                "SELECT result_json, created_at FROM vision_cache WHERE cache_key=?",
                (cache_key,),
            )).fetchone()
            if not row:
                return None
            try:
                created = datetime.fromisoformat(str(row["created_at"]))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except ValueError:
                created = datetime.min.replace(tzinfo=timezone.utc)
            if created < cutoff:
                await db.execute("DELETE FROM vision_cache WHERE cache_key=?", (cache_key,))
                await db.commit()
                return None
            try:
                value = json.loads(str(row["result_json"]))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

    async def put_vision_cache(self, cache_key: str, result: dict[str, Any]) -> None:
        payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        async with self.connection() as db:
            await db.execute(
                """INSERT INTO vision_cache(cache_key,result_json,created_at)
                VALUES (?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET result_json=excluded.result_json, created_at=excluded.created_at""",
                (cache_key, payload, utc_now_iso()),
            )
            await db.commit()

    async def log_search(
        self,
        vk_id: int,
        kind: str,
        request_type: str,
        query: str | None,
        engine: str | None,
        success: bool,
        details: dict | str | None = None,
    ) -> None:
        if isinstance(details, dict):
            details = json.dumps(details, ensure_ascii=False)
        async with self.connection() as db:
            await db.execute(
                """INSERT INTO search_logs(vk_id,kind,request_type,query,engine,success,details,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (vk_id, kind, request_type, query, engine, int(success), details, utc_now_iso()),
            )
            await db.commit()

    async def create_checkout_session(self, vk_id: int, package_key: str) -> str:
        token = secrets.token_urlsafe(32)
        now = utc_now_iso()
        async with self.connection() as db:
            while await (await db.execute("SELECT 1 FROM checkout_sessions WHERE token=?", (token,))).fetchone():
                token = secrets.token_urlsafe(32)
            await db.execute(
                """INSERT INTO checkout_sessions(token,vk_id,package_key,status,created_at,updated_at)
                VALUES (?,?,?,'created',?,?)""",
                (token, vk_id, package_key, now, now),
            )
            await db.commit()
        return token

    async def get_checkout_session(self, token: str) -> dict[str, Any] | None:
        async with self.connection() as db:
            row = await (await db.execute(
                "SELECT * FROM checkout_sessions WHERE token=?", (token,)
            )).fetchone()
            return dict(row) if row else None

    async def finalize_checkout_session(self, token: str, local_payment_id: str, payment_url: str) -> None:
        async with self.connection() as db:
            await db.execute(
                """UPDATE checkout_sessions SET status='invoice_created', local_payment_id=?,
                payment_url=?, updated_at=? WHERE token=?""",
                (local_payment_id, payment_url, utc_now_iso(), token),
            )
            await db.commit()

    async def create_payment(self, vk_id: int, package: Package, email: str) -> str:
        local_id = str(uuid.uuid4())
        async with self.connection() as db:
            await db.execute(
                """INSERT INTO payments(local_id,vk_id,package_key,balance_type,requests_count,amount,currency,email,status,created_at)
                VALUES (?,?,?,?,?,?,?,?,'creating',?)""",
                (
                    local_id,
                    vk_id,
                    package.key,
                    package.balance_type,
                    package.requests,
                    package.price,
                    package.currency,
                    email,
                    utc_now_iso(),
                ),
            )
            await db.commit()
        return local_id

    async def finalize_payment_creation(self, local_id: str, external_id: str, payment_url: str, raw_response: dict) -> None:
        async with self.connection() as db:
            await db.execute(
                """UPDATE payments SET external_id=?,payment_url=?,status='pending',raw_create_response=?
                WHERE local_id=?""",
                (external_id, payment_url, json.dumps(raw_response, ensure_ascii=False), local_id),
            )
            await db.commit()

    async def fail_payment_creation(self, local_id: str, details: str) -> None:
        async with self.connection() as db:
            await db.execute(
                "UPDATE payments SET status='create_failed',raw_create_response=? WHERE local_id=?",
                (details, local_id),
            )
            await db.commit()

    async def mark_payment_paid(self, external_id: str, raw_webhook: dict) -> tuple[bool, int, int, str, int] | None:
        async with self._lock:
            async with self.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                row = await (await db.execute("SELECT * FROM payments WHERE external_id=?", (external_id,))).fetchone()
                if not row:
                    await db.rollback()
                    return None
                btype = row["balance_type"] or "anikot"
                col = _balance_column(btype)
                if row["status"] == "paid":
                    user = await (await db.execute(f"SELECT {col} FROM users WHERE vk_id=?", (row["vk_id"],))).fetchone()
                    await db.rollback()
                    return False, int(row["vk_id"]), int(user[col]), btype, int(row["requests_count"])
                now = utc_now_iso()
                await db.execute(
                    "UPDATE payments SET status='paid',paid_at=?,raw_webhook=? WHERE local_id=?",
                    (now, json.dumps(raw_webhook, ensure_ascii=False), row["local_id"]),
                )
                await db.execute(
                    f"UPDATE users SET {col}={col}+?,updated_at=? WHERE vk_id=?",
                    (row["requests_count"], now, row["vk_id"]),
                )
                await db.execute(
                    "INSERT INTO ledger(vk_id,delta,balance_type,reason,ref,created_at) VALUES (?,?,?,'lava_payment',?,?)",
                    (row["vk_id"], row["requests_count"], btype, external_id, now),
                )
                user = await (await db.execute(f"SELECT {col} FROM users WHERE vk_id=?", (row["vk_id"],))).fetchone()
                await db.commit()
                return True, int(row["vk_id"]), int(user[col]), btype, int(row["requests_count"])

    async def stats(self) -> dict[str, int | float]:
        async with self.connection() as db:
            users = await (await db.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(account_blocked),0) AS b FROM users"
            )).fetchone()
            searches = await (await db.execute(
                """SELECT COALESCE(SUM(total_searches),0) AS c,
                COALESCE(SUM(total_pro_searches),0) AS p,
                COALESCE(SUM(total_proplus_searches),0) AS x FROM users"""
            )).fetchone()
            paid = await (await db.execute(
                "SELECT COUNT(*) AS c, COALESCE(SUM(amount),0) AS s FROM payments WHERE status='paid'"
            )).fetchone()
            return {
                "users": int(users["c"]),
                "blocked": int(users["b"]),
                "searches": int(searches["c"]),
                "pro_searches": int(searches["p"]),
                "proplus_searches": int(searches["x"]),
                "paid_orders": int(paid["c"]),
                "revenue": float(paid["s"]),
            }
