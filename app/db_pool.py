from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Small async pool for AniKot's single-process SQLite workload."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self.path = str(database.path)
        self.settings = database.settings
        self.size = max(1, min(int(os.getenv("DB_POOL_SIZE", "4")), 8))
        self._queue: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(maxsize=self.size)
        self._connections: list[aiosqlite.Connection] = []
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self.borrowed = 0
        self.peak_borrowed = 0

    async def _configure(self, conn: aiosqlite.Connection) -> None:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA temp_store=FILE")
        await conn.execute("PRAGMA mmap_size=0")
        await conn.execute("PRAGMA journal_size_limit=4194304")
        await conn.execute(f"PRAGMA cache_size=-{int(self.settings.db_cache_kib)}")
        await conn.commit()

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("SQLite pool is closed")
            for _ in range(self.size):
                conn = await aiosqlite.connect(self.path)
                await self._configure(conn)
                self._connections.append(conn)
                await self._queue.put(conn)
            self._started = True
            logger.info("SQLite pool started size=%s path=%s", self.size, self.path)

    @asynccontextmanager
    async def connection(self):
        if not self._started:
            await self.start()
        if self._closed:
            raise RuntimeError("SQLite pool is closed")

        conn = await self._queue.get()
        self.borrowed += 1
        self.peak_borrowed = max(self.peak_borrowed, self.borrowed)
        try:
            yield conn
        except Exception:
            if conn.in_transaction:
                await conn.rollback()
            raise
        finally:
            if conn.in_transaction:
                await conn.rollback()
            self.borrowed = max(0, self.borrowed - 1)
            if not self._closed:
                await self._queue.put(conn)

    def snapshot(self) -> dict[str, int]:
        return {
            "size": self.size,
            "borrowed": self.borrowed,
            "peak_borrowed": self.peak_borrowed,
            "available": self._queue.qsize() if self._started else 0,
        }

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._connections:
            try:
                if conn.in_transaction:
                    await conn.rollback()
                await conn.close()
            except Exception:
                logger.exception("Failed to close SQLite pooled connection")
        self._connections.clear()
        self._started = False


async def install_database_pool(database: Any) -> SQLiteConnectionPool:
    pool = SQLiteConnectionPool(database)
    await pool.start()
    database.connection = pool.connection
    return pool
