from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

import httpx
import uvicorn
from fastapi import FastAPI

from app.bot import build_bot
from app.config import settings
from app.db import Database
from app.db_pool import install_database_pool
from app.highload import install_highload_guard
from app.services.aiai import AIAIClient
from app.services.anime_detector import AnimeDetector
from app.services.lava import LavaClient
from app.search_ui import install_search_progress_cleanup
from app.support import install_support
from app.support_ai import SupportAI
from app.support_compat import patch_support_rule_registration
from app.web import build_web_app

logger = logging.getLogger("anikot")


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings.validate()

    db = Database(settings)
    db_pool = await install_database_pool(db)
    await db.init()

    limits = httpx.Limits(
        max_connections=settings.http_max_connections,
        max_keepalive_connections=settings.http_max_keepalive_connections,
        keepalive_expiry=settings.http_keepalive_expiry,
    )
    http_client = httpx.AsyncClient(
        limits=limits,
        follow_redirects=True,
        timeout=httpx.Timeout(settings.aiai_timeout, connect=15.0),
    )

    # Search traffic has priority capacity. Support AI gets its own gate so a
    # burst of support conversations cannot consume all recognition slots.
    search_ai_gate = asyncio.Semaphore(settings.aiai_max_concurrency)
    support_ai_concurrency = max(
        1, min(int(os.getenv("SUPPORT_AI_MAX_CONCURRENCY", "1")), 3)
    )
    support_ai_gate = asyncio.Semaphore(support_ai_concurrency)

    anikot_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_anikot_model,
        client=http_client,
        semaphore=search_ai_gate,
        timeout=settings.aiai_timeout,
    )
    pro_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_pro_model,
        client=http_client,
        semaphore=search_ai_gate,
        timeout=settings.aiai_timeout,
    )
    proplus_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_proplus_model,
        client=http_client,
        semaphore=search_ai_gate,
        timeout=settings.aiai_timeout,
    )
    support_ai = SupportAI(
        settings=settings,
        client=http_client,
        semaphore=support_ai_gate,
    )

    detector = AnimeDetector(
        anikot_ai=anikot_ai,
        pro_ai=pro_ai,
        proplus_ai=proplus_ai,
    )
    lava = LavaClient(settings, http_client)
    bot = build_bot(settings, db, detector, lava, http_client)

    load_guard = install_highload_guard(bot, settings)
    install_search_progress_cleanup()

    # vkbottle 4.11 CoroutineRule calls coro functions without Message.
    # Apply the compatibility correction before support registers its handler.
    patch_support_rule_registration()
    await install_support(bot, settings, db, support_ai)

    app.state.settings = settings
    app.state.db = db
    app.state.db_pool = db_pool
    app.state.bot = bot
    app.state.lava = lava
    app.state.http_client = http_client
    app.state.load_guard = load_guard
    app.state.runtime_ready = True

    bot_task = asyncio.create_task(bot.run_polling(), name="vk-long-polling")
    app.state.bot_task = bot_task

    logger.info(
        "AniKot 2.2.1 runtime started; HTTP=%s:%s search_ai=%s support_ai=%s "
        "http_pool=%s db_pool=%s max_inflight=%s",
        settings.web_host,
        settings.web_port,
        settings.aiai_max_concurrency,
        support_ai_concurrency,
        settings.http_max_connections,
        db_pool.size,
        load_guard.max_inflight,
    )

    try:
        yield
    finally:
        app.state.runtime_ready = False
        await _cancel_task(bot_task)
        await http_client.aclose()
        await db_pool.close()


app: FastAPI = build_web_app(lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level="info",
        access_log=False,
    )
