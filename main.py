from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI

from app.bot import build_bot
from app.config import settings
from app.db import Database
from app.services.aiai import AIAIClient
from app.services.anime_detector import AnimeDetector
from app.services.lava import LavaClient
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
    await db.init()

    anikot_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_anikot_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    pro_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_pro_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    proplus_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_proplus_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    detector = AnimeDetector(
        anikot_ai=anikot_ai,
        pro_ai=pro_ai,
        proplus_ai=proplus_ai,
    )
    lava = LavaClient(settings)
    bot = build_bot(settings, db, detector, lava)

    app.state.settings = settings
    app.state.db = db
    app.state.bot = bot
    app.state.lava = lava
    app.state.runtime_ready = True

    bot_task = asyncio.create_task(bot.run_polling(), name="vk-long-polling")
    app.state.bot_task = bot_task

    logger.info("AniKot runtime started; HTTP server owns %s:%s", settings.web_host, settings.web_port)

    try:
        yield
    finally:
        app.state.runtime_ready = False
        await _cancel_task(bot_task)


app: FastAPI = build_web_app(lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level="info",
    )
