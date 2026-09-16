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
from app.services.anilist import AniListClient
from app.services.anime_detector import AnimeDetector
from app.services.camie import CamieTagger
from app.services.lava import LavaClient
from app.services.model_manager import ModelManager
from app.services.pixai import PixAITagger
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
    """Start VK polling and model provisioning inside the one FastAPI process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings.validate()

    db = Database(settings)
    await db.init()

    model_manager = ModelManager(settings)
    camie = CamieTagger(settings.camie_model_path, settings.camie_metadata_path)
    pixai = PixAITagger(
        enabled=settings.pixai_enabled,
        model_path=settings.pixai_model_path,
        tags_path=settings.pixai_tags_path,
        general_threshold=settings.pixai_general_threshold,
        character_threshold=settings.pixai_character_threshold,
    )
    anikot_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_anikot_model,
        fallback_model=settings.aiai_anikot_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    pro_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_pro_model,
        fallback_model=settings.aiai_pro_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    proplus_ai = AIAIClient(
        api_key=settings.aiai_api_key,
        base_url=settings.aiai_base_url,
        preferred_model=settings.aiai_proplus_model,
        fallback_model=settings.aiai_proplus_model,
        timeout=settings.aiai_timeout,
        max_concurrency=settings.aiai_max_concurrency,
    )
    anilist = AniListClient()
    detector = AnimeDetector(
        camie=camie,
        pixai=pixai,
        anikot_ai=anikot_ai,
        pro_ai=pro_ai,
        proplus_ai=proplus_ai,
        anilist=anilist,
        local_threshold=settings.camie_threshold,
        top_k=settings.camie_top_k,
    )
    lava = LavaClient(settings)
    bot = build_bot(settings, db, detector, lava)

    app.state.settings = settings
    app.state.db = db
    app.state.bot = bot
    app.state.model_manager = model_manager
    app.state.lava = lava
    app.state.runtime_ready = True

    # Both jobs run in the background. Uvicorn itself is the only process that
    # owns PORT, which avoids the previous "address already in use" collision.
    bot_task = asyncio.create_task(bot.run_polling(), name="vk-long-polling")
    model_task = asyncio.create_task(model_manager.ensure_models(), name="model-provisioning")
    app.state.bot_task = bot_task
    app.state.model_task = model_task

    logger.info("AniKot runtime started; HTTP server owns %s:%s", settings.web_host, settings.web_port)

    try:
        yield
    finally:
        app.state.runtime_ready = False
        await _cancel_task(bot_task)
        await _cancel_task(model_task)


# IMPORTANT FOR BOTHOST:
# A conventional module-level FastAPI object makes it clear that this project
# already provides its own HTTP server. BotHost should not add its auto wrapper.
app: FastAPI = build_web_app(lifespan=lifespan)


if __name__ == "__main__":
    # BotHost injects PORT=3000. Keep the panel port at 3000 and do not set PORT manually.
    uvicorn.run(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level="info",
    )
