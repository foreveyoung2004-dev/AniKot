"""Optional manual model preparation helper.

BotHost users do not need to run this: main.py performs the same provisioning
on first start. This script is useful only for local/VPS preloading.
"""
from __future__ import annotations

import asyncio

from app.config import settings
from app.services.model_manager import ModelManager


async def main() -> None:
    settings.validate()
    status = await ModelManager(settings).ensure_models()
    print(status)


if __name__ == "__main__":
    asyncio.run(main())
