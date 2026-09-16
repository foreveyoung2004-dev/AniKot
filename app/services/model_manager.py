from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelStatus:
    camie_ready: bool
    pixai_ready: bool
    downloading: bool = False


class ModelManager:
    """Downloads local models into persistent storage on first start.

    Designed for Bothost: /app/data survives redeploys, so model weights are
    downloaded once and reused. Downloads run in a worker thread so the VK bot
    and HTTP health/webhook server can start immediately.
    """

    CAMIE_MODEL_FILE = "camie-tagger-v2.onnx"
    CAMIE_META_FILE = "camie-tagger-v2-metadata.json"
    PIXAI_MODEL_FILE = "model.onnx"
    PIXAI_TAGS_FILE = "selected_tags.csv"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._downloading = False
        self._lock = asyncio.Lock()

    @staticmethod
    def _valid(path: str | Path, min_bytes: int) -> bool:
        p = Path(path)
        try:
            return p.is_file() and p.stat().st_size >= min_bytes
        except OSError:
            return False

    def status(self) -> ModelStatus:
        camie_ready = self._valid(self.settings.camie_model_path, 100_000_000) and self._valid(
            self.settings.camie_metadata_path, 1_000
        )
        pixai_ready = (
            not self.settings.pixai_enabled
            or (
                self._valid(self.settings.pixai_model_path, 1_000_000_000)
                and self._valid(self.settings.pixai_tags_path, 100_000)
            )
        )
        return ModelStatus(camie_ready, pixai_ready, self._downloading)

    @staticmethod
    def _download_one(repo_id: str, filename: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()

        # local_dir avoids keeping a second full copy in the global HF cache.
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target.parent),
            )
        )
        if downloaded.resolve() != target.resolve():
            downloaded.replace(target)

    async def _download_with_retry(
        self,
        repo_id: str,
        filename: str,
        target: Path,
        min_bytes: int,
        label: str,
        expected_bytes: int | None = None,
    ) -> bool:
        if self._valid(target, min_bytes):
            return True

        target.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(target.parent).free
        # Keep a reserve for SQLite/temp files and Hugging Face download metadata.
        expected = expected_bytes or min_bytes
        required_free = int(expected * 1.08) + 150_000_000
        if free_bytes < required_free:
            logger.error(
                "Model setup: not enough free disk space for %s. Free=%s MB, required≈%s MB",
                label,
                free_bytes // 1_000_000,
                required_free // 1_000_000,
            )
            return False

        for attempt in range(1, self.settings.model_download_retries + 1):
            try:
                logger.info("Model setup: downloading %s (%s), attempt %s/%s", label, filename, attempt, self.settings.model_download_retries)
                await asyncio.to_thread(self._download_one, repo_id, filename, target)
                if not self._valid(target, min_bytes):
                    raise RuntimeError(f"Downloaded file is unexpectedly small: {target}")
                logger.info("Model setup: %s ready at %s", label, target)
                return True
            except Exception:
                logger.exception("Model setup: failed to download %s", label)
                if attempt < self.settings.model_download_retries:
                    await asyncio.sleep(min(5 * attempt, 15))
        return False

    async def ensure_models(self) -> ModelStatus:
        if not self.settings.auto_download_models:
            logger.info("AUTO_DOWNLOAD_MODELS=false: automatic model provisioning disabled")
            return self.status()

        async with self._lock:
            self._downloading = True
            try:
                await self._download_with_retry(
                    self.settings.camie_repo,
                    self.CAMIE_MODEL_FILE,
                    Path(self.settings.camie_model_path),
                    100_000_000,
                    "Camie Tagger v2 model",
                    expected_bytes=790_000_000,
                )
                await self._download_with_retry(
                    self.settings.camie_repo,
                    self.CAMIE_META_FILE,
                    Path(self.settings.camie_metadata_path),
                    1_000,
                    "Camie Tagger v2 metadata",
                    expected_bytes=2_000_000,
                )

                if self.settings.pixai_enabled and self.settings.auto_download_pixai:
                    await self._download_with_retry(
                        self.settings.pixai_repo,
                        self.PIXAI_MODEL_FILE,
                        Path(self.settings.pixai_model_path),
                        1_000_000_000,
                        "PixAI Tagger v0.9 ONNX",
                        expected_bytes=1_272_000_000,
                    )
                    await self._download_with_retry(
                        self.settings.pixai_repo,
                        self.PIXAI_TAGS_FILE,
                        Path(self.settings.pixai_tags_path),
                        100_000,
                        "PixAI selected tags",
                        expected_bytes=1_000_000,
                    )
                elif self.settings.pixai_enabled:
                    logger.info("AUTO_DOWNLOAD_PIXAI=false: Pro+ will use Camie + Qwen when PixAI files are absent")
            finally:
                self._downloading = False

        status = self.status()
        logger.info(
            "Model setup finished: camie_ready=%s pixai_ready=%s",
            status.camie_ready,
            status.pixai_ready,
        )
        return status
