from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import httpx

logger = logging.getLogger(__name__)


class AnimeTraceError(RuntimeError):
    """AnimeTrace request failed or is temporarily unavailable."""


@dataclass
class AnimeTraceResult:
    trace_id: str | None
    ai_generated: bool
    boxes: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    has_confident_box: bool
    model: str | None


class AnimeTraceClient:
    """Conservative client for AnimeTrace's public real-time recognition API.

    The public API currently has no per-user API key. To reduce the chance of
    hitting service protection, request starts are globally spaced, concurrency
    is bounded, and known overload/limit codes open a short circuit breaker.
    """

    TEMPORARY_CODES = {17702, 17706, 17731}
    SUCCESS_CODES = {0, 200, 17720}

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "https://api.animetrace.com",
        enabled: bool = True,
        timeout: float = 15.0,
        max_concurrency: int = 2,
        min_interval_ms: int = 500,
        preferred_model: str = "",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.timeout = max(3.0, timeout)
        self.preferred_model = preferred_model.strip()
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._rate_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._resolved_model: str | None = None
        self._min_interval = max(0.0, min_interval_ms / 1000.0)
        self._last_request = 0.0
        self._blocked_until = 0.0

        self.total = 0
        self.success = 0
        self.failures = 0
        self.rate_limited = 0
        self.last_code: int | None = None

    async def _wait_rate_slot(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            delay = self._min_interval - (now - self._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request = time.monotonic()

    async def _resolve_model(self) -> str | None:
        if self._resolved_model:
            return self._resolved_model

        async with self._model_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                response = await self.client.get(
                    f"{self.base_url}/v1/model/list",
                    timeout=min(self.timeout, 10.0),
                )
                response.raise_for_status()
                payload = response.json()
                models = payload.get("data") if isinstance(payload, dict) else None
                available = [
                    item
                    for item in (models or [])
                    if isinstance(item, dict) and item.get("enabled") and item.get("id")
                ]

                if self.preferred_model:
                    for item in available:
                        if str(item.get("id")) == self.preferred_model:
                            self._resolved_model = self.preferred_model
                            return self._resolved_model
                    logger.warning(
                        "AnimeTrace preferred model %r unavailable; using service default",
                        self.preferred_model,
                    )

                for item in available:
                    if item.get("default"):
                        self._resolved_model = str(item["id"])
                        return self._resolved_model

                if available:
                    self._resolved_model = str(available[0]["id"])
            except Exception:
                # Omitting model is valid; AnimeTrace then selects its server default.
                logger.exception("AnimeTrace model discovery failed; using server default")
            return self._resolved_model

    def _trip_breaker(self, seconds: float) -> None:
        self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)

    async def search_image(self, image_path: str) -> AnimeTraceResult | None:
        if not self.enabled:
            return None
        if time.monotonic() < self._blocked_until:
            raise AnimeTraceError("animetrace_circuit_open")

        async with self._semaphore:
            await self._wait_rate_slot()
            self.total += 1

            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            async with aiofiles.open(path, "rb") as fh:
                raw = await fh.read()

            data: dict[str, str] = {"is_multi": "1", "ai_detect": "1"}
            model = await self._resolve_model()
            if model:
                data["model"] = model

            try:
                response = await self.client.post(
                    f"{self.base_url}/v1/search",
                    data=data,
                    files={"file": (path.name, raw, mime)},
                    timeout=self.timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.failures += 1
                self._trip_breaker(10)
                raise AnimeTraceError("animetrace_transport_error") from exc
            finally:
                del raw

            if response.status_code == 429:
                self.rate_limited += 1
                self.failures += 1
                self._trip_breaker(60)
                raise AnimeTraceError("animetrace_rate_limited")
            if response.status_code == 503:
                self.failures += 1
                self._trip_breaker(20)
                raise AnimeTraceError("animetrace_busy")

            try:
                payload = response.json()
            except ValueError as exc:
                self.failures += 1
                raise AnimeTraceError("animetrace_invalid_json") from exc

            code_raw = payload.get("code") if isinstance(payload, dict) else None
            try:
                code = int(code_raw) if code_raw is not None else response.status_code
            except (TypeError, ValueError):
                code = response.status_code
            self.last_code = code

            if code == 17728:
                self.rate_limited += 1
                self.failures += 1
                self._trip_breaker(10 * 60)
                raise AnimeTraceError("animetrace_usage_limit")
            if code == 17704 or response.status_code == 403:
                self.failures += 1
                self._trip_breaker(5 * 60)
                raise AnimeTraceError("animetrace_maintenance")
            if code in self.TEMPORARY_CODES:
                self.failures += 1
                self._trip_breaker(20)
                raise AnimeTraceError(f"animetrace_temporary_{code}")
            if response.is_error:
                self.failures += 1
                raise AnimeTraceError(f"animetrace_http_{response.status_code}")
            if code not in self.SUCCESS_CODES:
                self.failures += 1
                raise AnimeTraceError(f"animetrace_code_{code}")

            boxes: list[dict[str, Any]] = []
            candidates: list[dict[str, Any]] = []
            has_confident_box = False
            raw_boxes = payload.get("data") if isinstance(payload, dict) else None
            for box_index, item in enumerate(raw_boxes or []):
                if not isinstance(item, dict):
                    continue
                not_confident = bool(item.get("not_confident"))
                has_confident_box = has_confident_box or not not_confident

                box_candidates: list[dict[str, Any]] = []
                for rank, candidate in enumerate(item.get("character") or []):
                    if not isinstance(candidate, dict):
                        continue
                    work = str(candidate.get("work") or "").strip()
                    character = str(candidate.get("character") or "").strip()
                    if not work:
                        continue
                    normalized = {
                        "work": work,
                        "character": character or "Неизвестно",
                        "rank": rank + 1,
                        "box_index": box_index,
                        "box_not_confident": not_confident,
                    }
                    box_candidates.append(normalized)
                    candidates.append(normalized)
                    if rank >= 4:
                        break

                boxes.append(
                    {
                        "box": item.get("box"),
                        "box_id": item.get("box_id"),
                        "not_confident": not_confident,
                        "candidates": box_candidates,
                    }
                )

            self.success += 1
            return AnimeTraceResult(
                trace_id=str(payload.get("trace_id") or "") or None,
                ai_generated=bool(payload.get("ai")),
                boxes=boxes,
                candidates=candidates,
                has_confident_box=has_confident_box,
                model=model,
            )

    def snapshot(self) -> dict[str, Any]:
        remaining = max(0.0, self._blocked_until - time.monotonic())
        return {
            "enabled": self.enabled,
            "model": self._resolved_model or self.preferred_model or "server-default",
            "total": self.total,
            "success": self.success,
            "failures": self.failures,
            "rate_limited": self.rate_limited,
            "last_code": self.last_code,
            "circuit_open_seconds": round(remaining, 1),
        }
