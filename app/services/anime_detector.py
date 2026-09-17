from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from .aiai import AIAIClient
from .animetrace import AnimeTraceClient, AnimeTraceError, AnimeTraceResult

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "accuracy-v4-hybrid"


@dataclass
class AnimeResult:
    title: str
    character: str
    country: str
    year: int | None
    episodes: int | None
    confidence: float | None
    engine: str
    analysis_version: str
    adult_content: bool = False
    minor_risk: bool = False
    usage: dict[str, Any] | None = None
    alternatives: list[dict[str, Any]] | None = None
    is_anime: bool = True
    anime_likelihood: float | None = None

    def __post_init__(self) -> None:
        # Cache entries from the old single-model architecture must be recomputed.
        if self.analysis_version != ANALYSIS_VERSION:
            raise ValueError("stale_analysis_version")


class AnimeDetector:
    """Tiered anime recognition.

    AniKot: Qwen3.5 Flash vision.
    Pro: AnimeTrace candidates -> DeepSeek verifier/normalizer.
    Pro+: AnimeTrace and Kimi K2.6 independently -> Kimi reconciliation.
    """

    def __init__(
        self,
        anikot_ai: AIAIClient,
        pro_ai: AIAIClient,
        proplus_ai: AIAIClient,
        animetrace: AnimeTraceClient | None = None,
    ):
        self.anikot_ai = anikot_ai
        self.pro_ai = pro_ai
        self.proplus_ai = proplus_ai
        self.animetrace = animetrace

    @staticmethod
    def _confidence(value: Any) -> float | None:
        try:
            if value is None:
                return None
            number = float(value)
            if 1.0 < number <= 100.0:
                number /= 100.0
            return max(0.0, min(1.0, number))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _integer(value: Any) -> int | None:
        if value in (None, "", "unknown", "неизвестно", "Неизвестно"):
            return None
        try:
            number = int(float(value))
            return number if number > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _country(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if any(token in raw for token in ("china", "chinese", "китай", "кнр")):
            return "Китай"
        return "Япония"

    @staticmethod
    def _merge_usage(
        first: dict[str, Any] | None,
        second: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        a = first if isinstance(first, dict) else {}
        b = second if isinstance(second, dict) else {}
        if not a and not b:
            return None
        merged: dict[str, Any] = {}
        for key in set(a) | set(b):
            av = a.get(key)
            bv = b.get(key)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                merged[key] = av + bv
            elif bv is not None:
                merged[key] = bv
            else:
                merged[key] = av
        return merged

    def _from_remote(
        self,
        remote: dict[str, Any],
        *,
        engine_override: str | None = None,
    ) -> AnimeResult | None:
        title = str(
            remote.get("title_ru")
            or remote.get("title")
            or remote.get("title_original")
            or ""
        ).strip()
        is_anime = bool(remote.get("is_anime", True))
        if not title:
            if is_anime:
                return None
            title = "Не аниме"

        character_raw = (
            remote.get("character_ru")
            or remote.get("character")
            or remote.get("character_original")
        )
        if not character_raw and isinstance(remote.get("characters"), list):
            character_raw = ", ".join(str(x) for x in remote["characters"][:2])
        character = str(character_raw or "Неизвестно").strip() or "Неизвестно"

        alternatives: list[dict[str, Any]] = []
        raw_alternatives = remote.get("alternatives")
        if isinstance(raw_alternatives, list):
            seen = {title.casefold()}
            for item in raw_alternatives[:4]:
                if not isinstance(item, dict):
                    continue
                alt_title = str(
                    item.get("title_ru")
                    or item.get("title")
                    or item.get("title_original")
                    or ""
                ).strip()
                if not alt_title or alt_title.casefold() in seen:
                    continue
                seen.add(alt_title.casefold())
                alternatives.append(
                    {
                        "title": alt_title,
                        "confidence": self._confidence(item.get("confidence")),
                    }
                )
                if len(alternatives) >= 2:
                    break

        model = engine_override or str(remote.get("_model") or "AI")
        if not engine_override and remote.get("_verified"):
            model = f"{model} + verifier"

        return AnimeResult(
            title=title,
            character=character,
            country=self._country(remote.get("country")),
            year=self._integer(remote.get("year")),
            episodes=self._integer(remote.get("episodes")),
            confidence=self._confidence(remote.get("confidence")),
            engine=model,
            analysis_version=ANALYSIS_VERSION,
            adult_content=bool(remote.get("adult_content", False)),
            minor_risk=bool(remote.get("minor_risk", False)),
            usage=remote.get("_usage") if isinstance(remote.get("_usage"), dict) else None,
            alternatives=alternatives,
            is_anime=is_anime,
            anime_likelihood=self._confidence(remote.get("anime_likelihood")),
        )

    async def _image(self, client: AIAIClient, path: str) -> AnimeResult | None:
        remote = await client.identify_anime_from_image(path)
        return self._from_remote(remote)

    async def _text(self, client: AIAIClient, query: str) -> AnimeResult | None:
        remote = await client.identify_anime_from_text(query)
        return self._from_remote(remote)

    async def _trace(self, path: str) -> AnimeTraceResult | None:
        if self.animetrace is None or not self.animetrace.enabled:
            return None
        try:
            result = await self.animetrace.search_image(path)
            if result and result.candidates:
                return result
        except AnimeTraceError as exc:
            logger.warning("AnimeTrace unavailable: %s", exc)
        except Exception:
            logger.exception("Unexpected AnimeTrace failure")
        return None

    @staticmethod
    def _trace_payload(trace: AnimeTraceResult) -> dict[str, Any]:
        return {
            "trace_id": trace.trace_id,
            "ai_generated": trace.ai_generated,
            "has_confident_box": trace.has_confident_box,
            "model": trace.model or "server-default",
            "candidates": trace.candidates[:12],
        }

    @staticmethod
    def _json_contract(extra: str = "") -> str:
        return (
            'Верни только JSON: {"is_anime":true,"anime_likelihood":0.0,'
            '"title_ru":"русское название","title_original":"оригинальное название",'
            '"character_ru":"имя персонажа по-русски или Неизвестно",'
            '"character_original":"оригинальное имя или null",'
            '"country":"Япония или Китай","year":null,"episodes":null,'
            '"confidence":0.0,"alternatives":[{"title_ru":"","confidence":0.0}],'
            '"adult_content":false,"minor_risk":false'
            + extra
            + '}. Не добавляй markdown или пояснения.'
        )

    async def _text_chat(
        self,
        client: AIAIClient,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 900,
        reasoning_effort: str = "low",
    ) -> dict[str, Any]:
        # AIAIClient._chat intentionally does not acquire its semaphore because
        # its public image/text methods already do. Hybrid calls are external to
        # those methods, so acquire the same gate here to preserve global limits.
        async with client._semaphore:  # noqa: SLF001 - internal service composition
            return await client._chat(  # noqa: SLF001 - internal service composition
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

    async def _verify_trace_with_deepseek(
        self,
        trace: AnimeTraceResult,
        safety: dict[str, Any] | None,
    ) -> AnimeResult | None:
        prompt = (
            "AnimeTrace распознал персонажей на кадре и вернул кандидатов ниже. "
            "Ты НЕ видишь исходное изображение. Твоя задача — проверить логическую согласованность кандидатов, "
            "сопоставить японские/китайские/английские названия с общеупотребимыми русскими названиями, "
            "убрать явные дубли и выбрать наиболее согласованный результат. Не выдумывай новый тайтл, которого нет "
            "среди кандидатов, кроме перевода/алиаса того же произведения. Кандидат rank=1 важнее последующих; "
            "box_not_confident=true означает слабое распознавание. Если несколько независимых боксов указывают на одно "
            "произведение, это усиливает результат. confidence оценивай консервативно: при отсутствии уверенного бокса "
            "не ставь выше 0.74. Год и число эпизодов указывай только если уверен. "
            "Данные AnimeTrace: "
            + json.dumps(self._trace_payload(trace), ensure_ascii=False, separators=(",", ":"))
            + ". "
            + self._json_contract()
        )
        remote = await self._text_chat(
            self.pro_ai,
            system=(
                "Ты проверяющий результатов специализированного поиска аниме. "
                "Не притворяйся, что видишь изображение; анализируй только предоставленные кандидаты."
            ),
            prompt=prompt,
            max_tokens=900,
            reasoning_effort="low",
        )

        if not trace.has_confident_box:
            confidence = self._confidence(remote.get("confidence"))
            if confidence is not None:
                remote["confidence"] = min(confidence, 0.74)

        if safety:
            remote["adult_content"] = bool(safety.get("adult_content"))
            remote["minor_risk"] = bool(safety.get("minor_risk"))
            if remote.get("anime_likelihood") is None:
                remote["anime_likelihood"] = safety.get("anime_likelihood")

        remote["_accuracy_revision"] = ANALYSIS_VERSION
        return self._from_remote(
            remote,
            engine_override=f"AnimeTrace + {remote.get('_model') or 'DeepSeek verifier'}",
        )

    async def _reconcile_proplus(
        self,
        trace: AnimeTraceResult,
        kimi_primary: dict[str, Any],
    ) -> AnimeResult | None:
        kimi_compact = {
            "title_ru": kimi_primary.get("title_ru") or kimi_primary.get("title"),
            "title_original": kimi_primary.get("title_original"),
            "character_ru": kimi_primary.get("character_ru") or kimi_primary.get("character"),
            "character_original": kimi_primary.get("character_original"),
            "country": kimi_primary.get("country"),
            "year": kimi_primary.get("year"),
            "episodes": kimi_primary.get("episodes"),
            "confidence": kimi_primary.get("confidence"),
            "alternatives": kimi_primary.get("alternatives") or [],
            "evidence": kimi_primary.get("evidence") or {},
            "adult_content": bool(kimi_primary.get("adult_content", False)),
            "minor_risk": bool(kimi_primary.get("minor_risk", False)),
        }
        prompt = (
            "До получения данных AnimeTrace ты уже независимо проанализировал исходный кадр как vision-модель Kimi. "
            "Теперь нужно сверить независимый вывод Kimi с результатами AnimeTrace. Учитывай, что названия могут быть "
            "одним и тем же произведением на русском, английском, японском или китайском. Не соглашайся автоматически "
            "ни с одним источником. Если источники явно совпадают по произведению/персонажу — sources_agree=true. "
            "Если они конфликтуют — sources_agree=false, выбери наиболее обоснованный вариант и снизь confidence. "
            "Если AnimeTrace пометил все боксы как not_confident, не позволяй слабому Trace переопределить сильный "
            "независимый Kimi без веской причины. Сохрани conservative safety: adult_content/minor_risk не могут стать "
            "false, если Kimi уже отметил их true. "
            "Независимый Kimi: "
            + json.dumps(kimi_compact, ensure_ascii=False, separators=(",", ":"))
            + ". AnimeTrace: "
            + json.dumps(self._trace_payload(trace), ensure_ascii=False, separators=(",", ":"))
            + ". "
            + self._json_contract(',"sources_agree":true')
        )
        reconciled = await self._text_chat(
            self.proplus_ai,
            system=(
                "Ты финальный арбитр AniKot Pro+. Сверяй два независимых источника и возвращай только структурированный итог."
            ),
            prompt=prompt,
            max_tokens=950,
            reasoning_effort="medium",
        )

        reconciled["adult_content"] = bool(
            kimi_primary.get("adult_content") or reconciled.get("adult_content")
        )
        reconciled["minor_risk"] = bool(
            kimi_primary.get("minor_risk") or reconciled.get("minor_risk")
        )
        reconciled["_usage"] = self._merge_usage(
            kimi_primary.get("_usage"),
            reconciled.get("_usage"),
        )
        reconciled["_accuracy_revision"] = ANALYSIS_VERSION
        return self._from_remote(
            reconciled,
            engine_override=f"AnimeTrace + {reconciled.get('_model') or 'Kimi K2.6'}",
        )

    async def identify_image_anikot(self, path: str) -> AnimeResult | None:
        return await self._image(self.anikot_ai, path)

    async def identify_image_pro(self, path: str) -> AnimeResult | None:
        # Qwen is not a recognition source for Pro in the normal path. It runs in
        # parallel as a safety/fallback vision pass so the move to text-only
        # DeepSeek does not remove the existing adult/minor image safeguards.
        trace_task = asyncio.create_task(self._trace(path))
        safety_task = asyncio.create_task(self.anikot_ai.identify_anime_from_image(path))
        trace_out, safety_out = await asyncio.gather(
            trace_task,
            safety_task,
            return_exceptions=True,
        )

        trace = trace_out if isinstance(trace_out, AnimeTraceResult) else None
        safety = safety_out if isinstance(safety_out, dict) else None
        if isinstance(trace_out, Exception):
            logger.warning("Pro AnimeTrace task failed: %s", trace_out)
        if isinstance(safety_out, Exception):
            logger.warning("Pro safety vision task failed: %s", safety_out)

        if trace is not None:
            try:
                return await self._verify_trace_with_deepseek(trace, safety)
            except Exception:
                logger.exception("DeepSeek AnimeTrace verification failed")

        # AnimeTrace/DeepSeek unavailable: preserve availability with the visual
        # safety pass. The engine label makes fallback usage visible in logs.
        if safety is not None:
            safety["_accuracy_revision"] = ANALYSIS_VERSION
            return self._from_remote(
                safety,
                engine_override=f"{safety.get('_model') or 'Qwen3.5 Flash'} (Pro fallback)",
            )
        return None

    async def identify_image_proplus(self, path: str) -> AnimeResult | None:
        trace_task = asyncio.create_task(self._trace(path))
        kimi_task = asyncio.create_task(self.proplus_ai.identify_anime_from_image(path))
        trace_out, kimi_out = await asyncio.gather(
            trace_task,
            kimi_task,
            return_exceptions=True,
        )

        trace = trace_out if isinstance(trace_out, AnimeTraceResult) else None
        kimi = kimi_out if isinstance(kimi_out, dict) else None
        if isinstance(trace_out, Exception):
            logger.warning("Pro+ AnimeTrace task failed: %s", trace_out)
        if isinstance(kimi_out, Exception):
            logger.warning("Pro+ Kimi task failed: %s", kimi_out)

        if trace is not None and kimi is not None:
            try:
                return await self._reconcile_proplus(trace, kimi)
            except Exception:
                logger.exception("Pro+ reconciliation failed; keeping independent Kimi result")

        if kimi is not None:
            kimi["_accuracy_revision"] = ANALYSIS_VERSION
            return self._from_remote(
                kimi,
                engine_override=f"{kimi.get('_model') or 'Kimi K2.6'} (AnimeTrace fallback)",
            )

        # Last-resort availability path: use the Pro text verifier to normalize
        # Trace candidates when Kimi is temporarily unavailable.
        if trace is not None:
            try:
                return await self._verify_trace_with_deepseek(trace, None)
            except Exception:
                logger.exception("Pro+ Trace-only fallback failed")
        return None

    async def identify_title_anikot(self, query: str) -> AnimeResult | None:
        return await self._text(self.anikot_ai, query)

    async def identify_title_pro(self, query: str) -> AnimeResult | None:
        # AnimeTrace is image-only, so text/title queries go directly to DeepSeek.
        return await self._text(self.pro_ai, query)

    async def identify_title_proplus(self, query: str) -> AnimeResult | None:
        return await self._text(self.proplus_ai, query)
