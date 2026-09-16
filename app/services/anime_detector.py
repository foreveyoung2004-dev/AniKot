from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from .aiai import AIAIClient
from .anilist import AniListClient
from .camie import CamieTagger
from .content_guard import classify_tags
from .pixai import PixAITagger

logger = logging.getLogger(__name__)


@dataclass
class AnimeResult:
    title: str
    engine: str
    confidence: float | None = None
    media: dict[str, Any] | None = None
    note: str | None = None


@dataclass
class LocalImageCheck:
    result: AnimeResult | None
    explicit: bool = False
    explicit_tags: list[str] | None = None
    minor_risk: bool = False
    minor_tags: list[str] | None = None


@dataclass
class ImageAIResult:
    result: AnimeResult | None
    adult_content: bool = False
    minor_risk: bool = False
    reason: str | None = None


@dataclass
class ProPlusResult:
    result: AnimeResult | None
    adult_content: bool
    minor_risk: bool
    reason: str | None = None


class AnimeDetector:
    def __init__(
        self,
        camie: CamieTagger,
        anikot_ai: AIAIClient,
        pro_ai: AIAIClient,
        proplus_ai: AIAIClient,
        anilist: AniListClient,
        pixai: PixAITagger | None = None,
        local_threshold: float = 0.492,
        top_k: int = 5,
    ):
        self.camie = camie
        self.anikot_ai = anikot_ai
        self.pro_ai = pro_ai
        self.proplus_ai = proplus_ai
        self.anilist = anilist
        self.pixai = pixai
        self.local_threshold = local_threshold
        self.top_k = top_k

    @staticmethod
    def _humanize_tag(tag: str) -> str:
        title = tag.replace("_", " ").strip()
        return re.sub(r"\s*\((series|anime|franchise)\)\s*$", "", title, flags=re.I)

    async def inspect_local_image(self, image_path: str) -> LocalImageCheck:
        if not self.camie.available:
            logger.warning("Camie files are not ready: %s", image_path)
            return LocalImageCheck(None)
        try:
            # Camie v2 author recommends ~0.492 for the macro-optimized profile.
            # We collect candidates from 0.35 and only accept a source after AniList
            # resolves it, which is more reliable than using one very high cutoff.
            grouped = await asyncio.to_thread(
                self.camie.predict,
                image_path,
                0.35,
                max(self.top_k, 24),
            )
            is_explicit, adult_hits, minor_risk, minor_hits = classify_tags(grouped)
            candidates = sorted(
                grouped.get("copyright", []),
                key=lambda x: x[1],
                reverse=True,
            )[: max(self.top_k, 8)]

            result: AnimeResult | None = None
            # Try several copyright/source candidates. A Danbooru source tag can
            # contain underscores/suffixes and the first candidate is not always
            # the one AniList can resolve.
            acceptance = max(0.42, min(self.local_threshold, 0.62))
            for tag, confidence in candidates:
                if confidence < acceptance:
                    continue
                title = self._humanize_tag(tag)
                media = await self.anilist.search(title)
                if not media:
                    continue
                result = AnimeResult(
                    media["title"].get("romaji") or title,
                    "Camie Tagger v2 (local) + AniList",
                    confidence,
                    media,
                    f"Локальный source-тег: {tag}",
                )
                is_explicit = is_explicit or bool(media.get("isAdult"))
                break

            # If no candidate crossed the normal acceptance threshold, try only
            # the strongest source candidate down to 0.35. We expose the lower
            # confidence in the result instead of silently pretending certainty.
            if result is None and candidates and candidates[0][1] >= 0.35:
                tag, confidence = candidates[0]
                title = self._humanize_tag(tag)
                media = await self.anilist.search(title)
                if media:
                    result = AnimeResult(
                        media["title"].get("romaji") or title,
                        "Camie Tagger v2 (local) + AniList",
                        confidence,
                        media,
                        f"Предварительное совпадение source-тега: {tag}",
                    )
                    is_explicit = is_explicit or bool(media.get("isAdult"))

            logger.info(
                "Camie search: result=%s top_sources=%s",
                result.title if result else None,
                [(t, round(float(c), 3)) for t, c in candidates[:5]],
            )
            return LocalImageCheck(result, is_explicit, adult_hits, minor_risk, minor_hits)
        except Exception:
            logger.exception("Camie local inference/search failed")
            return LocalImageCheck(None)

    async def _identify_image_with_ai(
        self,
        image_path: str,
        client: AIAIClient,
        tier_label: str,
    ) -> ImageAIResult:
        remote = await client.identify_anime_from_image(image_path)
        adult_content = bool(remote.get("adult_content", False))
        minor_risk = bool(remote.get("minor_risk", False))
        reason = str(remote.get("reason") or "") or None
        title = str(remote.get("title") or remote.get("title_ru") or "").strip()
        confidence = remote.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        media = await self.anilist.search(title) if title else None
        adult_content = adult_content or bool((media or {}).get("isAdult"))
        resolved = (media or {}).get("title", {}).get("romaji") or title
        if not resolved:
            return ImageAIResult(
                None,
                adult_content=adult_content,
                minor_risk=minor_risk,
                reason=reason or "AI не смог уверенно определить источник кадра.",
            )

        result = AnimeResult(
            resolved,
            f"{tier_label} / AIAI.BY / {remote.get('_model', 'AI')} + AniList",
            confidence,
            media,
            str(remote.get("evidence") or "") or None,
        )
        return ImageAIResult(result, adult_content=adult_content, minor_risk=minor_risk, reason=reason)

    async def identify_image_anikot(self, image_path: str) -> ImageAIResult:
        # Standard AniKot now uses the former Pro model instead of local Camie.
        return await self._identify_image_with_ai(
            image_path, self.anikot_ai, "AniKot"
        )

    async def identify_image_pro(self, image_path: str) -> ImageAIResult:
        # Pro uses the stronger Qwen3.5 397B vision-language tier.
        return await self._identify_image_with_ai(
            image_path, self.pro_ai, "AniKot Pro"
        )

    async def identify_image_proplus(self, image_path: str) -> ProPlusResult:
        """Highest-accuracy tier. GPT-5.4 handles general and adult anime identification.

        Local taggers remain only as a conservative safety pre-check for explicit
        material; they are not the Pro+ search engine.
        """
        local = await self.inspect_local_image(image_path)
        if local.minor_risk:
            return ProPlusResult(
                None,
                adult_content=local.explicit,
                minor_risk=True,
                reason="Есть риск несовершеннолетнего или неоднозначного возраста.",
            )

        remote = await self.proplus_ai.identify_anime_from_image(image_path)
        if bool(remote.get("minor_risk")):
            return ProPlusResult(
                None,
                adult_content=bool(remote.get("adult_content", local.explicit)),
                minor_risk=True,
                reason=str(remote.get("reason") or "Есть риск несовершеннолетнего или неоднозначного возраста."),
            )

        adult_content = bool(remote.get("adult_content", local.explicit))
        title = str(remote.get("title") or remote.get("title_ru") or "").strip()
        confidence = remote.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        media = await self.anilist.search(title) if title else None
        adult_content = adult_content or bool((media or {}).get("isAdult"))
        resolved = (media or {}).get("title", {}).get("romaji") or title
        if not resolved:
            return ProPlusResult(
                None,
                adult_content=adult_content,
                minor_risk=False,
                reason="Не удалось уверенно определить источник.",
            )
        result = AnimeResult(
            resolved,
            f"AniKot Pro+ / {remote.get('_model', 'AI')} + AniList",
            confidence,
            media,
            str(remote.get("evidence") or "") or None,
        )
        return ProPlusResult(result, adult_content=adult_content, minor_risk=False)

    async def identify_title_local(self, title: str) -> AnimeResult | None:
        # Direct catalogue hit is free and maximally reliable. If the spelling is
        # poor, the AniKot model normalizes the title before a second catalogue lookup.
        media = await self.anilist.search(title)
        if media:
            return AnimeResult(media["title"].get("romaji") or title, "AniKot / AniList", 1.0, media)
        normalized = await self.anikot_ai.normalize_title(title)
        canonical = str(normalized.get("title") or title).strip()
        media = await self.anilist.search(canonical)
        if not media:
            return None
        confidence = normalized.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return AnimeResult(
            media["title"].get("romaji") or canonical,
            f"AniKot / {normalized.get('_model', 'AI')} + AniList",
            confidence,
            media,
        )

    async def identify_title_pro(self, title: str) -> AnimeResult:
        direct = await self.anilist.search(title)
        if direct:
            return AnimeResult(direct["title"].get("romaji") or title, "AniKot Pro / AniList", 1.0, direct)
        normalized = await self.pro_ai.normalize_title(title)
        canonical = str(normalized.get("title") or title).strip()
        media = await self.anilist.search(canonical)
        confidence = normalized.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return AnimeResult(
            (media or {}).get("title", {}).get("romaji") or canonical,
            f"AniKot Pro / AIAI.BY / {normalized.get('_model', 'Qwen3-VL')} + AniList",
            confidence,
            media,
            "Название нормализовано через AI.",
        )

    async def identify_title_proplus(self, title: str) -> ProPlusResult:
        direct = await self.anilist.search(title)
        if direct:
            return ProPlusResult(
                AnimeResult(direct["title"].get("romaji") or title, "AniKot Pro+ / AniList", 1.0, direct),
                adult_content=bool(direct.get("isAdult")),
                minor_risk=False,
            )

        normalized = await self.proplus_ai.normalize_title(title)
        canonical = str(normalized.get("title") or title).strip()
        media = await self.anilist.search(canonical)
        confidence = normalized.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return ProPlusResult(
            AnimeResult(
                (media or {}).get("title", {}).get("romaji") or canonical,
                f"AniKot Pro+ / {normalized.get('_model', 'AI')} + AniList",
                confidence,
                media,
            ),
            adult_content=bool((media or {}).get("isAdult")),
            minor_risk=False,
        )

