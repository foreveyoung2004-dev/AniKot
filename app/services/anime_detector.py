from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .aiai import AIAIClient


@dataclass
class AnimeResult:
    title: str
    character: str
    country: str
    year: int | None
    episodes: int | None
    confidence: float | None
    engine: str
    adult_content: bool = False
    minor_risk: bool = False
    usage: dict[str, Any] | None = None
    alternatives: list[dict[str, Any]] | None = None
    is_anime: bool = True
    anime_likelihood: float | None = None


class AnimeDetector:
    """AniKot search engine using only the selected remote model for each tier."""

    def __init__(
        self,
        anikot_ai: AIAIClient,
        pro_ai: AIAIClient,
        proplus_ai: AIAIClient,
    ):
        self.anikot_ai = anikot_ai
        self.pro_ai = pro_ai
        self.proplus_ai = proplus_ai

    @staticmethod
    def _confidence(value: Any) -> float | None:
        try:
            if value is None:
                return None
            number = float(value)
            if number > 1.0 and number <= 100.0:
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

    def _from_remote(self, remote: dict[str, Any]) -> AnimeResult | None:
        # Prefer Russian-facing fields. Old cache/provider responses remain compatible via fallbacks.
        title = str(
            remote.get("title_ru")
            or remote.get("title")
            or remote.get("title_original")
            or ""
        ).strip()
        if not title:
            return None

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
            for item in raw_alternatives[:2]:
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

        return AnimeResult(
            title=title,
            character=character,
            country=self._country(remote.get("country")),
            year=self._integer(remote.get("year")),
            episodes=self._integer(remote.get("episodes")),
            confidence=self._confidence(remote.get("confidence")),
            engine=str(remote.get("_model") or "AI"),
            adult_content=bool(remote.get("adult_content", False)),
            minor_risk=bool(remote.get("minor_risk", False)),
            usage=remote.get("_usage") if isinstance(remote.get("_usage"), dict) else None,
            alternatives=alternatives,
            is_anime=bool(remote.get("is_anime", True)),
            anime_likelihood=self._confidence(remote.get("anime_likelihood")),
        )

    async def _image(self, client: AIAIClient, path: str) -> AnimeResult | None:
        return self._from_remote(await client.identify_anime_from_image(path))

    async def _text(self, client: AIAIClient, query: str) -> AnimeResult | None:
        return self._from_remote(await client.identify_anime_from_text(query))

    async def identify_image_anikot(self, path: str) -> AnimeResult | None:
        return await self._image(self.anikot_ai, path)

    async def identify_image_pro(self, path: str) -> AnimeResult | None:
        return await self._image(self.pro_ai, path)

    async def identify_image_proplus(self, path: str) -> AnimeResult | None:
        return await self._image(self.proplus_ai, path)

    async def identify_title_anikot(self, query: str) -> AnimeResult | None:
        return await self._text(self.anikot_ai, query)

    async def identify_title_pro(self, query: str) -> AnimeResult | None:
        return await self._text(self.pro_ai, query)

    async def identify_title_proplus(self, query: str) -> AnimeResult | None:
        return await self._text(self.proplus_ai, query)
