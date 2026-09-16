from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx


class AIAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        preferred_model: str,
        fallback_model: str,
        timeout: float = 90,
        max_concurrency: int = 6,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.preferred_model = preferred_model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self._resolved_model: str | None = None
        self._model_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def _resolve_model(self, client: httpx.AsyncClient) -> str:
        if self._resolved_model:
            return self._resolved_model
        async with self._model_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                ids = [str(x.get("id", "")) for x in data if x.get("id")]
                lower = {x.lower(): x for x in ids}
                preferred = self.preferred_model.lower()
                fallback = self.fallback_model.lower()
                if preferred in lower:
                    self._resolved_model = lower[preferred]
                else:
                    # Model IDs can differ only by punctuation/case between the
                    # public catalogue and /v1/models. Match a normalized ID
                    # before falling back to another model family.
                    norm = lambda v: re.sub(r"[^a-z0-9]+", "", v.lower())
                    wanted_norm = norm(self.preferred_model)
                    fuzzy = [
                        x for x in ids
                        if norm(x) == wanted_norm or norm(x).endswith(wanted_norm) or wanted_norm in norm(x)
                    ]
                    if fuzzy:
                        self._resolved_model = fuzzy[0]
                    elif fallback in lower:
                        self._resolved_model = lower[fallback]
                    else:
                        fallback_norm = norm(self.fallback_model)
                        fuzzy_fallback = [
                            x for x in ids
                            if norm(x) == fallback_norm or norm(x).endswith(fallback_norm) or fallback_norm in norm(x)
                        ]
                        self._resolved_model = fuzzy_fallback[0] if fuzzy_fallback else self.preferred_model
            except Exception:
                self._resolved_model = self.preferred_model
            return self._resolved_model

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {"raw": text}
            return json.loads(match.group(0))

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 700) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("AIAI_API_KEY не задан")
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                model = await self._resolve_model(client)
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                }
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if response.status_code >= 400 and model != self.fallback_model:
                    payload["model"] = self.fallback_model
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                result = self._parse_json(content)
                result["_model"] = data.get("model") or payload["model"]
                return result

    async def identify_anime_from_image(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "Определи аниме по кадру. Не выдумывай название. Если уверенность низкая, так и укажи. "
            "Одновременно классифицируй, относится ли кадр к явно adult/explicit аниме-контенту. "
            "minor_risk=true ставь только если adult_content=true и персонаж может быть несовершеннолетним "
            "или возраст неоднозначен. Не описывай сексуальные действия. "
            "Верни ТОЛЬКО JSON: {\"title\":\"каноническое название\",\"title_ru\":\"русское название или пусто\","
            "\"confidence\":0.0,\"characters\":[\"...\"],\"evidence\":\"кратко почему\","
            "\"adult_content\":false,\"minor_risk\":false,\"reason\":\"кратко\"}. "
            "confidence — число от 0 до 1."
        )
        return await self._chat(
            [
                {"role": "system", "content": "Ты эксперт по аниме и распознаванию кадров."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ]
        )

    async def normalize_title(self, title: str) -> dict[str, Any]:
        prompt = (
            f"Пользователь ищет аниме по названию: {title!r}. Исправь возможную опечатку или русскую транслитерацию. "
            "Верни ТОЛЬКО JSON: {\"title\":\"наиболее вероятное каноническое название для поиска\","
            "\"alternatives\":[\"...\"],\"confidence\":0.0}. Не придумывай, если не уверен."
        )
        return await self._chat(
            [
                {"role": "system", "content": "Ты нормализуешь названия аниме для каталожного поиска."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )

    async def identify_adult_anime_from_image(
        self,
        image_path: str,
        tag_hints: list[str] | None = None,
    ) -> dict[str, Any]:
        """Adult-title identification for AniKot Pro+.

        This method is intentionally limited to title/source identification.
        It must refuse/flag any case with underage or ambiguous-age indicators.
        """
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        hints = ", ".join((tag_hints or [])[:24])
        prompt = (
            "Задача: определить НАЗВАНИЕ/ИСТОЧНИК adult-аниме по кадру, без описания сексуальных действий "
            "и без ссылок на просмотр. Если персонаж может быть несовершеннолетним, выглядит несовершеннолетним "
            "или возраст неоднозначен, НЕ определяй тайтл и поставь minor_risk=true. "
            f"Подсказки локальных теггеров: {hints or 'нет'}. "
            "Верни ТОЛЬКО JSON: "
            "{\"title\":\"каноническое название или пусто\",\"title_ru\":\"русское название или пусто\","
            "\"confidence\":0.0,\"adult_content\":true,\"minor_risk\":false,"
            "\"reason\":\"кратко\",\"evidence\":\"только признаки источника/персонажа, без сексуального описания\"}."
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты классификатор источника аниме. Для adult-контента допускается только нейтральная "
                        "идентификация произведения. Любой риск несовершеннолетнего или неоднозначного возраста "
                        "означает minor_risk=true и пустой title."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
            max_tokens=500,
        )

    async def normalize_adult_title(self, title: str) -> dict[str, Any]:
        prompt = (
            f"Нормализуй только название adult-аниме для каталожного поиска: {title!r}. "
            "Не описывай сексуальный контент и не давай ссылки. Если запрос явно относится к сексуальному "
            "контенту с несовершеннолетним или возраст неоднозначен, верни minor_risk=true и пустой title. "
            "Верни ТОЛЬКО JSON: "
            "{\"title\":\"каноническое название или пусто\",\"confidence\":0.0,"
            "\"minor_risk\":false,\"reason\":\"кратко\"}."
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": "Ты нормализуешь названия для каталога; не генерируешь эротические описания.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
        )

