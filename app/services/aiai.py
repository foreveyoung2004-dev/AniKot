from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import aiofiles
import httpx


class AIAIClient:
    """Memory-conscious OpenAI-compatible async client for one AniKot tier.

    All tiers share one httpx.AsyncClient and one global semaphore supplied by main.py.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        preferred_model: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        timeout: float = 90,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.preferred_model = preferred_model
        self.timeout = timeout
        self.client = client
        self._resolved_model: str | None = None
        self._model_lock = asyncio.Lock()
        self._semaphore = semaphore

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        async with self._model_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                response = await self.client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=min(self.timeout, 30),
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                ids = [str(x.get("id", "")) for x in data if isinstance(x, dict) and x.get("id")]
                lower = {x.lower(): x for x in ids}
                wanted = self.preferred_model.lower()
                if wanted in lower:
                    self._resolved_model = lower[wanted]
                else:
                    wanted_norm = self._norm(self.preferred_model)
                    matches = [
                        model_id
                        for model_id in ids
                        if self._norm(model_id) == wanted_norm
                        or self._norm(model_id).endswith(wanted_norm)
                        or wanted_norm in self._norm(model_id)
                    ]
                    self._resolved_model = matches[0] if matches else self.preferred_model
            except Exception:
                self._resolved_model = self.preferred_model
            return self._resolved_model

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"raw": value}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {"raw": text}
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {"raw": value}

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 420) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("search_service_unavailable")
        model = await self._resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )
        result = self._parse_json(str(content))
        result["_model"] = data.get("model") or model
        usage = data.get("usage")
        if isinstance(usage, dict):
            result["_usage"] = usage
        return result

    @staticmethod
    def _result_contract() -> str:
        return (
            'Верни ТОЛЬКО JSON: '
            '{"title":"наиболее вероятное точное название",'
            '"character":"имя персонажа или Неизвестно",'
            '"country":"Япония или Китай","year":2024,"episodes":12,"confidence":0.0,'
            '"alternatives":[{"title":"другой вероятный тайтл","confidence":0.0}],'
            '"adult_content":false,"minor_risk":false}. '
            'confidence от 0 до 1. Даже если уверенность невысокая, обязательно верни '
            'наиболее вероятный title, если на изображении действительно аниме/дунхуа. '
            'Не оставляй title пустым только из-за сомнений. '
            'В alternatives верни до 2 других правдоподобных вариантов, если они есть. '
            'title можно оставить пустым только если это явно не аниме/дунхуа или визуальных данных недостаточно вообще. '
            'year/episodes могут быть null. Не выдумывай факты: сомнение отражай через confidence.'
        )

    async def identify_anime_from_image(self, image_path: str) -> dict[str, Any]:
        # The global gate is acquired before the file is read/base64-encoded.
        # This is the key RAM guard: queued searches do not hold large image strings.
        async with self._semaphore:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            async with aiofiles.open(path, "rb") as fh:
                raw = await fh.read()
            try:
                encoded = base64.b64encode(raw).decode("ascii")
            finally:
                del raw

            prompt = (
                "Определи аниме или дунхуа по кадру. Внутренне сравни до трёх кандидатов "
                "по персонажу, рисовке, одежде, фону, символам и тексту. "
                "Нужен конкретный тайтл/сезон, если это возможно. "
                "Для country используй Япония или Китай. "
                + self._result_contract()
            )
            try:
                return await self._chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Ты специалист по идентификации аниме и дунхуа. "
                                "Приоритет — точность. Не угадывай при недостатке данных."
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
                    max_tokens=420,
                )
            finally:
                del encoded

    async def identify_anime_from_text(self, query: str) -> dict[str, Any]:
        async with self._semaphore:
            prompt = (
                f"Пользователь ищет аниме или дунхуа по названию/описанию: {query!r}. "
                "Исправь опечатки, транслитерацию и альтернативные названия. "
                "Выбери наиболее вероятный тайтл, но не выдумывай сведения. "
                + self._result_contract()
            )
            return await self._chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Ты специалист по каталогам аниме и дунхуа. "
                            "Возвращай только сведения, в которых достаточно уверен."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=360,
            )
