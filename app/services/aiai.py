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
    """Small OpenAI-compatible client used by one AniKot search tier."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        preferred_model: str,
        timeout: float = 90,
        max_concurrency: int = 6,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.preferred_model = preferred_model
        self.timeout = timeout
        self._resolved_model: str | None = None
        self._model_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

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
                # Use the configured ID directly. The actual request will fail cleanly
                # and the caller will refund the user's search credit.
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

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 650) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("search_service_unavailable")
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                model = await self._resolve_model(client)
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
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
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    # Some OpenAI-compatible providers return content blocks.
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
            'Верни ТОЛЬКО JSON по схеме: '
            '{"title":"точное название",'
            '"character":"имя персонажа или Неизвестно",'
            '"country":"Япония или Китай",'
            '"year":2024,'
            '"episodes":12,'
            '"confidence":0.0,'
            '"adult_content":false,'
            '"minor_risk":false}. '
            'confidence — число от 0 до 1. Если точного ответа нет, не угадывай: '
            'оставь title пустым и поставь низкую confidence. '
            'year и episodes могут быть null, если нельзя определить надёжно. '
            'minor_risk=true только когда adult_content=true и есть признаки несовершеннолетнего '
            'или возраст персонажа в сексуальном контексте неоднозначен.'
        )

    async def identify_anime_from_image(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "Определи произведение по этому аниме/дунхуа-кадру. "
            "Перед финальным ответом внутренне сравни до трёх наиболее вероятных кандидатов по персонажу, "
            "дизайну, фону, рисовке, одежде, символам и тексту на кадре. "
            "Нужен именно тайтл конкретного произведения/сезона, а не название франшизы, если это возможно. "
            "Для country используй Япония для аниме и Китай для дунхуа. "
            "Не описывай сексуальные действия и не давай ссылки на adult-контент. "
            + self._result_contract()
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты профессиональный специалист по идентификации аниме и дунхуа. "
                        "Приоритет — точность; не выдумывай сведения, когда не уверен."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ]
        )

    async def identify_anime_from_text(self, query: str) -> dict[str, Any]:
        prompt = (
            f"Пользователь ищет аниме или дунхуа по названию/описанию: {query!r}. "
            "Исправь опечатки, русскую транслитерацию и альтернативные названия, затем выбери наиболее вероятный тайтл. "
            "Для country используй Япония или Китай. Не выдумывай сведения. "
            + self._result_contract()
        )
        return await self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты профессиональный специалист по каталогам аниме и дунхуа. "
                        "Возвращай только сведения, в которых достаточно уверен."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=550,
        )
