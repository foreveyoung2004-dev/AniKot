from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any

import aiofiles
import httpx

logger = logging.getLogger(__name__)


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
        self._structured_output_supported: bool | None = None

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
        if not text:
            return {"raw": ""}

        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"raw": value}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                try:
                    value = json.loads(match.group(0))
                    if isinstance(value, dict):
                        return value
                except json.JSONDecodeError:
                    pass

        # Safe fallback for providers/models that ignore JSON-only instructions.
        # We only extract an explicitly written title instead of guessing.
        title_match = re.search(
            r"(?:^|\n)\s*(?:название|title_ru|title|anime)\s*[:\-]\s*[\"']?([^\n\"']{2,120})",
            text,
            flags=re.I,
        )
        if title_match:
            title = title_match.group(1).strip(" .,:;-")
            confidence_match = re.search(
                r"(?:уверенность|confidence)\s*[:\-]\s*(\d+(?:\.\d+)?)\s*%?",
                text,
                flags=re.I,
            )
            confidence: float | None = None
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    if confidence > 1:
                        confidence /= 100.0
                except ValueError:
                    confidence = None
            return {
                "title": title,
                "confidence": confidence if confidence is not None else 0.45,
                "is_anime": True,
                "anime_likelihood": 0.7,
                "raw": text[:1200],
            }
        return {"raw": text[:1200]}

    async def _chat(self, messages: list[dict[str, Any]], max_tokens: int = 1100) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("search_service_unavailable")

        model = await self._resolve_model()
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # GPT-5 family can spend part of the completion budget on reasoning.
            # 400 tokens was too small for some vision replies and could leave no visible JSON.
            "max_tokens": max_tokens,
        }
        if self._structured_output_supported is not False:
            payload["response_format"] = {"type": "json_object"}
            payload["reasoning_effort"] = "low"

        async def send(body: dict[str, Any]) -> httpx.Response:
            return await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self.timeout,
            )

        response = await send(payload)

        # Some proxy/model combinations may reject response_format/reasoning_effort.
        # Retry once without those fields; a rejected 400 request is not a model result.
        if response.status_code == 400 and "response_format" in payload:
            logger.warning(
                "Structured output rejected for model=%s; retrying compatibility mode",
                model,
            )
            self._structured_output_supported = False
            fallback = dict(payload)
            fallback.pop("response_format", None)
            fallback.pop("reasoning_effort", None)
            response = await send(fallback)
        elif response.is_success and "response_format" in payload:
            self._structured_output_supported = True

        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )

        parsed = self._parse_json(str(content))
        parsed["_model"] = data.get("model") or model
        usage = data.get("usage")
        if isinstance(usage, dict):
            parsed["_usage"] = usage

        logger.info(
            "AI result model=%s finish=%s title=%r confidence=%r is_anime=%r anime_likelihood=%r usage=%s",
            parsed.get("_model"),
            choice.get("finish_reason"),
            parsed.get("title_ru") or parsed.get("title"),
            parsed.get("confidence"),
            parsed.get("is_anime"),
            parsed.get("anime_likelihood"),
            usage if isinstance(usage, dict) else None,
        )
        if not (parsed.get("title_ru") or parsed.get("title")):
            raw_preview = str(parsed.get("raw") or content or "")[:500].replace("\n", " ")
            logger.warning(
                "AI returned no title model=%s finish=%s content_len=%s preview=%r",
                parsed.get("_model"),
                choice.get("finish_reason"),
                len(str(content or "")),
                raw_preview,
            )
        return parsed

    @staticmethod
    def _result_contract() -> str:
        return (
            'Верни ТОЛЬКО JSON без пояснений: '
            '{"is_anime":true,"anime_likelihood":0.0,'
            '"title_ru":"общеупотребимое русское название",'
            '"title_original":"оригинальное/международное название",'
            '"character_ru":"имя персонажа по-русски или Неизвестно",'
            '"character_original":"оригинальное имя персонажа или null",'
            '"country":"Япония или Китай","year":2024,"episodes":12,"confidence":0.0,'
            '"alternatives":[{"title_ru":"русское название","title_original":"оригинальное название","confidence":0.0}],'
            '"adult_content":false,"minor_risk":false}. '
            'КРИТИЧЕСКИ ВАЖНО: title_ru, character_ru и title_ru внутри alternatives должны быть на русском языке. '
            'Для известных тайтлов используй именно распространённое официальное/фэндомное русское название, а не ромадзи. '
            'Пример: Kenja no Mago -> Внук мудреца; Shingeki no Kyojin -> Атака титанов. '
            'Имена персонажей записывай кириллицей: Sicily von Claude -> Сицилия фон Клод. '
            'title_original/character_original нужны только как внутренние поля и пользователю не показываются. '
            'is_anime=true, если основное содержимое кадра — аниме/дунхуа/анимационный персонаж, даже если вокруг интерфейс TikTok/VK/YouTube. '
            'Для фото людей, игр, мемов и другого не-аниме ставь is_anime=false. anime_likelihood — уверенность от 0 до 1, что на изображении есть аниме/дунхуа. '
            'confidence от 0 до 1 относится именно к определению конкретного тайтла. '
            'Если это аниме, не оставляй title_ru пустым только из-за сомнений: верни лучший вариант, а сомнение отрази через confidence и alternatives. '
            'title_ru можно оставить пустым только если это явно не аниме/дунхуа или визуальных данных недостаточно вообще. '
            'year/episodes могут быть null. Не выдумывай факты.'
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
                "Задача: максимально точно определить аниме или дунхуа по одному изображению. "
                "Перед ответом молча выполни многошаговую проверку и НЕ выводи ход рассуждений. "
                "1) Отдели сам аниме-кадр от интерфейса TikTok/VK/YouTube, рамок, кнопок, водяных знаков и текста соцсети. "
                "2) Определи визуальные признаки: персонаж, пол/возрастной образ, цвет и форма волос/глаз, одежда, форма, аксессуары, оружие, эмблемы, окружение, эпоха, палитра и стиль анимации. "
                "3) Прочитай видимый текст/OCR, логотипы и субтитры, но используй подписи соцсети только как вспомогательную подсказку, а не как единственное доказательство. "
                "4) Сформируй минимум 3 возможных кандидата, сравни их с кадром и отбрось варианты с явными противоречиями. "
                "5) Для финального кандидата проверь, действительно ли такой персонаж/форма/сцена совместимы с этим тайтлом. "
                "6) Если можно определить конкретный сезон/часть — учитывай его, но в title_ru оставляй общеупотребимое русское название произведения. "
                "7) Для популярных аниме не занижай confidence только из-за интерфейса соцсети или неполного кадра. "
                "В финальном JSON верни только результат, без рассуждений. "
                "Для country используй только Япония или Китай. "
                + self._result_contract()
            )
            try:
                return await self._chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Ты специализированный эксперт по визуальной идентификации аниме и дунхуа. "
                                "Твоя задача — не описывать изображение, а определить конкретный тайтл и персонажа. "
                                "Используй визуальные признаки, узнаваемость дизайна персонажа, форму, символику, фон, стиль студии и видимый текст. "
                                "Игнорируй UI соцсетей как часть произведения. "
                                "Не выбирай первый знакомый вариант: сначала внутренне сравни несколько кандидатов и только затем выбери лучший. "
                                "Все пользовательские названия и имена возвращай на русском языке. "
                                "Если это аниме, обязательно дай лучший вероятный вариант и честную confidence."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"}},
                            ],
                        },
                    ],
                    max_tokens=1200,
                )
            finally:
                del encoded

    async def identify_anime_from_text(self, query: str) -> dict[str, Any]:
        async with self._semaphore:
            prompt = (
                f"Пользователь ищет аниме или дунхуа по названию/описанию: {query!r}. "
                "Исправь опечатки, транслитерацию и альтернативные названия. "
                "Определи общеупотребимое русское название тайтла и русскую запись имени персонажа. "
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
                max_tokens=800,
            )
