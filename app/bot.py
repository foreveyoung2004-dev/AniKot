from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiofiles
import httpx
from vkbottle import GroupEventType
from vkbottle.bot import Bot, Message

from .config import PACKAGES, Settings
from .db import Database
from .keyboards import (
    age_confirmation_keyboard,
    legal_keyboard,
    main_keyboard,
    package_confirm_keyboard,
    profile_keyboard,
    referral_choice_keyboard,
    referral_input_keyboard,
    referral_keyboard,
    result_keyboard,
    search_cancel_keyboard,
    search_mode_keyboard,
    shop_categories_keyboard,
    shop_packages_keyboard,
    subscription_keyboard,
)
from .services.anime_detector import AnimeDetector, AnimeResult
from .services.content_guard import explicit_text
from .services.lava import LavaClient

logger = logging.getLogger(__name__)


def _vk_text_link(label: str, url: str) -> str:
    clean = (url or "").strip()
    return f"[{clean}|{label}]" if clean else label


def balance_label(balance_type: str) -> str:
    return {
        "anikot": "AniKot",
        "pro": "AniKot Pro",
        "proplus": "AniKot Pro+",
    }.get(balance_type, balance_type)


def format_result(result: AnimeResult) -> str:
    confidence = "Неизвестно"
    if result.confidence is not None:
        confidence = f"{max(0.0, min(1.0, result.confidence)) * 100:.0f}%"
    return "\n".join(
        [
            f"🎬 Название: {result.title}",
            f"👤 Персонаж: {result.character or 'Неизвестно'}",
            f"🌍 Страна: {result.country}",
            f"📅 Год: {result.year if result.year is not None else 'Неизвестно'}",
            f"📺 Количество эпизодов: {result.episodes if result.episodes is not None else 'Неизвестно'}",
            f"🎯 Уверенность: {confidence}",
        ]
    )


_EMOJI_PREFIXES = (
    "🐾", "✅", "❌", "⚠️", "⛔", "🎁", "🔎", "🔍", "✨", "💎", "👤", "👥",
    "🛒", "💳", "🎬", "📷", "📸", "🔗", "⚙️", "📊", "💰", "📢", "🛡️", "⏳",
    "🤔", "😿", "💸", "🚫", "🕒", "🔞",
)


def _decorate_text(text: str) -> str:
    value = str(text or "")
    if value.startswith(_EMOJI_PREFIXES):
        return value
    lower = value.lower()
    rules = (
        (("режим", "настрой"), "⚙️"),
        (("регистрац",), "✅"),
        (("ссылк", "реферал"), "🔗"),
        (("подпис", "групп"), "👥"),
        (("бонус", "награ"), "🎁"),
        (("оплат", "магаз"), "💳"),
        (("поиск", "аниме", "кадр", "фото"), "🔎"),
        (("профил", "аккаунт"), "👤"),
        (("ошиб", "не удалось"), "⚠️"),
        (("начис", "готов"), "✅"),
        (("отправ",), "📷"),
    )
    for keys, emoji in rules:
        if any(key in lower for key in keys):
            return f"{emoji} {value}"
    return f"🐾 {value}"


def format_uncertain_result(result: AnimeResult) -> str:
    confidence = "Неизвестно"
    if result.confidence is not None:
        confidence = f"{max(0.0, min(1.0, result.confidence)) * 100:.0f}%"

    lines = [
        "🤔 Точного совпадения пока нет, но наиболее вероятный вариант:",
        "",
        f"🎬 Название: {result.title}",
        f"👤 Персонаж: {result.character or 'Неизвестно'}",
        f"🌍 Страна: {result.country}",
        f"📅 Год: {result.year if result.year is not None else 'Неизвестно'}",
        f"📺 Эпизоды: {result.episodes if result.episodes is not None else 'Неизвестно'}",
        f"🎯 Уверенность: {confidence}",
    ]

    if result.alternatives:
        lines.extend(["", "🔀 Другие возможные варианты:"])
        for alt in result.alternatives[:2]:
            alt_conf = alt.get("confidence")
            conf_text = f"{alt_conf * 100:.0f}%" if isinstance(alt_conf, (int, float)) else "?"
            lines.append(f"• {alt.get('title', 'Неизвестно')} — {conf_text}")

    lines.extend(["", "💸 Запрос не списан."])
    return "\n".join(lines)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _photo_url_from_attachments(attachments: Any, max_side: int) -> str | None:
    for attachment in attachments or []:
        atype = _field(attachment, "type")
        photo = _field(attachment, "photo")
        if photo is None and atype == "photo":
            photo = _field(attachment, "object")
        if photo is None:
            continue

        candidates: list[tuple[int, int, str]] = []
        for size in _field(photo, "sizes", []) or []:
            url = _field(size, "url")
            if not url:
                continue
            width = int(_field(size, "width", 0) or 0)
            height = int(_field(size, "height", 0) or 0)
            candidates.append((width, height, str(url)))
        if not candidates:
            continue

        fitting = [x for x in candidates if max(x[0], x[1]) <= max_side]
        if fitting:
            return max(fitting, key=lambda x: x[0] * x[1])[2]

        return min(candidates, key=lambda x: max(x[0], x[1]) or 10**9)[2]
    return None


async def _vision_cache_key(path: Path, mode: str, settings: Settings) -> str:
    digest = hashlib.sha256()
    async with aiofiles.open(path, "rb") as fh:
        while True:
            chunk = await fh.read(256 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    model = {
        "anikot": settings.aiai_anikot_model,
        "pro": settings.aiai_pro_model,
        "proplus": settings.aiai_proplus_model,
    }.get(mode, mode)
    return f"ru-accuracy-v2:{model}:{digest.hexdigest()}"


def extract_photo_url(message: Message, max_side: int) -> str | None:
    direct = _photo_url_from_attachments(getattr(message, "attachments", None), max_side)
    if direct:
        return direct
    reply = getattr(message, "reply_message", None)
    if reply:
        found = _photo_url_from_attachments(_field(reply, "attachments", []), max_side)
        if found:
            return found
    for forwarded in getattr(message, "fwd_messages", None) or []:
        found = _photo_url_from_attachments(_field(forwarded, "attachments", []), max_side)
        if found:
            return found
    return None


async def download_image(
    url: str,
    target_dir: str,
    client: httpx.AsyncClient,
    max_bytes: int,
) -> Path:
    """Stream a VK image to disk without buffering the whole response in RAM."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    path = Path(target_dir) / f"{uuid.uuid4()}.jpg"
    total = 0
    try:
        async with client.stream("GET", url, timeout=30) as response:
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise ValueError("image_too_large")
            async with aiofiles.open(path, "wb") as fh:
                async for chunk in response.aiter_bytes(64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("image_too_large")
                    await fh.write(chunk)
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _extract_ref_payload(message: Message) -> str | None:
    ref = _field(message, "ref")
    return str(ref) if ref else None


def _parse_referral_token(value: str | None) -> str | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw.startswith("anikot_"):
        return raw.removeprefix("anikot_").strip() or None
    try:
        parsed = urlparse(raw)
        qs = parse_qs(parsed.query)
        ref = (qs.get("ref") or [None])[0]
        if ref and str(ref).startswith("anikot_"):
            return str(ref).removeprefix("anikot_").strip() or None
    except Exception:
        return None
    return None


def _referral_base_url(settings: Settings) -> str:
    if settings.vk_referral_base_url:
        return settings.vk_referral_base_url.rstrip("?")
    if settings.vk_group_url:
        try:
            slug = urlparse(settings.vk_group_url).path.strip("/").split("/")[-1]
            if slug:
                return f"https://vk.me/{slug}"
        except Exception:
            pass
    return f"https://vk.me/club{settings.vk_group_id}"


def _referral_link(settings: Settings, token: str) -> str:
    base = _referral_base_url(settings)
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}ref=anikot_{token}"


def _shop_root_text() -> str:
    return "🛒 Магазин AniKot\n\nВыберите тип запросов:"


def _shop_category_text(balance_type: str) -> str:
    title = {
        "anikot": "🔎 AniKot",
        "pro": "✨ AniKot Pro",
        "proplus": "💎 AniKot Pro+",
    }[balance_type]
    packages = [p for p in PACKAGES.values() if p.balance_type == balance_type]
    return title + "\n\n" + "\n".join(f"• {p.label}" for p in packages)


def _package_button_label(package) -> str:
    icon = {"anikot": "🔎", "pro": "✨", "proplus": "💎"}.get(package.balance_type, "🎟️")
    return f"{icon} {package.label.replace(' запросов', '')}"


def _threshold(settings: Settings, mode: str) -> float:
    return {
        "anikot": settings.anikot_min_confidence,
        "pro": settings.pro_min_confidence,
        "proplus": settings.proplus_min_confidence,
    }.get(mode, settings.anikot_min_confidence)


def _reliable(result: AnimeResult | None, threshold: float) -> bool:
    return bool(result and result.confidence is not None and result.confidence >= threshold)


def build_bot(settings: Settings, db: Database, detector: AnimeDetector, lava: LavaClient, http_client: httpx.AsyncClient) -> Bot:
    bot = Bot(settings.vk_token)
    search_tasks: dict[int, asyncio.Task] = {}
    search_context: dict[int, str] = {}
    shop_views: dict[int, str] = {}
    referral_check_lock = asyncio.Lock()
    last_referral_check = 0.0
    setattr(bot, "_anikot_search_tasks", search_tasks)

    async def answer(message: Message, text: str, **kwargs: Any):
        return await message.answer(_decorate_text(text), **kwargs)

    async def is_member(vk_id: int) -> bool:
        result = await bot.api.groups.is_member(group_id=settings.vk_group_id, user_id=vk_id)
        return bool(result) if isinstance(result, int) else bool(getattr(result, "member", result))

    def is_admin(vk_id: int) -> bool:
        return vk_id in settings.admin_ids

    async def user_main_keyboard(vk_id: int) -> str:
        user = await db.get_user(vk_id)
        show_bonus = not bool((user or {}).get("subscription_bonus_claimed"))
        return main_keyboard(show_subscription_bonus=show_bonus)

    async def send_home(message: Message, text: str = "🐾 AniKot\n\nВыберите нужный раздел:") -> None:
        await answer(message, text, keyboard=await user_main_keyboard(message.from_id))

    async def consume_search(vk_id: int, balance_type: str) -> int | str | None:
        if is_admin(vk_id):
            return "∞"
        return await db.consume_request(vk_id, balance_type)

    async def notify_mature_referrals() -> None:
        nonlocal last_referral_check
        now = time.monotonic()
        if now - last_referral_check < 60 or referral_check_lock.locked():
            return
        async with referral_check_lock:
            now = time.monotonic()
            if now - last_referral_check < 60:
                return
            last_referral_check = now
        try:
            rewards = await db.release_mature_referrals()
            for reward in rewards:
                try:
                    await bot.api.messages.send(
                        peer_id=reward["referrer_vk_id"],
                        random_id=0,
                        message=f"🎁 Реферальная награда: +{reward['amount']} AniKot Pro.",
                        keyboard=await user_main_keyboard(reward["referrer_vk_id"]),
                    )
                except Exception:
                    logger.exception("Referral notification failed")
        except Exception:
            logger.exception("Referral processing failed")

    async def send_legal(message: Message) -> None:
        agreement = _vk_text_link("Пользовательское соглашение", settings.user_agreement_url)
        privacy = _vk_text_link("Политику конфиденциальности", settings.privacy_policy_url)
        await answer(message, 
            f"Продолжая регистрацию, вы принимаете {agreement} и {privacy}.",
            keyboard=legal_keyboard(),
        )

    async def send_referral_question(message: Message, user: dict[str, Any]) -> None:
        has_candidate = bool(user.get("referral_candidate_token"))
        text = (
            "Реферальная ссылка обнаружена. Использовать её для регистрации?"
            if has_candidate
            else "У вас есть реферальная ссылка?"
        )
        await answer(message, text, keyboard=referral_choice_keyboard(has_candidate))

    async def finish_registration(message: Message) -> None:
        await db.complete_registration(message.from_id)
        await send_home(
            message,
            f"✅ Регистрация завершена.\n🎁 Начислено: +{settings.registration_bonus_requests} AniKot за регистрацию.",
        )

    async def handle_onboarding(message: Message, user: dict[str, Any], text: str) -> bool:
        state = user.get("onboarding_state") or "legal"
        if state == "legal":
            if text == "✅ Согласен":
                await db.accept_legal(message.from_id)
                user = await db.get_user(message.from_id) or user
                await send_referral_question(message, user)
            else:
                await send_legal(message)
            return True

        if state == "referral_choice":
            candidate = user.get("referral_candidate_token")
            if text == "✅ Использовать реферальную ссылку" and candidate:
                ok, _ = await db.apply_referral_token(message.from_id, str(candidate))
                if ok:
                    await finish_registration(message)
                else:
                    await db.set_referral_candidate(message.from_id, None)
                    await answer(message, "Ссылка не подошла.", keyboard=referral_choice_keyboard(False))
                return True
            if text in {"Да, есть ссылка", "🔗 Да, есть ссылка"}:
                await db.set_onboarding_state(message.from_id, "referral_link")
                await answer(message, "Отправьте реферальную ссылку одним сообщением.", keyboard=referral_input_keyboard())
                return True
            if text in {"Нет, продолжить", "➡️ Нет, продолжить", "Продолжить без ссылки", "➡️ Продолжить без ссылки"}:
                await finish_registration(message)
                return True
            await send_referral_question(message, user)
            return True

        if state == "referral_link":
            if text in {"Продолжить без ссылки", "➡️ Продолжить без ссылки"}:
                await finish_registration(message)
                return True
            token = _parse_referral_token(text)
            if not token:
                await answer(message, "Не удалось распознать ссылку.", keyboard=referral_input_keyboard())
                return True
            ok, _ = await db.apply_referral_token(message.from_id, token)
            if not ok:
                await answer(message, "Эта ссылка не подошла.", keyboard=referral_input_keyboard())
                return True
            await finish_registration(message)
            return True

        await finish_registration(message)
        return True

    async def route_to_proplus(message: Message) -> None:
        await db.set_search_mode(message.from_id, "proplus")
        user = await db.get_user(message.from_id)
        if settings.proplus_require_age_confirmation and not (user or {}).get("age_confirmed_at"):
            await answer(message, 
                "Для этого запроса требуетсся AniKot Pro+ и подтверждения возраста 18+.",
                keyboard=age_confirmation_keyboard(),
            )
            return
        await answer(message, 
            "Для этого запроса требуется AniKot Pro+. Отправьте запрос ещё раз.",
            keyboard=result_keyboard(),
        )

    async def run_search(
        message: Message,
        mode: str,
        photo_url: str | None,
        query_text: str,
    ) -> None:
        vk_id = message.from_id
        balance_type = mode
        temp_path: Path | None = None
        try:
            if photo_url:
                temp_path = await download_image(photo_url, settings.temp_dir, http_client, settings.image_max_bytes)
                cache_key = await _vision_cache_key(temp_path, mode, settings)
                cached = await db.get_vision_cache(cache_key)
                if cached:
                    try:
                        found = AnimeResult(**cached)
                        logger.info("Vision cache hit mode=%s key=%s", mode, cache_key[:28])
                    except (TypeError, ValueError):
                        found = None
                else:
                    found = None

                if found is None:
                    result = {
                        "anikot": detector.identify_image_anikot,
                        "pro": detector.identify_image_pro,
                        "proplus": detector.identify_image_proplus,
                    }[mode]
                    found = await result(str(temp_path))
                    if found is not None:
                        await db.put_vision_cache(cache_key, asdict(found))

                kind = "image"
                logged_query = None
            else:
                result = {
                    "anikot": detector.identify_title_anikot,
                    "pro": detector.identify_title_pro,
                    "proplus": detector.identify_title_proplus,
                }[mode]
                found = await result(query_text)
                kind = "title"
                logged_query = query_text

            user = await db.get_user(vk_id) or {}

            if (
                photo_url
                and found
                and not found.is_anime
                and (
                    found.anime_likelihood is None
                    or found.anime_likelihood <= settings.non_anime_max_likelihood
                )
            ):
                await db.log_search(
                    vk_id,
                    kind,
                    mode,
                    None,
                    found.engine,
                    False,
                    {
                        "reason": "non_anime",
                        "anime_likelihood": found.anime_likelihood,
                    },
                )
                if is_admin(vk_id):
                    await answer(
                        message,
                        "🚫 На изображении не обнаружено аниме или дунхуа. 💸 Запрос не списан.",
                        keyboard=result_keyboard(),
                    )
                    return

                warning = await db.register_non_anime_warning(vk_id)
                if warning.get("blocked"):
                    await answer(
                        message,
                        f"⛔ На изображении не обнаружено аниме или дунхуа.\n"
                        f"⚠️ Предупреждение {warning['warnings']}/{warning['limit']}.\n"
                        f"🕒 Доступ к AniKot заблокирован на {settings.non_anime_block_days} дней.\n"
                        f"💸 Запрос не списан.",
                        keyboard=result_keyboard(),
                    )
                else:
                    await answer(
                        message,
                        f"🚫 На изображении не обнаружено аниме или дунхуа.\n"
                        f"⚠️ Предупреждение {warning['warnings']}/{warning['limit']}.\n"
                        f"После {warning['limit']} предупреждений доступ блокируется на "
                        f"{settings.non_anime_block_days} дней.\n"
                        f"💸 Запрос не списан.",
                        keyboard=result_keyboard(),
                    )
                return

            if found and found.minor_risk:
                await db.log_search(vk_id, kind, mode, logged_query, found.engine, False, "content_rejected")
                await answer(message, "⛔ Этот запрос не может быть обработан. 💸 Запрос не списан.", keyboard=await user_main_keyboard(vk_id))
                return

            if found and found.adult_content:
                if mode != "proplus":
                    await route_to_proplus(message)
                    return
                if settings.proplus_require_age_confirmation and not user.get("age_confirmed_at"):
                    await answer(message, 
                        "🔞 Для продолжения подтвердите возраст 18+. 💸 Запрос не списан.",
                        keyboard=age_confirmation_keyboard(),
                    )
                    return

            if not _reliable(found, _threshold(settings, mode)):
                await db.log_search(
                    vk_id,
                    kind,
                    mode,
                    logged_query,
                    found.engine if found else None,
                    False,
                    {
                        "reason": "low_confidence",
                        "title": found.title if found else None,
                        "confidence": found.confidence if found else None,
                        "alternatives": found.alternatives if found else [],
                    },
                )
                if found and found.title:
                    charged_balance = await consume_search(vk_id, balance_type)
                    if charged_balance is None:
                        await answer(
                            message,
                            f"💳 Запросы {balance_label(balance_type)} закончились. Результат не выдан.",
                            keyboard=await user_main_keyboard(vk_id),
                        )
                        return
                    uncertain_text = format_uncertain_result(found).replace(
                        "💸 Запрос не списан.",
                        "💳 Списан 1 запрос за полученный вероятный результат.",
                    )
                    await answer(message, uncertain_text, keyboard=result_keyboard())
                else:
                    next_hint = {
                        "anikot": " ✨ Попробуйте режим Pro.",
                        "pro": " 💎 Попробуйте режим Pro+.",
                        "proplus": "",
                    }[mode]
                    await answer(
                        message,
                        "😿 AniKot не смог выделить даже вероятный тайтл по этому кадру. "
                        "💸 Запрос не списан." + next_hint,
                        keyboard=result_keyboard(),
                    )
                return

            charged_balance = await consume_search(vk_id, balance_type)
            if charged_balance is None:
                await answer(
                    message,
                    f"💳 Запросы {balance_label(balance_type)} закончились. Результат не списан и не выдан.",
                    keyboard=await user_main_keyboard(vk_id),
                )
                return

            await db.log_search(
                vk_id,
                kind,
                mode,
                logged_query,
                found.engine,
                True,
                {
                    "title": found.title,
                    "confidence": found.confidence,
                    "usage": found.usage or {},
                    "alternatives": found.alternatives or [],
                },
            )
            await answer(message, format_result(found), keyboard=result_keyboard())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Search failed")
            await db.log_search(vk_id, "image" if photo_url else "title", mode, query_text or None, None, False, str(exc))
            await answer(message, "⚠️ Произошла ошибка. 💸 Запрос не списан.", keyboard=await user_main_keyboard(vk_id))
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
                if settings.low_memory_mode:
                    gc.collect()
            task = search_tasks.get(vk_id)
            if task is asyncio.current_task():
                search_tasks.pop(vk_id, None)
                search_context.pop(vk_id, None)

    @bot.on.message()
    async def all_messages(message: Message):
        if message.from_id <= 0:
            return

        await notify_mature_referrals()
        user, _created = await db.ensure_user(message.from_id)
        text = (message.text or "").strip()
        lower = text.lower()
        photo_url = extract_photo_url(message, settings.vk_image_max_side)

        ref_payload = _extract_ref_payload(message)
        ref_token = _parse_referral_token(ref_payload)
        if ref_token and not user.get("registration_completed_at") and not user.get("referrer_vk_id"):
            await db.set_referral_candidate(message.from_id, ref_token)
            user = await db.get_user(message.from_id) or user

        if is_admin(message.from_id) and lower.startswith("/unblock "):
            parts = text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                reset = len(parts) >= 3 and parts[2].lower() in {"reset", "clear", "сброс"}
                await db.unblock_account(int(parts[1]), reset_strikes=reset)
                await answer(message, "Аккаунт разблокирован.", keyboard=await user_main_keyboard(message.from_id))
            return

        if not user.get("registration_completed_at"):
            await handle_onboarding(message, user, text)
            return

        blocked, strikes, block_reason, blocked_until = await db.is_account_blocked(message.from_id)
        if blocked:
            if block_reason == "non_anime_abuse" and blocked_until:
                try:
                    until_dt = datetime.fromisoformat(blocked_until)
                    until_text = until_dt.strftime("%d.%m.%Y %H:%M UTC")
                except Exception:
                    until_text = blocked_until
                await answer(
                    message,
                    f"⛔ Доступ к AniKot временно заблокирован за повторную отправку не-аниме изображений.\n"
                    f"🕒 Блокировка действует до {until_text}.",
                    keyboard=await user_main_keyboard(message.from_id),
                )
            else:
                await answer(
                    message,
                    f"⛔ Аккаунт AniKot заблокирован.\n⚠️ Страйки: {strikes}/{settings.unsubscribe_strike_limit}.",
                    keyboard=await user_main_keyboard(message.from_id),
                )
            return

        # Search cancellation is processed before any navigation.
        if text in {"Отмена", "🛑 Отмена"}:
            task = search_tasks.get(message.from_id)
            if task and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                await send_home(message, "🛑 Поиск отменён. 💸 Запрос не списан.")
            else:
                await send_home(message)
            return

        # Home navigation.
        if lower in {"start", "/start", "начать", "меню", "назад", "◀️ назад"}:
            await send_home(message)
            return

        if text in {"Режим поиска", "⚙️ Режим поиска"}:
            current = (await db.get_user(message.from_id) or {}).get("search_mode") or "anikot"
            current_name = {"anikot": "Обычный", "pro": "Pro", "proplus": "Pro+"}.get(current, "Обычный")
            await answer(message, f"Режим поиска AniKot\n\nТекущий режим: {current_name}", keyboard=search_mode_keyboard())
            return

        if text in {"Обычный", "Pro", "Pro+", "🔎 Обычный", "✨ Pro", "💎 Pro+"}:
            mode = {
                "Обычный": "anikot", "🔎 Обычный": "anikot",
                "Pro": "pro", "✨ Pro": "pro",
                "Pro+": "proplus", "💎 Pro+": "proplus",
            }[text]
            await db.set_search_mode(message.from_id, mode)
            await send_home(message, f"✅ Режим поиска: { {'anikot': '🔎 Обычный', 'pro': '✨ Pro', 'proplus': '💎 Pro+'}[mode] }")
            return

        if text == "✅ Мне есть 18 лет":
            await db.confirm_adult_age(message.from_id)
            await db.set_search_mode(message.from_id, "proplus")
            await answer(message, "✅ Возраст подтверждён. Отправьте запрос ещё раз.", keyboard=result_keyboard())
            return

        if text in {"Профиль", "👤 Профиль"}:
            user = await db.get_user(message.from_id) or user
            unlimited = is_admin(message.from_id)
            normal_balance = "∞" if unlimited else user["requests_balance"]
            pro_balance = "∞" if unlimited else user["pro_balance"]
            proplus_balance = "∞" if unlimited else user["proplus_balance"]
            await answer(message, 
                f"👤 Профиль AniKot\n\n"
                f"🔎 AniKot: {normal_balance}\n"
                f"✨ AniKot Pro: {pro_balance}\n"
                f"💎 AniKot Pro+: {proplus_balance}\n"
                f"📊 Поисков: {user['total_searches']}\n"
                f"⚠️ Страйки: {user['unsubscribe_strikes']}/{settings.unsubscribe_strike_limit}\n"
                f"🚫 Предупреждения за не-аниме: {int(user.get('non_anime_warnings') or 0)}/{settings.non_anime_warning_limit}",
                keyboard=profile_keyboard(),
            )
            return

        if text in {"Рефералы", "👥 Рефералы"}:
            stats = await db.referral_stats(message.from_id)
            link = _referral_link(settings, stats["token"])
            earned = stats["credited"] * settings.referral_reward_pro
            await answer(message, 
                "👥 Реферальная система\n\n"
                f"За друга: +{settings.referral_reward_pro} AniKot Pro\n"
                f"Проверка нового пользователя: {settings.referral_freeze_days} дня\n"
                f"На проверке: {stats['pending']}\n"
                f"Зачислено друзей: {stats['credited']}\n"
                f"Получено AniKot Pro: {earned}\n\n"
                f"Ваша ссылка:\n{link}",
                keyboard=referral_keyboard(),
            )
            return

        if text in {"Бонус за подписку", "🎁 Бонус за подписку"}:
            user = await db.get_user(message.from_id) or user
            if user.get("subscription_bonus_claimed"):
                await send_home(message)
                return
            group_line = settings.vk_group_url or f"https://vk.com/club{settings.vk_group_id}"
            await answer(message, 
                f"🎁 За подписку: +{settings.subscription_bonus_requests} AniKot.\n\n"
                f"После получения бонуса отписка даёт 1 страйк. "
                f"{settings.unsubscribe_strike_limit} страйка — блокировка аккаунта.",
                keyboard=subscription_keyboard(group_line),
            )
            return

        if text in {"Проверить подписку", "✅ Проверить подписку"} or lower == "проверить подписку":
            try:
                member = await is_member(message.from_id)
            except Exception:
                logger.exception("Subscription check failed")
                await answer(message, "Произошла ошибка. Попробуйте позже.", keyboard=await user_main_keyboard(message.from_id))
                return
            if not member:
                await answer(message, 
                    "Подписка пока не найдена.",
                    keyboard=subscription_keyboard(settings.vk_group_url or f"https://vk.com/club{settings.vk_group_id}"),
                )
                return
            claimed, _balance = await db.claim_subscription_bonus(message.from_id)
            await send_home(
                message,
                f"✅ Начислено: +{settings.subscription_bonus_requests} AniKot." if claimed else "Бонус уже был получен.",
            )
            return

        # Shop root and categories.
        if text in {"Купить запросы", "🛒 Купить запросы"}:
            shop_views[message.from_id] = "root"
            await answer(message, _shop_root_text(), keyboard=shop_categories_keyboard())
            return

        if text in {"🔎 AniKot", "✨ Pro", "💎 Pro+"}:
            balance_type = {"🔎 AniKot": "anikot", "✨ Pro": "pro", "💎 Pro+": "proplus"}[text]
            shop_views[message.from_id] = balance_type
            await answer(message, _shop_category_text(balance_type), keyboard=shop_packages_keyboard(balance_type))
            return

        if text in {"◀ Назад", "◀️ Назад"}:
            state = shop_views.get(message.from_id)
            if state and state != "root":
                shop_views[message.from_id] = "root"
                await answer(message, _shop_root_text(), keyboard=shop_categories_keyboard())
            else:
                shop_views.pop(message.from_id, None)
                await send_home(message)
            return

        if text in {"◀ В магазин", "🛒 В магазин"}:
            shop_views[message.from_id] = "root"
            await answer(message, _shop_root_text(), keyboard=shop_categories_keyboard())
            return

        for key, package in PACKAGES.items():
            legacy_package_label = package.label.replace(" запросов", "")
            if text in {_package_button_label(package), legacy_package_label}:
                if not settings.public_base_url:
                    await answer(message, "Оплата временно недоступна. Попробуйте позже.", keyboard=await user_main_keyboard(message.from_id))
                    return
                token = await db.create_checkout_session(message.from_id, key)
                checkout_url = f"{settings.public_base_url}/checkout/{token}"
                await answer(message, 
                    f"Выбран пакет:\n{balance_label(package.balance_type)} — {package.label}\n\nНажмите «Оплатить» для продолжения.",
                    keyboard=package_confirm_keyboard(checkout_url),
                )
                return

        # Admin helpers. Admin search balances are always unlimited and are never decremented.
        if is_admin(message.from_id) and lower.startswith("/add "):
            parts = text.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                await db.add_requests(int(parts[1]), int(parts[2]), "admin", str(uuid.uuid4()), "anikot")
                await answer(message, "Начислено.", keyboard=await user_main_keyboard(message.from_id))
            elif len(parts) == 4 and parts[1].isdigit() and parts[2].lower() in {"pro", "proplus", "anikot"} and parts[3].isdigit():
                await db.add_requests(int(parts[1]), int(parts[3]), "admin", str(uuid.uuid4()), parts[2].lower())
                await answer(message, "Начислено.", keyboard=await user_main_keyboard(message.from_id))
            return

        if is_admin(message.from_id) and lower == "/stats":
            stats = await db.stats()
            await answer(message, 
                f"👥 Пользователей: {stats['users']}\n"
                f"⛔ Заблокировано: {stats['blocked']}\n"
                f"🔎 Поисков: {stats['searches']}\n"
                f"✨ Pro: {stats['pro_searches']}\n"
                f"💎 Pro+: {stats['proplus_searches']}\n"
                f"💳 Оплат: {stats['paid_orders']}\n"
                f"💰 {stats['revenue']:.2f} {settings.lava_currency}",
                keyboard=await user_main_keyboard(message.from_id),
            )
            return

        user = await db.get_user(message.from_id) or user
        mode = user.get("search_mode") or "anikot"

        # Adult-title keywords are routed to Pro+ before any credit is spent.
        if text and explicit_text(text) and mode != "proplus":
            await route_to_proplus(message)
            return

        if not photo_url and not text:
            await send_home(message, "Отправьте фото, кадр или название аниме.")
            return

        existing = search_tasks.get(message.from_id)
        if existing and not existing.done():
            await answer(message, "Поиск уже выполняется.", keyboard=search_cancel_keyboard())
            return

        if len(search_tasks) >= settings.max_pending_searches:
            await answer(
                message,
                "⏳ Сейчас много запросов. Попробуйте ещё раз через несколько секунд — запрос не списан.",
                keyboard=await user_main_keyboard(message.from_id),
            )
            return

        # For explicit text in Pro+ require age before spending a search credit.
        if mode == "proplus" and text and explicit_text(text) and settings.proplus_require_age_confirmation and not user.get("age_confirmed_at"):
            await answer(message, "Для этого запроса подтвердите возраст 18+.", keyboard=age_confirmation_keyboard())
            return

        if not is_admin(message.from_id):
            user = await db.get_user(message.from_id) or user
            balance = {
                "anikot": int(user.get("requests_balance") or 0),
                "pro": int(user.get("pro_balance") or 0),
                "proplus": int(user.get("proplus_balance") or 0),
            }.get(mode, 0)
            if balance <= 0:
                await answer(
                    message,
                    f"💳 Запросы {balance_label(mode)} закончились.",
                    keyboard=await user_main_keyboard(message.from_id),
                )
                return

        await answer(message, "🔍 AniKot анализирует кадр..." if photo_url else "🔍 AniKot выполняет поиск...", keyboard=search_cancel_keyboard())
        task = asyncio.create_task(
            run_search(message, mode, photo_url, text),
            name=f"anikot-search-{message.from_id}",
        )
        search_tasks[message.from_id] = task
        search_context[message.from_id] = mode
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    @bot.on.raw_event(GroupEventType.WALL_REPLY_NEW, dataclass=dict)
    async def wall_reply_new(event: dict):
        try:
            obj = event.get("object") or {}
            if isinstance(obj, dict) and "comment" in obj:
                obj = obj["comment"]
            vk_id = int(obj.get("from_id") or 0)
            if vk_id <= 0:
                return
            await db.ensure_user(vk_id)
            event_key = f"wall_reply:{obj.get('post_id') or 0}:{obj.get('id') or obj.get('comment_id') or uuid.uuid4()}"
            result = await db.record_activity(vk_id, event_key, "wall_reply", str(obj.get("text") or ""))
            if result and result[0] > 0:
                try:
                    await bot.api.messages.send(
                        peer_id=vk_id,
                        random_id=0,
                        message=f"🎁 За активность: +{result[0]} AniKot.",
                        keyboard=await user_main_keyboard(vk_id),
                    )
                except Exception:
                    pass
        except Exception:
            logger.exception("Activity event failed")

    @bot.on.raw_event(GroupEventType.GROUP_LEAVE, dataclass=dict)
    async def group_leave(event: dict):
        try:
            obj = event.get("object") or {}
            vk_id = int(obj.get("user_id") or event.get("user_id") or 0)
            if vk_id <= 0:
                return
            await db.ensure_user(vk_id)
            event_ref = str(event.get("event_id") or obj.get("event_id") or f"group_leave:{vk_id}:{uuid.uuid4()}")
            info = await db.register_unsubscribe_strike(vk_id, event_ref=event_ref, details="GROUP_LEAVE after subscription bonus")
            if not info.get("applied"):
                return
            if info["blocked"]:
                message_text = f"⛔ Страйк {info['strikes']}/{settings.unsubscribe_strike_limit}. Аккаунт AniKot заблокирован."
            else:
                message_text = f"⚠️ Страйк {info['strikes']}/{settings.unsubscribe_strike_limit} за отписку после получения бонуса."
            try:
                await bot.api.messages.send(
                    peer_id=vk_id,
                    random_id=0,
                    message=message_text,
                    keyboard=await user_main_keyboard(vk_id),
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Unsubscribe strike event failed")

    return bot
