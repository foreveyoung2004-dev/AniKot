from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from vkbottle import GroupEventType
from vkbottle.bot import Bot, Message

from .config import PACKAGES, Settings
from .db import Database
from .keyboards import (
    age_confirmation_keyboard,
    buy_keyboard,
    legal_keyboard,
    main_keyboard,
    package_confirm_keyboard,
    referral_choice_keyboard,
    referral_input_keyboard,
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


def format_result(result: AnimeResult, balance: int | str, balance_type: str) -> str:
    media = result.media or {}
    titles = media.get("title") or {}
    lines = [f"🎬 {result.title}"]
    if titles.get("english") and titles.get("english") != result.title:
        lines.append(f"🇬🇧 {titles['english']}")
    if titles.get("native"):
        lines.append(f"🇯🇵 {titles['native']}")
    meta: list[str] = []
    if media.get("seasonYear"):
        meta.append(str(media["seasonYear"]))
    if media.get("format"):
        meta.append(str(media["format"]))
    if media.get("episodes"):
        meta.append(f"{media['episodes']} эп.")
    if media.get("averageScore"):
        meta.append(f"AniList {media['averageScore']}/100")
    if meta:
        lines.append(" • ".join(meta))
    if result.confidence is not None:
        lines.append(f"Уверенность: ~{max(0, min(1, result.confidence)) * 100:.0f}%")
    if result.note:
        lines.append(f"ℹ {result.note[:350]}")
    if media.get("siteUrl"):
        lines.append(f"AniList: {media['siteUrl']}")
    lines.append(f"\nОсталось {balance_label(balance_type)}: {balance}")
    return "\n".join(lines)


def _reliable(result: AnimeResult | None, minimum: float) -> bool:
    if result is None or result.confidence is None:
        return False
    return float(result.confidence) >= minimum


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _photo_url_from_attachments(attachments: Any) -> str | None:
    for attachment in attachments or []:
        atype = _field(attachment, "type")
        photo = _field(attachment, "photo")
        if photo is None and atype == "photo":
            photo = _field(attachment, "object")
        if photo is None:
            continue
        best, best_area = None, -1
        for size in _field(photo, "sizes", []) or []:
            url = _field(size, "url")
            if not url:
                continue
            width = int(_field(size, "width", 0) or 0)
            height = int(_field(size, "height", 0) or 0)
            area = width * height
            if area > best_area:
                best, best_area = str(url), area
        if best:
            return best
    return None


def extract_photo_url(message: Message) -> str | None:
    direct = _photo_url_from_attachments(getattr(message, "attachments", None))
    if direct:
        return direct
    reply = getattr(message, "reply_message", None)
    if reply:
        found = _photo_url_from_attachments(_field(reply, "attachments", []))
        if found:
            return found
    for forwarded in getattr(message, "fwd_messages", None) or []:
        found = _photo_url_from_attachments(_field(forwarded, "attachments", []))
        if found:
            return found
    return None


async def download_image(url: str, target_dir: str) -> Path:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    path = Path(target_dir) / f"{uuid.uuid4()}.jpg"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        if len(response.content) > 15 * 1024 * 1024:
            raise ValueError("image_too_large")
        path.write_bytes(response.content)
    return path


def _extract_ref_payload(message: Message) -> str | None:
    ref = _field(message, "ref")
    if ref:
        return str(ref)
    return None


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


def _shop_text() -> str:
    normal = [p for p in PACKAGES.values() if p.balance_type == "anikot"]
    pro = [p for p in PACKAGES.values() if p.balance_type == "pro"]
    proplus = [p for p in PACKAGES.values() if p.balance_type == "proplus"]
    lines = (
        ["💳 Магазин AniKot", "", "🔎 AniKot:"]
        + [f"• {p.label}" for p in normal]
        + ["", "✨ AniKot Pro:"]
        + [f"• {p.label}" for p in pro]
        + ["", "💎 AniKot Pro+:"]
        + [f"• {p.label}" for p in proplus]
    )
    return "\n".join(lines)


def build_bot(settings: Settings, db: Database, detector: AnimeDetector, lava: LavaClient) -> Bot:
    bot = Bot(settings.vk_token)

    async def is_member(vk_id: int) -> bool:
        result = await bot.api.groups.is_member(group_id=settings.vk_group_id, user_id=vk_id)
        return bool(result) if isinstance(result, int) else bool(getattr(result, "member", result))

    def is_admin(vk_id: int) -> bool:
        return vk_id in settings.admin_ids

    async def consume_search(vk_id: int, balance_type: str) -> int | str | None:
        if is_admin(vk_id):
            return "∞"
        return await db.consume_request(vk_id, balance_type)

    async def refund_search(vk_id: int, balance_type: str, ref: str) -> int | str:
        if is_admin(vk_id):
            return "∞"
        return await db.refund_search_request(vk_id, balance_type, ref)

    async def notify_mature_referrals() -> None:
        try:
            rewards = await db.release_mature_referrals()
            for reward in rewards:
                try:
                    await bot.api.messages.send(
                        peer_id=reward["referrer_vk_id"],
                        random_id=0,
                        message=(
                            f"🎁 Реферальная награда начислена: +{reward['amount']} AniKot Pro.\n"
                            f"Баланс AniKot Pro: {reward['balance']}."
                        ),
                        keyboard=main_keyboard(),
                    )
                except Exception:
                    logger.exception("Failed to send referral reward notification")
        except Exception:
            logger.exception("Referral maturity processing failed")

    async def send_legal(message: Message) -> None:
        agreement = _vk_text_link("Пользовательское соглашение", settings.user_agreement_url)
        privacy = _vk_text_link("Политику конфиденциальности", settings.privacy_policy_url)
        await message.answer(
            f"Продолжая регистрацию, вы принимаете {agreement} и {privacy}.",
            keyboard=legal_keyboard(),
        )

    async def send_referral_question(message: Message, user: dict[str, Any]) -> None:
        has_candidate = bool(user.get("referral_candidate_token"))
        if has_candidate:
            text = (
                "🔗 Реферальная ссылка обнаружена. Использовать её для регистрации?\n\n"
                f"Награда пригласившему будет начислена через {settings.referral_freeze_days} дня после регистрации."
            )
        else:
            text = "🔗 У тебя есть реферальная ссылка от друга?"
        await message.answer(text, keyboard=referral_choice_keyboard(has_candidate))

    async def finish_registration(message: Message) -> None:
        _created, user = await db.complete_registration(message.from_id)
        referral_note = ""
        if user.get("referrer_vk_id"):
            referral_note = (
                f"\n🔗 Реферальная связь закреплена на {settings.referral_freeze_days} дня. "
                "После этого пригласивший получит награду."
            )
        await message.answer(
            "✅ Регистрация завершена.\n\n"
            f"🎁 За регистрацию: +{settings.initial_pro_requests} AniKot Pro.\n"
            f"🎁 За подписку на сообщество: +{settings.subscription_bonus_pro} AniKot Pro.\n"
            f"👥 За приглашённого друга: +{settings.referral_reward_pro} AniKot Pro после {settings.referral_freeze_days} дней."
            + referral_note,
            keyboard=main_keyboard(),
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
                ok, _reason = await db.apply_referral_token(message.from_id, str(candidate))
                if ok:
                    await finish_registration(message)
                else:
                    await db.set_referral_candidate(message.from_id, None)
                    await message.answer(
                        "Не удалось применить эту ссылку. Можно продолжить без неё или отправить другую.",
                        keyboard=referral_choice_keyboard(False),
                    )
                return True
            if text == "🔗 Да, есть ссылка":
                await db.set_onboarding_state(message.from_id, "referral_link")
                await message.answer(
                    "Отправь реферальную ссылку одним сообщением.",
                    keyboard=referral_input_keyboard(),
                )
                return True
            if text in {"➡️ Нет, продолжить", "➡️ Продолжить без ссылки"}:
                await finish_registration(message)
                return True
            await send_referral_question(message, user)
            return True

        if state == "referral_link":
            if text == "➡️ Продолжить без ссылки":
                await finish_registration(message)
                return True
            token = _parse_referral_token(text)
            if not token:
                await message.answer(
                    "Не удалось распознать ссылку. Отправь её ещё раз или продолжи без неё.",
                    keyboard=referral_input_keyboard(),
                )
                return True
            ok, _reason = await db.apply_referral_token(message.from_id, token)
            if not ok:
                await message.answer(
                    "Эта ссылка не подошла. Отправь другую или продолжи без неё.",
                    keyboard=referral_input_keyboard(),
                )
                return True
            await finish_registration(message)
            return True

        await finish_registration(message)
        return True

    async def route_to_proplus(message: Message) -> None:
        await db.set_search_mode(message.from_id, "proplus")
        user = await db.get_user(message.from_id)
        if settings.proplus_require_age_confirmation and not (user or {}).get("age_confirmed_at"):
            await message.answer(
                "Этот запрос относится к контенту 18+. Для продолжения нужно подтвердить возраст.\n"
                "До подтверждения запросы не списываются.",
                keyboard=age_confirmation_keyboard(),
            )
            return
        await message.answer(
            "Для этого запроса нужен AniKot Pro+. Отправь запрос ещё раз — предыдущий запрос не списан.",
            keyboard=main_keyboard(),
        )

    @bot.on.message()
    async def all_messages(message: Message):
        if message.from_id <= 0:
            return

        await notify_mature_referrals()

        user, created = await db.ensure_user(message.from_id)
        text = (message.text or "").strip()
        lower = text.lower()
        photo_url = extract_photo_url(message)

        # Capture deep-link referral payload before legal acceptance.
        ref_payload = _extract_ref_payload(message)
        ref_token = _parse_referral_token(ref_payload)
        if ref_token and not user.get("registration_completed_at") and not user.get("referrer_vk_id"):
            await db.set_referral_candidate(message.from_id, ref_token)
            user = await db.get_user(message.from_id) or user

        # Admin recovery command stays available for blocked accounts.
        if is_admin(message.from_id) and lower.startswith("/unblock "):
            parts = text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                reset = len(parts) >= 3 and parts[2].lower() in {"reset", "clear", "сброс"}
                await db.unblock_account(int(parts[1]), reset_strikes=reset)
                await message.answer(
                    f"VK {parts[1]} разблокирован." + (" Страйки сброшены." if reset else ""),
                    keyboard=main_keyboard(),
                )
            return

        # Onboarding always happens before access to the bot.
        if not user.get("registration_completed_at"):
            await handle_onboarding(message, user, text)
            return

        blocked, strikes, reason = await db.is_account_blocked(message.from_id)
        if blocked:
            await message.answer(
                "⛔ Аккаунт AniKot заблокирован.\n"
                f"Страйки за отписку: {strikes}/{settings.unsubscribe_strike_limit}.\n"
                f"Причина: {reason or 'достигнут лимит страйков'}.",
                keyboard=main_keyboard(),
            )
            return

        user = await db.get_user(message.from_id) or user
        pending = str(user.get("pending_package") or "")

        # Payment navigation. Email is never requested in VK messages.
        if text in {"⬅ В магазин", "⬅ Отменить покупку"}:
            await db.set_pending_package(message.from_id, None)
            if text == "⬅ В магазин":
                await message.answer(_shop_text(), keyboard=buy_keyboard())
            else:
                await message.answer("Покупка отменена.", keyboard=main_keyboard())
            return

        if lower in {"start", "/start", "начать", "меню", "⬅ назад", "назад"}:
            await db.set_search_mode(message.from_id, "anikot")
            await message.answer("🐾 AniKot готов. Выбери режим поиска.", keyboard=main_keyboard())
            return

        if text == "🔎 AniKot":
            await db.set_search_mode(message.from_id, "anikot")
            await message.answer("🔎 AniKot включён. Пришли кадр, фото или название.", keyboard=main_keyboard())
            return

        if text == "✨ AniKot Pro":
            await db.set_search_mode(message.from_id, "pro")
            await message.answer("✨ AniKot Pro включён на следующий поиск. Пришли кадр, фото или название.", keyboard=main_keyboard())
            return

        if text == "💎 AniKot Pro+":
            await db.set_search_mode(message.from_id, "proplus")
            await message.answer("💎 AniKot Pro+ включён на следующий поиск. Пришли кадр, фото или название.", keyboard=main_keyboard())
            return

        if text == "✅ Мне есть 18 лет":
            await db.confirm_adult_age(message.from_id)
            await db.set_search_mode(message.from_id, "proplus")
            await message.answer("✅ Возраст подтверждён. Отправь запрос ещё раз.", keyboard=main_keyboard())
            return

        if text == "👤 Профиль":
            user = await db.get_user(message.from_id) or user
            unlimited = is_admin(message.from_id)
            normal_balance = "∞" if unlimited else user["requests_balance"]
            pro_balance = "∞" if unlimited else user["pro_balance"]
            proplus_balance = "∞" if unlimited else user["proplus_balance"]
            await message.answer(
                f"👤 AniKot ID: {message.from_id}\n"
                f"🔎 AniKot: {normal_balance}\n"
                f"✨ AniKot Pro: {pro_balance}\n"
                f"💎 AniKot Pro+: {proplus_balance}\n"
                f"📊 Всего поисков: {user['total_searches']}\n"
                f"⚠️ Страйки за отписку: {user['unsubscribe_strikes']}/{settings.unsubscribe_strike_limit}\n"
                f"🎁 Бонус за подписку: {'получен' if user['subscription_bonus_claimed'] else 'доступен'}",
                keyboard=main_keyboard(),
            )
            return

        if text == "👥 Рефералы":
            stats = await db.referral_stats(message.from_id)
            link = _referral_link(settings, stats["token"])
            earned = stats["credited"] * settings.referral_reward_pro
            await message.answer(
                "👥 Реферальная система AniKot\n\n"
                f"За каждого нового друга: +{settings.referral_reward_pro} AniKot Pro.\n"
                f"Награда замораживается на {settings.referral_freeze_days} дня после его регистрации.\n\n"
                f"⏳ На удержании: {stats['pending']}\n"
                f"✅ Зачислено друзей: {stats['credited']}\n"
                f"✨ Получено AniKot Pro: {earned}\n\n"
                f"Твоя ссылка:\n{link}",
                keyboard=main_keyboard(),
            )
            return

        if text == "🎁 Бонус за подписку":
            group_line = settings.vk_group_url or f"https://vk.com/club{settings.vk_group_id}"
            await message.answer(
                f"🎁 За подписку: +{settings.subscription_bonus_pro} AniKot Pro один раз.\n\n"
                f"⚠️ После получения бонуса каждая отписка = 1 страйк. "
                f"{settings.unsubscribe_strike_limit} страйка = блокировка аккаунта AniKot.",
                keyboard=subscription_keyboard(group_line),
            )
            return

        if text == "✅ Проверить подписку" or lower in {"проверить подписку", "проверить"}:
            try:
                member = await is_member(message.from_id)
            except Exception:
                logger.exception("Subscription check failed")
                await message.answer("Произошла ошибка. Попробуй позже.", keyboard=main_keyboard())
                return
            if not member:
                await message.answer("Подписка пока не найдена.", keyboard=subscription_keyboard(settings.vk_group_url or f"https://vk.com/club{settings.vk_group_id}"))
                return
            claimed, balance = await db.claim_subscription_bonus(message.from_id)
            await message.answer(
                f"✅ +{settings.subscription_bonus_pro} AniKot Pro. Баланс: {balance}."
                if claimed
                else f"Бонус уже был получен. Баланс AniKot Pro: {balance}.",
                keyboard=main_keyboard(),
            )
            return

        if text == "💳 Купить запросы":
            await db.set_pending_package(message.from_id, None)
            await message.answer(_shop_text(), keyboard=buy_keyboard())
            return

        for key, package in PACKAGES.items():
            if text.startswith((f"🔎 {key} ·", f"✨ {key} ·", f"💎 {key} ·")):
                if not settings.public_base_url:
                    await message.answer("Оплата временно недоступна. Попробуй позже.", keyboard=main_keyboard())
                    return
                token = await db.create_checkout_session(message.from_id, key)
                checkout_url = f"{settings.public_base_url}/checkout/{token}"
                await message.answer(
                    f"Выбран пакет:\n{package.label}\n\nНажми «Оплатить», чтобы перейти к оформлению покупки.",
                    keyboard=package_confirm_keyboard(checkout_url),
                )
                return

        # Admin commands are intentionally short and do not expose internal service details.
        if is_admin(message.from_id) and lower.startswith("/add "):
            parts = text.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                balance = await db.add_requests(int(parts[1]), int(parts[2]), "admin", str(uuid.uuid4()), "anikot")
                await message.answer(f"AniKot-баланс VK {parts[1]}: {balance}", keyboard=main_keyboard())
            elif len(parts) == 4 and parts[1].isdigit() and parts[2].lower() in {"pro", "proplus", "anikot"} and parts[3].isdigit():
                btype = parts[2].lower()
                balance = await db.add_requests(int(parts[1]), int(parts[3]), "admin", str(uuid.uuid4()), btype)
                await message.answer(f"Баланс {balance_label(btype)} VK {parts[1]}: {balance}", keyboard=main_keyboard())
            return

        if is_admin(message.from_id) and lower == "/stats":
            s = await db.stats()
            await message.answer(
                f"👥 Пользователей: {s['users']}\n"
                f"⛔ Заблокировано: {s['blocked']}\n"
                f"🔎 Всего поисков: {s['searches']}\n"
                f"✨ Pro: {s['pro_searches']}\n"
                f"💎 Pro+: {s['proplus_searches']}\n"
                f"💳 Оплат: {s['paid_orders']}\n"
                f"💰 {s['revenue']:.2f} {settings.lava_currency}",
                keyboard=main_keyboard(),
            )
            return

        user = await db.get_user(message.from_id) or user
        mode = user.get("search_mode") or "anikot"

        if text and explicit_text(text) and mode != "proplus":
            await route_to_proplus(message)
            return

        if photo_url:
            mode_name = {"anikot": "AniKot", "pro": "AniKot Pro", "proplus": "AniKot Pro+"}.get(mode, "AniKot")
            await message.answer(f"🔍 {mode_name} анализирует кадр…", keyboard=main_keyboard())
            temp_path: Path | None = None
            try:
                temp_path = await download_image(photo_url, settings.temp_dir)

                if mode == "anikot":
                    balance = await consume_search(message.from_id, "anikot")
                    if balance is None:
                        await message.answer("Запросы AniKot закончились.", keyboard=main_keyboard())
                        return
                    refund_ref = str(uuid.uuid4())
                    try:
                        checked = await detector.identify_image_anikot(str(temp_path))
                        if checked.adult_content:
                            returned = await refund_search(message.from_id, "anikot", refund_ref)
                            if checked.minor_risk:
                                await message.answer("Этот запрос не может быть обработан. Запрос AniKot возвращён.", keyboard=main_keyboard())
                            else:
                                await route_to_proplus(message)
                            return
                        if not _reliable(checked.result, settings.min_search_confidence):
                            await refund_search(message.from_id, "anikot", refund_ref)
                            await message.answer("AniKot не уверен в результате. Запрос возвращён — попробуй AniKot Pro.", keyboard=main_keyboard())
                            return
                        await db.log_search(message.from_id, "image", "anikot", None, checked.result.engine, True, {"title": checked.result.title})
                        await message.answer(format_result(checked.result, balance, "anikot"), keyboard=main_keyboard())
                    except Exception as exc:
                        logger.exception("AniKot image search failed")
                        await refund_search(message.from_id, "anikot", refund_ref)
                        await db.log_search(message.from_id, "image", "anikot", None, None, False, str(exc))
                        await message.answer("Произошла ошибка. Запрос возвращён.", keyboard=main_keyboard())
                    return

                if mode == "pro":
                    balance = await consume_search(message.from_id, "pro")
                    if balance is None:
                        await db.set_search_mode(message.from_id, "anikot")
                        await message.answer("Запросы AniKot Pro закончились.", keyboard=main_keyboard())
                        return
                    refund_ref = str(uuid.uuid4())
                    try:
                        checked = await detector.identify_image_pro(str(temp_path))
                        if checked.adult_content:
                            await refund_search(message.from_id, "pro", refund_ref)
                            if checked.minor_risk:
                                await db.set_search_mode(message.from_id, "anikot")
                                await message.answer("Этот запрос не может быть обработан. Запрос AniKot Pro возвращён.", keyboard=main_keyboard())
                            else:
                                await route_to_proplus(message)
                            return
                        if not _reliable(checked.result, settings.min_search_confidence):
                            await refund_search(message.from_id, "pro", refund_ref)
                            await db.set_search_mode(message.from_id, "anikot")
                            await message.answer("AniKot Pro не уверен в результате. Запрос возвращён — попробуй AniKot Pro+.", keyboard=main_keyboard())
                            return
                        await db.log_search(message.from_id, "image", "pro", None, checked.result.engine, True, {"title": checked.result.title})
                        await message.answer(format_result(checked.result, balance, "pro"), keyboard=main_keyboard())
                    except Exception as exc:
                        logger.exception("AniKot Pro image search failed")
                        await refund_search(message.from_id, "pro", refund_ref)
                        await db.log_search(message.from_id, "image", "pro", None, None, False, str(exc))
                        await message.answer("Произошла ошибка. Запрос возвращён.", keyboard=main_keyboard())
                    finally:
                        await db.set_search_mode(message.from_id, "anikot")
                    return

                # AniKot Pro+ is the highest-accuracy tier and works with any anime.
                balance = await consume_search(message.from_id, "proplus")
                if balance is None:
                    await db.set_search_mode(message.from_id, "anikot")
                    await message.answer("Запросы AniKot Pro+ закончились.", keyboard=main_keyboard())
                    return
                refund_ref = str(uuid.uuid4())
                try:
                    proplus = await detector.identify_image_proplus(str(temp_path))
                    if proplus.minor_risk:
                        await refund_search(message.from_id, "proplus", refund_ref)
                        await message.answer("Этот запрос не может быть обработан. Запрос AniKot Pro+ возвращён.", keyboard=main_keyboard())
                        return
                    if proplus.adult_content and settings.proplus_require_age_confirmation and not user.get("age_confirmed_at"):
                        await refund_search(message.from_id, "proplus", refund_ref)
                        await message.answer(
                            "Для этого запроса нужно подтвердить возраст 18+. Запрос AniKot Pro+ возвращён.",
                            keyboard=age_confirmation_keyboard(),
                        )
                        return
                    if not _reliable(proplus.result, settings.min_search_confidence):
                        await refund_search(message.from_id, "proplus", refund_ref)
                        await message.answer("AniKot Pro+ не смог определить тайтл с достаточной уверенностью. Запрос возвращён.", keyboard=main_keyboard())
                        return
                    await db.log_search(message.from_id, "image", "proplus", None, proplus.result.engine, True, {"title": proplus.result.title})
                    await message.answer(format_result(proplus.result, balance, "proplus"), keyboard=main_keyboard())
                except Exception as exc:
                    logger.exception("AniKot Pro+ image search failed")
                    await refund_search(message.from_id, "proplus", refund_ref)
                    await db.log_search(message.from_id, "image", "proplus", None, None, False, str(exc))
                    await message.answer("Произошла ошибка. Запрос возвращён.", keyboard=main_keyboard())
                finally:
                    await db.set_search_mode(message.from_id, "anikot")
                return
            finally:
                if temp_path:
                    temp_path.unlink(missing_ok=True)

        if text:
            if mode == "anikot":
                try:
                    result = await detector.identify_title_local(text)
                    if not result:
                        await message.answer("AniKot не нашёл точного совпадения. Запрос не списан — попробуй AniKot Pro.", keyboard=main_keyboard())
                        return
                    if (result.media or {}).get("isAdult"):
                        await route_to_proplus(message)
                        return
                    balance = await consume_search(message.from_id, "anikot")
                    if balance is None:
                        await message.answer("Запросы AniKot закончились.", keyboard=main_keyboard())
                        return
                    await db.log_search(message.from_id, "title", "anikot", text, result.engine, True, {"title": result.title})
                    await message.answer(format_result(result, balance, "anikot"), keyboard=main_keyboard())
                except Exception:
                    logger.exception("AniKot title search failed")
                    await message.answer("Произошла ошибка. Попробуй позже.", keyboard=main_keyboard())
                return

            if mode == "pro":
                balance = await consume_search(message.from_id, "pro")
                if balance is None:
                    await db.set_search_mode(message.from_id, "anikot")
                    await message.answer("Запросы AniKot Pro закончились.", keyboard=main_keyboard())
                    return
                refund_ref = str(uuid.uuid4())
                try:
                    result = await detector.identify_title_pro(text)
                    if (result.media or {}).get("isAdult"):
                        await refund_search(message.from_id, "pro", refund_ref)
                        await route_to_proplus(message)
                        return
                    if not _reliable(result, settings.min_search_confidence):
                        await refund_search(message.from_id, "pro", refund_ref)
                        await message.answer("AniKot Pro не уверен в результате. Запрос возвращён — попробуй AniKot Pro+.", keyboard=main_keyboard())
                        return
                    await db.log_search(message.from_id, "title", "pro", text, result.engine, True, {"title": result.title})
                    await message.answer(format_result(result, balance, "pro"), keyboard=main_keyboard())
                except Exception as exc:
                    logger.exception("AniKot Pro title search failed")
                    await refund_search(message.from_id, "pro", refund_ref)
                    await db.log_search(message.from_id, "title", "pro", text, None, False, str(exc))
                    await message.answer("Произошла ошибка. Запрос возвращён.", keyboard=main_keyboard())
                finally:
                    await db.set_search_mode(message.from_id, "anikot")
                return

            balance = await consume_search(message.from_id, "proplus")
            if balance is None:
                await db.set_search_mode(message.from_id, "anikot")
                await message.answer("Запросы AniKot Pro+ закончились.", keyboard=main_keyboard())
                return
            refund_ref = str(uuid.uuid4())
            try:
                proplus = await detector.identify_title_proplus(text)
                if proplus.minor_risk:
                    await refund_search(message.from_id, "proplus", refund_ref)
                    await message.answer("Этот запрос не может быть обработан. Запрос AniKot Pro+ возвращён.", keyboard=main_keyboard())
                    return
                if proplus.adult_content and settings.proplus_require_age_confirmation and not user.get("age_confirmed_at"):
                    await refund_search(message.from_id, "proplus", refund_ref)
                    await message.answer(
                        "Для этого запроса нужно подтвердить возраст 18+. Запрос AniKot Pro+ возвращён.",
                        keyboard=age_confirmation_keyboard(),
                    )
                    return
                if not _reliable(proplus.result, settings.min_search_confidence):
                    await refund_search(message.from_id, "proplus", refund_ref)
                    await message.answer("AniKot Pro+ не смог определить тайтл с достаточной уверенностью. Запрос возвращён.", keyboard=main_keyboard())
                    return
                await db.log_search(message.from_id, "title", "proplus", text, proplus.result.engine, True, {"title": proplus.result.title})
                await message.answer(format_result(proplus.result, balance, "proplus"), keyboard=main_keyboard())
            except Exception as exc:
                logger.exception("AniKot Pro+ title search failed")
                await refund_search(message.from_id, "proplus", refund_ref)
                await db.log_search(message.from_id, "title", "proplus", text, None, False, str(exc))
                await message.answer("Произошла ошибка. Запрос возвращён.", keyboard=main_keyboard())
            finally:
                await db.set_search_mode(message.from_id, "anikot")
            return

        await message.answer("Пришли фото, кадр или название аниме.", keyboard=main_keyboard())

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
                        message=f"🎁 За активность +{result[0]} AniKot. Баланс: {result[1]}.",
                        keyboard=main_keyboard(),
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
                message_text = (
                    f"⛔ Страйк {info['strikes']}/{settings.unsubscribe_strike_limit} за отписку.\n"
                    "Лимит достигнут — аккаунт AniKot заблокирован."
                )
            else:
                message_text = (
                    f"⚠️ Страйк {info['strikes']}/{settings.unsubscribe_strike_limit} за отписку после получения бонуса.\n"
                    f"После {settings.unsubscribe_strike_limit} страйков аккаунт AniKot будет заблокирован."
                )
            try:
                await bot.api.messages.send(
                    peer_id=vk_id,
                    random_id=0,
                    message=message_text,
                    keyboard=main_keyboard(),
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Unsubscribe strike event failed")

    return bot
