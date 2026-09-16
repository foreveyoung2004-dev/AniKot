from __future__ import annotations

import json
from typing import Iterable

from .config import PACKAGES


def _text(label: str) -> dict:
    return {"action": {"type": "text", "label": label}}


def _link(label: str, url: str) -> dict:
    return {"action": {"type": "open_link", "label": label, "link": url}}


def _inline(rows: Iterable[list[dict]]) -> str:
    return json.dumps(
        {"one_time": False, "inline": True, "buttons": list(rows)},
        ensure_ascii=False,
    )


def main_keyboard() -> str:
    """Main AniKot controls attached directly to the message."""
    return _inline(
        [
            [_text("🔎 AniKot"), _text("✨ AniKot Pro")],
            [_text("💎 AniKot Pro+"), _text("👤 Профиль")],
            [_text("💳 Купить запросы"), _text("🎁 Бонус за подписку")],
            [_text("👥 Рефералы")],
        ]
    )


def age_confirmation_keyboard() -> str:
    return _inline(
        [[_text("✅ Мне есть 18 лет"), _text("⬅ Назад")]]
    )


def legal_keyboard() -> str:
    # The documents themselves are clickable links inside the message text.
    # Only the consent action remains as an inline button attached to that message.
    return _inline([[ _text("✅ Согласен") ]])


def referral_choice_keyboard(has_candidate: bool = False) -> str:
    if has_candidate:
        return _inline(
            [
                [_text("✅ Использовать реферальную ссылку")],
                [_text("➡️ Продолжить без ссылки")],
            ]
        )
    return _inline(
        [[_text("🔗 Да, есть ссылка"), _text("➡️ Нет, продолжить")]]
    )


def referral_input_keyboard() -> str:
    return _inline([[_text("➡️ Продолжить без ссылки")]])


def subscription_keyboard(group_url: str) -> str:
    rows: list[list[dict]] = []
    if group_url:
        rows.append([_link("➕ Подписаться на сообщество", group_url)])
    rows.append([_text("✅ Проверить подписку")])
    rows.append([_text("⬅ Назад")])
    return _inline(rows)


def buy_keyboard() -> str:
    """All package buttons are inline and attached to the shop message."""
    rows: list[list[dict]] = []
    current: list[dict] = []
    icons = {"anikot": "🔎", "pro": "✨", "proplus": "💎"}
    for package in PACKAGES.values():
        icon = icons.get(package.balance_type, "💳")
        current.append(_text(f"{icon} {package.key} · {package.label}"))
        if len(current) == 2:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    rows.append([_text("⬅ Назад")])
    return _inline(rows)


def package_confirm_keyboard(payment_url: str) -> str:
    return _inline(
        [
            [_link("💳 Оплатить", payment_url)],
            [_text("⬅ В магазин")],
        ]
    )


def payment_link_keyboard(payment_url: str) -> str:
    return _inline(
        [
            [_link("💳 Оплатить", payment_url)],
            [_text("⬅ В магазин")],
        ]
    )
