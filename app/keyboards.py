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


def main_keyboard(show_subscription_bonus: bool = True) -> str:
    rows: list[list[dict]] = [
        [_text("Режим поиска")],
        [_text("Профиль"), _text("Купить запросы")],
    ]
    if show_subscription_bonus:
        rows.append([_text("Бонус за подписку"), _text("Рефералы")])
    else:
        rows.append([_text("Рефералы")])
    return _inline(rows)


def profile_keyboard() -> str:
    return _inline(
        [
            [_text("Купить запросы"), _text("Рефералы")],
            [_text("Назад")],
        ]
    )


def referral_keyboard() -> str:
    return _inline([[_text("Назад")]])


def search_mode_keyboard() -> str:
    return _inline(
        [
            [_text("Обычный"), _text("Pro"), _text("Pro+")],
            [_text("Назад")],
        ]
    )


def search_cancel_keyboard() -> str:
    return _inline([[_text("Отмена")]])


def result_keyboard() -> str:
    return _inline([[_text("Назад")]])


def age_confirmation_keyboard() -> str:
    return _inline([[_text("✅ Мне есть 18 лет")], [_text("Назад")]])


def legal_keyboard() -> str:
    return _inline([[_text("✅ Согласен")]])


def referral_choice_keyboard(has_candidate: bool = False) -> str:
    if has_candidate:
        return _inline(
            [
                [_text("✅ Использовать реферальную ссылку")],
                [_text("Продолжить без ссылки")],
            ]
        )
    return _inline([[_text("Да, есть ссылка"), _text("Нет, продолжить")]])


def referral_input_keyboard() -> str:
    return _inline([[_text("Продолжить без ссылки")]])


def subscription_keyboard(group_url: str) -> str:
    rows: list[list[dict]] = []
    if group_url:
        rows.append([_link("Подписаться", group_url)])
    rows.append([_text("Проверить подписку")])
    rows.append([_text("Назад")])
    return _inline(rows)


def shop_categories_keyboard() -> str:
    return _inline(
        [
            [_text("🔎 AniKot"), _text("✨ Pro"), _text("💎 Pro+")],
            [_text("◀ Назад")],
        ]
    )


def _packages_for(balance_type: str):
    return [p for p in PACKAGES.values() if p.balance_type == balance_type]


def shop_packages_keyboard(balance_type: str) -> str:
    packages = _packages_for(balance_type)
    rows: list[list[dict]] = []
    if balance_type == "anikot":
        rows.append([_text(p.label.replace(" запросов", "")) for p in packages])
    else:
        # Two compact rows: 10/30 together and 100 separately.
        first = packages[:2]
        rest = packages[2:]
        if first:
            rows.append([_text(p.label.replace(" запросов", "")) for p in first])
        if rest:
            rows.append([_text(p.label.replace(" запросов", "")) for p in rest])
    rows.append([_text("◀ Назад")])
    return _inline(rows)


def package_confirm_keyboard(payment_url: str) -> str:
    return _inline(
        [
            [_link("💳 Оплатить", payment_url)],
            [_text("◀ В магазин")],
        ]
    )
