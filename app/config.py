from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


def _default_data_dir() -> Path:
    explicit = os.getenv("DATA_DIR")
    if explicit:
        return Path(explicit)
    bothost_data = Path("/app/data")
    if bothost_data.exists():
        return bothost_data
    return Path("data")


DATA_DIR = _default_data_dir()


@dataclass(frozen=True)
class Package:
    key: str
    requests: int
    price: float
    currency: str
    label: str
    balance_type: str  # anikot | pro | proplus


def _parse_admin_ids(value: str) -> set[int]:
    result: set[int] = set()
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def _public_base_url() -> str:
    explicit = os.getenv("PUBLIC_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    domain = os.getenv("DOMAIN", "").strip()
    if not domain:
        return "https://anikot.bothost.tech"
    if domain.startswith("http://") or domain.startswith("https://"):
        return domain.rstrip("/")
    return f"https://{domain.rstrip('/')}"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


def _anikot_model() -> str:
    """Use Qwen3.5 Flash for ordinary AniKot and migrate the old Nano default.

    Existing BotHost deployments may still contain
    AIAI_ANIKOT_MODEL=gpt-5.4-nano. That value was the previous project
    default, so it is transparently upgraded to Qwen3.5 Flash. Any other
    explicit value remains an intentional override.
    """
    raw = os.getenv("AIAI_ANIKOT_MODEL", "").strip()
    if not raw or raw.lower() == "gpt-5.4-nano":
        return "qwen3.5-flash"
    return raw


def _proplus_model() -> str:
    """Use Kimi K2.6 for Pro+ and transparently migrate the old GPT-5.5 default.

    Existing BotHost deployments may still have AIAI_PROPLUS_MODEL=gpt-5.5 in
    their environment. Treat that legacy value as the old project default and
    move it to Kimi automatically. Any other explicit custom model remains an
    intentional override.
    """
    raw = os.getenv("AIAI_PROPLUS_MODEL", "").strip()
    if not raw or raw.lower() == "gpt-5.5":
        return "kimi-k2.6"
    return raw


def _load_packages(currency: str) -> Dict[str, Package]:
    # Product matrix for AniKot 2.0.
    default = {
        "A50": {"requests": 50, "price": 59, "balance_type": "anikot", "label": "50 запросов — 59 ₽"},
        "A150": {"requests": 150, "price": 129, "balance_type": "anikot", "label": "150 запросов — 129 ₽"},
        "P10": {"requests": 10, "price": 79, "balance_type": "pro", "label": "10 запросов — 79 ₽"},
        "P30": {"requests": 30, "price": 199, "balance_type": "pro", "label": "30 запросов — 199 ₽"},
        "P100": {"requests": 100, "price": 599, "balance_type": "pro", "label": "100 запросов — 599 ₽"},
        "X10": {"requests": 10, "price": 119, "balance_type": "proplus", "label": "10 запросов — 119 ₽"},
        "X30": {"requests": 30, "price": 299, "balance_type": "proplus", "label": "30 запросов — 299 ₽"},
        "X100": {"requests": 100, "price": 849, "balance_type": "proplus", "label": "100 запросов — 849 ₽"},
    }
    raw = os.getenv("SEARCH_PACKAGES_JSON")
    data = json.loads(raw) if raw else default
    allowed = {"anikot", "pro", "proplus"}
    packages: Dict[str, Package] = {}
    for key, value in data.items():
        balance_type = value.get("balance_type", "anikot")
        if balance_type not in allowed:
            raise ValueError(f"Unknown balance_type for package {key}: {balance_type}")
        packages[key] = Package(
            key=key,
            requests=int(value["requests"]),
            price=float(value["price"]),
            currency=value.get("currency", currency),
            label=value.get("label") or f'{value["requests"]} запросов — {value["price"]} {currency}',
            balance_type=balance_type,
        )
    return packages


@dataclass(frozen=True)
class Settings:
    vk_token: str = os.getenv("VK_TOKEN", "")
    vk_group_id: int = int(os.getenv("VK_GROUP_ID", "0") or 0)
    vk_group_url: str = os.getenv("VK_GROUP_URL", "")
    admin_ids: set[int] = frozenset(_parse_admin_ids(os.getenv("ADMIN_IDS", "")))

    # Registration / bonuses.
    # New users receive regular AniKot searches; referrals reward Pro searches.
    registration_bonus_requests: int = int(os.getenv("REGISTRATION_BONUS_REQUESTS", "1"))
    subscription_bonus_requests: int = int(os.getenv("SUBSCRIPTION_BONUS_REQUESTS", "2"))
    referral_reward_pro: int = int(os.getenv("REFERRAL_REWARD_PRO", "3"))
    referral_freeze_days: int = int(os.getenv("REFERRAL_FREEZE_DAYS", "3"))

    user_agreement_url: str = os.getenv("USER_AGREEMENT_URL", "")
    privacy_policy_url: str = os.getenv("PRIVACY_POLICY_URL", "")
    vk_referral_base_url: str = os.getenv("VK_REFERRAL_BASE_URL", "")

    unsubscribe_strike_limit: int = int(os.getenv("UNSUBSCRIBE_STRIKE_LIMIT", "3"))

    # Abuse protection for non-anime image submissions.
    non_anime_warning_limit: int = int(os.getenv("NON_ANIME_WARNING_LIMIT", "5"))
    non_anime_block_days: int = int(os.getenv("NON_ANIME_BLOCK_DAYS", "7"))
    non_anime_max_likelihood: float = float(os.getenv("NON_ANIME_MAX_LIKELIHOOD", "0.30"))

    # Community activity rewards regular AniKot requests.
    activity_actions_per_reward: int = int(os.getenv("ACTIVITY_ACTIONS_PER_REWARD", "3"))
    activity_reward_requests: int = int(os.getenv("ACTIVITY_REWARD_REQUESTS", "1"))
    activity_daily_reward_cap: int = int(os.getenv("ACTIVITY_DAILY_REWARD_CAP", "3"))
    activity_cooldown_seconds: int = int(os.getenv("ACTIVITY_COOLDOWN_SECONDS", "60"))
    activity_min_text_length: int = int(os.getenv("ACTIVITY_MIN_TEXT_LENGTH", "12"))

    data_dir: str = str(DATA_DIR)
    database_path: str = os.getenv("DATABASE_PATH", str(DATA_DIR / "bot.sqlite3"))
    temp_dir: str = os.getenv("TEMP_DIR", str(DATA_DIR / "tmp"))

    proplus_require_age_confirmation: bool = _env_bool("PROPLUS_REQUIRE_AGE_CONFIRMATION", True)

    # Three search tiers through AIAI.BY.
    aiai_api_key: str = os.getenv("AIAI_API_KEY", "")
    aiai_base_url: str = os.getenv("AIAI_BASE_URL", "https://api.aiai.by/v1")
    aiai_anikot_model: str = _anikot_model()
    aiai_pro_model: str = os.getenv("AIAI_PRO_MODEL", "gpt-5.4-mini")
    aiai_proplus_model: str = _proplus_model()
    aiai_timeout: float = float(os.getenv("AIAI_TIMEOUT", "90"))
    # Global cap for all search AI tiers together.
    aiai_max_concurrency: int = max(1, min(int(os.getenv("AIAI_MAX_CONCURRENCY", "3")), 3))

    # Low-memory transport/image settings.
    vk_image_max_side: int = int(os.getenv("VK_IMAGE_MAX_SIDE", "1280"))
    image_max_bytes: int = int(os.getenv("IMAGE_MAX_BYTES", str(6 * 1024 * 1024)))
    http_max_connections: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "8"))
    http_max_keepalive_connections: int = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "4"))
    http_keepalive_expiry: float = float(os.getenv("HTTP_KEEPALIVE_EXPIRY", "20"))
    low_memory_mode: bool = _env_bool("LOW_MEMORY_MODE", True)
    db_cache_kib: int = int(os.getenv("DB_CACHE_KIB", "1024"))
    max_pending_searches: int = int(os.getenv("MAX_PENDING_SEARCHES", "30"))
    vision_cache_ttl_days: int = int(os.getenv("VISION_CACHE_TTL_DAYS", "30"))

    # Confidence gates. The model's confidence is a heuristic, not a statistical guarantee.
    anikot_min_confidence: float = float(os.getenv("ANIKOT_MIN_CONFIDENCE", "0.55"))
    pro_min_confidence: float = float(os.getenv("PRO_MIN_CONFIDENCE", "0.65"))
    proplus_min_confidence: float = float(os.getenv("PROPLUS_MIN_CONFIDENCE", "0.75"))

    lava_api_key: str = os.getenv("LAVA_API_KEY", "")
    lava_webhook_key: str = os.getenv("LAVA_WEBHOOK_KEY", "")
    lava_offer_id: str = os.getenv("LAVA_OFFER_ID", "")
    lava_product_title: str = os.getenv("LAVA_PRODUCT_TITLE", "Пополнение запросов")
    lava_base_url: str = os.getenv("LAVA_BASE_URL", "https://gate.lava.top")
    lava_currency: str = os.getenv("LAVA_CURRENCY", "RUB")
    lava_success_url: str = os.getenv("LAVA_SUCCESS_URL", "")
    lava_failure_url: str = os.getenv("LAVA_FAILURE_URL", "")
    lava_cancel_url: str = os.getenv("LAVA_CANCEL_URL", "")

    public_base_url: str = _public_base_url()

    # BotHost injects PORT. WEB_PORT remains a local-development fallback.
    web_host: str = os.getenv("WEB_HOST", "0.0.0.0")
    web_port: int = int(os.getenv("PORT", os.getenv("WEB_PORT", "8080")))

    def validate(self) -> None:
        missing = []
        if not self.vk_token:
            missing.append("VK_TOKEN")
        if not self.vk_group_id:
            missing.append("VK_GROUP_ID")
        if missing:
            raise RuntimeError("Не заполнены обязательные переменные: " + ", ".join(missing))
        if self.unsubscribe_strike_limit < 1:
            raise RuntimeError("UNSUBSCRIBE_STRIKE_LIMIT должен быть >= 1")
        if self.non_anime_warning_limit < 1:
            raise RuntimeError("NON_ANIME_WARNING_LIMIT должен быть >= 1")
        if self.non_anime_block_days < 1:
            raise RuntimeError("NON_ANIME_BLOCK_DAYS должен быть >= 1")
        if not 0.0 <= self.non_anime_max_likelihood <= 1.0:
            raise RuntimeError("NON_ANIME_MAX_LIKELIHOOD должен быть от 0 до 1")
        if self.referral_freeze_days < 1:
            raise RuntimeError("REFERRAL_FREEZE_DAYS должен быть >= 1")
        if self.vk_image_max_side < 320:
            raise RuntimeError("VK_IMAGE_MAX_SIDE должен быть >= 320")
        if self.image_max_bytes < 262144:
            raise RuntimeError("IMAGE_MAX_BYTES слишком мал")
        if self.http_max_connections < 1 or self.http_max_keepalive_connections < 0:
            raise RuntimeError("Некорректные HTTP лимиты")
        if self.db_cache_kib < 256:
            raise RuntimeError("DB_CACHE_KIB должен быть >= 256")
        if self.max_pending_searches < self.aiai_max_concurrency:
            raise RuntimeError("MAX_PENDING_SEARCHES должен быть >= AIAI_MAX_CONCURRENCY")
        if self.vision_cache_ttl_days < 1:
            raise RuntimeError("VISION_CACHE_TTL_DAYS должен быть >= 1")
        for name, value in (
            ("ANIKOT_MIN_CONFIDENCE", self.anikot_min_confidence),
            ("PRO_MIN_CONFIDENCE", self.pro_min_confidence),
            ("PROPLUS_MIN_CONFIDENCE", self.proplus_min_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{name} должен быть от 0 до 1")

        for directory in (
            Path(self.data_dir),
            Path(self.database_path).parent,
            Path(self.temp_dir),
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
PACKAGES = _load_packages(settings.lava_currency)
