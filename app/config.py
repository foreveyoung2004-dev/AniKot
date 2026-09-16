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
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(DATA_DIR / "models")))

# Keep Hugging Face helper metadata/cache in persistent storage too.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf-cache"))


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


def _load_packages(currency: str) -> Dict[str, Package]:
    default = {
        "A100": {"requests": 100, "price": 59, "balance_type": "anikot", "label": "100 AniKot — 59 ₽"},
        "A300": {"requests": 300, "price": 129, "balance_type": "anikot", "label": "300 AniKot — 129 ₽"},
        "P10": {"requests": 10, "price": 79, "balance_type": "pro", "label": "10 AniKot Pro — 79 ₽"},
        "P30": {"requests": 30, "price": 199, "balance_type": "pro", "label": "30 AniKot Pro — 199 ₽"},
        "P100": {"requests": 100, "price": 599, "balance_type": "pro", "label": "100 AniKot Pro — 599 ₽"},
        "X10": {"requests": 10, "price": 119, "balance_type": "proplus", "label": "10 AniKot Pro+ — 119 ₽"},
        "X30": {"requests": 30, "price": 299, "balance_type": "proplus", "label": "30 AniKot Pro+ — 299 ₽"},
        "X100": {"requests": 100, "price": 849, "balance_type": "proplus", "label": "100 AniKot Pro+ — 849 ₽"},
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

    initial_requests: int = int(os.getenv("INITIAL_REQUESTS", "0"))
    initial_pro_requests: int = int(os.getenv("INITIAL_PRO_REQUESTS", "1"))
    initial_proplus_requests: int = int(os.getenv("INITIAL_PROPLUS_REQUESTS", "0"))
    subscription_bonus_pro: int = int(os.getenv("SUBSCRIPTION_BONUS_PRO", os.getenv("SUBSCRIPTION_BONUS", "5")))
    referral_reward_pro: int = int(os.getenv("REFERRAL_REWARD_PRO", "3"))
    referral_freeze_days: int = int(os.getenv("REFERRAL_FREEZE_DAYS", "3"))

    user_agreement_url: str = os.getenv("USER_AGREEMENT_URL", "")
    privacy_policy_url: str = os.getenv("PRIVACY_POLICY_URL", "")
    vk_referral_base_url: str = os.getenv("VK_REFERRAL_BASE_URL", "")

    unsubscribe_strike_limit: int = int(os.getenv("UNSUBSCRIBE_STRIKE_LIMIT", "3"))

    activity_actions_per_reward: int = int(os.getenv("ACTIVITY_ACTIONS_PER_REWARD", "3"))
    activity_reward_requests: int = int(os.getenv("ACTIVITY_REWARD_REQUESTS", "1"))
    activity_daily_reward_cap: int = int(os.getenv("ACTIVITY_DAILY_REWARD_CAP", "3"))
    activity_cooldown_seconds: int = int(os.getenv("ACTIVITY_COOLDOWN_SECONDS", "60"))
    activity_min_text_length: int = int(os.getenv("ACTIVITY_MIN_TEXT_LENGTH", "12"))

    data_dir: str = str(DATA_DIR)
    database_path: str = os.getenv("DATABASE_PATH", str(DATA_DIR / "bot.sqlite3"))
    temp_dir: str = os.getenv("TEMP_DIR", str(DATA_DIR / "tmp"))

    # Bothost-friendly model provisioning. Weights are never bundled in Git/ZIP.
    auto_download_models: bool = _env_bool("AUTO_DOWNLOAD_MODELS", True)
    auto_download_pixai: bool = _env_bool("AUTO_DOWNLOAD_PIXAI", True)
    model_download_retries: int = int(os.getenv("MODEL_DOWNLOAD_RETRIES", "3"))

    camie_repo: str = os.getenv("CAMIE_REPO", "Camais03/camie-tagger-v2")
    camie_model_path: str = os.getenv("CAMIE_MODEL_PATH", str(MODELS_DIR / "camie-tagger-v2.onnx"))
    camie_metadata_path: str = os.getenv("CAMIE_METADATA_PATH", str(MODELS_DIR / "camie-tagger-v2-metadata.json"))
    camie_threshold: float = float(os.getenv("CAMIE_THRESHOLD", "0.492"))
    camie_top_k: int = int(os.getenv("CAMIE_TOP_K", "5"))

    # Local ONNX PixAI tagger for Pro+. No torch/dghs-imgutils dependency needed.
    pixai_enabled: bool = _env_bool("PIXAI_ENABLED", True)
    pixai_repo: str = os.getenv("PIXAI_REPO", "deepghs/pixai-tagger-v0.9-onnx")
    pixai_model_path: str = os.getenv("PIXAI_MODEL_PATH", str(MODELS_DIR / "pixai-tagger-v0.9.onnx"))
    pixai_tags_path: str = os.getenv("PIXAI_TAGS_PATH", str(MODELS_DIR / "pixai-tagger-v0.9-tags.csv"))
    pixai_general_threshold: float = float(os.getenv("PIXAI_GENERAL_THRESHOLD", "0.30"))
    pixai_character_threshold: float = float(os.getenv("PIXAI_CHARACTER_THRESHOLD", "0.85"))

    proplus_require_age_confirmation: bool = _env_bool("PROPLUS_REQUIRE_AGE_CONFIRMATION", True)

    aiai_api_key: str = os.getenv("AIAI_API_KEY", "")
    aiai_base_url: str = os.getenv("AIAI_BASE_URL", "https://api.aiai.by/v1")
    # Separate AI tiers. AIAI_MODEL remains as a backwards-compatible alias
    # for the standard AniKot / Pro+ model.
    aiai_anikot_model: str = os.getenv(
        "AIAI_ANIKOT_MODEL",
        os.getenv("AIAI_MODEL", "qwen3-vl-32b-instruct"),
    )
    # AniKot Pro uses a newer, stronger native vision-language model.
    aiai_pro_model: str = os.getenv("AIAI_PRO_MODEL", "qwen3.5-397b-a17b")
    # Pro+ keeps the previous remote model; Camie + PixAI remain in its pipeline.
    aiai_proplus_model: str = os.getenv(
        "AIAI_PROPLUS_MODEL",
        os.getenv("AIAI_MODEL", "gpt-5.4"),
    )
    aiai_fallback_model: str = os.getenv("AIAI_FALLBACK_MODEL", "qwen3-vl-32b-instruct")
    aiai_timeout: float = float(os.getenv("AIAI_TIMEOUT", "90"))
    aiai_max_concurrency: int = int(os.getenv("AIAI_MAX_CONCURRENCY", "6"))
    min_search_confidence: float = float(os.getenv("MIN_SEARCH_CONFIDENCE", "0.70"))

    lava_api_key: str = os.getenv("LAVA_API_KEY", "")
    lava_webhook_key: str = os.getenv("LAVA_WEBHOOK_KEY", "")
    lava_offer_id: str = os.getenv("LAVA_OFFER_ID", "")
    lava_base_url: str = os.getenv("LAVA_BASE_URL", "https://gate.lava.top")
    lava_currency: str = os.getenv("LAVA_CURRENCY", "RUB")
    lava_success_url: str = os.getenv("LAVA_SUCCESS_URL", "")
    lava_failure_url: str = os.getenv("LAVA_FAILURE_URL", "")
    lava_cancel_url: str = os.getenv("LAVA_CANCEL_URL", "")

    # Public URL used for browser checkout. On BotHost DOMAIN is normally injected.
    # PUBLIC_BASE_URL can be set explicitly, e.g. https://anikot.bothost.tech
    public_base_url: str = _public_base_url()

    # Bothost injects PORT. WEB_PORT remains a local-development fallback.
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
        if self.referral_freeze_days < 1:
            raise RuntimeError("REFERRAL_FREEZE_DAYS должен быть >= 1")
        if self.model_download_retries < 1:
            raise RuntimeError("MODEL_DOWNLOAD_RETRIES должен быть >= 1")
        if not 0.0 <= self.min_search_confidence <= 1.0:
            raise RuntimeError("MIN_SEARCH_CONFIDENCE должен быть от 0 до 1")

        for directory in (
            Path(self.data_dir),
            Path(self.database_path).parent,
            Path(self.temp_dir),
            Path(self.camie_model_path).parent,
            Path(self.pixai_model_path).parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
PACKAGES = _load_packages(settings.lava_currency)
