from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import Package, Settings


def _deep_find(data: Any, names: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        for name in names:
            if name in data and data[name] not in (None, ""):
                return data[name]
        for value in data.values():
            found = _deep_find(value, names)
            if found not in (None, ""):
                return found
    elif isinstance(data, list):
        for item in data:
            found = _deep_find(item, names)
            if found not in (None, ""):
                return found
    return None


@dataclass
class LavaConfigStatus:
    ok: bool
    message: str
    product_title: str | None = None
    dynamic_price: bool | None = None


class LavaAPIError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

    def public_message(self) -> str:
        return "Не удалось выполнить операцию. Попробуйте позже."



def _format_lava_details(data: Any) -> str:
    if not data:
        return ""
    if isinstance(data, dict):
        details = data.get("details")
        if isinstance(details, dict):
            parts = [f"{k}: {v}" for k, v in details.items()]
            return "; ".join(parts)[:700]
        for key in ("message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value[:700]
    if isinstance(data, str):
        return data[:700]
    try:
        return json.dumps(data, ensure_ascii=False)[:700]
    except Exception:
        return str(data)[:700]


class LavaClient:
    def __init__(self, settings: Settings):
        self.s = settings

    @property
    def enabled(self) -> bool:
        return bool(self.s.lava_api_key and self.s.lava_offer_id)

    async def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "X-Api-Key": self.s.lava_api_key,
        }
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        async with httpx.AsyncClient(timeout=40, follow_redirects=True) as client:
            response = await client.request(
                method,
                f"{self.s.lava_base_url.rstrip('/')}{path}",
                headers=headers,
                **kwargs,
            )
        try:
            data: Any = response.json()
        except Exception:
            data = response.text[:2000]
        if response.status_code >= 400:
            raise LavaAPIError(
                response.status_code,
                f"Lava API {response.status_code}: {_format_lava_details(data) or response.reason_phrase}",
                data,
            )
        if not isinstance(data, dict):
            raise LavaAPIError(response.status_code, "LAVA вернула ответ не в формате JSON-объекта", data)
        return data

    async def check_config(self) -> LavaConfigStatus:
        """Validate API key + offer against the current LAVA account without creating a charge."""
        if not self.s.lava_api_key:
            return LavaConfigStatus(False, "LAVA_API_KEY не задан")
        if not self.s.lava_offer_id:
            return LavaConfigStatus(False, "LAVA_OFFER_ID не задан")
        # Dynamic-price products are hidden from the public feed. This is also
        # the same endpoint used to obtain the offerId during initial setup.
        try:
            data = await self._request_json(
                "GET", "/api/v2/products?feedVisibility=ONLY_HIDDEN"
            )
        except LavaAPIError as exc:
            return LavaConfigStatus(False, exc.public_message())

        items = data.get("items") or []
        for product in items if isinstance(items, list) else []:
            if not isinstance(product, dict):
                continue
            for offer in product.get("offers") or []:
                if isinstance(offer, dict) and str(offer.get("id")) == self.s.lava_offer_id:
                    dynamic = bool(product.get("isDynamicPrice"))
                    if not dynamic:
                        return LavaConfigStatus(
                            False,
                            "Offer найден, но у продукта выключена «Цена по запросу через API».",
                            str(product.get("title") or ""),
                            False,
                        )
                    return LavaConfigStatus(
                        True,
                        "LAVA API key и offerId подтверждены.",
                        str(product.get("title") or offer.get("name") or ""),
                        True,
                    )
        return LavaConfigStatus(
            False,
            "Этот LAVA_OFFER_ID не найден среди скрытых API-продуктов данного ключа.",
        )

    async def create_invoice(self, email: str, package: Package) -> tuple[str, str, dict[str, Any]]:
        if not self.enabled:
            raise LavaAPIError(None, "LAVA_API_KEY или LAVA_OFFER_ID не задан")

        currency = package.currency.upper()
        minimums = {"RUB": 50.0, "USD": 5.0, "EUR": 5.0}
        minimum = minimums.get(currency)
        if minimum is not None and package.price < minimum:
            raise LavaAPIError(
                400,
                f"Сумма {package.price:g} {currency} ниже минимальной {minimum:g} {currency}",
            )

        amount: int | float = int(package.price) if float(package.price).is_integer() else float(package.price)
        body: dict[str, Any] = {
            "email": email.strip(),
            "offerId": self.s.lava_offer_id.strip(),
            "currency": currency,
            "amount": amount,
        }
        # LAVA accepts only absolute HTTPS return URLs. Empty variables are omitted.
        if self.s.lava_success_url:
            body["successful_return_url"] = self.s.lava_success_url.strip()
        if self.s.lava_failure_url:
            body["failure_return_url"] = self.s.lava_failure_url.strip()
        if self.s.lava_cancel_url:
            body["cancel_return_url"] = self.s.lava_cancel_url.strip()

        data = await self._request_json("POST", "/api/v3/invoice", json=body)

        # payment.success webhooks use contractId. Prefer it when the create response
        # contains both contractId and invoiceId, otherwise fall back to invoiceId.
        external_id = _deep_find(data, ("contractId", "contract_id", "invoiceId", "invoice_id", "id"))
        payment_url = _deep_find(
            data,
            ("paymentUrl", "payment_url", "checkoutUrl", "checkout_url", "paymentLink", "payment_link", "url"),
        )
        if not external_id or not payment_url:
            raise LavaAPIError(
                200,
                "LAVA создала ответ, но AniKot не нашёл invoice/contract ID или paymentUrl.",
                data,
            )
        return str(external_id), str(payment_url), data
