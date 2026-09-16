from __future__ import annotations

import json
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


class LavaAPIError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, details: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def _format_details(data: Any) -> str:
    if not data:
        return ""
    if isinstance(data, dict):
        details = data.get("details")
        if isinstance(details, dict):
            return "; ".join(f"{k}: {v}" for k, v in details.items())[:700]
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
        self._resolved_offer_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.s.lava_api_key and (self.s.lava_offer_id or self.s.lava_product_title))

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
                _format_details(data) or response.reason_phrase,
                data,
            )
        if not isinstance(data, dict):
            raise LavaAPIError(response.status_code, "invalid_response", data)
        return data

    async def _discover_offer_id(self) -> str | None:
        """Resolve the current dynamic offer after a product was edited/republished.

        This prevents stale offer IDs from permanently breaking checkout. If several
        dynamic products exist, LAVA_PRODUCT_TITLE is used to select the AniKot one.
        """
        data = await self._request_json("GET", "/api/v2/products?feedVisibility=ONLY_HIDDEN")
        items = data.get("items") or []
        if not isinstance(items, list):
            return None

        candidates: list[tuple[str, str]] = []
        configured = (self.s.lava_offer_id or "").strip()
        wanted_title = (self.s.lava_product_title or "").strip().casefold()

        for product in items:
            if not isinstance(product, dict) or not bool(product.get("isDynamicPrice")):
                continue
            title = str(product.get("title") or "").strip()
            offers = product.get("offers") or []
            for offer in offers:
                if not isinstance(offer, dict) or not offer.get("id"):
                    continue
                offer_id = str(offer["id"])
                if configured and offer_id == configured:
                    self._resolved_offer_id = offer_id
                    return offer_id
                candidates.append((title, offer_id))

        if wanted_title:
            for title, offer_id in candidates:
                if title.casefold() == wanted_title:
                    self._resolved_offer_id = offer_id
                    return offer_id
            for title, offer_id in candidates:
                if wanted_title in title.casefold() or title.casefold() in wanted_title:
                    self._resolved_offer_id = offer_id
                    return offer_id

        if len(candidates) == 1:
            self._resolved_offer_id = candidates[0][1]
            return self._resolved_offer_id
        return None

    async def _create_with_offer(self, email: str, package: Package, offer_id: str) -> dict[str, Any]:
        currency = package.currency.upper()
        minimums = {"RUB": 50.0, "USD": 5.0, "EUR": 5.0}
        minimum = minimums.get(currency)
        if minimum is not None and package.price < minimum:
            raise LavaAPIError(400, "amount_below_minimum")

        amount: int | float = int(package.price) if float(package.price).is_integer() else float(package.price)
        body: dict[str, Any] = {
            "email": email.strip(),
            "offerId": offer_id.strip(),
            "currency": currency,
            "amount": amount,
        }
        if self.s.lava_success_url:
            body["successful_return_url"] = self.s.lava_success_url.strip()
        if self.s.lava_failure_url:
            body["failure_return_url"] = self.s.lava_failure_url.strip()
        if self.s.lava_cancel_url:
            body["cancel_return_url"] = self.s.lava_cancel_url.strip()
        return await self._request_json("POST", "/api/v3/invoice", json=body)

    async def create_invoice(self, email: str, package: Package) -> tuple[str, str, dict[str, Any]]:
        if not self.enabled:
            raise LavaAPIError(None, "payments_unavailable")

        offer_id = self._resolved_offer_id or self.s.lava_offer_id.strip()
        if not offer_id:
            offer_id = await self._discover_offer_id() or ""
        if not offer_id:
            raise LavaAPIError(None, "product_unavailable")

        try:
            data = await self._create_with_offer(email, package, offer_id)
        except LavaAPIError as exc:
            # A republished dynamic product may receive a new offers[].id. Refresh
            # once on not-found and retry transparently for the buyer.
            if exc.status_code != 404:
                raise
            refreshed = await self._discover_offer_id()
            if not refreshed or refreshed == offer_id:
                raise
            data = await self._create_with_offer(email, package, refreshed)

        external_id = _deep_find(data, ("contractId", "contract_id", "invoiceId", "invoice_id", "id"))
        payment_url = _deep_find(
            data,
            ("paymentUrl", "payment_url", "checkoutUrl", "checkout_url", "paymentLink", "payment_link", "url"),
        )
        if not external_id or not payment_url:
            raise LavaAPIError(200, "invalid_payment_response", data)
        return str(external_id), str(payment_url), data
