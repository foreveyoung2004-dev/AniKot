from __future__ import annotations

import hmac
import logging
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable
from urllib.parse import parse_qs
import html
import re

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .config import PACKAGES
from .keyboards import main_keyboard

logger = logging.getLogger(__name__)


def _external_id(payload: dict[str, Any]) -> str | None:
    for data in (payload, payload.get("data") if isinstance(payload.get("data"), dict) else {}):
        for key in ("contractId", "invoiceId", "contract_id", "invoice_id", "id"):
            if data.get(key):
                return str(data[key])
    return None


def _label(balance_type: str) -> str:
    return {
        "anikot": "AniKot",
        "pro": "AniKot Pro",
        "proplus": "AniKot Pro+",
    }.get(balance_type, balance_type)


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _checkout_page(package_label: str, action_url: str, error: str = "") -> str:
    safe_label = html.escape(package_label)
    safe_action = html.escape(action_url, quote=True)
    safe_error = html.escape(error)
    error_html = f'<p class="error">{safe_error}</p>' if safe_error else ''
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AniKot — оплата</title>
<style>
body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0e0b14;color:#fff;display:grid;min-height:100vh;place-items:center;padding:20px;box-sizing:border-box}}
.card{{width:min(440px,100%);background:#171220;border:1px solid #312441;border-radius:22px;padding:24px;box-shadow:0 18px 60px #0008}}
h1{{font-size:28px;margin:0 0 8px}} .muted{{color:#b7a9c8;margin:0 0 22px}}
.pkg{{background:#21182d;border-radius:15px;padding:14px 16px;margin-bottom:18px;font-weight:700}}
label{{display:block;margin:0 0 8px}} input{{width:100%;box-sizing:border-box;padding:14px 15px;border-radius:13px;border:1px solid #49365f;background:#0f0c14;color:#fff;font-size:16px;outline:none}}
button{{width:100%;border:0;border-radius:13px;padding:14px 16px;margin-top:14px;font-size:16px;font-weight:800;background:#8b5cf6;color:white;cursor:pointer}}
.error{{background:#3a1721;color:#ffd6df;border-radius:12px;padding:10px 12px}} .note{{font-size:13px;color:#9989ad;margin-top:14px;line-height:1.45}}
</style></head><body><main class="card"><h1>🐾 AniKot</h1><p class="muted">Оформление оплаты</p><div class="pkg">{safe_label}</div>{error_html}
<form method="post" action="{safe_action}"><label for="email">Email для оформления платежа</label><input id="email" name="email" type="email" autocomplete="email" required placeholder="name@example.com"><button type="submit">Перейти к оплате</button></form>
<p class="note">После продолжения откроется защищённая страница оплаты LAVA. После подтверждения платежа запросы будут начислены автоматически.</p></main></body></html>"""


def build_web_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager] | None = None,
) -> FastAPI:
    """Create the public FastAPI application.

    Runtime services are attached to ``app.state`` by main.py during lifespan
    startup. Keeping the FastAPI object at module/main scope makes the project
    unambiguously detectable as a web application by hosting platforms such as
    BotHost, avoiding a second automatic HTTP wrapper on the same PORT.
    """

    app = FastAPI(title="AniKot webhooks", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/")
    async def root():
        return {"ok": True, "service": "AniKot", "version": "1.5.1-bothost"}

    @app.get("/health")
    async def health(request: Request):
        model_manager = getattr(request.app.state, "model_manager", None)
        lava = getattr(request.app.state, "lava", None)
        models = model_manager.status() if model_manager else None
        return {
            "ok": True,
            "bot": "AniKot",
            "version": "1.5.1-bothost",
            "runtime_ready": bool(getattr(request.app.state, "runtime_ready", False)),
            "ready": bool(getattr(request.app.state, "runtime_ready", False)),
        }

    @app.get("/checkout/{token}", response_class=HTMLResponse)
    async def checkout_page(token: str, request: Request):
        db = getattr(request.app.state, "db", None)
        if db is None:
            return HTMLResponse("Оплата временно недоступна.", status_code=503)
        session = await db.get_checkout_session(token)
        if not session or session.get("package_key") not in PACKAGES:
            return HTMLResponse("Ссылка на оплату недействительна.", status_code=404)
        if session.get("payment_url"):
            return RedirectResponse(str(session["payment_url"]), status_code=303)
        package = PACKAGES[str(session["package_key"])]
        return HTMLResponse(_checkout_page(package.label, f"/checkout/{token}"))

    @app.post("/checkout/{token}")
    async def checkout_submit(token: str, request: Request):
        db = getattr(request.app.state, "db", None)
        lava = getattr(request.app.state, "lava", None)
        if db is None or lava is None:
            return HTMLResponse("Оплата временно недоступна.", status_code=503)
        session = await db.get_checkout_session(token)
        if not session or session.get("package_key") not in PACKAGES:
            return HTMLResponse("Ссылка на оплату недействительна.", status_code=404)
        if session.get("payment_url"):
            return RedirectResponse(str(session["payment_url"]), status_code=303)

        raw = (await request.body()).decode("utf-8", errors="ignore")
        form = parse_qs(raw)
        email = ((form.get("email") or [""])[0]).strip()
        package = PACKAGES[str(session["package_key"])]
        if not EMAIL_RE.match(email):
            return HTMLResponse(_checkout_page(package.label, f"/checkout/{token}", "Проверьте email и попробуйте ещё раз."), status_code=400)

        local_id = await db.create_payment(int(session["vk_id"]), package, email)
        try:
            external_id, payment_url, raw_response = await lava.create_invoice(email, package)
            await db.finalize_payment_creation(local_id, external_id, payment_url, raw_response)
            await db.finalize_checkout_session(token, local_id, payment_url)
            return RedirectResponse(payment_url, status_code=303)
        except Exception as exc:
            logger.exception("Payment checkout creation failed")
            await db.fail_payment_creation(local_id, str(exc))
            return HTMLResponse(_checkout_page(package.label, f"/checkout/{token}", "Не удалось продолжить оплату. Попробуйте позже."), status_code=503)

    @app.post("/lava/webhook")
    async def lava_webhook(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
    ):
        settings = getattr(request.app.state, "settings", None)
        db = getattr(request.app.state, "db", None)
        bot = getattr(request.app.state, "bot", None)
        if settings is None or db is None or bot is None:
            raise HTTPException(503, "service unavailable")

        if not settings.lava_webhook_key:
            raise HTTPException(503, "service unavailable")
        if not x_api_key or not hmac.compare_digest(x_api_key, settings.lava_webhook_key):
            raise HTTPException(401, "unauthorized")

        payload = await request.json()
        event_type = payload.get("eventType") or payload.get("event_type")
        if event_type != "payment.success":
            return {"ok": True, "ignored": event_type}

        external_id = _external_id(payload)
        if not external_id:
            return {"ok": True, "matched": False}

        result = await db.mark_payment_paid(external_id, payload)
        if not result:
            return {"ok": True, "matched": False}

        was_new, vk_id, balance, btype, requests_count = result
        if was_new:
            try:
                await bot.api.messages.send(
                    peer_id=vk_id,
                    random_id=0,
                    message=(f"✅ Оплата получена.\n"
                             f"Начислено: +{requests_count} {_label(btype)}."),
                    keyboard=main_keyboard(),
                )
            except Exception:
                logger.exception("Failed to notify VK user")
        return {"ok": True, "matched": True, "credited": was_new}

    return app
