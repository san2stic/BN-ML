from __future__ import annotations

import hashlib
import json
import logging
import smtplib
import ssl
import time
from email.mime.text import MIMEText
from typing import Any
from urllib import error, request


class Alerter:
    def __init__(self, config: dict[str, Any] | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.logger = logging.getLogger("bn_ml.alerts")
        self.config = config or {}

        mon_cfg = self.config.get("monitoring", {})
        alerting_cfg = mon_cfg.get("alerting", {}) if isinstance(mon_cfg, dict) else {}

        self.webhook_url = str(alerting_cfg.get("webhook_url", "")).strip()
        self.telegram_bot_token = str(alerting_cfg.get("telegram_bot_token", "")).strip()
        self.telegram_chat_id = str(alerting_cfg.get("telegram_chat_id", "")).strip()
        self.email_cfg = alerting_cfg.get("email", {}) if isinstance(alerting_cfg.get("email", {}), dict) else {}
        self.timeout_sec = float(alerting_cfg.get("timeout_sec", 4.0))
        self.dedupe_sec = max(0.0, float(alerting_cfg.get("dedupe_sec", 30.0)))
        self._last_sent_ts: dict[str, float] = {}

    def _is_duplicate(self, message: str) -> bool:
        if self.dedupe_sec <= 0:
            return False
        key = hashlib.sha1(message.encode("utf-8")).hexdigest()  # noqa: S324 - non-crypto dedupe key
        now = time.time()
        last = self._last_sent_ts.get(key, 0.0)
        if (now - last) < self.dedupe_sec:
            return True
        self._last_sent_ts[key] = now
        return False

    def _post_json(self, url: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_sec):
            return

    def _send_webhook(self, message: str) -> None:
        if not self.webhook_url:
            return
        self._post_json(self.webhook_url, {"text": message})

    def _send_telegram(self, message: str) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        self._post_json(url, {"chat_id": self.telegram_chat_id, "text": message})

    def _send_email(self, message: str) -> None:
        host = str(self.email_cfg.get("smtp_host", "")).strip()
        if not host:
            return
        port = int(self.email_cfg.get("smtp_port", 587))
        username = str(self.email_cfg.get("smtp_username", "")).strip()
        password = str(self.email_cfg.get("smtp_password", "")).strip()
        sender = str(self.email_cfg.get("from", username)).strip()
        recipients_raw = self.email_cfg.get("to", [])
        if isinstance(recipients_raw, str):
            recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
        else:
            recipients = [str(r).strip() for r in recipients_raw if str(r).strip()]
        if not sender or not recipients:
            return

        use_tls = bool(self.email_cfg.get("use_tls", True))

        mime = MIMEText(message, "plain", "utf-8")
        mime["Subject"] = str(self.email_cfg.get("subject_prefix", "[BN-ML ALERT]")) + " Runtime notification"
        mime["From"] = sender
        mime["To"] = ", ".join(recipients)

        context = ssl.create_default_context()
        with smtplib.SMTP(host=host, port=port, timeout=self.timeout_sec) as client:
            if use_tls:
                client.starttls(context=context)
            if username and password:
                client.login(username, password)
            client.sendmail(sender, recipients, mime.as_string())

    def send(self, message: str) -> None:
        if not self.enabled:
            return
        if self._is_duplicate(message):
            return
        self.logger.warning("ALERT | %s", message)

        for channel_fn, channel_name in [
            (self._send_webhook, "webhook"),
            (self._send_telegram, "telegram"),
            (self._send_email, "email"),
        ]:
            try:
                channel_fn(message)
            except (OSError, ValueError, error.URLError, smtplib.SMTPException) as exc:
                self.logger.warning("ALERT channel '%s' failed: %s", channel_name, exc)
