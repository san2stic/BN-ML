from __future__ import annotations

import json
from email import message_from_string

from monitoring.alerter import Alerter


def test_alerter_deduplicates_messages(monkeypatch) -> None:
    calls = {"webhook": 0}

    alerter = Alerter(
        config={"monitoring": {"alerting": {"webhook_url": "https://example.test", "dedupe_sec": 60}}},
        enabled=True,
    )

    def _fake_webhook(_message: str) -> None:
        calls["webhook"] += 1

    monkeypatch.setattr(alerter, "_send_webhook", _fake_webhook)
    monkeypatch.setattr(alerter, "_send_telegram", lambda _message: None)
    monkeypatch.setattr(alerter, "_send_email", lambda _message: None)

    alerter.send("same message")
    alerter.send("same message")
    assert calls["webhook"] == 1


def test_alerter_posts_webhook_payload(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = str(timeout)
        captured["body"] = req.data.decode("utf-8")
        return _Resp()

    monkeypatch.setattr("monitoring.alerter.request.urlopen", _fake_urlopen)

    alerter = Alerter(
        config={"monitoring": {"alerting": {"webhook_url": "https://hooks.test/path", "dedupe_sec": 0}}},
        enabled=True,
    )
    alerter.send("hello webhook")

    assert captured["url"] == "https://hooks.test/path"
    assert json.loads(captured["body"]) == {"text": "hello webhook"}


def test_alerter_sends_email(monkeypatch) -> None:
    sent: dict[str, str] = {}

    class _FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            sent["host"] = host
            sent["port"] = str(port)
            sent["timeout"] = str(timeout)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            sent["starttls"] = "1"

        def login(self, username: str, password: str):
            sent["login"] = f"{username}:{password}"

        def sendmail(self, sender: str, recipients: list[str], body: str):
            sent["sender"] = sender
            sent["recipients"] = ",".join(recipients)
            sent["body"] = body

    monkeypatch.setattr("monitoring.alerter.smtplib.SMTP", _FakeSMTP)

    alerter = Alerter(
        config={
            "monitoring": {
                "alerting": {
                    "dedupe_sec": 0,
                    "email": {
                        "smtp_host": "smtp.test",
                        "smtp_port": 587,
                        "smtp_username": "bot",
                        "smtp_password": "secret",
                        "from": "bot@test",
                        "to": ["ops@test"],
                        "use_tls": True,
                    }
                }
            }
        },
        enabled=True,
    )
    alerter.send("email body")

    mime = message_from_string(sent["body"])
    assert sent["host"] == "smtp.test"
    assert sent["sender"] == "bot@test"
    assert sent["recipients"] == "ops@test"
    assert "Runtime notification" in mime["Subject"]
