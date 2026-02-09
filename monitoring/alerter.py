from __future__ import annotations

import logging


class Alerter:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.logger = logging.getLogger("bn_ml.alerts")

    def send(self, message: str) -> None:
        if not self.enabled:
            return
        self.logger.warning("ALERT | %s", message)
