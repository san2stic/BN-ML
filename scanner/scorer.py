from __future__ import annotations


class OpportunityScorer:
    @staticmethod
    def technical_score(latest: dict) -> float:
        score = 50.0

        rsi = float(latest.get("rsi_14", 50))
        if rsi < 30:
            score += 20
        elif rsi > 70:
            score -= 20

        macd_hist = float(latest.get("macd_hist", 0))
        if macd_hist > 0:
            score += 10
        else:
            score -= 10

        bb_percent_b = float(latest.get("bb_percent_b", 0.5))
        if bb_percent_b < 0.2:
            score += 10
        elif bb_percent_b > 0.8:
            score -= 10

        return max(0.0, min(100.0, score))

    @staticmethod
    def momentum_score(latest: dict) -> float:
        roc_1h = float(latest.get("roc_1h", 0))
        roc_4h = float(latest.get("roc_4h", 0))
        volume_ratio = float(latest.get("volume_ratio_20", 1))

        score = 50 + roc_1h * 1.2 + roc_4h * 0.6 + (volume_ratio - 1) * 8
        return max(0.0, min(100.0, score))

    @staticmethod
    def global_score(ml_score: float, technical: float, momentum: float) -> float:
        return 0.50 * ml_score + 0.30 * technical + 0.20 * momentum
