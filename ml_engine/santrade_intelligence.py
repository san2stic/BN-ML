from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from bn_ml.domain_types import Opportunity


@dataclass
class SanTradeIntelligenceSnapshot:
    generated_at: str
    enabled: bool
    signal: str
    confidence: float
    market_score: float
    market_score_pct: float
    market_regime: str
    predicted_move_pct: float
    symbols_scanned: int
    opportunities_scanned: int
    buy_ratio: float
    sell_ratio: float
    hold_ratio: float
    avg_ml_score: float
    avg_technical_score: float
    avg_momentum_score: float
    avg_global_score: float
    avg_atr_ratio: float
    avg_spread_pct: float
    avg_correlation_btc: float
    score_dispersion: float
    model_samples: int
    model_ready: bool
    benchmark_symbol: str
    benchmark_price: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SanTradeIntelligence:
    CLASSES = np.asarray([0, 1, 2], dtype=int)
    ACTIONS = {0: "SELL", 1: "HOLD", 2: "BUY"}
    VECTOR_FIELDS = [
        "buy_ratio",
        "sell_ratio",
        "hold_ratio",
        "avg_ml_score",
        "avg_technical_score",
        "avg_momentum_score",
        "avg_global_score",
        "avg_atr_ratio",
        "avg_spread_pct",
        "avg_correlation_btc",
        "score_dispersion",
        "breadth_delta",
        "momentum_bias",
        "ml_bias",
        "risk_pressure",
        "heuristic_market_score",
    ]

    def __init__(self, config: dict[str, Any], data_manager: Any, logger: Any | None = None) -> None:
        self.config = config
        self.data_manager = data_manager
        self.logger = logger

        model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
        sti_cfg = model_cfg.get("santrade_intelligence", {}) if isinstance(model_cfg.get("santrade_intelligence", {}), dict) else {}

        self.enabled = bool(sti_cfg.get("enabled", True))
        self.min_pairs = max(3, int(sti_cfg.get("min_pairs", 8)))
        self.bearish_threshold = float(sti_cfg.get("bearish_threshold", -0.22))
        self.bullish_threshold = float(sti_cfg.get("bullish_threshold", 0.22))
        self.atr_risk_off_ratio = max(1e-9, float(sti_cfg.get("atr_risk_off_ratio", 0.018)))
        self.spread_risk_off_pct = max(1e-9, float(sti_cfg.get("spread_risk_off_pct", 0.18)))
        self.hold_target_band = abs(float(sti_cfg.get("online_target_hold_band_pct", 0.06))) / 100.0
        self.min_samples_for_model = max(8, int(sti_cfg.get("min_samples_for_model", 24)))
        self.max_blend_weight = float(np.clip(float(sti_cfg.get("model_blend_weight", 0.45)), 0.0, 1.0))
        self.persist_every_updates = max(1, int(sti_cfg.get("persist_every_updates", 5)))
        self.configured_benchmark_symbol = str(sti_cfg.get("benchmark_symbol", "")).strip().upper()

        state_path = str(sti_cfg.get("state_path", "artifacts/state/santrade_intelligence.joblib"))
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        random_state = int(model_cfg.get("random_state", 42))
        alpha = float(sti_cfg.get("online_alpha", 0.0005))
        self._scaler = StandardScaler()
        self._classifier = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            random_state=random_state,
            max_iter=1,
            tol=None,
            learning_rate="optimal",
        )

        self._vector_dim = len(self.VECTOR_FIELDS)
        self._scaler_fitted = False
        self._model_fitted = False
        self._model_samples = 0
        self._updates_since_persist = 0

        self._previous_vector: np.ndarray | None = None
        self._previous_benchmark_price: float = 0.0

        self._load_state()

    def update(
        self,
        pairs: Iterable[str],
        opportunities: Iterable[Opportunity],
        quote_asset: str = "USDT",
    ) -> SanTradeIntelligenceSnapshot:
        timestamp = datetime.now(timezone.utc).isoformat()
        pair_list = [str(pair).upper() for pair in pairs]
        opps = list(opportunities)

        if not self.enabled:
            return self._build_snapshot(
                generated_at=timestamp,
                signal="HOLD",
                confidence=0.0,
                market_score=0.0,
                market_regime="disabled",
                predicted_move_pct=0.0,
                symbols_scanned=len(pair_list),
                opportunities_scanned=len(opps),
                buy_ratio=0.0,
                sell_ratio=0.0,
                hold_ratio=1.0,
                avg_ml_score=0.0,
                avg_technical_score=0.0,
                avg_momentum_score=0.0,
                avg_global_score=0.0,
                avg_atr_ratio=0.0,
                avg_spread_pct=0.0,
                avg_correlation_btc=0.0,
                score_dispersion=0.0,
                benchmark_symbol=self._resolve_benchmark_symbol(quote_asset),
                benchmark_price=0.0,
            )

        stats = self._aggregate_market_stats(pair_list=pair_list, opportunities=opps)
        vector = self._vector_from_stats(stats)

        benchmark_symbol = self._resolve_benchmark_symbol(quote_asset)
        benchmark_price = self._fetch_benchmark_price(benchmark_symbol)

        self._train_previous_sample(current_price=benchmark_price)
        model_score, model_action, model_confidence = self._predict_model(vector)

        heuristic_score = float(stats["heuristic_market_score"])
        heuristic_signal = self._signal_from_score(heuristic_score)
        heuristic_confidence = float(stats["heuristic_confidence"])

        final_score = heuristic_score
        final_signal = heuristic_signal
        final_confidence = heuristic_confidence
        if model_score is not None and model_action is not None and model_confidence is not None:
            blend = self._blend_weight()
            final_score = float(np.clip((1.0 - blend) * heuristic_score + blend * model_score, -1.0, 1.0))
            final_signal = self._signal_from_score(final_score)
            final_confidence = float(
                np.clip((1.0 - blend) * heuristic_confidence + blend * model_confidence, 0.0, 100.0)
            )
            if final_signal == "HOLD" and model_action in {"BUY", "SELL"} and model_confidence >= 80.0:
                final_signal = model_action

        predicted_move_pct = float(np.clip(final_score * max(0.10, stats["avg_atr_ratio"] * 100.0 * 1.5), -12.0, 12.0))

        market_regime = self._resolve_regime(
            signal=final_signal,
            market_score=final_score,
            avg_atr_ratio=stats["avg_atr_ratio"],
            avg_spread_pct=stats["avg_spread_pct"],
            buy_ratio=stats["buy_ratio"],
            sell_ratio=stats["sell_ratio"],
            opportunities_scanned=len(opps),
        )

        self._previous_vector = vector
        self._previous_benchmark_price = benchmark_price

        snapshot = self._build_snapshot(
            generated_at=timestamp,
            signal=final_signal,
            confidence=final_confidence,
            market_score=final_score,
            market_regime=market_regime,
            predicted_move_pct=predicted_move_pct,
            symbols_scanned=len(pair_list),
            opportunities_scanned=len(opps),
            buy_ratio=stats["buy_ratio"],
            sell_ratio=stats["sell_ratio"],
            hold_ratio=stats["hold_ratio"],
            avg_ml_score=stats["avg_ml_score"],
            avg_technical_score=stats["avg_technical_score"],
            avg_momentum_score=stats["avg_momentum_score"],
            avg_global_score=stats["avg_global_score"],
            avg_atr_ratio=stats["avg_atr_ratio"],
            avg_spread_pct=stats["avg_spread_pct"],
            avg_correlation_btc=stats["avg_correlation_btc"],
            score_dispersion=stats["score_dispersion"],
            benchmark_symbol=benchmark_symbol,
            benchmark_price=benchmark_price,
        )

        self._updates_since_persist += 1
        if self._updates_since_persist >= self.persist_every_updates:
            self._persist_state()
            self._updates_since_persist = 0

        return snapshot

    def _aggregate_market_stats(self, pair_list: list[str], opportunities: list[Opportunity]) -> dict[str, float]:
        count = len(opportunities)
        if count == 0:
            return {
                "buy_ratio": 0.0,
                "sell_ratio": 0.0,
                "hold_ratio": 1.0,
                "avg_ml_score": 0.0,
                "avg_technical_score": 0.0,
                "avg_momentum_score": 0.0,
                "avg_global_score": 0.0,
                "avg_atr_ratio": 0.0,
                "avg_spread_pct": 0.0,
                "avg_correlation_btc": 0.0,
                "score_dispersion": 0.0,
                "breadth_delta": 0.0,
                "momentum_bias": 0.0,
                "ml_bias": 0.0,
                "risk_pressure": 0.0,
                "heuristic_market_score": 0.0,
                "heuristic_confidence": 0.0,
            }

        weights = np.asarray([max(float(o.orderbook_depth_usdt), 1.0) for o in opportunities], dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.ones(len(opportunities), dtype=float)
            weight_sum = float(len(opportunities))

        buy_count = sum(1 for o in opportunities if str(o.signal.action).upper() == "BUY")
        sell_count = sum(1 for o in opportunities if str(o.signal.action).upper() == "SELL")
        hold_count = max(0, count - buy_count - sell_count)

        buy_ratio = float(buy_count / max(count, 1))
        sell_ratio = float(sell_count / max(count, 1))
        hold_ratio = float(hold_count / max(count, 1))

        avg_ml = self._weighted_avg([float(o.ml_score) for o in opportunities], weights)
        avg_tech = self._weighted_avg([float(o.technical_score) for o in opportunities], weights)
        avg_mom = self._weighted_avg([float(o.momentum_score) for o in opportunities], weights)
        avg_global = self._weighted_avg([float(o.global_score) for o in opportunities], weights)
        avg_atr = self._weighted_avg([float(o.atr_ratio) for o in opportunities], weights)
        avg_spread = self._weighted_avg([float(o.spread_pct) for o in opportunities], weights)
        avg_corr = self._weighted_avg([float(o.correlation_with_btc) for o in opportunities], weights)

        score_dispersion = float(np.std([float(o.global_score) for o in opportunities]))
        breadth_delta = buy_ratio - sell_ratio
        momentum_bias = float(np.clip((avg_mom - 50.0) / 50.0, -1.0, 1.0))
        ml_bias = float(np.clip((avg_ml - 50.0) / 50.0, -1.0, 1.0))

        atr_pressure = max(0.0, (avg_atr / self.atr_risk_off_ratio) - 1.0)
        spread_pressure = max(0.0, (avg_spread / self.spread_risk_off_pct) - 1.0)
        correlation_pressure = max(0.0, avg_corr - 0.75)
        risk_pressure = float(np.clip(0.45 * atr_pressure + 0.35 * spread_pressure + 0.20 * correlation_pressure, 0.0, 3.0))

        heuristic_score = 0.55 * breadth_delta + 0.25 * momentum_bias + 0.20 * ml_bias - 0.30 * risk_pressure
        if len(pair_list) < self.min_pairs or count < self.min_pairs:
            heuristic_score *= 0.55
        heuristic_score = float(np.clip(heuristic_score, -1.0, 1.0))

        confidence = (abs(heuristic_score) * 100.0) + (abs(breadth_delta) * 30.0) - min(20.0, score_dispersion * 0.3)
        if len(pair_list) < self.min_pairs:
            confidence *= 0.65
        confidence = float(np.clip(confidence, 0.0, 100.0))

        return {
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "hold_ratio": hold_ratio,
            "avg_ml_score": avg_ml,
            "avg_technical_score": avg_tech,
            "avg_momentum_score": avg_mom,
            "avg_global_score": avg_global,
            "avg_atr_ratio": avg_atr,
            "avg_spread_pct": avg_spread,
            "avg_correlation_btc": avg_corr,
            "score_dispersion": score_dispersion,
            "breadth_delta": breadth_delta,
            "momentum_bias": momentum_bias,
            "ml_bias": ml_bias,
            "risk_pressure": risk_pressure,
            "heuristic_market_score": heuristic_score,
            "heuristic_confidence": confidence,
        }

    @staticmethod
    def _weighted_avg(values: list[float], weights: np.ndarray) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != weights.shape[0]:
            return float(np.mean(arr))
        denom = float(np.sum(weights))
        if denom <= 1e-12:
            return float(np.mean(arr))
        return float(np.sum(arr * weights) / denom)

    def _vector_from_stats(self, stats: dict[str, float]) -> np.ndarray:
        vector = [
            float(stats[field]) if field in stats else 0.0
            for field in self.VECTOR_FIELDS
        ]
        arr = np.asarray(vector, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size != self._vector_dim:
            return np.zeros(self._vector_dim, dtype=float)
        return arr

    def _train_previous_sample(self, current_price: float) -> None:
        if self._previous_vector is None:
            return
        if self._previous_vector.size != self._vector_dim:
            return
        if self._previous_benchmark_price <= 0.0 or current_price <= 0.0:
            return

        realized_return = (current_price / self._previous_benchmark_price) - 1.0
        if realized_return > self.hold_target_band:
            label = 2
        elif realized_return < -self.hold_target_band:
            label = 0
        else:
            label = 1

        x_prev = self._previous_vector.reshape(1, -1)
        self._scaler.partial_fit(x_prev)
        self._scaler_fitted = True
        x_scaled = self._scaler.transform(x_prev)

        if not self._model_fitted:
            self._classifier.partial_fit(x_scaled, np.asarray([label], dtype=int), classes=self.CLASSES)
            self._model_fitted = True
        else:
            self._classifier.partial_fit(x_scaled, np.asarray([label], dtype=int))

        self._model_samples += 1

    def _predict_model(self, vector: np.ndarray) -> tuple[float | None, str | None, float | None]:
        if not self._model_fitted or not self._scaler_fitted:
            return None, None, None
        if self._model_samples < self.min_samples_for_model:
            return None, None, None

        try:
            x_scaled = self._scaler.transform(vector.reshape(1, -1))
            proba = self._classifier.predict_proba(x_scaled)
            latest = np.asarray(proba[0], dtype=float)
            if latest.size < 3:
                return None, None, None
            action_idx = int(np.argmax(latest))
            model_score = float(np.clip(latest[2] - latest[0], -1.0, 1.0))
            model_confidence = float(np.clip(np.max(latest) * 100.0, 0.0, 100.0))
            return model_score, self.ACTIONS.get(action_idx, "HOLD"), model_confidence
        except Exception as exc:
            self._log_warning("SanTradeIntelligence model prediction failed: %s", exc)
            return None, None, None

    def _blend_weight(self) -> float:
        if self._model_samples <= 0:
            return 0.0
        dynamic = self._model_samples / (self._model_samples + 30.0)
        return float(np.clip(min(self.max_blend_weight, dynamic), 0.0, self.max_blend_weight))

    def _resolve_regime(
        self,
        *,
        signal: str,
        market_score: float,
        avg_atr_ratio: float,
        avg_spread_pct: float,
        buy_ratio: float,
        sell_ratio: float,
        opportunities_scanned: int,
    ) -> str:
        if opportunities_scanned < self.min_pairs:
            return "insufficient_data"

        risk_off = (avg_atr_ratio >= self.atr_risk_off_ratio) or (avg_spread_pct >= self.spread_risk_off_pct)
        if risk_off and sell_ratio >= buy_ratio:
            return "risk_off"

        if signal == "BUY" and market_score >= self.bullish_threshold:
            return "bull_acceleration"
        if signal == "SELL" and market_score <= self.bearish_threshold:
            return "bear_pressure"
        return "neutral"

    def _build_snapshot(
        self,
        *,
        generated_at: str,
        signal: str,
        confidence: float,
        market_score: float,
        market_regime: str,
        predicted_move_pct: float,
        symbols_scanned: int,
        opportunities_scanned: int,
        buy_ratio: float,
        sell_ratio: float,
        hold_ratio: float,
        avg_ml_score: float,
        avg_technical_score: float,
        avg_momentum_score: float,
        avg_global_score: float,
        avg_atr_ratio: float,
        avg_spread_pct: float,
        avg_correlation_btc: float,
        score_dispersion: float,
        benchmark_symbol: str,
        benchmark_price: float,
    ) -> SanTradeIntelligenceSnapshot:
        return SanTradeIntelligenceSnapshot(
            generated_at=generated_at,
            enabled=self.enabled,
            signal=str(signal).upper(),
            confidence=float(np.clip(confidence, 0.0, 100.0)),
            market_score=float(np.clip(market_score, -1.0, 1.0)),
            market_score_pct=float(np.clip((market_score + 1.0) * 50.0, 0.0, 100.0)),
            market_regime=str(market_regime),
            predicted_move_pct=float(predicted_move_pct),
            symbols_scanned=int(symbols_scanned),
            opportunities_scanned=int(opportunities_scanned),
            buy_ratio=float(np.clip(buy_ratio, 0.0, 1.0)),
            sell_ratio=float(np.clip(sell_ratio, 0.0, 1.0)),
            hold_ratio=float(np.clip(hold_ratio, 0.0, 1.0)),
            avg_ml_score=float(avg_ml_score),
            avg_technical_score=float(avg_technical_score),
            avg_momentum_score=float(avg_momentum_score),
            avg_global_score=float(avg_global_score),
            avg_atr_ratio=float(max(avg_atr_ratio, 0.0)),
            avg_spread_pct=float(max(avg_spread_pct, 0.0)),
            avg_correlation_btc=float(np.clip(avg_correlation_btc, -1.0, 1.0)),
            score_dispersion=float(max(score_dispersion, 0.0)),
            model_samples=int(self._model_samples),
            model_ready=bool(self._model_fitted and self._model_samples >= self.min_samples_for_model),
            benchmark_symbol=benchmark_symbol,
            benchmark_price=float(max(benchmark_price, 0.0)),
        )

    def _signal_from_score(self, score: float) -> str:
        if score >= self.bullish_threshold:
            return "BUY"
        if score <= self.bearish_threshold:
            return "SELL"
        return "HOLD"

    def _resolve_benchmark_symbol(self, quote_asset: str) -> str:
        if self.configured_benchmark_symbol:
            return self.configured_benchmark_symbol
        quote = str(quote_asset).strip().upper() or "USDT"
        return f"BTC/{quote}"

    def _fetch_benchmark_price(self, symbol: str) -> float:
        try:
            return float(self.data_manager.fetch_last_price(symbol))
        except Exception:
            fallback = "BTC/USDT"
            if symbol != fallback:
                try:
                    return float(self.data_manager.fetch_last_price(fallback))
                except Exception:
                    pass
        return 0.0

    def _persist_state(self) -> None:
        payload = {
            "scaler": self._scaler,
            "classifier": self._classifier,
            "scaler_fitted": self._scaler_fitted,
            "model_fitted": self._model_fitted,
            "model_samples": self._model_samples,
            "previous_vector": self._previous_vector,
            "previous_benchmark_price": self._previous_benchmark_price,
            "vector_dim": self._vector_dim,
        }
        try:
            joblib.dump(payload, self.state_path)
        except Exception as exc:
            self._log_warning("SanTradeIntelligence state persist failed: %s", exc)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = joblib.load(self.state_path)
        except Exception as exc:
            self._log_warning("SanTradeIntelligence state load failed: %s", exc)
            return

        if not isinstance(payload, dict):
            return

        try:
            vector_dim = int(payload.get("vector_dim", self._vector_dim))
        except (TypeError, ValueError):
            vector_dim = self._vector_dim
        if vector_dim != self._vector_dim:
            return

        scaler = payload.get("scaler")
        classifier = payload.get("classifier")
        if scaler is not None:
            self._scaler = scaler
        if classifier is not None:
            self._classifier = classifier

        self._scaler_fitted = bool(payload.get("scaler_fitted", False))
        self._model_fitted = bool(payload.get("model_fitted", False))
        try:
            self._model_samples = int(payload.get("model_samples", 0))
        except (TypeError, ValueError):
            self._model_samples = 0

        previous_vector = payload.get("previous_vector")
        if isinstance(previous_vector, np.ndarray) and previous_vector.size == self._vector_dim:
            self._previous_vector = previous_vector.astype(float)
        else:
            self._previous_vector = None

        try:
            self._previous_benchmark_price = float(payload.get("previous_benchmark_price", 0.0))
        except (TypeError, ValueError):
            self._previous_benchmark_price = 0.0

    def _log_warning(self, msg: str, *args: Any) -> None:
        if self.logger is None or not hasattr(self.logger, "warning"):
            return
        try:
            self.logger.warning(msg, *args)
        except Exception:
            return
