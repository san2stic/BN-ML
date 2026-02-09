from __future__ import annotations

from typing import Iterable

from bn_ml.domain_types import Opportunity
from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.fetch_data import BinanceDataManager
from data_manager.multi_timeframe import MultiTimeframeFeatureBuilder
from ml_engine.predictor import MLEnsemblePredictor
from scanner.scorer import OpportunityScorer


class MultiPairScanner:
    def __init__(self, config: dict, data_manager: BinanceDataManager, predictor: MLEnsemblePredictor) -> None:
        self.config = config
        self.data_manager = data_manager
        self.predictor = predictor
        self.cleaner = DataCleaner()
        self.features = FeatureEngineer()
        self.mtf_features = MultiTimeframeFeatureBuilder(
            config=config,
            data_manager=data_manager,
            cleaner=self.cleaner,
            feature_engineer=self.features,
        )
        self.scorer = OpportunityScorer()
        self.ohlcv_limit = int(config.get("scanner", {}).get("ohlcv_limit", 500))

    def scan(self, pairs: Iterable[str]) -> list[Opportunity]:
        selected, _ = self.scan_details(pairs)
        return selected

    def scan_details(self, pairs: Iterable[str]) -> tuple[list[Opportunity], list[Opportunity]]:
        opportunities: list[Opportunity] = []
        min_quote_volume = float(self.config.get("universe", {}).get("min_24h_volume_usdt", 1_000_000))

        for symbol in pairs:
            try:
                quote_volume = self.data_manager.fetch_quote_volume_24h(symbol)
                if quote_volume < min_quote_volume:
                    continue

                frame = self.mtf_features.build(symbol=symbol, limit=self.ohlcv_limit)
                feature_cols = self.features.feature_columns(frame)

                signal = self.predictor.predict(symbol=symbol, frame=frame, feature_columns=feature_cols)
                ml_score = signal.strength

                latest = frame.iloc[-1].to_dict()
                technical = self.scorer.technical_score(latest)
                momentum = self.scorer.momentum_score(latest)
                global_score = self.scorer.global_score(ml_score, technical, momentum)

                spread_pct = self.data_manager.fetch_spread_pct(symbol)
                depth_usdt = self.data_manager.fetch_orderbook_depth_usdt(symbol, depth_pct=0.5)
                atr_ratio = float(latest.get("atr_ratio", 0.01))
                # A conservative proxy including estimated fees/slippage floor.
                expected_net_profit_pct = max(0.05, (global_score - 50) * 0.05)
                correlation_btc = self.data_manager.estimate_correlation_with_btc(symbol)

                opportunities.append(
                    Opportunity(
                        symbol=symbol,
                        ml_score=ml_score,
                        technical_score=technical,
                        momentum_score=momentum,
                        global_score=global_score,
                        signal=signal,
                        spread_pct=spread_pct,
                        orderbook_depth_usdt=depth_usdt,
                        atr_ratio=atr_ratio,
                        expected_net_profit_pct=expected_net_profit_pct,
                        correlation_with_btc=correlation_btc,
                    )
                )
            except Exception:
                continue

        min_score = float(self.config.get("scanner", {}).get("min_global_score", 60))
        top_n = int(self.config.get("scanner", {}).get("top_n", 10))

        filtered = [o for o in opportunities if o.global_score >= min_score]
        filtered.sort(key=lambda o: o.global_score, reverse=True)
        return filtered[:top_n], opportunities
