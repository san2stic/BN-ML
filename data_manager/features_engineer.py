from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"]

        # --- EMAs ---
        for period in [9, 21, 50, 100, 200]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # EMA spread ratios (scale-invariant)
        df["ema_9_21_spread"] = df["ema_9"] / df["ema_21"].replace(0, np.nan) - 1
        df["ema_21_50_spread"] = df["ema_21"] / df["ema_50"].replace(0, np.nan) - 1
        df["ema_50_200_spread"] = df["ema_50"] / df["ema_200"].replace(0, np.nan) - 1

        # --- MACD ---
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- RSI(14) with Wilder's EWM smoothing ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # --- ADX + Directional Indicators (14) ---
        df = self._adx(df, high, low, close, period=14)

        # --- Stochastic %K(14,3) and %D(3) ---
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        raw_k = (close - low_14) / (high_14 - low_14).replace(0, np.nan) * 100
        df["stoch_k"] = raw_k.rolling(3).mean()
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # --- Williams %R(14) ---
        df["williams_r"] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100

        # --- CCI(20) ---
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # --- MFI(14) - Money Flow Index ---
        mf_raw = tp * volume
        mf_pos = pd.Series(np.where(tp > tp.shift(1), mf_raw, 0.0), index=df.index)
        mf_neg = pd.Series(np.where(tp < tp.shift(1), mf_raw, 0.0), index=df.index)
        mf_ratio = mf_pos.rolling(14).sum() / mf_neg.rolling(14).sum().replace(0, np.nan)
        df["mfi_14"] = 100 - (100 / (1 + mf_ratio))

        # --- RSI divergence proxy ---
        df["rsi_divergence"] = df["rsi_14"].diff(4) - close.pct_change(4) * 100

        # --- ATR ---
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / close.replace(0, np.nan)
        df["atr_acceleration"] = df["atr_14"].pct_change(4)

        # --- Bollinger Bands ---
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_mid"] = ma20
        df["bb_upper"] = ma20 + 2 * std20
        df["bb_lower"] = ma20 - 2 * std20
        width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_percent_b"] = (close - df["bb_lower"]) / width
        df["bb_bandwidth"] = width / df["bb_mid"].replace(0, np.nan)

        # --- Keltner Channel (20, 1.5) ---
        kc_mid = close.ewm(span=20, adjust=False).mean()
        kc_upper = kc_mid + 1.5 * df["atr_14"]
        kc_lower = kc_mid - 1.5 * df["atr_14"]
        kc_width = (kc_upper - kc_lower).replace(0, np.nan)
        df["keltner_pct"] = (close - kc_lower) / kc_width

        # --- Volatility regime ---
        returns = close.pct_change()
        vol_short = returns.rolling(14).std()
        vol_long = returns.rolling(96).std()
        df["vol_regime_ratio"] = vol_short / vol_long.replace(0, np.nan)

        # --- SuperTrend (10, 3.0) ---
        df = self._supertrend(df, high, low, close, period=10, multiplier=3.0)

        # --- Volume features ---
        signed_volume = np.where(close >= close.shift(1), volume, -volume)
        df["obv"] = pd.Series(signed_volume, index=df.index).cumsum()
        vol_ma20 = volume.rolling(20).mean().replace(0, np.nan)
        df["volume_ratio_20"] = volume / vol_ma20
        df["volume_acceleration"] = df["volume_ratio_20"].diff(4)

        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        df["ad_line"] = (clv.fillna(0) * volume).cumsum()

        # VWAP-proxy 24h (rolling 96 bars)
        vwap_96 = (close * volume).rolling(96, min_periods=20).sum() / volume.rolling(96, min_periods=20).sum().replace(0, np.nan)
        df["dist_vwap_24h"] = close / vwap_96.replace(0, np.nan) - 1

        # --- Rate of Change ---
        for period, label in [(4, "1h"), (16, "4h"), (96, "1d")]:
            df[f"roc_{label}"] = (close / close.shift(period) - 1) * 100

        # --- Distance features ---
        df["dist_high_24h"] = close / high.rolling(96).max() - 1
        df["dist_low_24h"] = close / low.rolling(96).min() - 1

        # --- Regime detection ---
        # Efficiency ratio (directional efficiency)
        for w in [20, 96]:
            net_move = close.diff(w).abs()
            path_length = close.diff().abs().rolling(w).sum().replace(0, np.nan)
            df[f"efficiency_ratio_{w}"] = net_move / path_length

        # Variance ratio (trending vs mean-reverting)
        var_short = returns.rolling(14).var()
        var_long = returns.rolling(56).var()
        df["variance_ratio"] = var_long / (4 * var_short).replace(0, np.nan)

        # --- Candle structure ---
        bar_range = (high - low).replace(0, np.nan)
        df["candle_body_ratio"] = (close - open_).abs() / bar_range
        df["upper_shadow_ratio"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / bar_range
        df["lower_shadow_ratio"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / bar_range

        # --- Past return statistics ---
        for h in [2, 4, 8, 16]:
            df[f"past_mean_return_{h}"] = returns.rolling(h).mean()

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return df

    @staticmethod
    def _adx(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out when the other is larger
        plus_dm = pd.Series(np.where(plus_dm > minus_dm, plus_dm, 0.0), index=df.index)
        minus_dm = pd.Series(np.where(minus_dm > plus_dm, minus_dm, 0.0), index=df.index)

        tr = pd.concat(
            [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)

        atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan)

        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        df["adx_14"] = dx.ewm(alpha=1 / period, adjust=False).mean()
        df["plus_di_14"] = plus_di
        df["minus_di_14"] = minus_di
        return df

    @staticmethod
    def _supertrend(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        hl2 = (high + low) / 2
        tr = pd.concat(
            [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        direction = pd.Series(1, index=df.index, dtype=float)
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

        df["supertrend_direction"] = direction
        return df

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        excluded = {"timestamp", "open", "high", "low", "close", "volume", "label", "label_edge", "symbol"}
        return [c for c in df.columns if c not in excluded]
