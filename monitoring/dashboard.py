from __future__ import annotations

import io
import json
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Keep imports stable when Streamlit is launched from non-root working dirs.
    sys.path.insert(0, str(PROJECT_ROOT))

from bn_ml.config import load_config
from data_manager.fetch_data import BinanceDataManager

CONFIG_PATH = (PROJECT_ROOT / "configs/bot.yaml").resolve()


def _resolve_runtime_path(raw_value: Any, fallback: str) -> Path:
    raw = str(raw_value).strip() if raw_value is not None else ""
    if not raw:
        raw = fallback
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_runtime_paths() -> tuple[Path, Path, Path]:
    default_db = _resolve_runtime_path(None, "artifacts/state/bn_ml.db")
    default_metrics_dir = _resolve_runtime_path(None, "artifacts/metrics")
    default_models = _resolve_runtime_path(None, "models")
    try:
        cfg = load_config(str(CONFIG_PATH))
    except Exception:
        return default_db, default_metrics_dir / "latest_scan.csv", default_models

    storage_cfg = cfg.get("storage", {})
    monitoring_cfg = cfg.get("monitoring", {})
    api_cfg = cfg.get("public_api", {})

    db_path = _resolve_runtime_path(storage_cfg.get("sqlite_path"), "artifacts/state/bn_ml.db")
    metrics_dir = _resolve_runtime_path(monitoring_cfg.get("metrics_dir"), "artifacts/metrics")
    models_dir = _resolve_runtime_path(api_cfg.get("models_dir"), "models")
    return db_path, metrics_dir / "latest_scan.csv", models_dir


DB_PATH, SCAN_PATH, MODELS_DIR = _load_runtime_paths()

TIMEFRAME_WINDOWS = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
    "all": None,
}

CHART_PANELS = ["Opportunity Heatmap", "Equity & Drawdown", "Cycle Flow", "Signal Matrix", "Portfolio Allocation", "PnL Distribution"]
TABLE_PANELS = ["Execution Blotter", "Cycle Feed", "Opportunity Book", "Model Performance", "Training & Downloads"]
SIDEBAR_PANELS = ["Watchlist", "Risk Flags", "Open Position Monitor"]
ALL_PANELS = CHART_PANELS + TABLE_PANELS + SIDEBAR_PANELS

st.set_page_config(page_title="BN-ML Trader Terminal", page_icon="BT", layout="wide")


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {row[1] for row in rows}
    return column in names


def _safe_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return df with only the requested columns; missing ones become NaN."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


@st.cache_data(ttl=15)
def load_dashboard_settings() -> dict[str, Any]:
    cfg = load_config(str(CONFIG_PATH))
    return cfg.get("monitoring", {}).get("dashboard", {})


@st.cache_data(ttl=2)
def load_runtime_data(db_path: str) -> dict[str, Any]:
    payload = {
        "account": {},
        "training_status": {},
        "trades": pd.DataFrame(),
        "cycles": pd.DataFrame(),
        "positions": pd.DataFrame(),
    }

    path = Path(db_path)
    if not path.exists():
        return payload

    conn = sqlite3.connect(path)
    try:
        if _table_exists(conn, "kv_state"):
            row = conn.execute("SELECT value_json FROM kv_state WHERE key='account_state'").fetchone()
            if row and row[0]:
                payload["account"] = _safe_json(row[0])
            training_row = conn.execute("SELECT value_json FROM kv_state WHERE key='training_status'").fetchone()
            if training_row and training_row[0]:
                payload["training_status"] = _safe_json(training_row[0])

        if _table_exists(conn, "trades"):
            trades = pd.read_sql_query(
                """
                SELECT id, ts, symbol, side, size_usdt, price, mode, extra_json
                FROM trades
                ORDER BY id DESC
                LIMIT 3000
                """,
                conn,
            )
            if not trades.empty:
                trades["ts"] = pd.to_datetime(trades["ts"], utc=True, errors="coerce")
                trades["extra"] = trades["extra_json"].apply(_safe_json)
                trades["reason"] = trades["extra"].apply(lambda x: x.get("reason"))
                trades["pnl_usdt"] = pd.to_numeric(trades["extra"].apply(lambda x: x.get("pnl_usdt")), errors="coerce").fillna(0)
                trades["pnl_pct"] = pd.to_numeric(trades["extra"].apply(lambda x: x.get("pnl_pct")), errors="coerce").fillna(0)
                trades["base_qty"] = pd.to_numeric(trades["extra"].apply(lambda x: x.get("base_qty")), errors="coerce")
                trades = trades.sort_values("ts").reset_index(drop=True)
            payload["trades"] = trades

        if _table_exists(conn, "cycles"):
            cycles = pd.read_sql_query(
                """
                SELECT id, ts, opportunities, opened_positions, data_json
                FROM cycles
                ORDER BY id DESC
                LIMIT 2000
                """,
                conn,
            )
            if not cycles.empty:
                cycles["ts"] = pd.to_datetime(cycles["ts"], utc=True, errors="coerce")
                cycles["data"] = cycles["data_json"].apply(_safe_json)
                cycles["open_positions"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("open_positions")), errors="coerce").fillna(0)
                cycles["full_closes"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("full_closes")), errors="coerce").fillna(0)
                cycles["partial_closes"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("partial_closes")), errors="coerce").fillna(0)
                cycles["daily_pnl_pct"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("daily_pnl_pct")), errors="coerce").fillna(0)
                cycles = cycles.sort_values("ts").reset_index(drop=True)
            payload["cycles"] = cycles

        if _table_exists(conn, "positions"):
            if _table_has_column(conn, "positions", "extra_json"):
                positions = pd.read_sql_query(
                    """
                    SELECT symbol, side, size_usdt, entry_price, stop_loss, take_profit_1, take_profit_2, opened_at, status, extra_json
                    FROM positions
                    WHERE status='OPEN'
                    ORDER BY opened_at DESC
                    """,
                    conn,
                )
            else:
                positions = pd.read_sql_query(
                    """
                    SELECT symbol, side, size_usdt, entry_price, stop_loss, take_profit_1, take_profit_2, opened_at, status
                    FROM positions
                    WHERE status='OPEN'
                    ORDER BY opened_at DESC
                    """,
                    conn,
                )
                positions["extra_json"] = "{}"

            if not positions.empty:
                positions["opened_at"] = pd.to_datetime(positions["opened_at"], utc=True, errors="coerce")
                positions["extra"] = positions["extra_json"].apply(_safe_json)
                positions["remaining_base_qty"] = pd.to_numeric(positions["extra"].apply(lambda x: x.get("remaining_base_qty")), errors="coerce")
            payload["positions"] = positions
    finally:
        conn.close()

    return payload


@st.cache_data(ttl=15)
def load_scan_data(path: str) -> pd.DataFrame:
    scan_path = Path(path)
    if not scan_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(scan_path)
        for col in [
            "confidence",
            "ml_score",
            "technical_score",
            "momentum_score",
            "global_score",
            "spread_pct",
            "depth_usdt",
            "correlation_btc",
            "last_price",
            "change_24h_pct",
            "quote_volume_24h",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _scan_age_seconds(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return float("inf")
    return max(0.0, datetime.now(timezone.utc).timestamp() - p.stat().st_mtime)


@st.cache_data(ttl=10)
def load_live_market_data(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    universe_cfg = cfg.get("universe", {})
    quote = str(cfg.get("base_quote", "USDT")).upper()
    min_volume = float(universe_cfg.get("min_24h_volume_usdt", 1_000_000))
    max_pairs = int(universe_cfg.get("max_pairs_scanned", 150))

    manager = BinanceDataManager(config=cfg, paper=False)
    pairs = manager.discover_pairs_by_quote(
        quote=quote,
        min_quote_volume_usdt=min_volume,
        max_pairs=max_pairs,
    )
    if not pairs:
        return pd.DataFrame()

    try:
        aux = load_live_aux_metrics(config_path)
    except Exception:
        aux = pd.DataFrame()
    aux_map: dict[str, dict[str, float]] = {}
    if not aux.empty:
        aux_map = {
            str(row["symbol"]): {
                "depth_usdt": float(row["depth_usdt"]) if pd.notna(row["depth_usdt"]) else np.nan,
                "correlation_btc": float(row["correlation_btc"]) if pd.notna(row["correlation_btc"]) else np.nan,
            }
            for _, row in aux.iterrows()
        }

    rows: list[dict[str, Any]] = []

    for symbol in pairs:
        try:
            ticker = manager.fetch_ticker(symbol)
        except Exception:
            continue

        last = float(ticker.get("last") or ticker.get("close") or 0.0)
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(last, 0.0)
        spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 and ask > 0 and bid > 0 else np.nan

        pct_24h = ticker.get("percentage")
        if pct_24h is None:
            open_24h = float(ticker.get("open") or 0.0)
            pct_24h = ((last / open_24h) - 1.0) * 100.0 if open_24h > 0 else 0.0
        pct_24h = float(pct_24h)

        quote_vol = float(ticker.get("quoteVolume") or 0.0)

        momentum_score = float(np.clip(50.0 + np.clip(pct_24h, -10.0, 10.0) * 4.5, 0.0, 100.0))
        spread_score = float(np.clip((0.20 - (spread_pct if np.isfinite(spread_pct) else 0.20)) / 0.20 * 100.0, 0.0, 100.0))
        liquidity_score = float(np.clip(np.log10(max(quote_vol, 1.0)) * 16.0, 0.0, 100.0))
        technical_score = float(np.clip(0.60 * spread_score + 0.40 * liquidity_score, 0.0, 100.0))
        global_score = float(np.clip(0.55 * momentum_score + 0.45 * technical_score, 0.0, 100.0))
        confidence = float(np.clip(abs(pct_24h) * 8.0 + technical_score * 0.35, 0.0, 100.0))

        if pct_24h >= 0.8:
            signal = "BUY"
        elif pct_24h <= -0.8:
            signal = "SELL"
        else:
            signal = "HOLD"

        rows.append(
            {
                "symbol": symbol,
                "signal": signal,
                "confidence": confidence,
                "ml_score": momentum_score,
                "technical_score": technical_score,
                "momentum_score": momentum_score,
                "global_score": global_score,
                "spread_pct": spread_pct,
                "depth_usdt": aux_map.get(symbol, {}).get("depth_usdt", np.nan),
                "correlation_btc": aux_map.get(symbol, {}).get("correlation_btc", np.nan),
                "last_price": last,
                "change_24h_pct": pct_24h,
                "quote_volume_24h": quote_vol,
                "source": "live_market",
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pick_btc_benchmark(manager: BinanceDataManager) -> tuple[str, pd.Series] | tuple[None, None]:
    candidates = ["BTC/USDT", "BTC/USDC", "BTC/FDUSD", "BTC/BUSD"]
    for symbol in candidates:
        try:
            df = manager.fetch_ohlcv(symbol=symbol, timeframe="15m", limit=160)
        except Exception:
            continue
        if df.empty:
            continue
        ret = pd.to_numeric(df["close"], errors="coerce").pct_change().dropna()
        if len(ret) >= 40:
            return symbol, ret
    return None, None


@st.cache_data(ttl=120)
def load_live_aux_metrics(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    universe_cfg = cfg.get("universe", {})
    quote = str(cfg.get("base_quote", "USDT")).upper()
    min_volume = float(universe_cfg.get("min_24h_volume_usdt", 1_000_000))
    max_pairs = int(universe_cfg.get("max_pairs_scanned", 150))

    manager = BinanceDataManager(config=cfg, paper=False)
    pairs = manager.discover_pairs_by_quote(
        quote=quote,
        min_quote_volume_usdt=min_volume,
        max_pairs=max_pairs,
    )
    if not pairs:
        return pd.DataFrame()
    btc_symbol, btc_ret = _pick_btc_benchmark(manager)
    rows: list[dict[str, Any]] = []

    for symbol in pairs:
        depth_usdt = np.nan
        corr = np.nan

        try:
            depth_usdt = float(manager.fetch_orderbook_depth_usdt(symbol=symbol, depth_pct=0.5))
        except Exception:
            depth_usdt = np.nan

        try:
            if symbol.startswith("BTC/"):
                corr = 1.0
            elif btc_symbol is not None and btc_ret is not None:
                frame = manager.fetch_ohlcv(symbol=symbol, timeframe="15m", limit=160)
                sret = pd.to_numeric(frame["close"], errors="coerce").pct_change().dropna()
                m = int(min(len(sret), len(btc_ret)))
                if m >= 40:
                    val = float(np.corrcoef(sret.iloc[-m:], btc_ret.iloc[-m:])[0, 1])
                    corr = 0.0 if np.isnan(val) else float(val)
        except Exception:
            corr = np.nan

        rows.append({"symbol": symbol, "depth_usdt": depth_usdt, "correlation_btc": corr})

    return pd.DataFrame(rows)


def compose_scan_frame(scan: pd.DataFrame, live_market: pd.DataFrame, scan_is_stale: bool) -> tuple[pd.DataFrame, str]:
    if live_market.empty and scan.empty:
        return pd.DataFrame(), "none"
    if scan.empty:
        # No scanner output available: fallback to heuristic live market view.
        return live_market.copy(), "live_market_fallback"

    merged = scan.copy()
    merged["source"] = "scanner_stale" if scan_is_stale else "scanner"

    # Enrich scanner rows with live prices/volumes without changing scanner signals.
    if not live_market.empty:
        enrich_cols = ["symbol", "last_price", "change_24h_pct", "quote_volume_24h"]
        live_enrich = live_market[enrich_cols].drop_duplicates("symbol")
        merged = merged.merge(live_enrich, on="symbol", how="left")

    if live_market.empty:
        return merged, "scanner_only"
    if scan_is_stale:
        return merged, "scanner_stale"
    return merged, "scanner_plus_live"


def draw_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --ink: #102320;
  --muted: #5f7370;
  --line: #d6dfdb;
  --panel: rgba(255,255,255,0.72);
  --panel-strong: rgba(255,255,255,0.90);
  --accent: #0f766e;
  --accent2: #e56b2f;
  --good: #0b8f5f;
  --bad: #be3e2b;
}

html, body, [class*="css"] {
  font-family: 'Sora', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, rgba(15,118,110,0.16), transparent 60%),
    radial-gradient(980px 450px at 92% 0%, rgba(229,107,47,0.12), transparent 56%),
    linear-gradient(160deg, #eef5f2 0%, #e8efec 44%, #e2ece8 100%);
  color: var(--ink);
}

[data-testid="stHeader"] { background: transparent; }
.main .block-container { padding-top: 1.05rem; }

.hero {
  border: 1px solid var(--line);
  border-radius: 24px;
  background: linear-gradient(140deg, var(--panel-strong), rgba(255,255,255,0.55));
  box-shadow: 0 16px 44px rgba(17,34,31,0.08);
  padding: 1.15rem 1.2rem;
  margin-bottom: .7rem;
}

.title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: .9rem;
  flex-wrap: wrap;
}

.title-main { font-size: clamp(1.4rem, 2.8vw, 2.05rem); font-weight: 700; letter-spacing: -0.03em; }
.title-sub { color: var(--muted); margin-top: .28rem; }

.chip {
  display: inline-flex;
  align-items: center;
  gap: .3rem;
  border-radius: 999px;
  padding: .3rem .64rem;
  border: 1px solid rgba(16,35,32,0.16);
  background: rgba(255,255,255,0.57);
  font-size: .72rem;
  font-family: 'JetBrains Mono', monospace;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: .58rem;
  margin-top: .65rem;
}

.kpi-cell {
  border: 1px solid var(--line);
  border-radius: 13px;
  background: var(--panel);
  padding: .6rem .64rem;
}

.kpi-label { font-size: .67rem; letter-spacing: .06em; color: var(--muted); text-transform: uppercase; }
.kpi-value { margin-top: .15rem; font-size: 1.08rem; font-weight: 700; }
.kpi-value.pos { color: var(--good); }
.kpi-value.neg { color: var(--bad); }

.sticky-col { position: sticky; top: .7rem; }

.panel {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--panel);
  box-shadow: 0 8px 24px rgba(14,34,31,.05);
  padding: .68rem;
  margin-bottom: .7rem;
}

.panel-title { font-size: .88rem; font-weight: 600; margin-bottom: .38rem; }
.section-head { margin: .35rem 0 .2rem; font-weight: 600; }

.watch-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: .45rem;
  border: 1px solid rgba(214,223,219,.95);
  border-radius: 11px;
  background: rgba(255,255,255,.6);
  padding: .4rem .38rem;
  margin-bottom: .32rem;
}

.watch-left { display: flex; align-items: center; gap: .45rem; min-width: 0; }
.dot { width: .53rem; height: .53rem; border-radius: 50%; }
.dot.buy { background: var(--good); }
.dot.sell { background: var(--bad); }
.dot.hold { background: #8a845d; }
.symbol { font-family: 'JetBrains Mono', monospace; font-size: .74rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.watch-right { font-family: 'JetBrains Mono', monospace; font-size: .72rem; }

[data-testid="stPlotlyChart"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: rgba(255,255,255,0.56);
  padding: .28rem;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
}

.pos-card {
  border: 1px solid var(--line);
  border-radius: 13px;
  background: var(--panel);
  padding: .55rem .6rem;
  margin-bottom: .5rem;
}
.pos-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: .35rem;
  gap: .3rem;
}
.pos-card-body {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: .3rem .5rem;
}
.pos-metric {
  display: flex;
  flex-direction: column;
}
.pos-label {
  font-size: .62rem;
  letter-spacing: .05em;
  color: var(--muted);
  text-transform: uppercase;
}
.pos-metric span:last-child {
  font-family: 'JetBrains Mono', monospace;
  font-size: .74rem;
  font-weight: 500;
}

@media (max-width: 1020px) { .sticky-col { position: static; } }
@media (max-width: 768px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .pos-card-body { grid-template-columns: repeat(2, 1fr); }
  .title-main { font-size: 1.1rem; }
}
@media (max-width: 480px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .pos-card-body { grid-template-columns: 1fr; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def draw_dark_css() -> None:
    st.markdown(
        """
<style>
:root {
  --ink: #e0ede9;
  --muted: #8fa8a0;
  --line: #2a3d38;
  --panel: rgba(18,28,25,0.85);
  --panel-strong: rgba(22,34,30,0.95);
  --accent: #14b8a6;
  --accent2: #f59e0b;
  --good: #34d399;
  --bad: #f87171;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, rgba(20,184,166,0.10), transparent 60%),
    radial-gradient(980px 450px at 92% 0%, rgba(245,158,11,0.08), transparent 56%),
    linear-gradient(160deg, #0c1512 0%, #101e1a 44%, #0d1814 100%);
  color: var(--ink);
}

.hero {
  border-color: var(--line);
  background: linear-gradient(140deg, var(--panel-strong), rgba(18,28,25,0.60));
  box-shadow: 0 16px 44px rgba(0,0,0,0.30);
}

.chip {
  border-color: rgba(224,237,233,0.18);
  background: rgba(18,28,25,0.70);
  color: var(--ink);
}

.kpi-cell {
  border-color: var(--line);
  background: var(--panel);
}

.kpi-label { color: var(--muted); }
.kpi-value { color: var(--ink); }
.kpi-value.pos { color: var(--good); }
.kpi-value.neg { color: var(--bad); }

.panel {
  border-color: var(--line);
  background: var(--panel);
  box-shadow: 0 8px 24px rgba(0,0,0,0.20);
}

.watch-row {
  border-color: var(--line);
  background: rgba(18,28,25,0.60);
}

.dot.buy { background: var(--good); }
.dot.sell { background: var(--bad); }
.dot.hold { background: #a8a060; }

.pos-card {
  border-color: var(--line);
  background: var(--panel);
}

[data-testid="stPlotlyChart"] {
  border-color: var(--line);
  background: rgba(18,28,25,0.60);
}

[data-testid="stDataFrame"] {
  border-color: var(--line);
}

.panel-title { color: var(--ink); }
.watch-right { color: var(--muted); }
.symbol { color: var(--ink); }
.title-main { color: var(--ink); }
.title-sub { color: var(--muted); }
</style>
        """,
        unsafe_allow_html=True,
    )


def apply_fullscreen_mode(enabled: bool) -> None:
    if not enabled:
        return
    st.markdown(
        """
<style>
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
footer { visibility: hidden !important; }
.main .block-container {
  max-width: 100% !important;
  padding-left: .65rem !important;
  padding-right: .65rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def inject_keyboard_shortcuts() -> None:
    st.markdown(
        """
<script>
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === 'r' || e.key === 'R') {
    window.location.reload();
  }
});
</script>
        """,
        unsafe_allow_html=True,
    )


def schedule_autorefresh(enabled: bool, seconds: int) -> None:
    if not enabled or seconds <= 0:
        return
    ms = max(1000, int(seconds * 1000))
    st.markdown(
        f"""
<script>
setTimeout(function() {{
  window.location.reload();
}}, {ms});
</script>
        """,
        unsafe_allow_html=True,
    )


def count_model_bundles() -> int:
    if not MODELS_DIR.exists():
        return 0
    return sum(1 for _ in MODELS_DIR.glob("*/metadata.json"))


@st.cache_data(ttl=60)
def load_model_metadata() -> pd.DataFrame:
    if not MODELS_DIR.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for meta_path in MODELS_DIR.glob("*/metadata.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        symbol = meta.get("symbol", meta_path.parent.name)
        m = meta.get("metrics", {})
        rows.append({
            "symbol": symbol,
            "trained_at": meta.get("trained_at", ""),
            "dataset_rows": int(meta.get("dataset_rows", 0)),
            "rf_accuracy": float(m.get("rf_train_accuracy", 0)),
            "rf_f1": float(m.get("rf_train_f1_macro", 0)),
            "xgb_accuracy": float(m.get("xgb_train_accuracy", 0)),
            "xgb_sharpe": float(m.get("xgb_hpo_walkforward_sharpe", 0)),
            "xgb_sortino": float(m.get("xgb_hpo_walkforward_sortino", 0)),
            "rf_sharpe": float(m.get("rf_hpo_walkforward_sharpe", 0)),
            "rf_sortino": float(m.get("rf_hpo_walkforward_sortino", 0)),
            "feature_count": len(meta.get("feature_columns", [])),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trained_at"] = pd.to_datetime(df["trained_at"], errors="coerce", utc=True)
    return df.sort_values("symbol").reset_index(drop=True)


def render_model_performance(models_df: pd.DataFrame) -> None:
    st.markdown("<div class='panel-title'>ML Model Performance</div>", unsafe_allow_html=True)
    if models_df.empty:
        st.info("No trained models found.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models_df["symbol"], y=models_df["xgb_accuracy"] * 100,
        name="XGB Accuracy %", marker_color="#0f766e",
    ))
    fig.add_trace(go.Bar(
        x=models_df["symbol"], y=models_df["rf_accuracy"] * 100,
        name="RF Accuracy %", marker_color="#e56b2f",
    ))
    fig.update_layout(
        title="Model Accuracy by Symbol", barmode="group",
        xaxis={"title": ""}, yaxis={"title": "Accuracy %", "range": [0, 100]},
        height=320,
    )
    st.plotly_chart(figure_style(fig), width="stretch")

    display = models_df[["symbol", "xgb_accuracy", "rf_accuracy", "xgb_sharpe", "rf_sharpe",
                          "xgb_sortino", "rf_sortino", "dataset_rows", "feature_count", "trained_at"]].copy()
    display["trained_at"] = display["trained_at"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        display, width="stretch", height=300,
        column_config={
            "xgb_accuracy": st.column_config.NumberColumn("XGB Acc", format="%.3f"),
            "rf_accuracy": st.column_config.NumberColumn("RF Acc", format="%.3f"),
            "xgb_sharpe": st.column_config.NumberColumn("XGB Sharpe", format="%.3f"),
            "rf_sharpe": st.column_config.NumberColumn("RF Sharpe", format="%.3f"),
            "xgb_sortino": st.column_config.NumberColumn("XGB Sortino", format="%.3f"),
            "rf_sortino": st.column_config.NumberColumn("RF Sortino", format="%.3f"),
        },
    )


@st.cache_data(ttl=60)
def load_model_bundle_catalog() -> pd.DataFrame:
    if not MODELS_DIR.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for bundle_dir in sorted(MODELS_DIR.iterdir()):
        if not bundle_dir.is_dir():
            continue
        files = [p for p in bundle_dir.rglob("*") if p.is_file()]
        if not files:
            continue
        meta: dict[str, Any] = {}
        meta_path = bundle_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = _safe_json(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        total_bytes = 0
        for file_path in files:
            try:
                total_bytes += int(file_path.stat().st_size)
            except OSError:
                continue
        rows.append(
            {
                "model_key": bundle_dir.name,
                "symbol": str(meta.get("symbol", bundle_dir.name)),
                "trained_at": str(meta.get("trained_at", "")),
                "file_count": len(files),
                "size_mb": float(total_bytes / (1024 * 1024)),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["trained_at"] = pd.to_datetime(df["trained_at"], errors="coerce", utc=True)
    return df.sort_values("symbol").reset_index(drop=True)


def _list_model_files(model_key: str | None = None) -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    target_dirs: list[Path]
    if model_key:
        candidate = MODELS_DIR / model_key
        target_dirs = [candidate] if candidate.exists() and candidate.is_dir() else []
    else:
        target_dirs = [p for p in MODELS_DIR.iterdir() if p.is_dir()]
    files: list[Path] = []
    for directory in target_dirs:
        files.extend([p for p in directory.rglob("*") if p.is_file()])
    return sorted(files)


def _build_model_archive(model_key: str | None = None) -> tuple[bytes, str, int, float]:
    files = _list_model_files(model_key=model_key)
    if not files:
        raise ValueError("No model files found to archive.")

    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if model_key:
        archive_name = f"bnml_models_{model_key}_{now_tag}.zip"
        archive_root = Path(model_key)
        base_dir = MODELS_DIR / model_key
    else:
        archive_name = f"bnml_models_all_{now_tag}.zip"
        archive_root = Path("models")
        base_dir = MODELS_DIR

    total_bytes = 0
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in files:
            try:
                rel_path = file_path.relative_to(base_dir)
            except ValueError:
                continue
            arcname = archive_root / rel_path
            bundle.write(file_path, arcname=str(arcname))
            try:
                total_bytes += int(file_path.stat().st_size)
            except OSError:
                continue
    return buffer.getvalue(), archive_name, len(files), float(total_bytes / (1024 * 1024))


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_ts(raw: Any) -> str:
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return "n/a"
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def render_training_and_downloads(training_status: dict[str, Any]) -> None:
    st.markdown("<div class='panel-title'>Model Training Progress & Downloads</div>", unsafe_allow_html=True)

    status = str(training_status.get("status", "idle")).strip().lower() or "idle"
    phase = str(training_status.get("phase", "waiting")).strip().lower() or "waiting"
    trigger = str(training_status.get("trigger", "unknown")).strip().lower() or "unknown"
    current_symbol = str(training_status.get("current_symbol", "")).strip()
    progress_pct = max(0.0, min(100.0, _to_float(training_status.get("progress_pct", 0.0))))

    status_label = {
        "running": "RUNNING",
        "completed": "COMPLETED",
        "failed": "FAILED",
        "idle": "IDLE",
    }.get(status, status.upper())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status_label)
    c2.metric("Trigger", trigger.upper())
    c3.metric("Progress", f"{progress_pct:.1f}%")
    c4.metric("Current Symbol", current_symbol or "-")

    st.progress(progress_pct / 100.0, text=f"{progress_pct:.1f}%")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Queued", _to_int(training_status.get("symbols_queued", 0)))
    d2.metric("Completed", _to_int(training_status.get("symbols_completed", 0)))
    d3.metric("Trained", _to_int(training_status.get("symbols_trained", 0)))
    d4.metric("Errors", _to_int(training_status.get("symbols_errors", 0)))

    st.caption(
        f"Phase: {phase.upper()} | Started: {_fmt_ts(training_status.get('started_at'))} | "
        f"Updated: {_fmt_ts(training_status.get('updated_at'))}"
    )
    if training_status.get("models_reloaded_at"):
        st.caption(
            f"Model reload: {_fmt_ts(training_status.get('models_reloaded_at'))} "
            f"({str(training_status.get('reload_reason', 'n/a'))})"
        )
    if training_status.get("last_error"):
        st.warning(str(training_status.get("last_error")))

    catalog = load_model_bundle_catalog()
    if catalog.empty:
        st.info("No model bundles found in `models/`.")
        return

    display = catalog.copy()
    display["trained_at"] = display["trained_at"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        display,
        width="stretch",
        height=260,
        column_config={
            "model_key": st.column_config.TextColumn("Bundle Key"),
            "symbol": st.column_config.TextColumn("Symbol"),
            "trained_at": st.column_config.TextColumn("Trained At"),
            "file_count": st.column_config.NumberColumn("Files", format="%d"),
            "size_mb": st.column_config.NumberColumn("Size MB", format="%.2f"),
        },
    )

    st.markdown("**Download Model Artifacts**")
    all_col, one_col = st.columns(2)

    with all_col:
        if st.button("Prepare full archive", key="prep-models-all", width="stretch"):
            try:
                archive_data, archive_name, file_count, size_mb = _build_model_archive(model_key=None)
                st.session_state["models_archive_all"] = {
                    "data": archive_data,
                    "name": archive_name,
                    "file_count": file_count,
                    "size_mb": size_mb,
                }
                st.success(f"Archive ready: {file_count} files ({size_mb:.2f} MB).")
            except ValueError as exc:
                st.warning(str(exc))

        all_archive = st.session_state.get("models_archive_all")
        if isinstance(all_archive, dict):
            st.download_button(
                "Download all models (.zip)",
                data=all_archive["data"],
                file_name=str(all_archive["name"]),
                mime="application/zip",
                key="dl-models-all",
                width="stretch",
            )

    with one_col:
        options: list[tuple[str, str]] = []
        for _, row in catalog.iterrows():
            key = str(row["model_key"])
            label = f"{str(row['symbol'])} ({key})"
            options.append((label, key))
        option_labels = [label for label, _ in options]
        option_to_key = {label: key for label, key in options}
        selected_label = st.selectbox(
            "Select one bundle",
            options=option_labels,
            key="select-model-bundle",
        )
        selected_key = option_to_key.get(selected_label, "")

        if st.button("Prepare selected bundle", key="prep-model-one", width="stretch"):
            try:
                archive_data, archive_name, file_count, size_mb = _build_model_archive(model_key=selected_key)
                st.session_state["models_archive_one"] = {
                    "data": archive_data,
                    "name": archive_name,
                    "file_count": file_count,
                    "size_mb": size_mb,
                    "model_key": selected_key,
                }
                st.success(f"Bundle ready: {file_count} files ({size_mb:.2f} MB).")
            except ValueError as exc:
                st.warning(str(exc))

        one_archive = st.session_state.get("models_archive_one")
        if isinstance(one_archive, dict) and str(one_archive.get("model_key", "")) == selected_key:
            st.download_button(
                "Download selected bundle (.zip)",
                data=one_archive["data"],
                file_name=str(one_archive["name"]),
                mime="application/zip",
                key="dl-models-one",
                width="stretch",
            )


def runtime_mode(cycles: pd.DataFrame, account: dict[str, Any], trades: pd.DataFrame) -> str:
    if not cycles.empty:
        latest = cycles.iloc[-1]
        latest_data = latest.get("data", {})
        if not isinstance(latest_data, dict) or not latest_data:
            latest_data = _safe_json(latest.get("data_json"))
        paper = latest_data.get("paper")
        if paper is True:
            return "PAPER"
        if paper is False:
            return "LIVE"

    if account.get("exchange_synced_at") or account.get("exchange_quote_asset"):
        return "LIVE"

    if not trades.empty and "mode" in trades.columns:
        mode = str(trades.iloc[-1].get("mode", "")).strip().lower()
        if mode == "paper":
            return "PAPER"
        if mode == "live":
            return "LIVE"

    return "UNKNOWN"


def format_money(x: float) -> str:
    return f"{x:,.2f}"


def format_pct(x: float) -> str:
    return f"{x:.2f}%"


def _trend_arrow(current: float, previous: float) -> str:
    if current > previous + 0.01:
        return " <span style='color:var(--good);font-size:.72rem;'>&#9650;</span>"
    if current < previous - 0.01:
        return " <span style='color:var(--bad);font-size:.72rem;'>&#9660;</span>"
    return " <span style='color:var(--muted);font-size:.72rem;'>&#9644;</span>"


def filter_by_timeframe(trades: pd.DataFrame, cycles: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = TIMEFRAME_WINDOWS.get(timeframe, None)
    if window is None:
        return trades, cycles

    now = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now - window)

    t = trades[trades["ts"] >= cutoff].copy() if not trades.empty else trades
    c = cycles[cycles["ts"] >= cutoff].copy() if not cycles.empty else cycles
    return t, c


def compute_kpi(
    account: dict[str, Any],
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    cycles: pd.DataFrame,
    trades_prev: pd.DataFrame | None = None,
) -> dict[str, float]:
    sell = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    realized = float(sell["pnl_usdt"].sum()) if not sell.empty else 0.0
    wins = float(sell[sell["pnl_usdt"] > 0]["pnl_usdt"].sum()) if not sell.empty else 0.0
    losses = abs(float(sell[sell["pnl_usdt"] < 0]["pnl_usdt"].sum())) if not sell.empty else 0.0
    profit_factor = wins / losses if losses > 1e-9 else (wins if wins > 0 else 0.0)
    win_count = len(sell[sell["pnl_usdt"] > 0]) if not sell.empty else 0
    win_rate = (win_count / len(sell) * 100) if not sell.empty and len(sell) > 0 else float(account.get("win_rate", 0.0)) * 100

    heartbeat_sec = -1.0
    if not cycles.empty and pd.notna(cycles["ts"].max()):
        heartbeat_sec = (datetime.now(timezone.utc) - cycles["ts"].max().to_pydatetime()).total_seconds()

    # Previous period for trend arrows
    prev_realized = 0.0
    prev_win_rate = 0.0
    prev_pf = 0.0
    if trades_prev is not None and not trades_prev.empty:
        sp = trades_prev[trades_prev["side"] == "SELL"]
        prev_realized = float(sp["pnl_usdt"].sum()) if not sp.empty else 0.0
        pw = len(sp[sp["pnl_usdt"] > 0]) if not sp.empty else 0
        prev_win_rate = (pw / len(sp) * 100) if not sp.empty and len(sp) > 0 else 0.0
        pw_sum = float(sp[sp["pnl_usdt"] > 0]["pnl_usdt"].sum()) if not sp.empty else 0.0
        pl_sum = abs(float(sp[sp["pnl_usdt"] < 0]["pnl_usdt"].sum())) if not sp.empty else 0.0
        prev_pf = pw_sum / pl_sum if pl_sum > 1e-9 else 0.0

    return {
        "total_capital": float(account.get("total_capital", 0.0)),
        "active_capital": float(account.get("active_capital", 0.0)),
        "daily_pnl_pct": float(account.get("daily_pnl_pct", 0.0)),
        "weekly_pnl_pct": float(account.get("weekly_pnl_pct", 0.0)),
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "realized": realized,
        "open_positions": float(len(positions)),
        "heartbeat_sec": heartbeat_sec,
        "prev_realized": prev_realized,
        "prev_win_rate": prev_win_rate,
        "prev_pf": prev_pf,
    }


def figure_style(fig: go.Figure) -> go.Figure:
    dark = st.session_state.get("dark_mode", False)
    text_color = "#e0ede9" if dark else "#14302b"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Sora", "color": text_color, "size": 12},
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    if dark:
        grid_color = "#2a3d38"
        fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    return fig


def chart_heatmap(scan: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if scan.empty or "symbol" not in scan.columns:
        fig.add_annotation(text="No opportunity map", showarrow=False)
        return figure_style(fig)

    data = scan.copy()
    data["global_score"] = pd.to_numeric(data.get("global_score", 0), errors="coerce").fillna(0)
    data["confidence"] = pd.to_numeric(data.get("confidence", 0), errors="coerce").fillna(0)
    data["signal"] = data.get("signal", "HOLD").fillna("HOLD").astype(str)

    signal_color = {"BUY": "#0b8f5f", "SELL": "#be3e2b", "HOLD": "#8b845d"}
    colors = [signal_color.get(str(s).upper(), "#0f766e") for s in data["signal"]]

    fig = go.Figure(
        go.Treemap(
            labels=data["symbol"],
            parents=["Market"] * len(data),
            values=np.clip(data["global_score"].values + 1.0, 1.0, None),
            marker={"colors": colors, "line": {"color": "rgba(255,255,255,0.7)", "width": 1}},
            customdata=np.stack([data["signal"].values, data["confidence"].values, data["global_score"].values], axis=-1),
            hovertemplate="<b>%{label}</b><br>Signal: %{customdata[0]}<br>Conf: %{customdata[1]:.2f}<br>Score: %{customdata[2]:.2f}<extra></extra>",
            texttemplate="<b>%{label}</b><br>%{customdata[0]}",
            root={"color": "rgba(0,0,0,0)"},
            tiling={"pad": 4},
        )
    )
    fig.update_layout(title="Opportunity Heatmap")
    return figure_style(fig)


@st.cache_data(ttl=120)
def load_btc_benchmark_series(config_path: str) -> pd.Series:
    try:
        cfg = load_config(config_path)
        manager = BinanceDataManager(config=cfg, paper=False)
        _, btc_ret = _pick_btc_benchmark(manager)
        if btc_ret is None:
            return pd.Series(dtype=float)
        return btc_ret.cumsum()
    except Exception:
        return pd.Series(dtype=float)


def chart_equity_drawdown(trades: pd.DataFrame, btc_cum_pnl: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    sell = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    if sell.empty:
        fig.add_annotation(text="No realized trades", showarrow=False)
        return figure_style(fig)

    sell = sell.sort_values("ts")
    sell["cum_pnl"] = sell["pnl_usdt"].cumsum()
    sell["running_max"] = sell["cum_pnl"].cummax()
    sell["drawdown"] = sell["cum_pnl"] - sell["running_max"]

    fig.add_trace(
        go.Scatter(
            x=sell["ts"],
            y=sell["cum_pnl"],
            mode="lines",
            name="Equity Delta",
            line={"width": 3, "color": "#0f766e"},
            fill="tozeroy",
            fillcolor="rgba(15,118,110,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell["ts"],
            y=sell["drawdown"],
            mode="lines",
            name="Drawdown",
            line={"width": 2, "color": "#be3e2b", "dash": "dot"},
            yaxis="y2",
        )
    )

    if btc_cum_pnl is not None and not btc_cum_pnl.empty:
        capital = sell["cum_pnl"].iloc[0] if not sell.empty else 1.0
        scale = abs(capital) if abs(capital) > 1 else 1.0
        fig.add_trace(
            go.Scatter(
                x=btc_cum_pnl.index,
                y=btc_cum_pnl.values * scale,
                mode="lines",
                name="BTC Benchmark",
                line={"width": 1.5, "color": "#f59e0b", "dash": "dash"},
                opacity=0.6,
            )
        )

    fig.update_layout(
        title="Equity & Drawdown",
        yaxis={"title": "USDT"},
        yaxis2={"title": "DD", "overlaying": "y", "side": "right", "showgrid": False},
    )
    return figure_style(fig)


def chart_cycle_flow(cycles: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if cycles.empty:
        fig.add_annotation(text="No cycle history", showarrow=False)
        return figure_style(fig)

    frame = cycles.tail(180)
    fig.add_trace(go.Bar(x=frame["ts"], y=frame["opportunities"], name="Opportunities", marker_color="rgba(15,118,110,0.72)"))
    fig.add_trace(
        go.Scatter(
            x=frame["ts"],
            y=frame["opened_positions"],
            name="Opened",
            mode="lines+markers",
            line={"width": 2.3, "color": "#e56b2f"},
            marker={"size": 5},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Cycle Flow",
        yaxis={"title": "Opp"},
        yaxis2={"title": "Opened", "overlaying": "y", "side": "right", "showgrid": False},
    )
    return figure_style(fig)


def chart_signal_map(scan: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if scan.empty or "confidence" not in scan.columns:
        fig.add_annotation(text="No scanner snapshot", showarrow=False)
        return figure_style(fig)

    data = scan.copy()
    data["signal"] = data.get("signal", "HOLD").fillna("HOLD").astype(str)
    colors = {"BUY": "#0b8f5f", "SELL": "#be3e2b", "HOLD": "#8b845d"}

    for sig in ["BUY", "HOLD", "SELL"]:
        sub = data[data["signal"].str.upper() == sig]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(sub.get("confidence"), errors="coerce"),
                y=pd.to_numeric(sub.get("global_score"), errors="coerce"),
                mode="markers+text",
                text=sub.get("symbol"),
                textposition="top center",
                marker={"size": 11, "color": colors[sig], "opacity": 0.86},
                name=sig,
            )
        )

    fig.update_layout(title="Signal Matrix", xaxis={"title": "Confidence"}, yaxis={"title": "Global Score"})
    return figure_style(fig)


def chart_portfolio_allocation(positions: pd.DataFrame, active_capital: float) -> go.Figure:
    fig = go.Figure()
    if positions.empty or active_capital <= 0:
        fig.add_annotation(text="No open positions", showarrow=False)
        return figure_style(fig)

    data = positions[["symbol", "size_usdt"]].copy()
    data["size_usdt"] = pd.to_numeric(data["size_usdt"], errors="coerce").fillna(0)
    allocated = data["size_usdt"].sum()
    free = max(0.0, active_capital - allocated)

    labels = list(data["symbol"]) + ["Free Capital"]
    values = list(data["size_usdt"]) + [free]
    palette = ["#0f766e", "#14b8a6", "#5eead4", "#99f6e4", "#ccfbf1",
               "#e56b2f", "#f59e0b", "#84cc16"]
    colors = palette[: len(data)] + ["#d6dfdb"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker={"colors": colors[: len(labels)], "line": {"color": "white", "width": 2}},
        textinfo="label+percent",
        textfont={"size": 11, "family": "Sora"},
        hovertemplate="<b>%{label}</b><br>%{value:.2f} USDT<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Portfolio Allocation", showlegend=False)
    return figure_style(fig)


def chart_pnl_distribution(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    sell = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    if sell.empty or "pnl_pct" not in sell.columns:
        fig.add_annotation(text="No closed trades", showarrow=False)
        return figure_style(fig)

    pnl = sell["pnl_pct"].dropna()
    if pnl.empty:
        fig.add_annotation(text="No PnL data", showarrow=False)
        return figure_style(fig)

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    if not wins.empty:
        fig.add_trace(go.Histogram(
            x=wins, name="Wins", marker_color="#0b8f5f", opacity=0.8,
            nbinsx=max(10, len(wins) // 3),
        ))
    if not losses.empty:
        fig.add_trace(go.Histogram(
            x=losses, name="Losses", marker_color="#be3e2b", opacity=0.8,
            nbinsx=max(10, len(losses) // 3),
        ))

    fig.update_layout(
        title="PnL Distribution (%)",
        xaxis={"title": "Return %"},
        yaxis={"title": "Count"},
        barmode="overlay",
    )
    return figure_style(fig)


def render_hero(
    mode: str,
    model_count: int,
    timeframe: str,
    auto_refresh_sec: int,
    kpi: dict[str, float],
    market_data_mode: str,
) -> None:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def vcls(v: float) -> str:
        if v > 0:
            return "pos"
        if v < 0:
            return "neg"
        return ""

    heartbeat = "N/A" if kpi["heartbeat_sec"] < 0 else f"{kpi['heartbeat_sec']:.0f}s"
    wr_arrow = _trend_arrow(kpi["win_rate"], kpi.get("prev_win_rate", kpi["win_rate"]))
    pf_arrow = _trend_arrow(kpi["profit_factor"], kpi.get("prev_pf", kpi["profit_factor"]))
    real_arrow = _trend_arrow(kpi["realized"], kpi.get("prev_realized", kpi["realized"]))

    st.markdown(
        f"""
<div class='hero'>
  <div class='title-row'>
    <div>
      <div class='title-main'>BN-ML Trader Terminal</div>
      <div class='title-sub'>Full-screen trading desk for live operations, scanning flow, and execution telemetry.</div>
    </div>
    <div style='display:flex; gap:.45rem; flex-wrap:wrap;'>
      <span class='chip'>MODE {mode}</span>
      <span class='chip'>MODELS {model_count}</span>
      <span class='chip'>TF {timeframe.upper()}</span>
      <span class='chip'>REFRESH {auto_refresh_sec}s</span>
      <span class='chip'>DATA {market_data_mode.upper()}</span>
      <span class='chip'>{now_utc}</span>
    </div>
  </div>
  <div class='kpi-grid'>
    <div class='kpi-cell'><div class='kpi-label'>Total Capital</div><div class='kpi-value'>{format_money(kpi['total_capital'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Active Capital</div><div class='kpi-value'>{format_money(kpi['active_capital'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Daily PnL</div><div class='kpi-value {vcls(kpi['daily_pnl_pct'])}'>{format_pct(kpi['daily_pnl_pct'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Weekly PnL</div><div class='kpi-value {vcls(kpi['weekly_pnl_pct'])}'>{format_pct(kpi['weekly_pnl_pct'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Win Rate</div><div class='kpi-value'>{format_pct(kpi['win_rate'])}{wr_arrow}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Profit Factor</div><div class='kpi-value'>{kpi['profit_factor']:.2f}{pf_arrow}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Open Positions</div><div class='kpi-value'>{int(kpi['open_positions'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Heartbeat</div><div class='kpi-value'>{heartbeat}</div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_watchlist(scan: pd.DataFrame, top_n: int = 14) -> None:
    st.markdown("<div class='panel-title'>Sticky Watchlist</div>", unsafe_allow_html=True)
    if scan.empty:
        st.info("No opportunities snapshot.")
        return

    frame = scan.sort_values("global_score", ascending=False).head(top_n).copy()
    frame["signal"] = frame.get("signal", "HOLD").astype(str)

    html = []
    for _, row in frame.iterrows():
        sig = str(row.get("signal", "HOLD")).upper()
        if sig not in {"BUY", "SELL", "HOLD"}:
            sig = "HOLD"
        dot = sig.lower()
        symbol = str(row.get("symbol", "N/A"))
        score = float(row.get("global_score", 0.0))
        conf = float(row.get("confidence", 0.0))
        price = pd.to_numeric(row.get("last_price"), errors="coerce")
        chg_24h = pd.to_numeric(row.get("change_24h_pct"), errors="coerce")
        px_txt = f"P {price:.6f}" if pd.notna(price) else ""
        chg_txt = f"24h {chg_24h:+.2f}%" if pd.notna(chg_24h) else ""
        live_suffix = " | ".join([v for v in [px_txt, chg_txt] if v])
        right = f"{sig} | G {score:.1f} | C {conf:.1f}"
        if live_suffix:
            right = f"{right} | {live_suffix}"
        chg_val = float(chg_24h) if pd.notna(chg_24h) else 0.0
        bar_width = min(abs(chg_val) * 8, 60)
        bar_color = "var(--good)" if chg_val >= 0 else "var(--bad)"
        bar_html = f"<div style='height:3px;width:{bar_width:.0f}px;background:{bar_color};border-radius:2px;margin-top:2px;'></div>"
        html.append(
            f"""
<div class='watch-row'>
  <div class='watch-left'><span class='dot {dot}'></span><span class='symbol'>{symbol}</span></div>
  <div class='watch-right'>{right}{bar_html}</div>
</div>
            """.strip()
        )
    st.markdown("\n".join(html), unsafe_allow_html=True)


def render_risk_flags(account: dict[str, Any], risk_config: dict[str, Any] | None = None) -> None:
    st.markdown("<div class='panel-title'>Risk Flags</div>", unsafe_allow_html=True)
    daily = float(account.get("daily_pnl_pct", 0.0))
    weekly = float(account.get("weekly_pnl_pct", 0.0))
    losses = int(account.get("consecutive_losses", 0))

    cb = (risk_config or {}).get("circuit_breakers", {})
    daily_limit = abs(float(cb.get("daily_drawdown_stop_pct", -5.0)))
    weekly_limit = 6.0
    loss_limit = int(cb.get("max_consecutive_losses", 3))

    flags = []
    if daily <= -daily_limit:
        flags.append("Daily drawdown breaker threshold reached.")
    if weekly <= -weekly_limit:
        flags.append("Weekly drawdown breaker threshold reached.")
    if losses >= loss_limit:
        flags.append(f"Consecutive losses: {losses}.")

    if not flags:
        st.success("No active risk breaker flags.")
    else:
        for item in flags:
            st.error(item)

    daily_usage = min(abs(daily) / daily_limit, 1.0) if daily_limit > 0 else 0.0
    weekly_usage = min(abs(weekly) / weekly_limit, 1.0) if weekly_limit > 0 else 0.0
    loss_usage = min(losses / loss_limit, 1.0) if loss_limit > 0 else 0.0

    st.caption("Daily DD")
    st.progress(daily_usage, text=f"{abs(daily):.2f}% / {daily_limit:.0f}%")
    st.caption("Weekly DD")
    st.progress(weekly_usage, text=f"{abs(weekly):.2f}% / {weekly_limit:.0f}%")
    st.caption("Consec. Losses")
    st.progress(loss_usage, text=f"{losses} / {loss_limit}")


def render_open_positions_panel(positions: pd.DataFrame) -> None:
    st.markdown("<div class='panel-title'>Open Position Monitor</div>", unsafe_allow_html=True)
    if positions.empty:
        st.caption("No open positions.")
        return

    mini = _safe_columns(positions, ["symbol", "size_usdt", "remaining_base_qty", "entry_price", "stop_loss"])
    mini["size_usdt"] = pd.to_numeric(mini["size_usdt"], errors="coerce")
    mini["remaining_base_qty"] = pd.to_numeric(mini["remaining_base_qty"], errors="coerce")
    st.dataframe(
        mini,
        width="stretch",
        height=220,
        column_config={
            "size_usdt": st.column_config.NumberColumn("Size", format="%.2f"),
            "remaining_base_qty": st.column_config.NumberColumn("Base Qty", format="%.8f"),
            "entry_price": st.column_config.NumberColumn("Entry", format="%.6f"),
            "stop_loss": st.column_config.NumberColumn("Stop", format="%.6f"),
        },
    )


def render_position_detail_cards(positions: pd.DataFrame, scan: pd.DataFrame) -> None:
    if positions.empty:
        return

    price_map: dict[str, float] = {}
    if not scan.empty and "symbol" in scan.columns and "last_price" in scan.columns:
        for _, row in scan.iterrows():
            sym = str(row.get("symbol", ""))
            px = pd.to_numeric(row.get("last_price"), errors="coerce")
            if sym and pd.notna(px):
                price_map[sym] = float(px)

    now = datetime.now(timezone.utc)
    cards_html: list[str] = []

    for _, pos in positions.iterrows():
        symbol = str(pos.get("symbol", "N/A"))
        entry = float(pos.get("entry_price", 0))
        sl = float(pos.get("stop_loss", 0))
        tp1 = float(pos.get("take_profit_1", 0))
        tp2 = float(pos.get("take_profit_2", 0))
        size = float(pos.get("size_usdt", 0))
        opened_at = pos.get("opened_at")

        extra = pos.get("extra", {})
        if not isinstance(extra, dict):
            extra = _safe_json(pos.get("extra_json", "{}"))
        trailing = bool(extra.get("trailing_active", False))
        tp1_hit = bool(extra.get("tp1_hit", False))
        tp2_hit = bool(extra.get("tp2_hit", False))

        current_price = price_map.get(symbol, np.nan)
        if pd.notna(current_price) and entry > 0:
            unrealized_pct = ((current_price / entry) - 1.0) * 100.0
            dist_sl_pct = ((current_price / sl) - 1.0) * 100.0 if sl > 0 else np.nan
            dist_tp1_pct = ((tp1 / current_price) - 1.0) * 100.0 if tp1 > 0 else np.nan
            dist_tp2_pct = ((tp2 / current_price) - 1.0) * 100.0 if tp2 > 0 else np.nan
        else:
            unrealized_pct = np.nan
            dist_sl_pct = np.nan
            dist_tp1_pct = np.nan
            dist_tp2_pct = np.nan

        if pd.notna(opened_at):
            try:
                dt = opened_at.to_pydatetime() if hasattr(opened_at, "to_pydatetime") else opened_at
                hours_held = (now - dt).total_seconds() / 3600.0
                time_str = f"{hours_held:.1f}h"
            except Exception:
                time_str = "N/A"
        else:
            time_str = "N/A"

        pnl_cls = "pos" if (pd.notna(unrealized_pct) and unrealized_pct > 0) else "neg" if (pd.notna(unrealized_pct) and unrealized_pct < 0) else ""
        pnl_str = f"{unrealized_pct:+.2f}%" if pd.notna(unrealized_pct) else "N/A"
        cur_str = f"{current_price:.6f}" if pd.notna(current_price) else "N/A"
        sl_str = f"{dist_sl_pct:+.2f}%" if pd.notna(dist_sl_pct) else "N/A"
        tp1_str = f"{dist_tp1_pct:+.2f}%" if pd.notna(dist_tp1_pct) else "N/A"
        tp2_str = f"{dist_tp2_pct:+.2f}%" if pd.notna(dist_tp2_pct) else "N/A"

        trail_badge = "<span class='chip' style='background:#14b8a6;color:white;font-size:.6rem;'>TRAIL</span>" if trailing else ""
        tp1_badge = "<span class='chip' style='background:#0b8f5f;color:white;font-size:.6rem;'>TP1</span>" if tp1_hit else ""
        tp2_badge = "<span class='chip' style='background:#0b8f5f;color:white;font-size:.6rem;'>TP2</span>" if tp2_hit else ""

        cards_html.append(f"""
<div class='pos-card'>
  <div class='pos-card-header'>
    <span class='symbol' style='font-size:.82rem;font-weight:600;'>{symbol}</span>
    <span>{trail_badge}{tp1_badge}{tp2_badge}</span>
  </div>
  <div class='pos-card-body'>
    <div class='pos-metric'><span class='pos-label'>Unrealized</span><span class='kpi-value {pnl_cls}' style='font-size:.88rem;'>{pnl_str}</span></div>
    <div class='pos-metric'><span class='pos-label'>Size</span><span>{size:.2f} USDT</span></div>
    <div class='pos-metric'><span class='pos-label'>Entry</span><span>{entry:.6f}</span></div>
    <div class='pos-metric'><span class='pos-label'>Current</span><span>{cur_str}</span></div>
    <div class='pos-metric'><span class='pos-label'>Dist SL</span><span>{sl_str}</span></div>
    <div class='pos-metric'><span class='pos-label'>Dist TP1</span><span>{tp1_str}</span></div>
    <div class='pos-metric'><span class='pos-label'>Dist TP2</span><span>{tp2_str}</span></div>
    <div class='pos-metric'><span class='pos-label'>Held</span><span>{time_str}</span></div>
  </div>
</div>""")

    st.markdown("\n".join(cards_html), unsafe_allow_html=True)


def render_execution_blotter(trades: pd.DataFrame) -> None:
    if trades.empty:
        st.info("No trades recorded.")
        return
    t = trades.sort_values("ts", ascending=False).head(220).copy()
    t["ts"] = t["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    blotter_cols = ["ts", "symbol", "side", "size_usdt", "base_qty", "price", "mode", "reason", "pnl_usdt", "pnl_pct"]
    st.dataframe(
        _safe_columns(t, blotter_cols),
        width="stretch",
        height=360,
        column_config={
            "size_usdt": st.column_config.NumberColumn("USDT", format="%.2f"),
            "base_qty": st.column_config.NumberColumn("Base Qty", format="%.8f"),
            "price": st.column_config.NumberColumn("Price", format="%.6f"),
            "pnl_usdt": st.column_config.NumberColumn("PnL USDT", format="%.2f"),
            "pnl_pct": st.column_config.NumberColumn("PnL %", format="%.2f"),
        },
    )


def render_cycle_feed(cycles: pd.DataFrame) -> None:
    if cycles.empty:
        st.info("No cycle logs yet.")
        return
    c = cycles.sort_values("ts", ascending=False).head(260).copy()
    c["ts"] = c["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cycle_cols = ["ts", "opportunities", "opened_positions", "open_positions", "full_closes", "partial_closes", "daily_pnl_pct"]
    st.dataframe(
        _safe_columns(c, cycle_cols),
        width="stretch",
        height=360,
        column_config={"daily_pnl_pct": st.column_config.NumberColumn("Daily PnL %", format="%.2f")},
    )


def render_opportunity_book(scan: pd.DataFrame) -> None:
    if scan.empty:
        st.info("No scan snapshot yet.")
        return
    s = scan.sort_values("global_score", ascending=False).copy()
    st.dataframe(
        s,
        width="stretch",
        height=360,
        column_config={
            "confidence": st.column_config.ProgressColumn("Conf", min_value=0, max_value=100, format="%.1f"),
            "ml_score": st.column_config.ProgressColumn("ML", min_value=0, max_value=100, format="%.1f"),
            "technical_score": st.column_config.ProgressColumn("Tech", min_value=0, max_value=100, format="%.1f"),
            "momentum_score": st.column_config.ProgressColumn("Mom", min_value=0, max_value=100, format="%.1f"),
            "global_score": st.column_config.ProgressColumn("Global", min_value=0, max_value=100, format="%.1f"),
            "spread_pct": st.column_config.NumberColumn("Spread %", format="%.4f"),
            "last_price": st.column_config.NumberColumn("Last", format="%.6f"),
            "change_24h_pct": st.column_config.NumberColumn("24h %", format="%.2f"),
            "quote_volume_24h": st.column_config.NumberColumn("Quote Vol 24h", format="%.2f"),
        },
    )


def main() -> None:
    draw_css()
    if st.session_state.get("dark_mode", False):
        draw_dark_css()
    settings = load_dashboard_settings()
    _full_cfg = load_config(str(CONFIG_PATH))
    _risk_cfg = _full_cfg.get("risk", {})

    default_full_screen = bool(settings.get("full_screen_default", True))
    default_auto_refresh_sec = int(settings.get("auto_refresh_sec", 5))
    default_timeframe = str(settings.get("timeframe_default", "4h")).lower()
    live_market_enabled = bool(settings.get("live_market_enabled", True))
    scan_stale_sec = int(settings.get("scan_stale_sec", 900))
    if default_timeframe not in TIMEFRAME_WINDOWS:
        default_timeframe = "4h"

    with st.sidebar:
        st.markdown("### Desk Controls")
        full_screen = st.toggle("Full-screen Desk", value=default_full_screen)
        auto_refresh_enabled = st.toggle("Auto Refresh", value=True)
        refresh_sec = st.slider("Refresh Interval (s)", min_value=5, max_value=60, value=default_auto_refresh_sec, step=5)
        timeframe = st.selectbox("Desk Timeframe", options=list(TIMEFRAME_WINDOWS.keys()), index=list(TIMEFRAME_WINDOWS.keys()).index(default_timeframe))
        watchlist_size = st.slider("Watchlist Size", min_value=6, max_value=24, value=14, step=1)
        dark_mode = st.toggle("Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode")
        detached_panels = st.multiselect(
            "Detachable Panels",
            options=ALL_PANELS,
            default=[],
            help="Selected panels move to the detachable workspace section.",
        )
        with st.expander("Keyboard Shortcuts", expanded=False):
            st.markdown(
                "| Key | Action |\n|-----|--------|\n| `R` | Refresh data |"
            )

    apply_fullscreen_mode(enabled=full_screen)

    runtime = load_runtime_data(str(DB_PATH))
    scan = load_scan_data(str(SCAN_PATH))
    scan_age_sec = _scan_age_seconds(str(SCAN_PATH))
    scan_is_stale = scan_age_sec > scan_stale_sec

    live_market = pd.DataFrame()
    if live_market_enabled:
        with st.spinner("Fetching live market data..."):
            try:
                live_market = load_live_market_data(str(CONFIG_PATH))
            except Exception as exc:
                st.toast(f"Live market data unavailable: {exc}", icon="\u26a0\ufe0f")

    scan, market_data_mode = compose_scan_frame(scan=scan, live_market=live_market, scan_is_stale=scan_is_stale)

    account = runtime["account"]
    training_status_raw = runtime.get("training_status", {})
    training_status = training_status_raw if isinstance(training_status_raw, dict) else {}
    trades = runtime["trades"]
    cycles = runtime["cycles"]
    positions = runtime["positions"]

    trades_tf, cycles_tf = filter_by_timeframe(trades=trades, cycles=cycles, timeframe=timeframe)

    # Previous period trades for trend arrows
    prev_trades = pd.DataFrame()
    window = TIMEFRAME_WINDOWS.get(timeframe)
    if window is not None and not trades.empty:
        now_utc_ts = datetime.now(timezone.utc)
        cutoff_current = pd.Timestamp(now_utc_ts - window)
        cutoff_prev = pd.Timestamp(now_utc_ts - window * 2)
        prev_trades = trades[(trades["ts"] >= cutoff_prev) & (trades["ts"] < cutoff_current)]

    mode = runtime_mode(cycles=cycles, account=account, trades=trades)
    model_count = count_model_bundles()
    models_df = load_model_metadata()
    try:
        _btc_bench = load_btc_benchmark_series(str(CONFIG_PATH))
    except Exception:
        _btc_bench = pd.Series(dtype=float)
    kpi = compute_kpi(account=account, trades=trades_tf, positions=positions, cycles=cycles_tf, trades_prev=prev_trades)

    if mode == "PAPER":
        st.warning("Mode PAPER actif: le capital affiché est simulé (pas le wallet Binance réel).")
    elif mode == "LIVE" and not account.get("exchange_synced_at"):
        st.warning("Mode LIVE actif mais capital non synchronisé avec Binance pour le moment.")

    if market_data_mode == "scanner_stale":
        st.warning(
            f"Snapshot scanner stale ({int(scan_age_sec)}s): signaux conservés depuis le dernier scan bot."
        )
    elif market_data_mode == "live_market_fallback":
        st.warning(
            "Aucun snapshot scanner disponible: signaux affichés en fallback live_market (heuristique dashboard)."
        )

    # Risk breaker toasts
    if float(account.get("daily_pnl_pct", 0.0)) <= -5.0:
        st.toast("RISK: Daily drawdown breaker active!", icon="\u26a0\ufe0f")
    if int(account.get("consecutive_losses", 0)) >= 3:
        st.toast("RISK: Consecutive losses breaker active!", icon="\u26a0\ufe0f")

    render_hero(
        mode=mode,
        model_count=model_count,
        timeframe=timeframe,
        auto_refresh_sec=refresh_sec,
        kpi=kpi,
        market_data_mode=market_data_mode,
    )

    if st.button("Refresh Now", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    left, right = st.columns([0.95, 2.25], gap="medium")

    with left:
        st.markdown("<div class='sticky-col'>", unsafe_allow_html=True)
        if "Watchlist" not in detached_panels:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            render_watchlist(scan=scan, top_n=watchlist_size)
            st.markdown("</div>", unsafe_allow_html=True)

        if "Risk Flags" not in detached_panels:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            render_risk_flags(account, risk_config=_risk_cfg)
            st.markdown("</div>", unsafe_allow_html=True)

        if "Open Position Monitor" not in detached_panels:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            render_open_positions_panel(positions)
            render_position_detail_cards(positions, scan)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        chart_renderers = {
            "Opportunity Heatmap": lambda: st.plotly_chart(chart_heatmap(scan), width="stretch"),
            "Equity & Drawdown": lambda: st.plotly_chart(chart_equity_drawdown(trades_tf, btc_cum_pnl=_btc_bench), width="stretch"),
            "Cycle Flow": lambda: st.plotly_chart(chart_cycle_flow(cycles_tf), width="stretch"),
            "Signal Matrix": lambda: st.plotly_chart(chart_signal_map(scan), width="stretch"),
            "Portfolio Allocation": lambda: st.plotly_chart(chart_portfolio_allocation(positions, kpi["active_capital"]), width="stretch"),
            "PnL Distribution": lambda: st.plotly_chart(chart_pnl_distribution(trades_tf), width="stretch"),
        }

        active_charts = [name for name in CHART_PANELS if name not in detached_panels]
        if active_charts:
            for i in range(0, len(active_charts), 2):
                cols = st.columns(2, gap="medium")
                for j, panel_name in enumerate(active_charts[i : i + 2]):
                    with cols[j]:
                        chart_renderers[panel_name]()

        table_renderers = {
            "Execution Blotter": lambda: render_execution_blotter(trades_tf),
            "Cycle Feed": lambda: render_cycle_feed(cycles_tf),
            "Opportunity Book": lambda: render_opportunity_book(scan),
            "Model Performance": lambda: render_model_performance(models_df),
            "Training & Downloads": lambda: render_training_and_downloads(training_status),
        }

        active_tables = [name for name in TABLE_PANELS if name not in detached_panels]
        if active_tables:
            tabs = st.tabs(active_tables)
            for tab, name in zip(tabs, active_tables):
                with tab:
                    table_renderers[name]()

    detached = [name for name in detached_panels if name in ALL_PANELS]
    if detached:
        st.markdown("### Detachable Workspace")
        tabs = st.tabs(detached)
        for tab, name in zip(tabs, detached):
            with tab:
                if name in chart_renderers:
                    chart_renderers[name]()
                elif name == "Watchlist":
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    render_watchlist(scan=scan, top_n=watchlist_size)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif name == "Risk Flags":
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    render_risk_flags(account, risk_config=_risk_cfg)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif name == "Open Position Monitor":
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    render_open_positions_panel(positions)
                    render_position_detail_cards(positions, scan)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif name in table_renderers:
                    table_renderers[name]()

    inject_keyboard_shortcuts()
    schedule_autorefresh(enabled=auto_refresh_enabled, seconds=refresh_sec)


if __name__ == "__main__":
    main()
