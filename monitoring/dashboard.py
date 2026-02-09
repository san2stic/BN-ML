from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bn_ml.config import load_config
from data_manager.fetch_data import BinanceDataManager

DB_PATH = Path("artifacts/state/bn_ml.db")
SCAN_PATH = Path("artifacts/metrics/latest_scan.csv")
MODELS_DIR = Path("models")

TIMEFRAME_WINDOWS = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "1w": pd.Timedelta(days=7),
    "all": None,
}

CHART_PANELS = ["Opportunity Heatmap", "Equity & Drawdown", "Cycle Flow", "Signal Matrix"]
TABLE_PANELS = ["Execution Blotter", "Cycle Feed", "Opportunity Book"]
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


@st.cache_data(ttl=15)
def load_dashboard_settings() -> dict[str, Any]:
    cfg = load_config("configs/bot.yaml")
    return cfg.get("monitoring", {}).get("dashboard", {})


@st.cache_data(ttl=15)
def load_runtime_data(db_path: str) -> dict[str, Any]:
    payload = {"account": {}, "trades": pd.DataFrame(), "cycles": pd.DataFrame(), "positions": pd.DataFrame()}

    path = Path(db_path)
    if not path.exists():
        return payload

    conn = sqlite3.connect(path)
    try:
        if _table_exists(conn, "kv_state"):
            row = conn.execute("SELECT value_json FROM kv_state WHERE key='account_state'").fetchone()
            if row and row[0]:
                payload["account"] = _safe_json(row[0])

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

    aux = load_live_aux_metrics(config_path)
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

@media (max-width: 1020px) { .sticky-col { position: static; } }
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


def filter_by_timeframe(trades: pd.DataFrame, cycles: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = TIMEFRAME_WINDOWS.get(timeframe, None)
    if window is None:
        return trades, cycles

    now = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now - window)

    t = trades[trades["ts"] >= cutoff].copy() if not trades.empty else trades
    c = cycles[cycles["ts"] >= cutoff].copy() if not cycles.empty else cycles
    return t, c


def compute_kpi(account: dict[str, Any], trades: pd.DataFrame, positions: pd.DataFrame, cycles: pd.DataFrame) -> dict[str, float]:
    sell = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    realized = float(sell["pnl_usdt"].sum()) if not sell.empty else 0.0
    wins = float(sell[sell["pnl_usdt"] > 0]["pnl_usdt"].sum()) if not sell.empty else 0.0
    losses = abs(float(sell[sell["pnl_usdt"] < 0]["pnl_usdt"].sum())) if not sell.empty else 0.0
    profit_factor = wins / losses if losses > 1e-9 else (wins if wins > 0 else 0.0)

    heartbeat_sec = -1.0
    if not cycles.empty and pd.notna(cycles["ts"].max()):
        heartbeat_sec = (datetime.now(timezone.utc) - cycles["ts"].max().to_pydatetime()).total_seconds()

    return {
        "total_capital": float(account.get("total_capital", 0.0)),
        "active_capital": float(account.get("active_capital", 0.0)),
        "daily_pnl_pct": float(account.get("daily_pnl_pct", 0.0)),
        "weekly_pnl_pct": float(account.get("weekly_pnl_pct", 0.0)),
        "win_rate": float(account.get("win_rate", 0.0)) * 100,
        "profit_factor": float(profit_factor),
        "realized": realized,
        "open_positions": float(len(positions)),
        "heartbeat_sec": heartbeat_sec,
    }


def figure_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Sora", "color": "#14302b", "size": 12},
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def chart_heatmap(scan: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if scan.empty or "symbol" not in scan.columns:
        fig.add_annotation(text="No opportunity map", showarrow=False)
        return figure_style(fig)

    data = scan.copy()
    data["global_score"] = pd.to_numeric(data.get("global_score", 0), errors="coerce").fillna(0)
    data["confidence"] = pd.to_numeric(data.get("confidence", 0), errors="coerce").fillna(0)
    data["signal"] = data.get("signal", "HOLD").astype(str)

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


def chart_equity_drawdown(trades: pd.DataFrame) -> go.Figure:
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
    data["signal"] = data.get("signal", "HOLD").astype(str)
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
    <div class='kpi-cell'><div class='kpi-label'>Win Rate</div><div class='kpi-value'>{format_pct(kpi['win_rate'])}</div></div>
    <div class='kpi-cell'><div class='kpi-label'>Profit Factor</div><div class='kpi-value'>{kpi['profit_factor']:.2f}</div></div>
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
        html.append(
            f"""
<div class='watch-row'>
  <div class='watch-left'><span class='dot {dot}'></span><span class='symbol'>{symbol}</span></div>
  <div class='watch-right'>{right}</div>
</div>
            """.strip()
        )
    st.markdown("\n".join(html), unsafe_allow_html=True)


def render_risk_flags(account: dict[str, Any]) -> None:
    st.markdown("<div class='panel-title'>Risk Flags</div>", unsafe_allow_html=True)
    daily = float(account.get("daily_pnl_pct", 0.0))
    weekly = float(account.get("weekly_pnl_pct", 0.0))
    losses = int(account.get("consecutive_losses", 0))

    flags = []
    if daily <= -5:
        flags.append("Daily drawdown breaker threshold reached.")
    if weekly <= -6:
        flags.append("Weekly drawdown breaker threshold reached.")
    if losses >= 3:
        flags.append(f"Consecutive losses: {losses}.")

    if not flags:
        st.success("No active risk breaker flags.")
    else:
        for item in flags:
            st.error(item)


def render_open_positions_panel(positions: pd.DataFrame) -> None:
    st.markdown("<div class='panel-title'>Open Position Monitor</div>", unsafe_allow_html=True)
    if positions.empty:
        st.caption("No open positions.")
        return

    mini = positions[["symbol", "size_usdt", "remaining_base_qty", "entry_price", "stop_loss"]].copy()
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


def render_execution_blotter(trades: pd.DataFrame) -> None:
    if trades.empty:
        st.info("No trades recorded.")
        return
    t = trades.sort_values("ts", ascending=False).head(220).copy()
    t["ts"] = t["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(
        t[["ts", "symbol", "side", "size_usdt", "base_qty", "price", "mode", "reason", "pnl_usdt", "pnl_pct"]],
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
    st.dataframe(
        c[["ts", "opportunities", "opened_positions", "open_positions", "full_closes", "partial_closes", "daily_pnl_pct"]],
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
            "confidence": st.column_config.NumberColumn("Conf", format="%.2f"),
            "ml_score": st.column_config.NumberColumn("ML", format="%.2f"),
            "technical_score": st.column_config.NumberColumn("Tech", format="%.2f"),
            "momentum_score": st.column_config.NumberColumn("Mom", format="%.2f"),
            "global_score": st.column_config.NumberColumn("Global", format="%.2f"),
            "spread_pct": st.column_config.NumberColumn("Spread %", format="%.4f"),
            "last_price": st.column_config.NumberColumn("Last", format="%.6f"),
            "change_24h_pct": st.column_config.NumberColumn("24h %", format="%.2f"),
            "quote_volume_24h": st.column_config.NumberColumn("Quote Vol 24h", format="%.2f"),
        },
    )


def main() -> None:
    draw_css()
    settings = load_dashboard_settings()

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
        detached_panels = st.multiselect(
            "Detachable Panels",
            options=ALL_PANELS,
            default=[],
            help="Selected panels move to the detachable workspace section.",
        )

    apply_fullscreen_mode(enabled=full_screen)

    runtime = load_runtime_data(str(DB_PATH))
    scan = load_scan_data(str(SCAN_PATH))
    scan_age_sec = _scan_age_seconds(str(SCAN_PATH))
    scan_is_stale = scan_age_sec > scan_stale_sec

    live_market = pd.DataFrame()
    if live_market_enabled:
        live_market = load_live_market_data("configs/bot.yaml")

    scan, market_data_mode = compose_scan_frame(scan=scan, live_market=live_market, scan_is_stale=scan_is_stale)

    account = runtime["account"]
    trades = runtime["trades"]
    cycles = runtime["cycles"]
    positions = runtime["positions"]

    trades_tf, cycles_tf = filter_by_timeframe(trades=trades, cycles=cycles, timeframe=timeframe)

    mode = runtime_mode(cycles=cycles, account=account, trades=trades)
    model_count = count_model_bundles()
    kpi = compute_kpi(account=account, trades=trades_tf, positions=positions, cycles=cycles_tf)

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
            render_risk_flags(account)
            st.markdown("</div>", unsafe_allow_html=True)

        if "Open Position Monitor" not in detached_panels:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            render_open_positions_panel(positions)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        chart_renderers = {
            "Opportunity Heatmap": lambda: st.plotly_chart(chart_heatmap(scan), use_container_width=True),
            "Equity & Drawdown": lambda: st.plotly_chart(chart_equity_drawdown(trades_tf), use_container_width=True),
            "Cycle Flow": lambda: st.plotly_chart(chart_cycle_flow(cycles_tf), use_container_width=True),
            "Signal Matrix": lambda: st.plotly_chart(chart_signal_map(scan), use_container_width=True),
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
                    render_risk_flags(account)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif name == "Open Position Monitor":
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    render_open_positions_panel(positions)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif name in table_renderers:
                    table_renderers[name]()

    schedule_autorefresh(enabled=auto_refresh_enabled, seconds=refresh_sec)


if __name__ == "__main__":
    main()
