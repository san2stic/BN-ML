const connectionStatus = document.getElementById("connection-status");
const snapshotTime = document.getElementById("snapshot-time");
const accountCapital = document.getElementById("account-capital");
const buyCount = document.getElementById("buy-count");
const holdCount = document.getElementById("hold-count");
const sellCount = document.getElementById("sell-count");
const tableBody = document.getElementById("predictions-body");
const trainingStatus = document.getElementById("training-status");
const trainingCurrentSymbol = document.getElementById("training-current-symbol");
const trainingMeta = document.getElementById("training-meta");
const trainingProgressBar = document.getElementById("training-progress-bar");
const trainingProgressText = document.getElementById("training-progress-text");
const trainingQueued = document.getElementById("training-queued");
const trainingCompleted = document.getElementById("training-completed");
const trainingTrained = document.getElementById("training-trained");
const trainingErrors = document.getElementById("training-errors");
const modelsBundleCount = document.getElementById("models-bundle-count");
const modelsBundleSelect = document.getElementById("models-bundle-select");
const downloadAllModels = document.getElementById("download-all-models");
const downloadSelectedModel = document.getElementById("download-selected-model");
const marketIndexValue = document.getElementById("market-index-value");
const marketIndexMeta = document.getElementById("market-index-meta");
const marketIndexBenchmark = document.getElementById("market-index-benchmark");
const marketIndexUpdatedAt = document.getElementById("market-index-updated-at");
const marketIndexLine = document.getElementById("market-index-line");

const LIMIT = 20;
const MARKET_HISTORY_LIMIT = 240;
const MAX_WS_RETRIES = 4;

let socket = null;
let reconnectTimer = null;
let reconnectAttempts = 0;
let websocketDisabled = false;

let marketSocket = null;
let marketReconnectTimer = null;
let marketReconnectAttempts = 0;
let marketWebsocketDisabled = false;
let marketSeries = [];

function fmtNumber(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0";
  return num.toFixed(digits);
}

function fmtMoney(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "n/a";
  return `${num.toFixed(2)} USDT`;
}

function fmtInt(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0";
  return String(Math.max(0, Math.trunc(num)));
}

function normalizeSignal(value) {
  const signal = String(value || "HOLD").toUpperCase();
  if (signal === "BUY" || signal === "SELL" || signal === "HOLD") {
    return signal;
  }
  return "HOLD";
}

function renderRows(rows) {
  const counters = { BUY: 0, HOLD: 0, SELL: 0 };
  rows.forEach((row) => {
    const signal = normalizeSignal(row.signal);
    counters[signal] = (counters[signal] || 0) + 1;
  });

  buyCount.textContent = String(counters.BUY || 0);
  holdCount.textContent = String(counters.HOLD || 0);
  sellCount.textContent = String(counters.SELL || 0);

  tableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const signal = normalizeSignal(row.signal);
    tr.innerHTML = `
      <td>${row.symbol || "-"}</td>
      <td><span class="signal-pill ${signal}">${signal}</span></td>
      <td>${fmtNumber(row.confidence)}</td>
      <td>${fmtNumber(row.global_score)}</td>
      <td>${fmtNumber(row.spread_pct, 4)}</td>
      <td>${fmtNumber(row.depth_usdt, 0)}</td>
    `;
    tableBody.appendChild(tr);
  });
}

function setDisconnected() {
  connectionStatus.textContent = "disconnected";
  connectionStatus.style.color = "#ff6a76";
}

function setConnected() {
  connectionStatus.textContent = "live websocket";
  connectionStatus.style.color = "#2fe3a5";
}

function setPolling() {
  connectionStatus.textContent = "http polling";
  connectionStatus.style.color = "#f5c86a";
}

async function refreshAccount() {
  try {
    const res = await fetch("/api/v1/account");
    if (!res.ok) return;
    const payload = await res.json();
    accountCapital.textContent = fmtMoney(payload.total_capital);
  } catch (err) {
    accountCapital.textContent = "n/a";
  }
}

async function refreshPredictionsFallback() {
  try {
    const res = await fetch(`/api/v1/predictions?limit=${LIMIT}`);
    if (!res.ok) return;
    const payload = await res.json();
    renderRows(payload.rows || []);
    snapshotTime.textContent = payload.generated_at || "n/a";
  } catch (err) {
    setDisconnected();
  }
}

function setTrainingStatus(status) {
  const text = String(status || "idle").toLowerCase();
  trainingStatus.textContent = text;
  if (text === "running") {
    trainingStatus.style.color = "#2fe3a5";
    return;
  }
  if (text === "completed") {
    trainingStatus.style.color = "#7bd0ff";
    return;
  }
  if (text === "failed") {
    trainingStatus.style.color = "#ff6a76";
    return;
  }
  trainingStatus.style.color = "#9eb7e8";
}

async function refreshTrainingStatus() {
  try {
    const ts = Date.now();
    const res = await fetch(`/api/v1/training/status?ts=${ts}`, { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();

    setTrainingStatus(payload.status);
    trainingCurrentSymbol.textContent = `symbol: ${payload.current_symbol || "-"}`;
    trainingMeta.textContent = `phase: ${payload.phase || "waiting"} | trigger: ${payload.trigger || "n/a"}`;

    const progress = Number(payload.progress_pct);
    const bounded = Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 0;
    trainingProgressBar.style.width = `${bounded.toFixed(1)}%`;
    trainingProgressText.textContent = `${bounded.toFixed(1)}%`;

    trainingQueued.textContent = fmtInt(payload.symbols_queued);
    trainingCompleted.textContent = fmtInt(payload.symbols_completed);
    trainingTrained.textContent = fmtInt(payload.symbols_trained);
    trainingErrors.textContent = fmtInt(payload.symbols_errors);
  } catch (err) {
    setTrainingStatus("unavailable");
  }
}

function setSelectedDownloadLink(modelKey) {
  const key = String(modelKey || "").trim();
  if (!key) {
    downloadSelectedModel.href = "#";
    downloadSelectedModel.classList.add("disabled");
    downloadSelectedModel.setAttribute("aria-disabled", "true");
    return;
  }
  downloadSelectedModel.href = `/api/v1/models/download?model_key=${encodeURIComponent(key)}`;
  downloadSelectedModel.classList.remove("disabled");
  downloadSelectedModel.setAttribute("aria-disabled", "false");
}

async function refreshModelBundles() {
  try {
    const res = await fetch("/api/v1/models");
    if (!res.ok) return;
    const payload = await res.json();
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    modelsBundleCount.textContent = String(rows.length);
    downloadAllModels.href = "/api/v1/models/download";

    const previousValue = modelsBundleSelect.value;
    modelsBundleSelect.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = rows.length > 0 ? "Select a bundle..." : "No bundle available";
    modelsBundleSelect.appendChild(placeholder);

    rows.forEach((row) => {
      const key = String(row.model_key || "").trim();
      if (!key) return;
      const symbol = String(row.symbol || key);
      const option = document.createElement("option");
      option.value = key;
      option.textContent = `${symbol} (${key})`;
      modelsBundleSelect.appendChild(option);
    });

    if (previousValue && rows.some((row) => String(row.model_key || "") === previousValue)) {
      modelsBundleSelect.value = previousValue;
    } else {
      modelsBundleSelect.value = "";
    }

    setSelectedDownloadLink(modelsBundleSelect.value);
  } catch (err) {
    modelsBundleCount.textContent = "0";
    setSelectedDownloadLink("");
  }
}

function normalizeMarketPoint(point) {
  if (!point) return null;
  const ts = point.ts || point.generated_at;
  if (!ts) return null;

  const indexValue = Number(point.index_value);
  if (!Number.isFinite(indexValue)) return null;

  const maybeScore = Number(point.market_score);
  const marketScore = Number.isFinite(maybeScore) ? maybeScore : (indexValue / 50.0) - 1.0;
  const signal = normalizeSignal(point.signal);

  return {
    ts: String(ts),
    index_value: Math.max(0, Math.min(100, indexValue)),
    market_score: Math.max(-1, Math.min(1, marketScore)),
    confidence: Math.max(0, Math.min(100, Number(point.confidence) || 0)),
    signal,
    market_regime: String(point.market_regime || "unknown"),
    profile: String(point.profile || "neutral"),
  };
}

function upsertMarketPoint(point) {
  const normalized = normalizeMarketPoint(point);
  if (!normalized) return;
  const last = marketSeries.length ? marketSeries[marketSeries.length - 1] : null;
  if (last && last.ts === normalized.ts) {
    marketSeries[marketSeries.length - 1] = normalized;
  } else {
    marketSeries.push(normalized);
  }
  if (marketSeries.length > MARKET_HISTORY_LIMIT) {
    marketSeries = marketSeries.slice(-MARKET_HISTORY_LIMIT);
  }
}

function setMarketTone(signal) {
  const normalized = normalizeSignal(signal);
  if (normalized === "BUY") {
    marketIndexValue.style.color = "#2fe3a5";
    return;
  }
  if (normalized === "SELL") {
    marketIndexValue.style.color = "#ff6a76";
    return;
  }
  marketIndexValue.style.color = "#ffca66";
}

function renderMarketChart() {
  if (!marketIndexLine) return;
  if (!marketSeries.length) {
    marketIndexLine.setAttribute("points", "");
    return;
  }

  const width = 800;
  const height = 220;
  if (marketSeries.length === 1) {
    const y = height - (marketSeries[0].index_value / 100) * height;
    marketIndexLine.setAttribute("points", `0,${y.toFixed(2)} ${width},${y.toFixed(2)}`);
  } else {
    const denominator = Math.max(1, marketSeries.length - 1);
    const points = marketSeries.map((item, idx) => {
      const x = (idx / denominator) * width;
      const y = height - (item.index_value / 100) * height;
      return `${x.toFixed(2)},${Math.max(0, Math.min(height, y)).toFixed(2)}`;
    });
    marketIndexLine.setAttribute("points", points.join(" "));
  }

  const latest = marketSeries[marketSeries.length - 1];
  if (latest.signal === "SELL") {
    marketIndexLine.style.stroke = "#ff6a76";
  } else if (latest.signal === "BUY") {
    marketIndexLine.style.stroke = "#2fe3a5";
  } else {
    marketIndexLine.style.stroke = "#ffca66";
  }
}

function applyMarketLatest(latest) {
  if (!latest) return;
  const signal = normalizeSignal(latest.signal);
  const indexValue = Number(latest.index_value);
  const confidence = Number(latest.confidence);
  const benchmarkSymbol = String(latest.benchmark_symbol || "-");
  const benchmarkPrice = Number(latest.benchmark_price);

  marketIndexValue.textContent = Number.isFinite(indexValue) ? fmtNumber(indexValue, 1) : "50.0";
  marketIndexMeta.textContent = `signal: ${signal} | regime: ${latest.market_regime || "unknown"} | profile: ${latest.profile || "neutral"}`;
  marketIndexBenchmark.textContent = Number.isFinite(benchmarkPrice) && benchmarkPrice > 0
    ? `benchmark: ${benchmarkSymbol} ${fmtNumber(benchmarkPrice, 2)}`
    : `benchmark: ${benchmarkSymbol}`;
  marketIndexUpdatedAt.textContent = `updated: ${latest.generated_at || "n/a"} | confidence: ${
    Number.isFinite(confidence) ? fmtNumber(confidence, 1) : "0.0"
  }%`;

  setMarketTone(signal);
}

function processMarketPayload(payload) {
  if (!payload) return;
  const rows = Array.isArray(payload.rows) ? payload.rows : [];
  if (rows.length) {
    marketSeries = [];
    rows.forEach((row) => upsertMarketPoint(row));
  }

  const latest = payload.latest || null;
  if (latest) {
    applyMarketLatest(latest);
    upsertMarketPoint({
      ts: latest.generated_at,
      index_value: latest.index_value,
      market_score: latest.market_score,
      confidence: latest.confidence,
      signal: latest.signal,
      market_regime: latest.market_regime,
      profile: latest.profile,
    });
  }

  renderMarketChart();
}

async function refreshMarketIndex() {
  try {
    const ts = Date.now();
    const res = await fetch(`/api/v1/market/index/history?limit=${MARKET_HISTORY_LIMIT}&ts=${ts}`, { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    processMarketPayload(payload);
  } catch (err) {
    marketIndexUpdatedAt.textContent = "updated: n/a";
  }
}

function scheduleReconnect() {
  if (websocketDisabled) {
    return;
  }
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }
  const delay = Math.min(15000, 1500 * Math.max(1, reconnectAttempts));
  reconnectTimer = setTimeout(connectSocket, delay);
}

function connectSocket() {
  if (websocketDisabled) {
    return;
  }

  if (!("WebSocket" in window)) {
    websocketDisabled = true;
    setPolling();
    return;
  }

  if (socket && socket.readyState <= 1) {
    return;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/predictions?limit=${LIMIT}`;
  socket = new WebSocket(wsUrl);

  socket.addEventListener("open", () => {
    reconnectAttempts = 0;
    setConnected();
  });

  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      const payload = message.payload || {};
      renderRows(payload.rows || []);
      snapshotTime.textContent = payload.generated_at || "n/a";
    } catch (err) {
      console.error("Invalid websocket payload", err);
    }
  });

  socket.addEventListener("close", async () => {
    reconnectAttempts += 1;
    setDisconnected();
    await refreshPredictionsFallback();
    if (reconnectAttempts >= MAX_WS_RETRIES) {
      websocketDisabled = true;
      setPolling();
      return;
    }
    scheduleReconnect();
  });

  socket.addEventListener("error", async () => {
    reconnectAttempts += 1;
    setDisconnected();
    await refreshPredictionsFallback();
    if (reconnectAttempts >= MAX_WS_RETRIES) {
      websocketDisabled = true;
      setPolling();
      try {
        socket.close();
      } catch (err) {
        // noop
      }
      return;
    }
    scheduleReconnect();
  });
}

function scheduleMarketReconnect() {
  if (marketWebsocketDisabled) {
    return;
  }
  if (marketReconnectTimer) {
    clearTimeout(marketReconnectTimer);
  }
  const delay = Math.min(15000, 1500 * Math.max(1, marketReconnectAttempts));
  marketReconnectTimer = setTimeout(connectMarketSocket, delay);
}

function connectMarketSocket() {
  if (marketWebsocketDisabled) {
    return;
  }

  if (!("WebSocket" in window)) {
    marketWebsocketDisabled = true;
    return;
  }

  if (marketSocket && marketSocket.readyState <= 1) {
    return;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}/ws/market/intelligence?limit=${MARKET_HISTORY_LIMIT}`;
  marketSocket = new WebSocket(wsUrl);

  marketSocket.addEventListener("open", () => {
    marketReconnectAttempts = 0;
  });

  marketSocket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type !== "market_intelligence") return;
      processMarketPayload(message.payload || {});
    } catch (err) {
      console.error("Invalid market websocket payload", err);
    }
  });

  marketSocket.addEventListener("close", async () => {
    marketReconnectAttempts += 1;
    await refreshMarketIndex();
    if (marketReconnectAttempts >= MAX_WS_RETRIES) {
      marketWebsocketDisabled = true;
      return;
    }
    scheduleMarketReconnect();
  });

  marketSocket.addEventListener("error", async () => {
    marketReconnectAttempts += 1;
    await refreshMarketIndex();
    if (marketReconnectAttempts >= MAX_WS_RETRIES) {
      marketWebsocketDisabled = true;
      try {
        marketSocket.close();
      } catch (err) {
        // noop
      }
      return;
    }
    scheduleMarketReconnect();
  });
}

modelsBundleSelect.addEventListener("change", () => {
  setSelectedDownloadLink(modelsBundleSelect.value);
});

refreshAccount();
refreshPredictionsFallback();
refreshTrainingStatus();
refreshModelBundles();
refreshMarketIndex();
connectSocket();
connectMarketSocket();

setInterval(refreshAccount, 10000);
setInterval(refreshPredictionsFallback, 15000);
setInterval(refreshTrainingStatus, 4000);
setInterval(refreshModelBundles, 15000);
setInterval(refreshMarketIndex, 15000);
