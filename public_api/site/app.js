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

const LIMIT = 20;
let socket = null;
let reconnectTimer = null;
let reconnectAttempts = 0;
let websocketDisabled = false;
const MAX_WS_RETRIES = 4;

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

function renderRows(rows) {
  const counters = { BUY: 0, HOLD: 0, SELL: 0 };
  rows.forEach((row) => {
    const signal = String(row.signal || "HOLD").toUpperCase();
    counters[signal] = (counters[signal] || 0) + 1;
  });

  buyCount.textContent = String(counters.BUY || 0);
  holdCount.textContent = String(counters.HOLD || 0);
  sellCount.textContent = String(counters.SELL || 0);

  tableBody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const signal = String(row.signal || "HOLD").toUpperCase();
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

modelsBundleSelect.addEventListener("change", () => {
  setSelectedDownloadLink(modelsBundleSelect.value);
});

refreshAccount();
refreshPredictionsFallback();
refreshTrainingStatus();
refreshModelBundles();
connectSocket();
setInterval(refreshAccount, 10000);
setInterval(refreshPredictionsFallback, 15000);
setInterval(refreshTrainingStatus, 4000);
setInterval(refreshModelBundles, 15000);
