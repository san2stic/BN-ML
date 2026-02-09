const connectionStatus = document.getElementById("connection-status");
const snapshotTime = document.getElementById("snapshot-time");
const accountCapital = document.getElementById("account-capital");
const buyCount = document.getElementById("buy-count");
const holdCount = document.getElementById("hold-count");
const sellCount = document.getElementById("sell-count");
const tableBody = document.getElementById("predictions-body");

const LIMIT = 20;
let socket = null;
let reconnectTimer = null;

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

function scheduleReconnect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }
  reconnectTimer = setTimeout(connectSocket, 3000);
}

function connectSocket() {
  if (socket && socket.readyState <= 1) {
    return;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws/predictions?limit=${LIMIT}`);

  socket.addEventListener("open", () => {
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
    setDisconnected();
    await refreshPredictionsFallback();
    scheduleReconnect();
  });

  socket.addEventListener("error", async () => {
    setDisconnected();
    await refreshPredictionsFallback();
  });
}

refreshAccount();
refreshPredictionsFallback();
connectSocket();
setInterval(refreshAccount, 10000);
setInterval(refreshPredictionsFallback, 15000);

