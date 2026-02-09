# Docker Deployment

## Prerequisites
- Docker + Docker Compose v2
- `.env` configuré (`BINANCE_API_KEY`, `BINANCE_API_SECRET`) si mode live
- `.env` configuré avec `RUNPOD_API_KEY` si `model-sync-runpod` est activé
- `configs/bot.yaml` valide

## Stack incluse
- `bot-paper` ou `bot-live` (runtime trading)
- `trainer-auto` (retrain périodique des modèles, conteneur dédié)
- `model-sync-runpod` (trigger endpoint RunPod + pull modèles quotidien)
- `dashboard` (Streamlit Trader Terminal)
- `api` (site public + API temps réel + `/metrics`)
- `prometheus` (scrape monitoring)
- `grafana` (visualisation)

Ports exposés:
- API + site public: `8000`
- Dashboard Streamlit: `8501`
- Prometheus: `9090`
- Grafana: `3000` (login par défaut `admin` / `admin`)

## Build
```bash
docker compose build
```

## CI Build & Publish (GHCR)
Workflow: `.github/workflows/publish-docker-ghcr.yml`

Publication automatique vers:
- `ghcr.io/<owner>/bn-ml-trading-bot:latest` (push sur `main`)
- `ghcr.io/<owner>/bn-ml-trading-bot:vX.Y.Z` (tag/release)
- `ghcr.io/<owner>/bn-ml-trading-bot:sha-...` (tag commit)

Validation CI:
- en pull request, l’image est buildée (sans push)

## Start Paper Stack
```bash
docker compose --profile paper up -d bot-paper model-sync-runpod dashboard api prometheus grafana
```

## Start Live Stack
```bash
docker compose --profile live up -d bot-live model-sync-runpod dashboard api prometheus grafana
```

Important:
- `bot-paper` / `bot-live` démarrent avec `--disable-retrain`.
- en mode RunPod, la mise à jour des modèles est assurée par `model-sync-runpod` (daemon quotidien `--role runpod_client`).
- si tu n'utilises pas RunPod, démarre `trainer-auto` à la place de `model-sync-runpod`.

## URLs
- Site public/API: `http://localhost:8000`
- Docs OpenAPI: `http://localhost:8000/docs`
- WebSocket prédictions: `ws://localhost:8000/ws/predictions`
- Dashboard trading: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Ops services (on demand)
Trainer:
```bash
docker compose --profile ops run --rm trainer
```

Rapport DoD:
```bash
docker compose --profile ops run --rm report
```

## Runtime persistence
Le compose persiste automatiquement:
- `artifacts/` via volume `bnml_artifacts`
- `models/` via volume `bnml_models`
- données Prometheus/Grafana via `prometheus_data` et `grafana_data`

Au démarrage, le service `volume-init` applique automatiquement `chown 10001:10001` sur `artifacts/` et `models/` pour éviter les erreurs `PermissionError` quand le bot tourne en `appuser`.

## Monitoring metrics (Prometheus)
Métriques exposées via `api:/metrics`:
- `bnml_scan_rows`, `bnml_scan_age_seconds`, `bnml_scan_signal_count`
- `bnml_account_total_capital`, `bnml_account_active_capital`
- `bnml_account_daily_pnl_pct`, `bnml_account_weekly_pnl_pct`
- `bnml_open_positions`, `bnml_total_trades`, `bnml_total_cycles`
- `bnml_market_drift_detected`

## Stop
```bash
docker compose down
```

Stop + suppression volumes:
```bash
docker compose down -v
```

## Troubleshooting
- API non accessible:
  - `docker compose logs -f api`
  - vérifier `docker compose ps`
- `PermissionError: [Errno 13] ... artifacts/...`:
  - relancer la stack (le service `volume-init` corrige les droits):
    - `docker compose down`
    - `docker compose --profile paper up -d bot-paper model-sync-runpod dashboard api prometheus grafana`
  - si besoin, correction manuelle one-shot:
    - `docker compose run --rm --user root bot-paper sh -lc "mkdir -p /app/artifacts /app/models && chown -R 10001:10001 /app/artifacts /app/models"`
- Dashboard vide:
  - vérifier que `bot-paper`/`bot-live` écrit bien `artifacts/metrics/latest_scan.csv`
- Grafana sans datasource:
  - redémarrer `grafana` après `prometheus`
  - vérifier les mounts `monitoring/grafana/provisioning/*`
