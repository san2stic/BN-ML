# BN-ML - Binance Spot ML Trading Bot

[![CI](https://github.com/san2stic/BN-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/san2stic/BN-ML/actions/workflows/ci.yml)
[![Security](https://github.com/san2stic/BN-ML/actions/workflows/security.yml/badge.svg)](https://github.com/san2stic/BN-ML/actions/workflows/security.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Bot de trading Binance Spot orienté production avec:
- entraînement ML par symbole
- hyperparameter optimization walk-forward (objectif Sharpe/Sortino/Accuracy)
- scanner multi-paires
- risk/money management strict
- dashboard web "Trader Terminal" auto-lançable avec le bot

## 1) Prérequis

- Python 3.10+
- Compte Binance Spot (API key + secret)
- macOS (M1) ou Linux (Ubuntu recommandé pour CUDA)

Installation:

```bash
# Option 1 (PyPI, install par nom)
python3 -m venv .venv
source .venv/bin/activate
pip install bn-ml-trading-bot

# Option 2 (auto-install depuis GitHub)
python3 -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/san2stic/BN-ML.git"

# Option 3 (dev local depuis le repo cloné)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Publication PyPI automatique: voir `docs/pypi_trusted_publisher.md`.

## 2) Configuration

Fichier principal: `configs/bot.yaml`

Variables d'environnement (dans `.env` ou shell):

```bash
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
BINANCE_TESTNET=false
```

Modes:
- `environment: paper` pour simulation
- `environment: live` pour réel
- en live, `exchange.testnet` doit être `false`
- `risk.capital_total: auto` pour utiliser automatiquement le capital réel Binance (authentifié) en mode live
- `data.paper_market_data_mode: live` pour paper trading sur données de marché Binance réelles (fallback synthétique si indisponible)

Universe dynamique:
- `universe.dynamic_base_quote_pairs: true` pour scanner toutes les paires `*/base_quote` (ex: `*/USDC`)
- `universe.max_pairs_scanned` pour limiter la charge/rate limit
- `universe.train_dynamic_pairs: true` pour entraîner par symbole sur cet univers
- `universe.train_max_pairs` pour limiter le nombre de modèles entraînés
- `universe.train_missing_only: true` pour n'entraîner que les modèles manquants/anciens
- `universe.model_max_age_hours: 24` seuil d'ancienneté modèle avant retrain

Mode liste utilisateur (trader uniquement des paires choisies):
- `universe.user_selected_only: true` active le mode restreint
- `universe.user_selected_pairs: [BTC/USDC, ETH/USDC, ...]` liste prioritaire des paires autorisées
- fallback compatibilité: si `user_selected_pairs` est vide, le bot utilise `universe.pairs`
- en mode restreint, l'univers dynamique est ignoré pour le scan et l'entraînement

Auto-retrain en fond (thread séparé):
- `model.auto_retrain_enabled: true` active le worker d'entraînement asynchrone
- `model.retrain_check_interval_sec: 30` fréquence de vérification (`x`) des conditions de retrain
- `model.retrain_interval_hours: 6` intervalle minimal entre retrains complets
- `model.auto_train_missing_models: true` entraîne automatiquement les bundles manquants
- `model.auto_train_missing_batch_size: 5` limite les auto-trains manquants par batch
- `model.auto_train_missing_disable_hpo: true` accélère l'auto-train manquant (désactive HPO sur ce flux)

Monitoring et alerting runtime:
- `monitoring.realtime_prices.enabled: true` active le flux websocket Binance pour suivi prix temps réel (positions + paires scannées)
- `monitoring.alerting.webhook_url` pour webhook externe
- `monitoring.alerting.telegram_bot_token` + `monitoring.alerting.telegram_chat_id` pour Telegram
- `monitoring.alerting.email.*` pour SMTP email
- `model.drift.*` active le monitoring runtime de drift régime (KS + ratio de volatilité)
- `risk.circuit_breakers.drift_block_enabled: true` bloque les nouvelles entrées si drift détecté

Backups automatiques:
- `storage.backup.enabled: true`
- `storage.backup.interval_minutes: 60`
- `storage.backup.keep_last: 48`
- `storage.backup.base_dir: artifacts/backups`

Analyse multi-timeframe (MTF):
- `model.multi_timeframe.enabled: true` active la fusion multi-timeframe
- `model.multi_timeframe.base_timeframe: 15m` timeframe principal
- `model.multi_timeframe.timeframes: [1h, 4h, 1d]` timeframes supérieurs fusionnés
- `model.multi_timeframe.feature_columns` features HTF à projeter sur le timeframe de base
- `model.multi_timeframe.min_candles_per_timeframe` / `max_candles_per_timeframe` / `extra_candles_buffer` pour contrôler la profondeur de chaque HTF
- les signaux de confluence sont ajoutés automatiquement (`mtf_trend_consensus`, `mtf_macd_consensus`, `mtf_confluence_score`, etc.)
- `model.lstm.enabled: true` active un modèle séquentiel entraîné et inféré dans l’ensemble (backend `sequence_mlp`, compatible joblib)

## 3) Commandes principales

Si tu as installé le package via `pip install bn-ml-trading-bot` (ou `pip install git+...`), utilise les commandes `bnml-*`.
Si tu exécutes depuis le repo cloné, les commandes `python3 -m scripts.*` restent disponibles.

Entraîner les modèles par symbole:

```bash
bnml-trainer
```

Entraînement incrémental (manquants/anciens uniquement):

```bash
bnml-trainer --train-missing-only
```

Boucle d'entraînement automatique (conteneur/process dédié):

```bash
bnml-trainer-auto --train-missing-only
```

Forcer entraînement complet:

```bash
bnml-trainer --train-all
```

Entraîner seulement certains symboles:

```bash
bnml-trainer --symbol BTC/USDC --symbol ETH/USDC
```

Lancer un cycle unique bot (paper):

```bash
bnml-bot --once --paper
```

Campagne DoD paper 30 jours (checks quotidiens + rapport final):

```bash
bnml-dod-30d --days 30 --disable-retrain
```

Lancer en live continu:

```bash
bnml-bot --live
```

Le dashboard est auto-lancé en mode continu si `monitoring.dashboard.auto_launch_with_bot: true`.

Lancer sans dashboard:

```bash
bnml-bot --live --no-dashboard
```

Lancer dashboard seul:

```bash
streamlit run monitoring/dashboard.py
```

Backtest baseline:

```bash
bnml-backtest --paper
```

Kill switch (fermeture d'urgence):

```bash
bnml-kill-switch
```

Check quotidien DoD:

```bash
bnml-dod-check --fail-on-violation
```

Rapport synthese DoD:

```bash
python3 -m scripts.generate_dod_report --days 30
```

API publique (site + endpoints + websocket):

```bash
bnml-api
```

## 4) Docker Full Stack (bot + monitoring + site public)

Build:

```bash
docker compose build
```

Paper stack complète:

```bash
docker compose --profile paper up -d bot-paper trainer-auto dashboard api prometheus grafana
```

Live stack complète:

```bash
docker compose --profile live up -d bot-live trainer-auto dashboard api prometheus grafana
```

Exposer le dashboard via Cloudflare Tunnel (URL publique dans les logs):

```bash
docker compose --profile paper --profile tunnel up -d bot-paper trainer-auto dashboard api prometheus grafana cloudflared
docker compose logs -f cloudflared
```

Note Docker: `bot-paper`/`bot-live` sont lancés avec `--disable-retrain`; le retrain est délégué au conteneur `trainer-auto`.

Endpoints:
- site public + API: `http://localhost:8000`
- docs API: `http://localhost:8000/docs`
- websocket predictions: `ws://localhost:8000/ws/predictions`
- dashboard Trader Terminal: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Détails ops/volumes/troubleshooting: `docs/deployment_docker.md`.

## 5) Dashboard "Trader Terminal"

Le dashboard inclut:
- mode full-screen desk
- auto-refresh (5s par défaut)
- filtre timeframe (`15m`, `1h`, `4h`, `1d`, `1w`, `all`)
- panneaux détachables (workspace par onglets)
- watchlist sticky, heatmap opportunités, equity/drawdown, blotter, cycle feed
- monitor live des prix via websocket quand activé

Paramètres dans `configs/bot.yaml`:
- `monitoring.dashboard.auto_launch_with_bot`
- `monitoring.dashboard.address`
- `monitoring.dashboard.port`
- `monitoring.dashboard.auto_refresh_sec`
- `monitoring.dashboard.full_screen_default`
- `monitoring.dashboard.timeframe_default`
- `monitoring.dashboard.live_market_enabled`
- `monitoring.dashboard.scan_stale_sec`

## 6) Modèles ML par crypto

Chaque symbole a son bundle dédié:
- `models/<SYMBOL_KEY>/rf.joblib`
- `models/<SYMBOL_KEY>/xgb.joblib`
- `models/<SYMBOL_KEY>/metadata.json`

Le trainer applique:
- sélection de features
- fusion de features multi-timeframe alignées temporellement (sans look-ahead)
- validation temporelle
- HPO walk-forward orienté rentabilité nette (Sharpe/Sortino/Accuracy + rendement net + pénalité drawdown/turnover)
- accélération XGBoost selon le hardware
- retrain en arrière-plan (thread dédié), sans bloquer la boucle de scan
- ensemble pondéré par performance hors-échantillon avec seuils de décision BUY/SELL
- labels dynamiques adaptés à la volatilité et aux coûts d'exécution

Réglages HPO:
- `model.hpo.enabled`
- `model.hpo.max_samples`
- `model.hpo.max_splits`
- `model.hpo.rf_trials`
- `model.hpo.xgb_trials`
- `model.hpo.objective_weight_*`
- `model.execution.cost_multiplier`
- `model.labeling.*`
- `model.ensemble.*`
- `model.train_ohlcv_limit`
- `scanner.ohlcv_limit`

## 7) Hardware (Mac M1 + Ubuntu RTX 2070 Super)

Probe matériel:

```bash
python3 -m scripts.hardware_probe
```

Mac M1:
- mettre `model.acceleration.mode: cpu`

Ubuntu + RTX:
- mettre `model.acceleration.mode: auto` (ou `cuda`)
- installer build XGBoost compatible CUDA

Le fallback CPU est géré via `model.acceleration.allow_cuda_fallback`.

## 8) Sécurité live

Le bot bloque le démarrage live si:
- clés API absentes
- `exchange.testnet` encore à `true`
- pas de market data valide
- impossible de lire le solde authentifié

Recommandations:
- API key trading avec whitelist IP
- commencer par paper trading prolongé
- garder `scripts.kill_switch` prêt
- brancher au moins un canal `monitoring.alerting` (webhook/Telegram/email) avant live continu

## 9) Fichiers importants

- `configs/bot.yaml`: configuration runtime
- `scripts/run_bot.py`: boucle de trading
- `scripts/run_santrade_intelligence.py`: moteur global marché standalone (SanTradeIntelligence)
- `scripts/run_trainer.py`: entraînement per-symbol
- `monitoring/dashboard.py`: interface web
- `public_api/app.py`: API publique + websocket + endpoint metrics
- `artifacts/state/bn_ml.db`: état persistant (positions/trades/cycles)
- `artifacts/logs/dashboard.log`: logs Streamlit
- `artifacts/metrics/latest_scan.csv`: snapshot complet des paires scannées
- `artifacts/metrics/latest_opportunities.csv`: opportunités retenues par les filtres
- `artifacts/reports/dod/`: checks quotidiens + rapport DoD
- `docs/runbook_incident.md`: procedures incident
- `docs/deployment_docker.md`: deploiement Docker

## 10) Checklist avant push Git

```bash
python3 -m pytest -q
python3 -m scripts.run_bot --once --paper --disable-retrain --no-dashboard
python3 -m scripts.hardware_probe
```

Optionnel:

```bash
python3 -m scripts.run_trainer --paper --symbol BTC/USDC
```

Si tu pushes un setup live, ne commite jamais `.env` ou des secrets.

## Contribution et sécurité

- Guide contribution: `CONTRIBUTING.md`
- Politique sécurité: `SECURITY.md`
- Template PR: `.github/PULL_REQUEST_TEMPLATE.md`
- Templates issues: `.github/ISSUE_TEMPLATE/`

## 11) Sync modèles (GitHub ou RunPod)

Le script unique est `python3 -m scripts.model_sync`.

### Mode A: GitHub publisher/client

Objectif:
- `publisher`: entraîne puis pousse les modèles vers un repo Git dédié
- `client`: pull le repo et remplace le dossier local `models/`

One-shot:

```bash
python3 -m scripts.model_sync publish --config configs/bot.yaml --repo-dir /ABS/PATH/model-registry-repo
python3 -m scripts.model_sync pull --config configs/bot.yaml --repo-dir /ABS/PATH/model-registry-repo --models-dir models
```

Daemon quotidien:

```bash
python3 -m scripts.model_sync daemon --role publisher --repo-dir /ABS/PATH/model-registry-repo
python3 -m scripts.model_sync daemon --role client --repo-dir /ABS/PATH/model-registry-repo
```

### Mode B: RunPod endpoint (trigger + pull archive)

Objectif:
- appeler un endpoint RunPod qui lance l'entraînement
- attendre la fin du job
- télécharger l'archive des modèles et remplacer `models/`

Configuration (`configs/bot.yaml`):

```yaml
model_sync:
  runpod_client:
    schedule: "03:00"
  runpod:
    enabled: true
    trigger_url: "https://api.runpod.ai/v2/<ENDPOINT_ID>/run"
    status_url_template: "https://api.runpod.ai/v2/<ENDPOINT_ID>/status/{job_id}"
    api_key_env: RUNPOD_API_KEY
    trigger_method: POST
    trigger_payload: {}
    headers: {}
    request_timeout_sec: 20
    poll_interval_sec: 10
    job_timeout_sec: 3600
    download_timeout_sec: 300
    extract_subdir: models
```

One-shot:

```bash
python3 -m scripts.model_sync runpod-pull --config configs/bot.yaml --models-dir models
```

Daemon quotidien:

```bash
python3 -m scripts.model_sync daemon --role runpod_client --config configs/bot.yaml --models-dir models
```

## 12) Docker (RunPod quotidien)

Le compose inclut un service dédié `model-sync-runpod` qui exécute le daemon quotidien (`--role runpod_client`).
Le compose inclut aussi un service dédié `santrade-intelligence-paper` / `santrade-intelligence-live`
qui exécute `scripts.run_santrade_intelligence` en boucle (sans exécution d'ordres).
L'image Docker utilise un CMD par défaut train-only (`scripts.runpod_train_only`), pas `run_bot`.

```bash
# stack paper avec SanTradeIntelligence + sync RunPod
docker compose --profile paper --profile runpod up -d bot-paper santrade-intelligence-paper model-sync-runpod dashboard api prometheus grafana

# stack live avec SanTradeIntelligence + sync RunPod
docker compose --profile live --profile runpod up -d bot-live santrade-intelligence-live model-sync-runpod dashboard api prometheus grafana
```

Dans `.env`, définir `RUNPOD_API_KEY=...`.
