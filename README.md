# BN-ML - Binance Spot ML Trading Bot

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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

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

Universe dynamique:
- `universe.dynamic_base_quote_pairs: true` pour scanner toutes les paires `*/base_quote` (ex: `*/USDC`)
- `universe.max_pairs_scanned` pour limiter la charge/rate limit
- `universe.train_dynamic_pairs: true` pour entraîner par symbole sur cet univers
- `universe.train_max_pairs` pour limiter le nombre de modèles entraînés
- `universe.train_missing_only: true` pour n'entraîner que les modèles manquants/anciens
- `universe.model_max_age_hours: 24` seuil d'ancienneté modèle avant retrain

Auto-retrain en fond (thread séparé):
- `model.auto_retrain_enabled: true` active le worker d'entraînement asynchrone
- `model.retrain_check_interval_sec: 30` fréquence de vérification (`x`) des conditions de retrain
- `model.retrain_interval_hours: 6` intervalle minimal entre retrains complets
- `model.auto_train_missing_models: true` entraîne automatiquement les bundles manquants
- `model.auto_train_missing_batch_size: 5` limite les auto-trains manquants par batch
- `model.auto_train_missing_disable_hpo: true` accélère l'auto-train manquant (désactive HPO sur ce flux)

## 3) Commandes principales

Entraîner les modèles par symbole:

```bash
python3 -m scripts.run_trainer
```

Entraînement incrémental (manquants/anciens uniquement):

```bash
python3 -m scripts.run_trainer --train-missing-only
```

Forcer entraînement complet:

```bash
python3 -m scripts.run_trainer --train-all
```

Entraîner seulement certains symboles:

```bash
python3 -m scripts.run_trainer --symbol BTC/USDC --symbol ETH/USDC
```

Lancer un cycle unique bot (paper):

```bash
python3 -m scripts.run_bot --once --paper
```

Lancer en live continu:

```bash
python3 -m scripts.run_bot --live
```

Le dashboard est auto-lancé en mode continu si `monitoring.dashboard.auto_launch_with_bot: true`.

Lancer sans dashboard:

```bash
python3 -m scripts.run_bot --live --no-dashboard
```

Lancer dashboard seul:

```bash
streamlit run monitoring/dashboard.py
```

Backtest baseline:

```bash
python3 -m scripts.run_backtest --paper
```

Kill switch (fermeture d'urgence):

```bash
python3 -m scripts.kill_switch
```

## 4) Dashboard "Trader Terminal"

Le dashboard inclut:
- mode full-screen desk
- auto-refresh (5s par défaut)
- filtre timeframe (`15m`, `1h`, `4h`, `1d`, `1w`, `all`)
- panneaux détachables (workspace par onglets)
- watchlist sticky, heatmap opportunités, equity/drawdown, blotter, cycle feed

Paramètres dans `configs/bot.yaml`:
- `monitoring.dashboard.auto_launch_with_bot`
- `monitoring.dashboard.address`
- `monitoring.dashboard.port`
- `monitoring.dashboard.auto_refresh_sec`
- `monitoring.dashboard.full_screen_default`
- `monitoring.dashboard.timeframe_default`
- `monitoring.dashboard.live_market_enabled`
- `monitoring.dashboard.scan_stale_sec`

## 5) Modèles ML par crypto

Chaque symbole a son bundle dédié:
- `models/<SYMBOL_KEY>/rf.joblib`
- `models/<SYMBOL_KEY>/xgb.joblib`
- `models/<SYMBOL_KEY>/metadata.json`

Le trainer applique:
- sélection de features
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

## 6) Hardware (Mac M1 + Ubuntu RTX 2070 Super)

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

## 7) Sécurité live

Le bot bloque le démarrage live si:
- clés API absentes
- `exchange.testnet` encore à `true`
- pas de market data valide
- impossible de lire le solde authentifié

Recommandations:
- API key trading avec whitelist IP
- commencer par paper trading prolongé
- garder `scripts.kill_switch` prêt

## 8) Fichiers importants

- `configs/bot.yaml`: configuration runtime
- `scripts/run_bot.py`: boucle de trading
- `scripts/run_trainer.py`: entraînement per-symbol
- `monitoring/dashboard.py`: interface web
- `artifacts/state/bn_ml.db`: état persistant (positions/trades/cycles)
- `artifacts/logs/dashboard.log`: logs Streamlit
- `artifacts/metrics/latest_scan.csv`: snapshot complet des paires scannées
- `artifacts/metrics/latest_opportunities.csv`: opportunités retenues par les filtres

## 9) Checklist avant push Git

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
