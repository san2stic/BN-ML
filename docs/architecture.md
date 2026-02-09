# BN-ML Architecture

## Vue d'ensemble

Le système est composé de 5 blocs:

1. `data_manager/`: récupération Binance + features techniques
2. `ml_engine/`: entraînement, HPO, prédiction
3. `scanner/`: scoring multi-paires et ranking opportunités
4. `trader/`: gestion des risques, ordres, positions
5. `monitoring/`: logs, alertes, dashboard Streamlit

## Flux runtime (boucle bot)

Ordre d'exécution dans `scripts/run_bot.py`:

1. Gestion des positions ouvertes (SL/TP/trailing/time stop, closes partielles et full close)
2. Scan des paires configurées
3. Prédiction BUY/SELL/HOLD + confiance
4. Validation risk manager (exposition, taille, limites globales)
5. Exécution ordre via order manager (paper/live)
6. Persistance état/trades/cycles dans SQLite
7. Export snapshot positions et métriques

En mode continu, le dashboard peut être auto-lancé en sous-processus Streamlit.

## Retrain asynchrone

Le retrain est géré par un worker en thread séparé:

- vérification périodique des conditions via `model.retrain_check_interval_sec`
- retrain adaptatif selon `model.retrain_interval_hours` et la dégradation de perf
- auto-train immédiat des modèles manquants (`model.auto_train_missing_models`)
- auto-train manquant par batch (`model.auto_train_missing_batch_size`)
- hot-reload des composants ML (`predictor`/`scanner`) après entraînement réussi

## Modèles ML par symbole

Chaque paire possède son modèle dédié:

- `models/<SYMBOL_KEY>/rf.joblib`
- `models/<SYMBOL_KEY>/xgb.joblib`
- `models/<SYMBOL_KEY>/metadata.json`

Pipeline d'entraînement (`scripts/run_trainer.py` + `ml_engine/trainer.py`):

1. Chargement OHLCV + features
2. Construction label multi-classe (SELL/HOLD/BUY)
3. Feature selection
4. HPO walk-forward (RF + XGB) avec objectif pondéré:
   - Sharpe
   - Sortino
   - Accuracy
5. Entraînement final et sérialisation modèles + métadonnées

Mode incrémental:
- `train_missing_only` entraîne uniquement les symboles sans bundle valide
- retrain aussi les bundles plus vieux que `model_max_age_hours`

## Accélération hardware

Détection via `bn_ml/hardware.py`:

- `mode: auto`: CUDA si GPU NVIDIA détecté, sinon CPU
- `mode: cpu`: force CPU (recommandé sur Apple Silicon)
- `mode: cuda`: force CUDA (avec fallback CPU optionnel)

Paramètres:
- `model.acceleration.mode`
- `model.acceleration.cpu_n_jobs`
- `model.acceleration.allow_cuda_fallback`

## Garde-fous live

Avant d'entrer en boucle live:

1. Vérification clés API Binance
2. Vérification `exchange.testnet: false`
3. Vérification market data (ticker valide)
4. Vérification accès balance authentifié

Si un check échoue, le bot s'arrête immédiatement.

## Scanner et décision

Le scanner produit un score global par paire basé sur:

- score ML
- score technique
- score momentum

Le scanner peut fonctionner en univers dynamique:
- découverte automatique des paires `*/base_quote` sur Binance Spot (ex: `*/USDC`)
- filtrage volume 24h + exclusions (tokens à levier / fiat / stable-stable)
- cache de l'univers pour limiter les appels API

Puis le risk manager valide selon contraintes:

- limite nombre de positions
- limite exposition portefeuille
- contraintes de sizing
- circuit breakers

## Persistance et sorties

Stockage local actuel:

- SQLite: `artifacts/state/bn_ml.db`
- logs: `artifacts/logs/`
- métriques: `artifacts/metrics/`
- modèles: `models/`

Sorties principales:

- historique cycles
- historique trades
- positions ouvertes/fermées
- résumé backtest (`artifacts/metrics/backtest_summary.csv`)
- scan complet (`artifacts/metrics/latest_scan.csv`)
- opportunités retenues (`artifacts/metrics/latest_opportunities.csv`)

## Dashboard Trader Terminal

`monitoring/dashboard.py` lit les artefacts runtime et expose:

- watchlist sticky
- heatmap opportunités
- equity/drawdown
- execution blotter
- cycle feed
- panneaux détachables
- mode full-screen + auto-refresh

Configuration via:

- `monitoring.dashboard.auto_launch_with_bot`
- `monitoring.dashboard.address`
- `monitoring.dashboard.port`
- `monitoring.dashboard.auto_refresh_sec`
- `monitoring.dashboard.full_screen_default`
- `monitoring.dashboard.timeframe_default`
