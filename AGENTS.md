# AGENTS.md - BN-ML Binance Spot ML Trading Bot

Derniere mise a jour: 2026-02-09 (P0+P1+P2 code/tests integres localement).

## 1) Mission
Construire un systeme de trading Binance Spot pilote par ML, avec priorite absolue:

`robustesse risque > performance brute`

Le bot doit privilegier la preservation du capital, meme si cela reduit le nombre de trades.

## 2) Perimetre
- Marche: Binance Spot uniquement.
- Univers: paires liquides filtrees par volume 24h (>= 1M quote).
- Horizon: intraday / swing court.
- Signaux: `BUY`, `SELL`, `HOLD` + confiance `0-100`.

Hors perimetre:
- futures / margin / leverage
- market making HFT
- arbitrage inter-exchange

## 3) Etat reel du projet (audit 2026-02-09)
Statut global:
- Scanner multi-paires: `PARTIEL` (polling 5 min; websocket prix temps reel actif pour positions/paires suivies, pas encore event-driven scanner complet).
- Execution ordres: `OK` (paper/live, precision/filters exchange, retry).
- Money management: `OK` (budgets risque jour/semaine bloques en pre-trade + breakers drawdown/pertes/vol/drift).
- Entrainement adaptatif: `OK` (RF/XGB/LGB, HPO walk-forward, retrain async).
- Monitoring: `OK` (dashboard + alerting webhook/Telegram/email + backups runtime).
- Paper trading: `OK` (end-to-end), avec mode donnees reelles Binance en paper (+ fallback synthetique).

## 4) Architecture actuelle
```text
bn_ml/
  config.py
  env.py
  exchange.py
  hardware.py
  state_store.py
  symbols.py
  types.py
data_manager/
  fetch_data.py
  data_cleaner.py
  features_engineer.py
  multi_timeframe.py
ml_engine/
  trainer.py
  predictor.py
  validator.py
  adaptive_trainer.py
scanner/
  opportunity_scanner.py
  scorer.py
trader/
  order_manager.py
  position_manager.py
  risk_manager.py
  exit_manager.py
monitoring/
  logger.py
  alerter.py
  dashboard.py
scripts/
  run_bot.py
  run_trainer.py
  run_backtest.py
  kill_switch.py
  hardware_probe.py
configs/
  bot.yaml
tests/
docs/
```

## 5) Comportement runtime actuel
Boucle principale (`scripts/run_bot.py`):
1. sync capital live (si mode live)
2. gestion des positions ouvertes (SL, trailing, TP partiels, time stop)
3. scan univers dynamique ou statique
4. prediction ML + scoring opportunites
5. validation risk manager
6. execution ordre BUY (paper/live)
7. persistance SQLite + exports metrics CSV
8. retrain asynchrone en thread dedie (si active)

## 6) Stack ML actuelle
Modeles actifs:
- `RandomForestClassifier`
- `XGBoostClassifier`
- `LightGBMClassifier` (si dispo)
- `SequenceMLP` via chemin `model.lstm` (modele sequence integre a l'ensemble et serialise `lstm.joblib`)

Non implemente a date:
- `LSTM` deep learning natif (TensorFlow/PyTorch) non obligatoire; backend sequence operationnel livre via sklearn.

Points cle:
- labels multiclasses `SELL/HOLD/BUY` avec edge dynamique base vol + ATR + couts
- HPO Optuna (fallback grid)
- Purged time-series CV
- pond. d'ensemble basee sur metriques hors echantillon
- calibration proba (isotonic/sigmoid) optionnelle
- decision BUY/SELL via seuils/marge, sinon HOLD

## 7) Features actuelles
Familles implementees:
- tendance: EMA 9/21/50/100/200, MACD, ADX/DI, SuperTrend
- momentum: RSI, Stoch K/D, Williams %R, CCI, MFI
- volatilite: ATR/ATR ratio, Bollinger %B, Keltner, vol regime
- volume: OBV, volume ratio, Acc/Dist, distance VWAP
- structure: ROC 1h/4h/1d, distance high/low 24h, patterns bougie
- derivees: spreads EMA, divergence RSI proxy, confluence multi-timeframe

Contrainte active:
- feature selection automatique avec limite (`model.feature_limit`, defaut 50)

## 8) Risk & Money Management (etat actuel)
Actif aujourd'hui:
- buckets capital (active/reserve/buffer) via config
- max positions simultanees
- sizing Kelly borne min/max + ajustement ATR ratio
- filtre spread / profondeur carnet / correlation BTC / confiance min
- SL initial ATR, trailing ATR apres activation
- TP partiels 50% puis 30%, puis fermeture du reliquat
- hard stop position (drawdown), time stop
- circuit breakers: drawdown journalier, pertes consecutives, volatilite ratio

Points a renforcer:
- calibration fine des seuils drift/vol selon regime reel (a ajuster apres paper run long)
- hardening resilience reseau pour alerting externe (retry file/outbox)

## 9) Data & execution
Data manager:
- live: CCXT Binance Spot
- paper: mode `data.paper_market_data_mode=live` (historique+ticker Binance) avec fallback synthetique deterministe
- univers dynamique `*/base_quote` avec filtres volume + exclusions assets non souhaites

Execution:
- paper et live via `OrderManager`
- validation precision exchange + minQty/minNotional en live
- retry avec backoff sur appels exchange
- preflight live strict: cles API, testnet=false, ticker valide, balance authentifiee

## 10) Monitoring, stockage, artefacts
Stockage:
- SQLite: `artifacts/state/bn_ml.db`
- snapshots positions: `artifacts/state/open_positions.json`
- metrics scan/opportunites: `artifacts/metrics/latest_scan.csv`, `artifacts/metrics/latest_opportunities.csv`
- logs: `artifacts/logs/`

Monitoring:
- dashboard Streamlit "Trader Terminal" (auto-launch possible)
- alerting externe: webhook / Telegram / SMTP email (+ dedupe)
- realtime price monitor websocket Binance (miniTicker)
- backups runtime periodiques (`artifacts/backups/`)
- kill switch disponible: `python3 -m scripts.kill_switch`

## 11) Qualite et tests
Etat tests au 2026-02-09:
- `60 passed` via `python3 -m pytest -q`

Couverture fonctionnelle verifiee:
- retrain adaptatif/background worker
- risk manager
- order manager constraints
- exit manager
- predictor weighted ensemble + callback model manquant
- trainer HPO + labeling
- multi-timeframe builder
- state store

## 12) Gaps prioritaires avant "v1 robuste"
P0 (bloquant live scale):
1. `DONE` verrouiller l'application stricte des limites risque jour/semaine en pre-trade
2. `DONE` alimenter/calculer `market_volatility_ratio` en runtime
3. `DONE` remplacer le mode paper synthetique par un paper base sur donnees reelles (historique + ticker live)

P1 (fortement recommande):
1. `DONE` websocket temps reel top paires (monitoring positions + paires suivies)
2. `DONE` alerting externe (Telegram/Email/webhook)
3. `DONE` backup horaire automatise modeles + etat

P2 (evolution ML):
1. `DONE (backend sequence_mlp)` integrer chemin sequence `model.lstm` (train + infer + metadata)
2. `DONE` gestion explicite du regime change (KS drift + vol ratio runtime)

## 13) Roadmap mise a jour
Phase 1 - Fondations: `MAJORITAIREMENT DONE`
- scaffold, config, pipeline data/features, paper E2E

Phase 2 - ML: `EN COURS AVANCE`
- labels dynamiques, trainer RF/XGB/LGB, walk-forward/HPO, model bundles per symbol
- sequence `model.lstm` operationnel (backend sequence_mlp) + drift monitoring runtime en place

Phase 3 - Trading live controle: `EN COURS`
- scanner, risk manager, order/position manager, dashboard
- reste: alerting externe + enforcement risque complet

Phase 4 - Hardening: `A PRIORISER`
- tests integration resilience, optimisation latence/couts API
- runbook incident + documentation deploiement Docker: `DONE` (`docs/runbook_incident.md`, `docs/deployment_docker.md`)

## 14) Definition of Done v1 (reconfirmee)
Le systeme est "pret v1" uniquement si:
1. paper trading stable >= 30 jours sur donnees de marche reelles
2. zero violation des contraintes risque
3. rapport backtest >= 6 mois disponible et reproductible
4. dashboard live + alerting operationnels
5. documentation de deploiement Docker + procedures incident complete

## 15) Regles d'execution non-negociables
- Toujours verifier les preconditions risque avant chaque ordre.
- Si doute sur qualite signal ou liquidite: `HOLD`.
- Ne jamais hardcoder des secrets/API keys.
- En live, ne jamais bypass les checks preflight.
- Toute modification du risk manager doit etre couverte par tests.
- Priorite absolue: preservation du capital.
