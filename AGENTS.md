# AGENT.MD - Binance Spot ML Trading Bot

## 1) Mission
Construire un systeme de trading crypto pour Binance Spot, pilote par ML, avec:
- scanner multi-paires en temps reel
- execution d'ordres robuste
- money management strict
- entrainement adaptatif anti-overfitting
- monitoring, alerting, et mode paper trading

Objectif prioritaire: robustesse risque > performance brute.

## 2) Perimetre
- Marche: Binance Spot uniquement
- Univers: paires liquides (USDT majoritairement), filtre volume 24h > 1M USDT
- Horizon principal: intraday / swing court
- Signaux: BUY / SELL / HOLD + score de confiance 0-100

Hors perimetre initial:
- futures / margin / leverage
- market making haute frequence
- arbitrage inter-exchange

## 3) Principes d'architecture
- Python 3.10+
- Approche modulaire, testable, observable
- Separation stricte:
  - data ingestion
  - feature engineering
  - model training/prediction
  - risk controls
  - order execution
  - monitoring
- Config centrale YAML surchargeable par variables d'environnement
- Aucune cle API en dur dans le code

## 4) Structure projet cible
```text
data_manager/
  fetch_data.py
  features_engineer.py
  data_cleaner.py
ml_engine/
  trainer.py
  predictor.py
  validator.py
scanner/
  opportunity_scanner.py
  scorer.py
trader/
  order_manager.py
  position_manager.py
  risk_manager.py
monitoring/
  logger.py
  alerter.py
  dashboard.py
configs/
  bot.yaml
scripts/
  run_bot.py
  run_backtest.py
  run_trainer.py
tests/
docs/
```

## 5) Modele ML (v1)
Ensemble de 3 modeles:
- RandomForestClassifier
- XGBoostClassifier
- LSTM (classification sequence)

Sorties:
- classe {BUY, SELL, HOLD}
- force de signal (regression 0-100 ou calibration probabiliste)

Fusion:
- weighted voting
- poids adaptes selon performance recente 7 jours
- signal valide seulement si confiance >= 65%

## 6) Features (v1)
Inclure les familles:
- tendance: EMA(9/21/50/100/200), MACD, ADX/DI, SuperTrend, Ichimoku
- momentum: RSI, Stoch RSI, Stoch, Williams %R, CCI, MFI
- volatilite: Bollinger + %B, ATR, Keltner, vol historique
- volume: OBV, VWAP, volume ratio, Acc/Dist
- structure: pivots, HH/LL, ROC(1h/4h/1D), correlation BTC
- derivees: divergences, ecarts EMA, distance highs/lows 24h/7j/30j, spread normalise

Contrainte:
- selection continue pour conserver ~40-50 features max.

## 7) Entrainement adaptatif
- Fenetre glissante: 30-60 jours
- Re-entrainement: toutes les 4-6h ou trigger volatilite
- Validation: walk-forward + TimeSeriesSplit (K=5)
- Anti-overfitting:
  - early stopping
  - regularisation L1/L2
  - contraintes arbres (max_depth, min_samples_leaf)
  - dropout LSTM
- Regime change:
  - monitor Sharpe/WinRate/Drawdown
  - re-entrainement force si degradation >15% sur 24h
  - test KS sur drift de distribution

## 8) Scanner multi-paires
- Scan complet: toutes les 5 minutes
- Monitoring positions: toutes les 30 secondes
- WebSocket temps reel: top 20 paires

Scoring:
- ML score (50%)
- Technique score (30%)
- Momentum score (20%)
- Score global = 0.5*ML + 0.3*Technique + 0.2*Momentum

Filtres execution:
- spread <= 0.15%
- profondeur carnet >= 50k USDT a +/-0.5%
- exclusion gaps recents >3%
- controle correlation BTC selon exposition existante

## 9) Risk & Money Management (non-negociable)
- Buckets capital: 60% actif / 30% reserve / 10% buffer
- Max 5 positions simultanees
- Sizing: Kelly modifie borne [1%, 20%] du capital actif
- Ajustement volatilite via ATR ratio
- Stop initial: 1.5 * ATR(14)
- Trailing: actif apres +2%, distance 1 * ATR
- Time stop: sortie si stagnation >48h
- Hard stop par position: -3%
- TP partiels: 50% / 30% / 20%
- Risque max: 2% jour, 6% semaine
- Circuit breakers:
  - drawdown journalier <= -5%
  - 3 pertes consecutives
  - volatilite marche >150% moyenne 30j

## 10) Securite operationnelle
- API scanner en read-only
- API trading avec whitelist IP
- Respect strict rate limit Binance
- Retry + exponential backoff
- Backup modeles/positions toutes les heures
- Kill switch: fermeture immediate de toutes positions
- Paper trading obligatoire avant live

## 11) Metriques cibles
Trading:
- Sharpe > 1.5
- Sortino > 2.0
- Win rate > 55%
- Profit factor > 1.8
- Max drawdown < 15%

ML:
- Accuracy val > 60%
- Precision BUY > 65%
- Recall > 55%
- F1 > 0.60
- ROC AUC > 0.70

Ops:
- Uptime > 99.5%
- Latence execution < 500ms
- Slippage reel < 0.08%
- Scan complet < 30s

## 12) Roadmap implementation
Phase 1 - Fondations:
1. scaffold projet + config YAML
2. data manager (historique + websocket)
3. moteur indicateurs + pipeline features
4. mode paper trading end-to-end

Phase 2 - ML:
1. labels + dataset builder
2. trainer RF/XGB/LSTM
3. validation walk-forward + backtest 6 mois
4. model registry + versionning

Phase 3 - Trading live controle:
1. scanner multi-paires
2. risk manager complet
3. order + position manager
4. alerting Telegram/Email + dashboard Streamlit

Phase 4 - Hardening:
1. tests integration et resilience
2. optimisation latence/couts API
3. runbook prod + procedures incident

## 13) Definition of Done (v1)
Le systeme est "pret v1" si:
1. paper trading stable >= 30 jours
2. aucune violation des contraintes de risque
3. rapport backtest >= 6 mois disponible
4. dashboard live + alerting operationnels
5. documentation de deploiement Docker complete

## 14) Regles d'execution
- Toujours verifier les preconditions risque avant ordre.
- Si doute sur la qualite du signal: HOLD.
- Priorite absolue: preservation du capital.
