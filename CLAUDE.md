# CLAUDE.md - Guide de développement pour BN-ML

## Vue d'ensemble du projet

**BN-ML** est un bot de trading ML pour Binance Spot avec gestion stricte du risque. La priorité absolue est **robustesse > performance brute** - la préservation du capital est non négociable.

### Caractéristiques principales

- **Trading ML multi-modèles** : Ensemble RF + XGBoost + LightGBM + LSTM avec vote pondéré
- **Entraînement per-symbol** : Bundles ML dédiés par paire avec métadonnées
- **Multi-timeframe** : Fusion des timeframes supérieurs (1h, 4h, 1d) sur base 15m sans look-ahead bias
- **Risk-first** : Validation pré-trade complète (sizing, corrélations, limites portfolio, circuit breakers)
- **Paper + Live** : Modes paper (synthetic + live market fallback) et live avec exchange réel
- **Monitoring complet** : Dashboard Streamlit, API FastAPI, Prometheus/Grafana, alertes Telegram/email
- **Retraining adaptatif** : Worker thread qui ré-entraîne automatiquement sur dégradation

## Architecture du projet

```
BN-ML/
├── bn_ml/              # Core framework (config, env, exchange, state)
├── data_manager/       # Data fetching, cleaning, features, multi-timeframe
├── ml_engine/          # Trainer, predictor, validators, drift, SanTradeIntelligence
├── scanner/            # Multi-pair scanning et scoring
├── trader/             # Risk management, orders, positions, exits
├── monitoring/         # Logging, alerting, dashboard, realtime prices
├── public_api/         # FastAPI server avec websocket
├── scripts/            # Points d'entrée CLI (bot, trainer, backtest, etc.)
├── configs/            # Configuration YAML (bot.yaml)
├── tests/              # 60+ tests pytest
├── models/             # Bundles ML per-symbol (RF, XGB, LGB, LSTM)
├── artifacts/          # État runtime, logs, métriques, backups
└── docs/               # Architecture, déploiement, runbooks
```

**93 fichiers Python** au total, organisation modulaire stricte avec séparation des responsabilités.

## Stack technique

- **Langage** : Python 3.10+
- **ML** : scikit-learn, xgboost, lightgbm, optuna (HPO)
- **Data** : pandas, numpy
- **Exchange** : ccxt 4.2.0+
- **Web** : FastAPI, Streamlit, Uvicorn
- **Monitoring** : Prometheus, Plotly, websockets
- **Tests** : pytest, httpx
- **Déploiement** : Docker, docker-compose
- **Accélération** : CUDA via XGBoost (optionnel)

## Points d'entrée principaux

### Commandes CLI

Tous les scripts sont exposés via setuptools dans `pyproject.toml` :

| Commande | Script | Usage |
|----------|--------|-------|
| `bnml-bot` | `scripts/run_bot.py` | Boucle principale du bot (paper/live) |
| `bnml-trainer` | `scripts/run_trainer.py` | Entraînement one-shot des modèles |
| `bnml-trainer-auto` | `scripts/run_trainer_auto.py` | Daemon de retraining automatique |
| `bnml-backtest` | `scripts/run_backtest.py` | Backtest baseline |
| `bnml-kill-switch` | `scripts/kill_switch.py` | Urgence : ferme toutes les positions |
| `bnml-dod-check` | `scripts/check_dod_daily.py` | Checks DoD quotidiens |
| `bnml-dod-30d` | `scripts/run_dod_30d.py` | Campagne DoD 30 jours |
| `bnml-hardware-probe` | `scripts/hardware_probe.py` | Détection CUDA/CPU |
| `bnml-model-sync` | `scripts/model_sync.py` | Sync modèles GitHub/RunPod |
| `bnml-santrade-intel` | `scripts/run_santrade_intelligence.py` | Market intelligence standalone |
| `bnml-api` | `public_api/app.py:run` | Serveur REST API + websocket |

### Boucle principale du bot

**Flow** (`run_bot.py`) :
1. Charger config + env
2. Initialiser data manager, predictor, scanner, risk/order/position managers
3. Auto-lancer dashboard Streamlit (si configuré)
4. Démarrer websocket realtime prices (optionnel)
5. Démarrer worker de retraining background (si activé)
6. **Boucle principale** :
   - Sync capital (mode live)
   - Gérer positions ouvertes (SL/TP/trailing/time stops)
   - Scanner l'univers
   - Prédire signaux + scorer opportunités
   - Valider contraintes risque
   - Exécuter ordres (paper/live)
   - Persister état + métriques
   - Backup artifacts (horaire)
   - Sleep jusqu'au prochain scan

## Configuration

### Fichier principal : `configs/bot.yaml`

**Sections clés** :

- **`environment`** : `paper` ou `live`
- **`exchange`** : Paramètres API Binance, rate limits, retries
- **`data`** : `paper_market_data_mode` (live/synthetic)
- **`universe`** : Découverte de paires, scan dynamique, filtres training
- **`scanner`** : Intervalle scan, top-N selection, scores min, spread/depth/liquidity
- **`model`** : HPO settings, feature limits, labeling parameters, multi-timeframe, acceleration
- **`risk`** : Capital buckets, limites positions, drawdown caps, circuit breakers
- **`monitoring`** : Dashboard auto-launch, alertes (webhook/Telegram/email), realtime prices
- **`storage`** : Chemin SQLite, intervalles backup

### Variables d'environnement

Créer un fichier `.env` à partir de `.env.example` :

```bash
BINANCE_API_KEY=              # API key Binance
BINANCE_API_SECRET=           # Secret Binance
BINANCE_TESTNET=false         # Testnet ou mainnet
RUNPOD_API_KEY=               # Key RunPod pour training cloud
TELEGRAM_BOT_TOKEN=           # Token bot Telegram pour alertes
TELEGRAM_CHAT_ID=             # Chat ID Telegram
EMAIL_SMTP_HOST=              # SMTP host pour email
EMAIL_SMTP_PORT=587           # Port SMTP
EMAIL_USERNAME=               # Username email
EMAIL_PASSWORD=               # Password email
BNML_API_CORS_ORIGINS=*       # CORS origins pour API
BNML_API_WS_POLL_SEC=2.0      # Polling websocket
CLOUDFLARED_URL=              # URL tunnel Cloudflare
```

**Important** : Les clés API doivent être définies avant trading live (preflight checks enforced).

## Bases de données et schémas

### SQLite : `artifacts/state/bn_ml.db`

**Tables principales** :

```sql
-- Positions ouvertes (par symbol)
positions (
    symbol TEXT PRIMARY KEY,
    side TEXT,              -- "LONG"
    size_usdt REAL,
    entry_price REAL,
    stop_loss REAL,
    take_profit_1 REAL,     -- TP1 (50% exit)
    take_profit_2 REAL,     -- TP2 (30% exit)
    opened_at TEXT,
    status TEXT,            -- "OPEN" ou "CLOSED"
    extra_json TEXT         -- Trailing SL, TP hits, qty tracking
)

-- Historique trades
trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    symbol TEXT,
    side TEXT,              -- "BUY" ou "SELL"
    size_usdt REAL,
    price REAL,
    mode TEXT,              -- "paper" ou "live"
    extra_json TEXT         -- Order ID, fees, slippage
)

-- Cycles de scan
cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    opportunities INTEGER,
    opened_positions INTEGER,
    data_json TEXT
)

-- Key-value store générique
kv_state (
    key TEXT PRIMARY KEY,
    value_json TEXT,
    updated_at TEXT
)

-- Historique métriques modèles
model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    metrics_json TEXT       -- Scores validation per-model
)
```

### Bundles de modèles (per-symbol)

```
models/<SYMBOL_KEY>/
├── rf.joblib              # RandomForest
├── xgb.joblib             # XGBoost
├── lgb.joblib             # LightGBM (optionnel)
├── lstm.joblib            # Sequence MLP (optionnel)
└── metadata.json          # Métadonnées training
```

**Exemple metadata.json** :
```json
{
  "symbol": "BTC/USDC",
  "trained_at": "2026-02-10T12:34:56Z",
  "feature_columns": [...],
  "ensemble_weights": {"rf": 0.4, "xgb": 0.35, "lgb": 0.15, "lstm": 0.1},
  "validation_metrics": {...},
  "decision_params": {"min_buy_proba": 0.42, "min_sell_proba": 0.42, ...}
}
```

## API publique

### Endpoints REST (FastAPI sur port 8000)

| Méthode | Chemin | Description |
|---------|--------|-------------|
| GET | `/` | Page d'index (HTML) |
| GET | `/healthz` | Health check |
| GET | `/docs` | Documentation OpenAPI |
| GET | `/api/runtime/summary` | Résumé bot (capital, positions, equity) |
| GET | `/api/runtime/account` | État du compte |
| GET | `/api/runtime/positions` | Positions ouvertes |
| GET | `/api/runtime/trades` | Historique trades (paginé) |
| GET | `/api/models` | Liste bundles modèles |
| GET | `/api/models/download` | Télécharger tous modèles (zip) |
| GET | `/api/market/index` | Snapshot index marché |
| GET | `/api/market/index/history` | Historique index (paginé) |
| GET | `/api/market/intelligence` | Snapshot SanTradeIntelligence |
| GET | `/api/training/status` | Statut training modèles |
| GET | `/api/metrics` | Métriques Prometheus |

### Endpoints WebSocket

| Chemin | Description |
|--------|-------------|
| `/ws/predictions` | Stream prédictions temps réel |

**Features** :
- Middleware CORS (origins configurables)
- Instrumentation Prometheus
- Téléchargement archive modèles streaming
- Gestion erreurs avec HTTPException

## Tests

**Framework** : pytest (~60 tests passants)

### Catégories de tests

| Catégorie | Fichiers | Focus |
|-----------|----------|-------|
| Config | `test_config.py` | Chargement YAML, accès dotted keys |
| Data Manager | `test_data_manager_paper_market_data.py` | Modes paper/live market data |
| Trainer | `test_trainer_*.py` | Flow EnsembleTrainer, feature selection, HPO, LSTM |
| ML Engine | `test_predictor_*.py`, `test_drift_monitor.py` | Inférence, détection drift, sequence models |
| Scanner | `test_multi_timeframe_builder.py` | Fusion features multi-timeframe |
| Trader | `test_order_manager.py`, `test_exit_manager.py`, `test_risk_manager.py` | Contraintes ordres, logique exits, validation risk |
| Monitoring | `test_logger.py`, `test_alerter.py`, `test_realtime_prices.py` | Logging, alertes, websocket prices |
| State | `test_state_store.py`, `test_backup_manager.py` | Persistance SQLite, backups |
| API | `test_public_api_*.py` | Endpoints FastAPI, accès données |
| Intégration | `test_run_bot_*.py` | Cycles bot complets |
| Scripts | `test_run_trainer_*.py`, `test_runpod_train_only.py`, `test_santrade_intelligence.py` | Workflows scripts |

### Exécuter les tests

```bash
# Tous les tests
python3 -m pytest -q

# Subset avec verbose
python3 -m pytest tests/test_trainer_*.py -v

# Coverage
python3 -m pytest --cov=bn_ml --cov-report=html
```

## Déploiement

### Local (développement)

```bash
# Setup environnement
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# Lancer bot en paper
bnml-bot --config configs/bot.yaml

# Entraîner modèles
bnml-trainer --config configs/bot.yaml

# Dashboard
streamlit run monitoring/dashboard.py
```

### Docker

**Services disponibles** (docker-compose.yml) :

| Service | Profil | Port | Description |
|---------|--------|------|-------------|
| `bot-paper` | paper | - | Bot en mode paper |
| `bot-live` | live | - | Bot en mode live |
| `trainer-auto` | paper/live | - | Retraining automatique background |
| `santrade-intelligence-paper` | paper | - | Market intelligence paper |
| `santrade-intelligence-live` | live | - | Market intelligence live |
| `dashboard` | paper/live | 8501 | Dashboard Streamlit |
| `api` | paper/live | 8000 | API FastAPI |
| `prometheus` | ops | 9090 | Prometheus metrics |
| `grafana` | ops | 3000 | Grafana dashboards |
| `model-sync-runpod` | runpod | - | Sync training RunPod daily |
| `cloudflared` | paper/live | - | Tunnel public URL |

**Commandes Docker** :

```bash
# Build
docker compose build

# Lancer stack paper
docker compose --profile paper up -d bot-paper trainer-auto dashboard api

# Lancer stack live + runpod
docker compose --profile live --profile runpod up -d \
  bot-live trainer-auto dashboard api model-sync-runpod

# Monitoring stack
docker compose --profile ops up -d prometheus grafana

# Logs
docker compose logs -f bot-paper

# Stop
docker compose down
```

**Volumes persistants** :
- `bnml_artifacts` : États runtime, logs, métriques, backups
- `bnml_models` : Bundles ML per-symbol

### CI/CD

**GitHub Actions workflows** :

1. **ci.yml** : Push/PR → Lint (ruff) + pytest + build package (Python 3.10, 3.11)
2. **security.yml** : Dependabot scanning dépendances
3. **publish-pypi.yml** : Publish PyPI sur tag release (trusted publisher)
4. **publish-docker-ghcr.yml** : Build image Docker → GHCR
5. **release-please.yml** : Auto-bump version + GitHub releases

## Principes de design et décisions architecturales

### 1. Risk-First Philosophy

**"Robustesse risque > performance brute"**

- Préservation capital non négociable
- Validation pré-trade exhaustive (sizing, corrélations, limites)
- Circuit breakers (drawdown, VaR)
- Stop-loss obligatoires, take-profit scaling (50%/30%/20%)
- Time stops (max holding period)
- Preflight checks strictes avant live (API keys, capital min, exchange connectivité)

### 2. Ensemble ML adaptatif

**Modèles multiples avec vote pondéré** :
- RandomForest (robustesse baseline)
- XGBoost (gradient boosting performance)
- LightGBM (rapidité, grandes features)
- LSTM (optionnel, séquences temporelles)

**Poids d'ensemble** : Optimisés via HPO sur validation set

**Prédiction** : Vote pondéré des probabilités calibrées → BUY/SELL/HOLD

### 3. Entraînement per-symbol

Chaque paire trading a son propre bundle ML :
- Features spécifiques symbol (volatilité, liquidité, spread)
- Labels dynamiques adaptés à l'ATR et volatilité
- Métadonnées persistées (feature_columns, weights, thresholds)
- Retraining déclenché sur drift detection (PSI, KS)

### 4. Multi-timeframe fusion

**Projection sans look-ahead bias** :
- Base timeframe : 15m (trading)
- Higher timeframes : 1h, 4h, 1d (contexte macro)
- Features higher TF projetées sur base via forward-fill aligné
- Pas de leak futur, pure past data

### 5. Walk-Forward validation

**Time-series CV avec purge** :
- Split temporel (70% train, 30% val)
- Purge overlap pour éviter label leakage
- Évaluation sur périodes non vues chronologiquement
- Métriques : Accuracy, Precision, Recall, F1 per-class

### 6. Dynamic labeling

**Labels 3-classes (SELL/HOLD/BUY)** :
- Thresholds adaptatifs basés sur ATR et volatilité
- Zone neutre (HOLD) pour éviter bruit
- Regarde-ahead configurable (ex: 10 bars ahead)
- Équilibrage via SMOTE si nécessaire

### 7. Scanner multi-paires

**Flow** :
1. Fetch universe pairs (filtres liquidité, spread, depth)
2. Prédictions ML sur chaque pair
3. Score composite (ML + technical + momentum)
4. Ranking + top-N selection
5. Validation risk constraints
6. Exécution ordres

### 8. Monitoring complet

**Niveaux de monitoring** :
- **Logs** : Structurés, rotatifs, niveaux (DEBUG/INFO/WARN/ERROR)
- **Alertes** : Webhook, Telegram, email (threshold breach, erreurs critiques)
- **Dashboard** : Streamlit temps réel (equity curve, positions, heatmap opportunités, progress training)
- **API** : FastAPI pour intégrations externes + websocket streaming
- **Métriques** : Prometheus exportées, Grafana dashboards (optionnel)

### 9. État persisté

**StateStore SQLite** :
- Positions ouvertes
- Historique trades
- Cycles scan
- Key-value générique (capital, métriques)
- Model metrics history

**Backups automatiques** :
- Artifacts/ backupés horaire
- Models/ backupés après retraining
- Rotation old backups (config)

### 10. SanTradeIntelligence

**Agrégation market-wide** :
- Collecte signaux ML de tous symbols universe
- Ensemble SGD pour signal global (bullish/bearish/neutral)
- Profiles sauvegardés (timestamp, score, classe, probas, symbols contributeurs)
- API dédiée pour snapshot et historique

## Conventions de code

### Style Python

- **Version** : Python 3.10+ (f-strings, type hints, walrus operator)
- **Type hints** : Annotations complètes sur fonctions
- **Naming** :
  - Classes : `PascalCase` (EnsembleTrainer, MultiPairScanner)
  - Fonctions : `snake_case` (_build_exchange, _safe_json)
  - Constants : `UPPER_CASE` (LABEL_TO_ACTION, CLASSES)
  - Privées : `_prefixed`
- **Imports** : Organisés (stdlib, third-party, local)
- **Docstrings** : Minimales mais présentes où complexe
- **Comments** : Expliquer le "pourquoi", pas le "quoi"
- **Logging** : Logger par module (`logger = logging.getLogger(__name__)`)

### Linting

- **ruff** : Checks E9, F63, F7, F82 sur CI
- Pas d'enforce black/isort, mais idiomes Python propres

## Workflow Git

### Branches

- **main** : Production
- **san2stic-dev** : Development actuel
- **Recommandé** : `codex/<topic>` pour features (ex: `codex/santrade-intel`)

### Process

1. Créer branch depuis `main`
2. Implémenter + tester localement
3. Lancer `pytest -q` + single bot cycle
4. Commit atomique avec message clair (français ou anglais)
5. Push + créer PR avec checklist (template `.github/PULL_REQUEST_TEMPLATE.md`)
6. CI runs (lint, tests, package build)
7. Merge après approval
8. Release-please auto-bump version + publish PyPI + Docker

### Release

- Version dans `pyproject.toml` (actuellement 0.1.0)
- Manifest release-please dans `.release-please-manifest.json`
- Publish automatique PyPI (trusted publisher) + GHCR (Docker)

## Règles de développement pour Claude

### ⚠️ Règles critiques - TOUJOURS suivre

1. **JAMAIS modifier la logique de risque sans tests exhaustifs**
   - RiskManager est critique : toute modification doit inclure tests unitaires + intégration
   - Preflight checks ne doivent JAMAIS être contournés
   - Stop-loss/take-profit logique doit rester conservative

2. **JAMAIS committer de secrets**
   - API keys, tokens, passwords dans `.env` uniquement
   - `.env` est dans `.gitignore`
   - Vérifier avant commit : `git diff --staged`

3. **TOUJOURS tester avant PR**
   - `pytest -q` doit passer (60 tests)
   - Lancer un cycle bot complet en paper mode
   - Vérifier logs pour warnings/errors

4. **Bias vers HOLD**
   - En cas de doute, mieux rester neutre (HOLD) que prendre position
   - Thresholds conservateurs par défaut (min_buy_proba: 0.42)
   - Zone neutre large pour filtrer bruit

5. **Documentation obligatoire**
   - Nouvelles features : documenter dans README.md
   - Changements architecture : mettre à jour AGENTS.md
   - Scripts complexes : ajouter docstrings
   - Config params : commenter dans bot.yaml

### 🔧 Guidelines de développement

#### Ajout de nouvelles features

1. **Planification** :
   - Vérifier alignement avec risk-first philosophy
   - Identifier impacts sur components existants
   - Définir tests nécessaires

2. **Implémentation** :
   - Créer branch `codex/<feature>`
   - Suivre conventions naming/typing
   - Ajouter logs appropriés (niveau DEBUG/INFO)
   - Gérer erreurs gracefully (try/except, fallbacks)

3. **Testing** :
   - Tests unitaires pour logique pure
   - Tests intégration si interaction avec exchange/state
   - Mock exchange API dans tests (ccxt.async_support mock)

4. **Documentation** :
   - Docstrings sur fonctions publiques
   - README.md si nouvelle commande CLI
   - bot.yaml si nouveau config param

#### Modification de code existant

1. **Comprendre l'existant** :
   - Lire code + tests associés
   - Vérifier usages dans codebase
   - Identifier dépendances

2. **Refactoring safe** :
   - Tests doivent rester verts
   - Backward compatibility si API publique
   - Migration path si breaking change nécessaire

3. **Review** :
   - Self-review avant push
   - Expliquer rationale dans PR description
   - Lier issues/discussions pertinentes

#### Debugging

1. **Logs** :
   - Activer DEBUG level dans bot.yaml : `log_level: DEBUG`
   - Logs dans `artifacts/logs/bnml.log` (rotatif)
   - Chercher patterns d'erreurs, warnings

2. **State inspection** :
   - SQLite : `sqlite3 artifacts/state/bn_ml.db`
   - Queries : `SELECT * FROM positions;`, `SELECT * FROM trades ORDER BY ts DESC LIMIT 10;`

3. **Backtrace** :
   - Python traceback complet dans logs
   - Identifier module/ligne source
   - Reproduire localement en paper mode

4. **Dashboard** :
   - Vérifier equity curve pour anomalies
   - Heatmap positions pour identifier patterns
   - Model progress pour détecter training issues

### 📋 Checklist pré-commit

- [ ] `pytest -q` passe (60 tests)
- [ ] Lancer 1 cycle bot complet en paper mode sans erreur
- [ ] Aucun secret dans diff staged
- [ ] Logs propres (pas d'erreurs inattendues)
- [ ] Documentation à jour si nécessaire
- [ ] Code suit conventions (typing, naming, imports)
- [ ] Commit message clair et atomique

### 📋 Checklist pré-PR

- [ ] Tous commits passent checklist pré-commit
- [ ] Branch à jour avec `main` (rebase si nécessaire)
- [ ] PR description remplie (contexte, changements, tests)
- [ ] Template PR checklist complété
- [ ] CI passe (lint + tests + build)
- [ ] Reviewer assigné si applicable

### 🚨 Incidents et kill-switch

**Si problème en live** :

1. **Kill-switch immédiat** :
   ```bash
   bnml-kill-switch --config configs/bot.yaml
   ```
   Ferme toutes positions immédiatement (market orders).

2. **Stop bot** :
   ```bash
   # Si bot en foreground
   Ctrl+C

   # Si bot en docker
   docker compose down
   ```

3. **Investigation** :
   - Consulter logs : `artifacts/logs/bnml.log`
   - Vérifier state DB : `sqlite3 artifacts/state/bn_ml.db`
   - Review recent trades : `SELECT * FROM trades ORDER BY ts DESC LIMIT 20;`

4. **Runbook** :
   - Suivre procédures dans `docs/runbook_incident.md`
   - Documenter incident (timestamp, symptômes, actions, résolution)
   - Postmortem si nécessaire

### 🔐 Sécurité

**Production** :
- API keys en `.env` uniquement (JAMAIS commit)
- Rotation keys régulière (recommandé : mensuel)
- IP whitelist Binance API (si possible)
- Monitoring alertes actives (Telegram/email)
- Backup réguliers (artifacts/ + models/)

**Testnet** :
- Utiliser Binance Testnet pour tests live-like : `BINANCE_TESTNET=true`
- API keys testnet séparées (pas de fonds réels)

**Paper mode** :
- Mode safe par défaut : `environment: paper`
- Aucun ordre réel envoyé
- Utile pour dev, backtest, validation

### 📚 Documentation utile

- **README.md** : Installation, configuration, commandes
- **AGENTS.md** : État du projet, architecture, roadmap, DoD
- **CONTRIBUTING.md** : Workflow contribution, standards techniques
- **docs/architecture.md** : Design détaillé système
- **docs/deployment_docker.md** : Déploiement Docker, troubleshooting
- **docs/runbook_incident.md** : Procédures incident
- **SECURITY.md** : Reporting vulnérabilités, best practices

### 🎯 Priorités actuelles (selon AGENTS.md)

**P0 (Bloquant prod)** :
- ✅ Tous résolus (v0.1.0 stable)

**P1 (Haute priorité)** :
- Retraining adaptatif complet (drift → auto-retrain → reload models)
- Position sizing adaptatif (Kelly criterion, volatility-adjusted)
- Monitoring avancé (Prometheus + Grafana dashboards détaillés)

**P2 (Moyen terme)** :
- Backtest robuste (walk-forward, slippage, fees, realistic fills)
- Portfolio optimization (corrélations, diversification)
- Alertes avancées (anomaly detection, performance decay)

**Roadmap** : Voir section "Priorités et roadmap" dans AGENTS.md

### 🧪 Testing guidelines

**Structure test** :
```python
# tests/test_<module>.py
import pytest
from bn_ml.<module> import <Class>

def test_<feature>_<scenario>():
    """Test <what> when <condition>."""
    # Arrange
    instance = <Class>(...)

    # Act
    result = instance.method(...)

    # Assert
    assert result == expected
```

**Mocking exchange** :
```python
# Mock ccxt exchange
from unittest.mock import MagicMock, patch

@patch("bn_ml.exchange.ccxt.binance")
def test_fetch_ohlcv(mock_binance):
    mock_binance.return_value.fetch_ohlcv.return_value = [[...]]
    # Test logic
```

**Fixtures utiles** :
```python
@pytest.fixture
def config():
    """Minimal config for tests."""
    return {
        "exchange": {"name": "binance", "enableRateLimit": True},
        "risk": {"max_position_size_usdt": 100}
    }

@pytest.fixture
def state_store(tmp_path):
    """Temporary state store."""
    db_path = tmp_path / "test.db"
    return StateStore(str(db_path))
```

### 💡 Tips pour Claude

**Quand ajouter une feature ML** :
1. Vérifier impact sur pipeline training (FeatureEngineer, EnsembleTrainer)
2. Tester avec/sans feature (ablation study)
3. Documenter feature dans metadata.json
4. Ajouter test de feature engineering

**Quand modifier risk logic** :
1. Tests exhaustifs (unit + integration)
2. Simulation sur données historiques
3. Review approfondie (critical path)
4. Documentation risque résiduel

**Quand debugger model performance** :
1. Vérifier drift : `ml_engine/drift_monitor.py`
2. Métriques validation : `artifacts/state/bn_ml.db` table `model_metrics`
3. Feature importance : XGBoost/LightGBM `.feature_importances_`
4. Confusion matrix : Logger dans training loop

**Quand optimiser performance** :
1. Profiler avec `cProfile` ou `py-spy`
2. Identifier bottlenecks (data fetch, feature engineering, inference)
3. Optimiser requêtes ccxt (batch, caching)
4. Paralléliser scanner si nécessaire (multiprocessing)

**Quand ajouter endpoint API** :
1. Définir dans `public_api/app.py`
2. Ajouter route FastAPI avec type hints
3. Documenter dans docstring (OpenAPI auto-gen)
4. Tester avec `tests/test_public_api_*.py` (httpx)

## Composants clés détaillés

### BinanceDataManager (`data_manager/data_manager.py`)

**Responsabilité** : Abstraction CCXT pour paper/live modes

**Modes** :
- **Live** : API réelle Binance
- **Paper** : Synthetic (OHLCV simulé) + Live market fallback

**Méthodes principales** :
- `fetch_ohlcv(symbol, timeframe, limit)` : Fetch OHLCV bars
- `fetch_ticker(symbol)` : Prix actuel + volume
- `get_balance()` : Capital disponible
- `create_order(symbol, side, amount, price=None)` : Créer ordre (paper/live)
- `fetch_order_status(order_id)` : Statut ordre
- `cancel_order(order_id)` : Annuler ordre

**Configuration** : Section `exchange` + `data` dans bot.yaml

### FeatureEngineer (`data_manager/feature_engineer.py`)

**Responsabilité** : Génération features techniques + multi-timeframe

**Indicators** :
- Trend : EMA 9/21/50/200, MACD, ADX
- Momentum : RSI 14, Stochastic, CCI, MFI, Williams %R
- Volatility : ATR, Bollinger Bands, Keltner Channels
- Volume : OBV, VWAP, CMF
- Patterns : Candlestick patterns detection

**Multi-timeframe** :
- Fetch higher TF (1h, 4h, 1d)
- Compute indicators per TF
- Project sur base TF (15m) via forward-fill aligné

**Configuration** : Section `model.features` dans bot.yaml

### EnsembleTrainer (`ml_engine/trainer.py`)

**Responsabilité** : Entraînement ensemble RF+XGB+LGB+LSTM avec HPO

**Pipeline** :
1. Fetch data (via BinanceDataManager)
2. Feature engineering (via FeatureEngineer)
3. Labeling (DynamicLabeler) → BUY/SELL/HOLD
4. Split temporel (70/30 train/val avec purge)
5. HPO (Optuna) : trials parallèles, pruning, best params
6. Entraînement final avec best params
7. Calibration probabilités (CalibratedClassifierCV)
8. Validation metrics (accuracy, precision, recall, F1)
9. Sauvegarde bundle (modèles + metadata.json)

**Configuration** : Section `model` dans bot.yaml

### MLEnsemblePredictor (`ml_engine/predictor.py`)

**Responsabilité** : Chargement bundles + inférence temps réel

**Flow** :
1. Charger modèles depuis `models/<SYMBOL_KEY>/`
2. Fetch OHLCV récent (via BinanceDataManager)
3. Feature engineering (via FeatureEngineer)
4. Prédictions par modèle (RF, XGB, LGB, LSTM)
5. Vote pondéré des probabilités
6. Décision finale : BUY/SELL/HOLD avec confiance

**Sortie** :
```python
{
    "action": "BUY",           # ou "SELL", "HOLD"
    "confidence": 0.68,        # Probabilité classe prédite
    "probas": {                # Probabilités 3 classes
        "SELL": 0.12,
        "HOLD": 0.20,
        "BUY": 0.68
    }
}
```

**Configuration** : Section `model` dans bot.yaml

### MultiPairScanner (`scanner/multi_pair_scanner.py`)

**Responsabilité** : Scanner univers + scoring opportunités

**Flow** :
1. Fetch universe pairs (filtres liquidité/spread/depth)
2. Pour chaque pair :
   - Prédiction ML (via MLEnsemblePredictor)
   - Score technique (trend, momentum, volatility)
   - Score composite = ML + technical
3. Ranking pairs par score
4. Filtrage (min_score, max_pairs)
5. Retour top-N opportunités

**Sortie** :
```python
[
    {
        "symbol": "BTC/USDC",
        "action": "BUY",
        "confidence": 0.72,
        "score": 8.5,          # Score composite 0-10
        "price": 42350.0
    },
    ...
]
```

**Configuration** : Section `scanner` dans bot.yaml

### RiskManager (`trader/risk_manager.py`)

**Responsabilité** : Validation pré-trade STRICTE

**Checks** :
- Position sizing (max_position_size_usdt, capital disponible)
- Limites portfolio (max_open_positions, max_exposure_pct)
- Corrélations inter-positions (max_correlation)
- Circuit breakers (max_drawdown_pct, max_loss_per_day)
- Capital minimum (min_capital_usdt)
- Symétrie long/short (actuellement long-only)

**Sortie** :
```python
{
    "approved": True,          # False si refusé
    "reason": None,            # Message si refusé
    "adjusted_size_usdt": 95.0 # Size ajusté si besoin
}
```

**Configuration** : Section `risk` dans bot.yaml

### OrderManager (`trader/order_manager.py`)

**Responsabilité** : Exécution ordres paper/live avec validation exchange

**Modes** :
- **Paper** : Simulation fills (prix market + slippage)
- **Live** : API Binance réelle avec retry + error handling

**Validations** :
- Exchange constraints (minNotional, minQty, stepSize, tickSize)
- Balance suffisant
- Symbol trading activé

**Types ordres** :
- Market (exécution immédiate)
- Limit (prix spécifié)

**Configuration** : Section `exchange` + `environment` dans bot.yaml

### PositionManager (`trader/position_manager.py`)

**Responsabilité** : Tracking positions ouvertes + SL/TP/trailing

**État position** :
```python
{
    "symbol": "BTC/USDC",
    "side": "LONG",
    "size_usdt": 100.0,
    "entry_price": 42000.0,
    "stop_loss": 41300.0,      # -1.67%
    "take_profit_1": 42800.0,  # +1.9% (50% exit)
    "take_profit_2": 43400.0,  # +3.3% (30% exit)
    "opened_at": "2026-02-10T10:00:00Z",
    "trailing_stop": False,
    "tp1_hit": False,
    "tp2_hit": False
}
```

**Logique** :
- Sync avec exchange (live mode)
- Trailing stop après TP1 hit (optionnel)
- Exit scaling : TP1 50%, TP2 30%, final 20%
- Time stop : fermeture auto après max_holding_period
- Persiste dans SQLite (table `positions`)

**Configuration** : Section `risk` dans bot.yaml

### ExitManager (`trader/exit_manager.py`)

**Responsabilité** : Gestion exits (SL, TP, trailing, time stops)

**Types exits** :
- **Stop-loss** : Perte max définie (-1.5% par défaut)
- **Take-profit scaling** : TP1 +2%, TP2 +3.5%
- **Trailing stop** : Active après TP1 hit (optionnel)
- **Time stop** : Fermeture après max_holding_period (24h par défaut)

**Flow chaque cycle** :
1. Pour chaque position ouverte :
2. Fetch prix actuel
3. Check SL hit → close market order
4. Check TP1/TP2 hit → partial close
5. Check trailing stop breach → close
6. Check time stop expired → close
7. Update position state

**Configuration** : Section `risk` dans bot.yaml

### SanTradeIntelligence (`ml_engine/santrade_intelligence.py`)

**Responsabilité** : Agrégation market-wide des signaux ML

**Flow** :
1. Collecte prédictions de tous symbols universe
2. Scores globaux (bullish/bearish/neutral) par symbol
3. Ensemble SGD pour signal market-wide
4. Output :
   ```python
   {
       "timestamp": "2026-02-10T12:00:00Z",
       "market_signal": "BULLISH",   # ou "BEARISH", "NEUTRAL"
       "confidence": 0.65,
       "probas": {"BULLISH": 0.65, "NEUTRAL": 0.20, "BEARISH": 0.15},
       "contributing_symbols": ["BTC/USDC", "ETH/USDC", ...],
       "total_symbols": 15
   }
   ```

**Utilisation** :
- Contexte macro pour décisions trading
- Filter opportunités selon market regime
- Ajustement sizing/risk selon market confidence

**Configuration** : Section `santrade_intelligence` dans bot.yaml

### Dashboard Streamlit (`monitoring/dashboard.py`)

**Responsabilité** : Monitoring temps réel avec panels détachables

**Panels** :
- **Overview** : Capital, positions, PnL, equity curve
- **Positions** : Tableau positions ouvertes avec SL/TP/trailing
- **Opportunities** : Heatmap opportunités scanner avec scores
- **Trades** : Historique trades avec filtres
- **Models** : Status training, drift detection, progress bars
- **Metrics** : Métriques Prometheus, alertes, logs récents

**Features** :
- Auto-refresh (1-5s configurable)
- Panels détachables (ouvrir dans nouvel onglet)
- Filtres interactifs (symbol, date range, action)
- Plotly charts interactifs

**Configuration** : Section `monitoring.dashboard` dans bot.yaml

**Lancement** :
```bash
streamlit run monitoring/dashboard.py
# Ou auto-launch via bot : dashboard.auto_launch: true
```

## Environnement de développement recommandé

### IDE setup

**VSCode** :
- Extensions : Python, Pylance, Ruff, pytest
- Settings :
  ```json
  {
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["-q"],
    "editor.formatOnSave": false,
    "files.exclude": {
      "**/__pycache__": true,
      "**/*.pyc": true,
      ".pytest_cache": true
    }
  }
  ```

**PyCharm** :
- Interpreter : .venv Python 3.10+
- Test runner : pytest
- Code style : Default (pas de black enforced)

### Outils utiles

**Data exploration** :
```bash
# SQLite browser
sqlite3 artifacts/state/bn_ml.db
# Ou GUI : DB Browser for SQLite

# Logs tail
tail -f artifacts/logs/bnml.log

# Model inspection
python3 -c "import joblib; m = joblib.load('models/BTC_USDC/xgb.joblib'); print(m.feature_importances_)"
```

**Debugging** :
```python
# Breakpoints avec pdb
import pdb; pdb.set_trace()

# Ou ipdb (plus features)
import ipdb; ipdb.set_trace()

# Remote debugging (VSCode/PyCharm)
# Configurer launch.json ou Run configuration
```

**Profiling** :
```bash
# cProfile
python3 -m cProfile -o profile.stats scripts/run_bot.py --config configs/bot.yaml
python3 -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# py-spy (sampling profiler, low overhead)
py-spy record -o profile.svg -- python3 scripts/run_bot.py --config configs/bot.yaml
```

## FAQ et troubleshooting

### Q: Bot ne démarre pas, erreur "API keys not configured"

**R:** Créer `.env` avec `BINANCE_API_KEY` et `BINANCE_API_SECRET`. En mode paper, utiliser des dummy keys (préflight checks allégés).

### Q: Tests échouent avec "module not found"

**R:** Installer en mode editable : `pip install -e .`

### Q: Dashboard ne s'auto-lance pas

**R:** Vérifier `bot.yaml` : `monitoring.dashboard.auto_launch: true` et `port: 8501` libre.

### Q: Ordres refusés "minNotional not met"

**R:** Augmenter `risk.max_position_size_usdt` dans bot.yaml. Binance requiert minimum ~10-15 USDT par ordre.

### Q: Prédictions toujours HOLD

**R:** Vérifier thresholds dans metadata.json : `min_buy_proba` et `min_sell_proba`. Si trop élevés, zone neutre large. Recommandé : 0.38-0.42.

### Q: Drift détecté mais pas de retraining

**R:** Activer retraining auto : `model.retraining.enabled: true` et lancer `bnml-trainer-auto` ou worker thread dans bot.

### Q: Paper mode donne résultats différents de backtest

**R:** Paper mode utilise prix market actuels (live fallback) + slippage réaliste. Backtest plus simplifié (close prices). Normal d'avoir écarts.

### Q: Comment passer de paper à live ?

**R:**
1. Vérifier preflight : capital min, API keys valides, IP whitelisted
2. Éditer bot.yaml : `environment: live`
3. Tester avec capital minimum d'abord (ex: 100 USDT)
4. Monitoring alertes actives
5. Lancer : `bnml-bot --config configs/bot.yaml`

### Q: Kill-switch ne ferme pas toutes positions

**R:** Vérifier logs pour erreurs exchange. En cas de panne API Binance, attendre rétablissement. Kill-switch utilise market orders (exécution immédiate normalement).

### Q: Comment ajouter nouvelle paire trading ?

**R:**
1. Entraîner modèle : `bnml-trainer --config configs/bot.yaml --symbol BTC/USDC`
2. Bundle sauvegardé dans `models/BTC_USDC/`
3. Ajouter symbol dans universe : `universe.training_symbols` ou discovery via `universe.discovery.enabled: true`
4. Redémarrer bot

### Q: GPU pas détecté pour XGBoost

**R:**
1. Vérifier CUDA : `nvcc --version`
2. Installer xgboost avec GPU : `pip install xgboost[gpu]`
3. Hardware probe : `bnml-hardware-probe`
4. Config : `model.acceleration.gpu_enable: true`

### Q: API publique non accessible

**R:**
1. Lancer : `bnml-api --config configs/bot.yaml`
2. Vérifier port 8000 libre : `lsof -i :8000`
3. CORS : définir `BNML_API_CORS_ORIGINS` dans .env
4. Healthcheck : `curl http://localhost:8000/healthz`

## Ressources additionnelles

### Documentation externe

- **CCXT** : https://docs.ccxt.com/
- **Binance API** : https://binance-docs.github.io/apidocs/spot/en/
- **FastAPI** : https://fastapi.tiangolo.com/
- **Streamlit** : https://docs.streamlit.io/
- **XGBoost** : https://xgboost.readthedocs.io/
- **Optuna** : https://optuna.readthedocs.io/

### Communauté et support

- **Issues GitHub** : https://github.com/san2stic/BN-ML/issues
- **Discussions** : https://github.com/san2stic/BN-ML/discussions
- **Security** : Voir SECURITY.md pour reporting vulnérabilités

### Contribution

Voir CONTRIBUTING.md pour :
- Setup environnement local
- Workflow recommandé (branches codex/*)
- Pre-PR checks (pytest, single bot cycle)
- Standards techniques (no secrets, risk logic tests, HOLD bias)
- Commits atomiques, PR context

---

**Version CLAUDE.md** : 1.0.0 (2026-02-10)
**Dernière mise à jour** : 2026-02-10
**Auteur** : Analysé et généré par Claude (Anthropic) pour le projet BN-ML
