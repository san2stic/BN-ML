# GitHub Model Sync

Ce document décrit le flux de distribution des modèles BN-ML via un repository GitHub dédié.

## 1) Architecture cible

- `publisher` (serveur d'entraînement):
  - entraîne les modèles quotidiennement
  - pousse le dossier modèles vers un repo GitHub
- `client` (serveur bot consumer):
  - récupère les derniers modèles depuis GitHub
  - remplace son dossier local `models/`

Le script utilisé est `python3 -m scripts.model_sync`.

## 2) Paramètres principaux

Dans `configs/bot.yaml`:

```yaml
model_sync:
  repo_dir: "/abs/path/to/model-registry-repo"
  remote: "origin"
  branch: "main"
  repo_models_subdir: "models"
  publisher:
    schedule: "00:00"
    train_before_push: true
  client:
    schedule: "06:00"
```

## 3) Publisher

One-shot:

```bash
python3 -m scripts.model_sync publish --config configs/bot.yaml
```

Par défaut:
- lance l'entraînement (`train_before_push: true`)
- synchronise le dossier local `models/` vers `<repo_dir>/models`
- commit/push uniquement s'il y a des changements

Options utiles:
- `--skip-training`
- `--symbol BTC/USDC --symbol ETH/USDC`
- `--train-missing-only` ou `--train-all`
- `--commit-message "chore(models): daily refresh"`

## 4) Client

One-shot:

```bash
python3 -m scripts.model_sync pull --config configs/bot.yaml
```

Comportement:
- `git pull --ff-only` depuis la branche configurée
- miroir de `<repo_dir>/models` vers le dossier local `models/`

## 5) Scheduler quotidien

Daemon intégré:

```bash
python3 -m scripts.model_sync daemon --role publisher --config configs/bot.yaml
python3 -m scripts.model_sync daemon --role client --config configs/bot.yaml
```

Ou cron:

```cron
0 0 * * * cd /ABS/PATH/BN-ML && /usr/bin/python3 -m scripts.model_sync publish --config configs/bot.yaml >> artifacts/logs/model_sync_publisher.log 2>&1
0 6 * * * cd /ABS/PATH/BN-ML && /usr/bin/python3 -m scripts.model_sync pull --config configs/bot.yaml >> artifacts/logs/model_sync_client.log 2>&1
```

## 6) Sécurité opérationnelle

- utiliser un repo GitHub dédié aux modèles
- configurer l'auth Git (SSH ou token) côté machine
- ne jamais stocker de secrets dans le repo modèles
- garder le worktree propre (comportement bloquant par défaut)
