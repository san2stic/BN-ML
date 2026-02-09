# Model Sync (GitHub + RunPod)

Ce document couvre les deux modes de synchronisation des modèles BN-ML:
- distribution via repo GitHub (publisher/client)
- distribution via endpoint RunPod (trigger + polling + download archive)

Le script utilisé est `python3 -m scripts.model_sync`.

## 1) Mode GitHub

Configuration:

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

Publisher one-shot:

```bash
python3 -m scripts.model_sync publish --config configs/bot.yaml
```

Client one-shot:

```bash
python3 -m scripts.model_sync pull --config configs/bot.yaml
```

Daemon quotidien:

```bash
python3 -m scripts.model_sync daemon --role publisher --config configs/bot.yaml
python3 -m scripts.model_sync daemon --role client --config configs/bot.yaml
```

## 2) Mode RunPod

Configuration:

```yaml
model_sync:
  runpod_client:
    schedule: "03:00"
  runpod:
    enabled: true
    trigger_url: "https://api.runpod.ai/v2/<ENDPOINT_ID>/run"
    status_url_template: "https://api.runpod.ai/v2/<ENDPOINT_ID>/status/{job_id}"
    api_key_env: "RUNPOD_API_KEY"
    trigger_method: "POST"
    trigger_payload:
      input:
        prompt: "Train BN-ML models only (no trading bot execution) and return a downloadable models archive"
    headers: {}
    request_timeout_sec: 20
    poll_interval_sec: 10
    job_timeout_sec: 3600
    download_timeout_sec: 300
    extract_subdir: "models"
```

`.env`:

```bash
RUNPOD_API_KEY=rp_...
```

One-shot (trigger + attente + pull):

```bash
python3 -m scripts.model_sync runpod-pull --config configs/bot.yaml --models-dir models
```

Daemon quotidien:

```bash
python3 -m scripts.model_sync daemon --role runpod_client --config configs/bot.yaml --models-dir models
```

## 3) Docker

`docker-compose.yml` inclut le service `model-sync-runpod`:

```bash
docker compose --profile paper up -d bot-paper model-sync-runpod dashboard api prometheus grafana
docker compose --profile live up -d bot-live model-sync-runpod dashboard api prometheus grafana
```

Image RunPod recommandée:
- construire depuis `Dockerfile` (CMD par défaut = `scripts.runpod_train_only`, donc pas de `run_bot`)
- ou lancer explicitement:

```bash
python -m scripts.runpod_train_only --paper --train-missing-only --models-dir models --archive-path artifacts/exports/models_latest.zip
```

## 4) Sécurité opérationnelle

- ne jamais commiter de clé API RunPod
- conserver `extract_subdir: models` pour limiter l'extraction au sous-dossier attendu
- en mode Git, garder un worktree propre (blocage par défaut)
- le handler RunPod doit exécuter uniquement le training et produire une archive `models`, jamais `scripts.run_bot`
