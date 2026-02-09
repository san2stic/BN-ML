# Docker Deployment

## Prerequisites
- Docker + Docker Compose v2
- `.env` configure avec credentials Binance si mode live
- `configs/bot.yaml` valide

## Build
```bash
docker compose build
```

## Profiles
- `paper`: campagne DoD paper 30 jours + dashboard
- `live`: runtime live + dashboard

## Start Paper DoD Campaign
```bash
docker compose --profile paper up -d bot-paper dashboard
```

Verifier:
```bash
docker compose ps
docker compose logs -f bot-paper
```

## Start Live Runtime
```bash
docker compose --profile live up -d bot-live dashboard
```

## Generate DoD Summary Report
```bash
docker compose --profile paper run --rm report
```

Artefacts:
- `artifacts/reports/dod/dod_v1_summary.md`
- `artifacts/reports/dod/daily/*.json`

## Stop Services
```bash
docker compose down
```

## Recommended Operations
1. Lancer `python3 -m scripts.check_dod_daily --fail-on-violation` quotidiennement.
2. Garder `storage.backup.enabled: true`.
3. En live, ne jamais desactiver preflight et circuit breakers.

## Troubleshooting
- Dashboard inaccessible:
  - verifier port `8501`
  - verifier `docker compose logs dashboard`
- Bot ne demarre pas en live:
  - verifier `.env` (`BINANCE_API_KEY`, `BINANCE_API_SECRET`)
  - verifier `exchange.testnet: false`
- Aucun rapport genere:
  - verifier presence `artifacts/state/bn_ml.db`
  - executer `docker compose run --rm report`
