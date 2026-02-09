# Incident Runbook

## Scope
- Binance Spot runtime incidents for BN-ML in `paper` or `live`.
- Priorite absolue: preservation du capital.

## Severity
- `SEV-1`: risque capital immediat (ordres incoherents, pertes anormales, boucle ordres).
- `SEV-2`: trading degrade (data stale, drift/volatility breakers permanents, alerting KO).
- `SEV-3`: observabilite/deploiement (dashboard indisponible, backup en echec).

## Immediate Actions (SEV-1)
1. Activer le kill switch:
```bash
python3 -m scripts.kill_switch --config configs/bot.yaml
```
2. Stopper le bot:
```bash
pkill -f "python -m scripts.run_bot" || true
```
3. Capturer les artefacts:
```bash
python3 -m scripts.generate_dod_report --days 7
```
4. Archiver logs/state:
- `artifacts/logs/`
- `artifacts/state/bn_ml.db`
- `artifacts/backups/`

## Daily Incident Triage
1. Verifier checks DoD:
```bash
python3 -m scripts.check_dod_daily --fail-on-violation
```
2. Inspecter les violations dans:
- `artifacts/reports/dod/daily/<YYYY-MM-DD>.json`
3. Verifier dernier resume:
- `artifacts/reports/dod/dod_v1_summary.md`

## Common Scenarios

### 1) Flux marche indisponible / stale
- Symptomes: scan vide, prix figes, erreurs exchange repetitives.
- Actions:
  1. verifier connectivite Binance/public API.
  2. redemarrer bot.
  3. si persistant, passer `paper_market_data_mode: synthetic` temporairement (paper uniquement).

### 2) Breakers risque declenches en continu
- Symptomes: `Skip ... risk budget/circuit breaker active`.
- Actions:
  1. verifier `daily_realized_usdt`, `weekly_realized_usdt`, `market_volatility_ratio`, `market_drift_detected` dans `kv_state.account_state`.
  2. maintenir `HOLD` jusqu'a normalisation regime.
  3. ne pas bypass les checks.

### 3) Alerting externe KO
- Symptomes: logs alertes locales mais pas de Telegram/webhook/email.
- Actions:
  1. verifier config `monitoring.alerting.*`.
  2. verifier credentials/reseau sortant.
  3. garder monitoring dashboard actif jusqu'a remediation.

### 4) Backup runtime absent
- Symptomes: pas de nouveau dossier `artifacts/backups/`.
- Actions:
  1. verifier `storage.backup.enabled: true`.
  2. verifier permissions ecriture dossier.
  3. lancer backup force via cycle runtime puis recontrole.

## Recovery Checklist
1. Incident contenu (positions fermees si necessaire).
2. Cause racine identifiee.
3. Patch + tests executes (`python3 -m pytest -q`).
4. Reprise en `paper` avec surveillance renforcee.
5. Postmortem court:
- timeline
- impact
- cause
- action corrective
- action preventive
