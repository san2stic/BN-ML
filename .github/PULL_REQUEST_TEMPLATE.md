## Résumé

Décrivez ce que change cette PR et pourquoi.

## Type de changement

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor
- [ ] Documentation
- [ ] CI / tooling

## Validation

- [ ] `python3 -m pytest -q`
- [ ] Tests ou checks ciblés ajoutés/mis à jour
- [ ] Aucun secret/API key commité

## Checklist risque (obligatoire si logique trading)

- [ ] Les préconditions risque sont conservées avant ordre
- [ ] Le comportement par défaut reste `HOLD` en cas de doute
- [ ] Les limites daily/weekly ne sont pas contournées
- [ ] Les changements RiskManager sont couverts par tests

## Notes de déploiement

Impacts potentiels en runtime, migration de config, rollback.
