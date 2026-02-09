# Contribuer à BN-ML

Merci de contribuer.
La priorité du projet est non négociable: `robustesse risque > performance brute`.

## Setup local

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Workflow recommandé

1. Créer une branche depuis `main` (`codex/<topic>` recommandé).
2. Implémenter des changements petits et ciblés.
3. Ajouter/adapter les tests dans `tests/`.
4. Exécuter les vérifications avant push.

## Vérifications minimales avant PR

```bash
python3 -m pytest -q
python3 -m scripts.run_bot --once --paper --disable-retrain --no-dashboard
```

## Standards techniques

- Ne jamais commiter de secrets/API keys.
- En live, ne jamais contourner les checks preflight.
- Toute modification de logique risque doit inclure des tests.
- Préférer `HOLD` si qualité signal/liquidité incertaine.
- Garder les changements rétro-compatibles côté config autant que possible.

## Commits et PR

- Commits atomiques avec message explicite.
- PR avec contexte, impacts et stratégie de rollback si nécessaire.
- Utiliser la checklist du template PR.
