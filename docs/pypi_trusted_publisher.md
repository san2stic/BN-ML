# Publication PyPI (Trusted Publisher)

Objectif: publier automatiquement `bn-ml-trading-bot` sur PyPI à chaque release GitHub.

Workflow utilisé: `.github/workflows/publish-pypi.yml` (job `publish`).

## 1) Préparer PyPI

1. Créer un compte sur https://pypi.org et activer la 2FA.
2. Si le projet `bn-ml-trading-bot` n'existe pas encore:
   - aller dans `Your account` -> `Publishing`
   - créer un `pending publisher` pour ce nom de projet
   - cela permet un premier publish sans token API

## 2) Configurer Trusted Publisher sur PyPI

Dans PyPI (`Your account` -> `Publishing` ou dans le projet existant):
- `Publishing` -> `Add a new pending publisher`
- Renseigner:
  - `PyPI project name`: `bn-ml-trading-bot`
  - `Owner`: `san2stic`
  - `Repository name`: `BN-ML`
  - `Workflow name`: `Publish PyPI`
  - `Environment name`: `pypi`

Important: le `Workflow name` doit correspondre exactement à `name:` dans le workflow GitHub.

## 3) Vérifier côté GitHub

Le workflow est déjà prêt avec:
- `permissions: id-token: write` (OIDC)
- `environment: pypi`
- publication via `pypa/gh-action-pypi-publish@release/v1`

Aucun token `PYPI_API_TOKEN` n'est nécessaire avec Trusted Publishing.

## 4) Déclenchement

- Automatique: à chaque GitHub Release `published`.
- Manuel: `Actions` -> `Publish PyPI` -> `Run workflow`.

## 5) Installation finale

Après la première publication réussie:

```bash
pip install bn-ml-trading-bot
```
