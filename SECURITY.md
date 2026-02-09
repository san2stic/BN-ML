# Politique de sécurité

## Signaler une vulnérabilité

Ne créez pas d'issue publique pour une faille de sécurité.
Ouvrez une GitHub Security Advisory (privée) ou contactez le mainteneur du dépôt en privé.

Informations utiles dans le signalement:

- Description du problème
- Surface impactée (module/fichier)
- Conditions d'exploitation
- Impact estimé (fonds, exécution, confidentialité)
- Preuve de concept minimale
- Proposition de mitigation si possible

## Portée sensible

Ce dépôt manipule:

- décisions d'achat/vente spot
- logique de risk management
- connecteurs exchange/API
- persistance d'état et alerting runtime

Toute vulnérabilité pouvant dégrader la préservation du capital est traitée en priorité haute.

## Bonnes pratiques

- Jamais de secrets/API keys dans le code ou les logs.
- Rotation immédiate des clés en cas d'exposition.
- Vérifier strictement les checks preflight avant mode live.
- Maintenir les dépendances à jour (Dependabot + audit).
