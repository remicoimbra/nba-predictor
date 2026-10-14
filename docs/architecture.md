# Architecture

Ce document explique comment le projet est construit et pourquoi. Pour lancer l'application, voir le [README](../README.md).

## Vue d'ensemble

```
PC de développement (IP résidentielle)
  nba_api ──> data/collectors ──> data/preprocessing ──> data/models
                                        │                    │
                          team_state.json, schedule.json    xgboost_v1.pkl, feature_columns.json,
                                        │                    model_metrics.json
                                        ▼                    ▼
                                 api/ (FastAPI) ── lit les fichiers, prédit à la demande
                                        │ JSON
                                        ▼
                                 front/ (Next.js) ── 3 pages
```

Il n'y a pas de base de données : des fichiers CSV et JSON font office de stockage. C'est un choix assumé à cette échelle (8 279 matchs, 30 équipes), qui évite une dépendance de plus. Le schéma SQL cible est décrit plus bas.

## Pipeline de données (`data/`)

| Étape | Script | Entrée → sortie |
| --- | --- | --- |
| Collecte | `collectors/fetch_games.py` | `nba_api` → `raw/games_raw_all_seasons.csv` (1 ligne par équipe) puis `raw/games_history.csv` (1 ligne par match) |
| Nettoyage | `preprocessing/clean_data.py` | → `processed/games_clean.csv` (typage strict, cible `home_win`, tri chronologique) |
| Features | `preprocessing/feature_engineering.py` | → `processed/game_features.csv` |
| Entraînement | `models/train_model.py` | → `saved_models/*.pkl`, `feature_columns.json`, `model_metrics.json` |
| État courant | `preprocessing/team_state.py` | → `processed/team_state.json` |
| Calendrier | `collectors/fetch_schedule.py` | → `processed/schedule.json` (J-3 à J+14) |

### Les variables du modèle

Pour chaque équipe, calculées **avant** le match :

- **Elo** : note qui monte après une victoire, d'autant plus que l'adversaire était fort et l'écart large (même principe que FiveThirtyEight). Régression de 25 % vers la moyenne (1500) entre deux saisons.
- **Bilan** : pourcentage de victoires de la saison, spécifique au contexte (domicile ou extérieur), et sur les 10 derniers matchs.
- **Points marqués et encaissés** : moyennes sur 5 et 10 matchs.
- **Net Rating estimé** : (points marqués − encaissés) rapportés aux possessions estimées à partir du boxscore.
- **Jours de repos**, plafonnés à 7.
- **Différences** domicile − extérieur de chacune, généralement plus discriminantes qu'une paire de colonnes séparées.

### Éviter la fuite temporelle

Une feature qui utilise une information connue seulement après le match donne un modèle excellent en test et inutile en production. Trois protections :

1. Chaque statistique glissante est décalée d'un match (`shift(1)`), et l'Elo enregistré est celui d'**avant** le match.
2. Le jeu de test est la saison la plus récente (2025-26), jamais mélangée à l'entraînement : pas de découpage aléatoire.
3. `game_features.csv` contient aussi le boxscore brut du match joué (`home_fga`, `away_oreb`…), connu seulement après coup. `train_model.py` choisit ses colonnes par **liste blanche de motifs** (`FEATURE_PATTERNS`) et jamais par liste noire : une nouvelle colonne ne devient pas une feature par accident.

### `features_lib.py` : une seule définition des formules

L'entraînement calcule les features sur tout l'historique (vectorisé, pandas). L'API les calcule pour un seul match futur à partir de `team_state.json`. Si les deux codes divergeaient, le modèle tournerait sans erreur mais avec de moins bonnes prédictions, sans qu'on le voie. `features_lib.py` contient donc les constantes (fenêtres, paramètres Elo, plafond de repos) et les fonctions pures utilisées des deux côtés. La non-régression a été vérifiée : l'ancienne et la nouvelle implémentation de l'Elo donnent un écart de 0,0 sur les 8 279 matchs.

### Cas particuliers gérés à la prédiction

- **Changement de saison** : la saison d'un match est déduite de sa date (frontière en août). Si elle est postérieure à celle de `team_state.json`, on reproduit ce que fait l'entraînement : Elo régressé de 25 % et bilans de la saison remis à zéro (manquants). Les moyennes glissantes ne sont pas remises à zéro, comme à l'entraînement.
- **Valeurs manquantes** : laissées en `NaN`, XGBoost les gère nativement, plutôt qu'une valeur arbitraire qui biaiserait la prédiction.
- **Jours de repos** : plafonnés à 7 à l'entraînement et en prédiction (`REST_DAYS_CAP`). Sans plafond, les ~100 matchs d'ouverture de saison de l'historique (130 à 288 jours après le dernier match, à cause de l'intersaison et de la « bulle » de 2020) donnent des valeurs que le modèle voit trop peu. Mesuré sur 5 graines : le plafond ne change pas la qualité de façon mesurable (69,1 % sans plafond contre 68,6 % avec, pour un écart-type de 0,4 point entre graines). Une date antérieure au dernier match connu donne un repos manquant plutôt qu'un nombre négatif.

## API (`api/`)

FastAPI, un routeur par domaine. La logique est dans `services/`, les routes dans `routers/`.

- `services/predictor_service.py` : charge une seule fois le modèle, `feature_columns.json` et `team_state.json`, construit les features d'un match et prédit. Relit `team_state.json` quand sa date de modification change, sans redémarrage. Calcule aussi l'explication : les contributions XGBoost de type SHAP (`pred_contribs`) en log-odds, sommées par famille de statistiques. La sigmoïde de (biais + somme des contributions) redonne exactement la probabilité affichée.
- `services/schedule_service.py` : lit `schedule.json`. Un appel direct à la NBA n'a lieu que si `NBA_LIVE_SCHEDULE_FALLBACK` est actif (développement local).
- `routers/games.py` : `/games/today`, `/games/calendar`, `/games/{game_id}`. Erreurs distinguées : 400 date mal formée, 404 date non synchronisée, 502 calendrier NBA indisponible, 503 modèle non prêt. Un match avec une équipe inconnue renvoie `prediction: null` sans faire échouer le reste de la liste.
- `routers/model.py` : `/model/metrics`.

### Limite connue : pas d'historique par date

`team_state.json` ne contient que l'état **actuel** des équipes. Interroger une date passée donne donc les matchs réels de ce jour mais avec l'Elo et la forme d'aujourd'hui : utile pour tester l'interface, inutile pour un vrai backtest.

### Le score estimé

Le modèle ne classe que victoire ou défaite. Le score affiché vient d'une heuristique simple (forme offensive et défensive récente) et doit être lu comme un ordre de grandeur.

## Front (`front/`)

Next.js (App Router), JavaScript, Tailwind CSS v4 (configuration dans `globals.css`, pas de `tailwind.config.js`).

- **Pages** : `/` (matchs du jour, date dans l'URL), `/match?id=…&date=…`, `/modele`. Le front est un export statique (`output: "export"`) : aucune route dynamique côté serveur, d'où l'identifiant du match en paramètre plutôt que dans le chemin.
- **Couleurs** : chaque couleur est déclarée une fois en `light-dark(clair, sombre)`. `npm run check:contrast` vérifie toutes les paires (7:1 pour le texte, 3:1 pour les graphiques) : à relancer après toute modification de couleur. Le rouge NBA n'est jamais utilisé pour du texte (contraste insuffisant).
- **Graphiques** : SVG et HTML maison, sans bibliothèque. Chacun a un `aria-label`, un tableau de données alternatif (« Voir les données ») et une infobulle au survol.
- **Animations** : `motion`, avec respect de `prefers-reduced-motion`.
- **Logos** : les 60 fichiers SVG sont versionnés dans `public/logos/` et servis par le front, car `cdn.nba.com` est bloqué depuis certains réseaux.
- **Calendrier** : composant fait maison (bandeau de 14 jours + grille mensuelle, clavier complet : flèches, Début/Fin, Page précédente/suivante, Entrée, Échap).

## Pourquoi la synchronisation part du PC

`stats.nba.com` ne répond pas aux requêtes venant d'IP de datacenters (le serveur laisse la connexion ouverte sans répondre), et `cdn.nba.com` renvoie un 403 (Akamai). Testé depuis un VPS OVH : les deux sont bloqués. L'architecture de déploiement en découle :

```
PC (tâche planifiée Windows, 1 fois par jour)
  infra/sync_to_vps.ps1 : fetch_games --refresh → clean_data → team_state → fetch_schedule
        │  scp vers un fichier .tmp puis mv atomique (SSH par clé)
        ▼
Serveur : data/processed/team_state.json + schedule.json
        ▲ relus automatiquement par l'API quand leur date de modification change
Internet ──HTTPS──> Caddy ──> /     fichiers statiques du front (export Next.js)
                          └─> /api  API FastAPI, service systemd isolé (NBA_LIVE_SCHEDULE_FALLBACK=0)
```

Détails du déploiement (isolation de l'API, redéploiement, retour arrière) : [deploiement.md](deploiement.md).

L'API sur le serveur ne fait aucun appel réseau vers la NBA. Si le PC reste éteint plusieurs jours, les prédictions s'appuient sur un état de plus en plus ancien, puis le calendrier ne couvre plus la date du jour (404) : le service se dégrade, il ne tombe pas.

## Schéma de base de données (cible, non implémenté)

Si les fichiers deviennent insuffisants : PostgreSQL local (conteneur Docker). Tables prévues : `teams`, `games`, `game_features`, `predictions` (version du modèle, vainqueur prédit, probabilité), `model_metrics`. Les colonnes de `game_features` devront suivre les vraies variables du modèle (Elo, Net Rating, bilan selon le contexte), pas le schéma initial du projet.

## Décisions et pistes

| Sujet | Décision |
| --- | --- |
| Elo plutôt que bilan seul | L'Elo pondère par la force de l'adversaire : +2 points de précision, la variable la plus importante du modèle |
| Bilan selon le contexte **en plus** du bilan global | Le remplacer abîmait la précision (échantillon divisé par deux, environ 20 matchs par saison) |
| Régression logistique avec `StandardScaler` | L'Elo (1300 à 1700) et les pourcentages (0 à 1) ont des échelles très différentes |
| Matchs de présaison | Gardés, avec un badge « Présaison » : le modèle n'est entraîné que sur la saison régulière |
| Pistes | Réglage des hyperparamètres, validation croisée temporelle sur plusieurs saisons (le gain de l'Elo n'est mesuré que sur une saison de test), vrai modèle de régression pour le score |
