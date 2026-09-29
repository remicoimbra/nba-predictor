# CLAUDE.md — NBA Predictor

Ce fichier sert de contexte pour reprendre le projet dans une future session avec Claude. Il décrit l'objectif, l'architecture, la stack, et l'état d'avancement réel du code (pas juste le plan initial).

## Contexte

Projet portfolio réalisé par un étudiant en 2ᵉ année de BUT MMI (spécialité Dev). L'objectif est de démontrer des compétences fullstack + une intro au Machine Learning, via une application de prédiction de scores et résultats de matchs NBA.

## Objectif du projet

Application hybride Python/JavaScript qui :

1. Récupère des données historiques NBA (`nba_api`)
2. Entraîne un modèle ML (scikit-learn / XGBoost) pour prédire l'issue des matchs (victoire + score approximatif)
3. Expose les prédictions via une API REST (FastAPI)
4. Affiche le tout sur un dashboard web moderne (Next.js + Tailwind CSS)

## Stack technique

| Domaine         | Techno                                                                    |
| --------------- | ------------------------------------------------------------------------- |
| Data & ML       | Python, `nba_api`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib` |
| Back-End / API  | Python, FastAPI, uvicorn                                                  |
| Base de données | PostgreSQL ou Supabase (pas encore branchée)                              |
| Front-End       | Next.js (App Router) + Tailwind CSS, JavaScript (pas TypeScript)          |
| Hébergement     | VPS personnel de l'utilisateur (pas encore déployé, voir "Déploiement")   |

## Arborescence du projet

```
nba-predictor/
│
├── data/                          # Scripts Python : collecte & ML
│   ├── collectors/
│   │   ├── fetch_games.py          # FAIT — voir "État du pipeline ML" ci-dessous
│   │   ├── fetch_schedule.py       # FAIT (2026-09-28) — calendrier J-3 → J+14 → processed/schedule.json (lu par l'API)
│   │   ├── fetch_players_stats.py  # pas encore développé
│   │   └── fetch_teams_stats.py    # pas encore développé
│   ├── preprocessing/
│   │   ├── clean_data.py           # FAIT
│   │   ├── feature_engineering.py  # FAIT — v3 (Pace/Net Rating, win_pct contexte, Elo), refactoré cette session
│   │   ├── features_lib.py         # FAIT (nouveau, cette session) — logique Elo/rolling stats PARTAGÉE entre feature_engineering.py (batch) et predictor_service.py (live)
│   │   └── team_state.py           # FAIT — construit team_state.json (état courant par équipe) ; + opponent_id/elo_before par match récent et elo_history (30 pts) depuis la refonte UI
│   ├── models/
│   │   ├── train_model.py          # FAIT — LogReg + XGBoost comparés ; sauvegarde feature_columns.json ET model_metrics.json (lu par GET /model/metrics)
│   │   ├── evaluate_model.py       # pas encore développé (séparé de train_model.py)
│   │   └── saved_models/           # logreg_baseline.pkl, xgboost_v1.pkl, feature_columns.json, model_metrics.json (non versionnés)
│   ├── raw/                        # games_raw_all_seasons.csv, games_history.csv (non versionnés)
│   ├── processed/                  # games_clean.csv, game_features.csv, team_state.json (non versionnés)
│   ├── notebooks/
│   └── requirements.txt            # FAIT — versions de scikit-learn/xgboost/joblib alignées sur api/requirements.txt (les .pkl sont rechargés par l'API)
│
├── api/                            # Back-end FastAPI — vraies prédictions branchées cette session
│   ├── venv/                       # environnement virtuel Python (non versionné) — Python 3.14
│   ├── app/
│   │   ├── main.py                 # FAIT (réécrit cette session) — squelette FastAPI + CORS, logique déportée dans routers/
│   │   ├── core/
│   │   │   ├── config.py           # FAIT (2026-09-28) — NBA_LIVE_SCHEDULE_FALLBACK (CORS/URL encore en dur ailleurs)
│   │   │   └── database.py         # à créer (connexion Supabase/Postgres)
│   │   ├── models/                 # à créer (ORM SQLAlchemy)
│   │   ├── schemas/                # à créer (schémas Pydantic)
│   │   ├── routers/
│   │   │   ├── games.py            # FAIT — /games/today (+ résumé équipes), /games/calendar, /games/{game_id} (fiche complète)
│   │   │   ├── model.py            # FAIT (refonte UI) — /model/metrics (lit model_metrics.json)
│   │   │   ├── predictions.py      # à créer
│   │   │   ├── teams.py            # à créer
│   │   │   └── players.py          # à créer
│   │   └── services/
│   │       ├── predictor_service.py    # FAIT (cette session) — charge xgboost_v1.pkl + team_state.json, construit les features live et prédit
│   │       ├── schedule_service.py     # FAIT — lit processed/schedule.json ; appel direct ScoreboardV3 en repli (dev local uniquement)
│   │       └── nba_sync_service.py     # VIDE, OBSOLÈTE — remplacé par infra/sync_to_vps.ps1 (NBA bloquée depuis le VPS), à supprimer
│   ├── tests/
│   └── requirements.txt            # FAIT — pip freeze du venv (⚠️ sous PowerShell 5.1, `pip freeze > requirements.txt` écrit en UTF-16 : reconvertir en UTF-8)
│
├── front/                          # Next.js + Tailwind — ADAPTÉ au nouveau format cette session (voir "Adaptation front" plus bas)
│   ├── src/
│   │   ├── app/                    # voir "Refonte UI" plus bas
│   │   │   ├── layout.js           # polices (Inter + Barlow Condensed), script de thème anti-flash, Header, lien d'évitement
│   │   │   ├── template.js         # fondu d'entrée des pages (motion)
│   │   │   ├── globals.css         # palette light-dark() des 2 thèmes (vérifiée AAA par scripts/check-contrast.mjs)
│   │   │   ├── page.js             # matchs du jour → components/HomeView.js (sous Suspense, lit ?date=)
│   │   │   ├── match/[id]/page.js  # fiche match → components/MatchView.js
│   │   │   └── modele/page.js      # page « Le modèle » → components/ModelView.js
│   │   ├── components/
│   │   │   ├── calendar/           # DatePicker (bandeau de jours) + CalendarPopover (grille mensuelle clavier) + dayStatus
│   │   │   ├── charts/             # graphiques SVG/HTML maison animés (motion) : ProbabilityBar, WinGauge, FactorsChart, ComparisonBars, FormChart, EloLine, ModelCharts, DataTable, Tooltip
│   │   │   ├── TeamBits.js         # TeamBadge (logo, sinon tricode), FormPills, PreseasonBadge
│   │   │   ├── GameCard.js, DaySummary.js, States.js, Header.js, ThemeToggle.js, Providers.js
│   │   ├── lib/                    # format.js (dates/nombres fr-FR), useApi.js (fetch + états), charts.js (échelles, useWidth, transitions)
│   │   └── services/
│   │       └── api.js              # getGames, getGame, getCalendar, getModelMetrics ; base = NEXT_PUBLIC_API_URL (défaut http://127.0.0.1:8000)
│   ├── public/logos/               # logos des 30 équipes, {team_id}-L.svg (fond clair) / -D.svg (fond sombre), VERSIONNÉS
│   ├── scripts/check-contrast.mjs  # `npm run check:contrast` : toutes les paires de couleurs >= 7:1 (texte) / 3:1 (graphiques)
│   ├── scripts/fetch-logos.mjs     # `npm run fetch:logos` : (re)télécharge public/logos/ depuis cdn.nba.com
│   ├── package.json
│   └── postcss.config.mjs          # Tailwind v4 : config "CSS-first" (@import "tailwindcss" dans globals.css), PAS de tailwind.config.js
│
├── infra/
│   ├── docker-compose.yml          # fichier vide (à écrire)
│   ├── Dockerfile.api              # fichier vide (à écrire)
│   ├── Dockerfile.data             # fichier vide (probablement inutile : la synchro tourne sur le PC, pas sur le VPS)
│   └── sync_to_vps.ps1             # FAIT (2026-09-28) — synchro quotidienne PC → VPS, voir "Déploiement"
│
├── docs/
│   └── architecture.md             # à compléter (peut reprendre le contenu de ce fichier)
│
└── README.md
```

## Choix faits lors du setup Next.js (`create-next-app`)

Ces choix conditionnent la structure du code, à respecter si on régénère ou étend le projet :

- **TypeScript** : Non (JavaScript pur)
- **ESLint** : Oui
- **Tailwind CSS** : Oui
- **React Compiler** : Non (feature expérimentale, pas nécessaire pour ce projet)
- **`src/` directory** : Oui
- **App Router** : Oui (pas Pages Router)
- **Alias d'import personnalisé** : Non (garde `@/*` par défaut)

## Flux de données prévu (cible finale)

```
1. COLLECTE (data/collectors/) — nba_api → BDD (raw_games)
2. NETTOYAGE & FEATURES (data/preprocessing/) → BDD (processed_features)
3. ENTRAÎNEMENT (data/models/) → modèle exporté (.pkl / format XGBoost)
4. API (api/) → charge le modèle, prédit à la demande, stocke dans "predictions"
5. FRONT (front/) → fetch les endpoints, affiche dashboard
6. AUTOMATISATION (optionnel) → scheduler quotidien (cron / APScheduler)
```

**État actuel du flux : les étapes 1 → 2 → 3 → 4 → 5 sont fonctionnelles et validées de bout en bout (1→2→3 sur données réelles, 7 saisons, 8 279 matchs ; 4 testée via TestClient + mocks côté Claude, et confirmée fonctionnelle en conditions réelles côté utilisateur ; 5 adaptée au nouveau format de réponse et testée visuellement côté utilisateur avec de vraies données, voir "Adaptation front" plus bas). `/games/today` sert désormais de VRAIES prédictions XGBoost, affichées correctement dans le dashboard. Depuis la refonte UI du 2026-09-28, le front a 3 pages (matchs du jour, fiche match avec graphiques explicatifs, page « Le modèle »), voir "Refonte UI". Aucune base de données n'est encore connectée (les CSV + `team_state.json` font office de BDD pour l'instant). L'étape 6 (automatisation) n'est toujours pas commencée : `team_state.json` doit pour l'instant être régénéré à la main (`python preprocessing/team_state.py`).**

## Schéma de base de données (cible, pas encore implémenté)

```sql
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    nba_team_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(5),
    conference VARCHAR(10)
);

CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    nba_game_id VARCHAR(20) UNIQUE NOT NULL,
    game_date DATE NOT NULL,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    season VARCHAR(9),
    status VARCHAR(20)
);

CREATE TABLE game_features (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    home_avg_pts_last5 FLOAT,
    away_avg_pts_last5 FLOAT,
    home_win_pct FLOAT,
    away_win_pct FLOAT,
    home_rest_days INTEGER,
    away_rest_days INTEGER
    -- NOTE : le schéma cible ci-dessus date du setup initial et est
    -- maintenant en retard sur les vraies colonnes de game_features.csv
    -- (voir "État du pipeline ML" plus bas pour la liste à jour : Elo,
    -- Net Rating, win_pct contextuel, etc.). À mettre à jour si/quand la
    -- BDD est vraiment branchée.
);

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES games(id),
    model_version VARCHAR(20),
    predicted_winner_id INTEGER REFERENCES teams(id),
    win_probability FLOAT,
    predicted_home_score INTEGER,
    predicted_away_score INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(20),
    trained_at TIMESTAMP,
    accuracy FLOAT,
    log_loss FLOAT,
    mae_score FLOAT
);
```

## Endpoints API (cible)

```
# Matchs
GET  /games
GET  /games/{game_id}            # FAIT (refonte UI) — fiche complète, ?date= facultatif (sinon cherché dans schedule.json)
GET  /games/calendar             # FAIT (refonte UI) — {today, live_fallback, dates: {date: nb_matchs}}
GET  /games/today                # FAIT — vraies prédictions XGBoost (cette session). Paramètre optionnel ?date=YYYY-MM-DD (utile en intersaison / pour tester sur une date passée déjà en historique — voir limite dans "Intégration modèle → API" plus bas)

# Équipes
GET  /teams
GET  /teams/{team_id}/stats

# Prédictions
GET  /predictions
GET  /predictions/{game_id}
POST /predictions/generate

# Modèle
GET  /model/metrics              # FAIT (refonte UI) — contenu de model_metrics.json, 404 si absent
GET  /model/version

# Admin
POST /admin/sync/games
POST /admin/retrain
```

## État du pipeline ML (mis à jour après la session Elo/XGBoost)

### Collecte (`data/collectors/fetch_games.py`)

- 7 saisons régulières (`2019-20` → `2025-26`) via `nba_api` (`LeagueGameFinder`).
- `games_raw_all_seasons.csv` (16 578 lignes, 1 ligne par équipe par match) → recombiné en `games_history.csv` (8 279 matchs, 1 ligne par match).
- **Changement cette session** : `recombine_games()` conserve maintenant les stats de boxscore par équipe (FGM, FGA, FG3M, FTA, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS), en plus des scores. Nécessaire pour estimer Pace/Net Rating (voir plus bas). `main()` réutilise `games_raw_all_seasons.csv` s'il existe déjà au lieu de re-taper `stats.nba.com` (`load_or_fetch_raw()`).

### Nettoyage (`data/preprocessing/clean_data.py`)

- Typage strict (dates, ID, scores), cible `home_win`, tri chronologique anti-fuite.
- **Changement cette session** : type et nettoie aussi les nouvelles colonnes de boxscore (`BOX_SCORE_COLS`), avec repli silencieux si un ancien `games_history.csv` (sans ces colonnes) est utilisé.

### Feature engineering (`data/preprocessing/feature_engineering.py`) — v3

Réécrit en profondeur cette session, en 3 vagues successives :

1. **v1 (déjà en place avant cette session)** : `avg_pts_last5`, `win_pct` (global), `rest_days`.
2. **v2 — features avancées** :
   - `avg_pts_scored/allowed_last{5,10}` (fenêtres 5 ET 10, avant seulement 5).
   - `pace_est`, `off_rtg`, `def_rtg`, `net_rtg` : estimés par équipe à partir de son propre boxscore (`Pace ≈ FGA - OREB + TOV + 0.4*FTA`), en l'absence du boxscore adverse aligné sur la même ligne. Approximation raisonnable pour un projet portfolio, mais **contribution modeste au modèle final** (net_rtg est solide mais loin derrière win_pct/Elo en importance).
   - `win_pct_last10` : forme récente glissante, indépendante de la saison.
   - Features différentielles `diff_*` (home − away) : généralement plus discriminantes qu'une paire home/away séparée pour un modèle.
3. **v3 — contexte domicile/extérieur + Elo** :
   - `win_pct_context` : win % spécifique au contexte du match (domicile pour l'équipe qui reçoit, extérieur pour celle qui se déplace). **Ajouté EN PLUS de `win_pct` global, pas en remplacement** — un remplacement pur avait légèrement dégradé l'accuracy (dilution d'échantillon, ~20 matchs par saison au lieu de ~40). En complément du global, il apporte un vrai signal (3ᵉ feature XGBoost).
   - **`elo` (`home_elo`/`away_elo`/`diff_elo`)** : rating Elo par équipe, calculé séquentiellement match par match sur tout l'historique chronologique (pas équipe par équipe comme le reste — un match met à jour les 2 équipes en même temps). Formule classique + multiplicateur d'écart de score façon FiveThirtyEight + régression vers la moyenne entre saisons (25%). **C'est de loin la feature la plus forte du pipeline** (voir résultats ci-dessous) : contrairement à win_pct, elle pondère par la force de l'adversaire.
   - Détection automatique des colonnes disponibles (`has_box_score()`) : le script fonctionne même sans les colonnes de boxscore (désactive juste Pace/Net Rating), pas de crash si `games_clean.csv` est une ancienne version.
   - ⚠️ Piège corrigé en route : `game_features.csv` transporte aussi les colonnes de boxscore BRUT du match courant (`home_fga`, `away_oreb`...), connues seulement APRÈS le match → fuite temporelle si utilisées comme features. `train_model.py` s'en protège avec une liste blanche de motifs (`FEATURE_PATTERNS`), jamais une liste noire.

### Modèles (`data/models/train_model.py`)

- Split temporel strict : train sur saisons ≠ `2025-26`, test sur `2025-26` uniquement.
- Détection automatique des colonnes de features via liste blanche de motifs (`win_pct`, `rest_days`, `avg_pts_scored_last`, `avg_pts_allowed_last`, `net_rtg_last`, `elo`) — pas de liste figée à maintenir à la main.
- **Deux modèles entraînés et comparés systématiquement** : Régression Logistique (dans un `Pipeline` avec `StandardScaler` — nécessaire depuis l'ajout de l'Elo, qui vit sur une échelle ~1300-1700 très différente de `win_pct` en 0-1 ; corrige aussi le `ConvergenceWarning`) et XGBoost (`n_estimators=300, max_depth=3, learning_rate=0.05`).
- Les deux modèles sont sauvegardés dans `saved_models/` (`logreg_baseline.pkl` est maintenant le pipeline complet scaler+modèle, `xgboost_v1.pkl`).

### Résultats obtenus (progression au fil de la session, saison test 2025-26, 1 186 matchs)

| Étape                                             | Accuracy LogReg | Accuracy XGBoost | Log loss (meilleur) |
| ------------------------------------------------- | --------------- | ---------------- | ------------------- |
| Baseline ("home gagne toujours")                  | 55,1%           | —                | —                   |
| v1 (win_pct global, avg_pts_last5, rest_days)     | 66,75%          | 67,4%            | 0,605               |
| v3 sans Elo (+ Pace/Net Rating + win_pct_context) | 66,4%           | 67,6%            | 0,606               |
| **v3 + Elo + scaling LogReg (état final)**        | **68,6%**       | **69,6%**        | **0,596**           |

**Feature la plus importante dans les deux modèles : `diff_elo`** (loin devant `diff_win_pct`, qui était dominante avant l'ajout de l'Elo). `diff_net_rtg_last10` reste un signal solide en 2ᵉ position. `win_pct_context` apporte un complément modeste mais réel.

## Intégration modèle → API (session courante)

Objectif de la session : remplacer les données fake de `/games/today` par de vraies prédictions XGBoost. **Fait, testé, validé en conditions réelles.**

### `data/preprocessing/features_lib.py` (nouveau)

Extraction de toute la logique Elo + rolling stats/win% de `feature_engineering.py` vers un module partagé, pour que l'entraînement (batch, tout l'historique) et la prédiction (live, un seul match) utilisent EXACTEMENT les mêmes formules — sans ce module, un ajustement de fenêtre ou de formule Elo fait un jour dans `feature_engineering.py` pourrait dériver silencieusement de ce qu'utilise l'API.

- `elo_update()`, `elo_expected_home()`, `elo_mov_multiplier()` : fonctions pures, un match à la fois.
- `compute_elo_timeline()` : passe séquentielle sur tout l'historique (utilisée par `feature_engineering.py`), renvoie aussi l'état Elo final par équipe.
- `elo_for_target_season()` : version légère de la régression inter-saison, ne nécessite QUE le rating courant + sa saison de référence (pas tout l'historique) — c'est ce qu'utilise `predictor_service.py` à partir de `team_state.json`.
- `rolling_mean_last_n()`, `win_pct()`, `rest_days()`, `net_rtg_from_boxscore()` : fonctions pures réutilisables en live.
- Non-régression vérifiée : refactoring de `compute_elo_ratings()` dans `feature_engineering.py` donnant un écart de `0.0000000000` avec l'implémentation originale sur un jeu de test synthétique (cas limites inclus : ligne corrompue, nouvelle équipe, changement de saison) — puis confirmé sur les vraies données (8 279 matchs) : accuracies inchangées (68,6 %/69,6 %).

### `data/preprocessing/team_state.py` (nouveau)

Construit `data/processed/team_state.json` à partir de `games_clean.csv` : un instantané de l'état COURANT de chaque équipe (Elo, jusqu'à 10 derniers matchs avec boxscore, résultats de la saison en cours dom/ext/global, date du dernier match). C'est le pont entre le pipeline batch (qui a besoin de tout l'historique) et l'API (qui n'a besoin que de l'état actuel pour prédire un match à venir) — évite de recharger/retraiter 8000+ matchs à chaque requête.

- ⚠️ **`team_state.json` n'est PAS un instantané historique par date** : il ne contient que l'état le plus récent. Interroger `/games/today?date=` avec une date passée utilise donc l'Elo/la forme ACTUELS des équipes, pas ceux réels à cette date passée — utile pour un test bout-en-bout, pas pour un vrai backtest historique.
- À régénérer manuellement pour l'instant : `python preprocessing/team_state.py` (voir `nba_sync_service.py`, pas encore fait, pour l'automatisation).
- Testé et confirmé sur les vraies données : 30 équipes traitées, saison `2025-26` correctement détectée.

### `data/models/train_model.py` (modifié)

Sauvegarde désormais `data/models/saved_models/feature_columns.json` (liste ordonnée des colonnes utilisées à l'entraînement) en plus des `.pkl`. Nécessaire car XGBoost est strict sur l'ordre/les noms de colonnes : sans ce fichier, `predictor_service.py` devrait deviner ou recopier à la main l'ordre des features, avec un risque de désynchronisation silencieuse (prédiction fausse sans erreur visible).

### `api/app/services/predictor_service.py` (nouveau)

Cœur de l'intégration. Charge le modèle + `team_state.json` + `feature_columns.json` une seule fois (pas à chaque requête). Pour un match `home_team_id` vs `away_team_id` :

1. Récupère l'état des deux équipes dans `team_state.json`.
2. Construit les 31 features exactement comme à l'entraînement, via `features_lib.py`.
3. Gère les `None`/NaN nativement (XGBoost les supporte) plutôt que d'imputer une valeur arbitraire — utile en tout début de saison (moins de 5-10 matchs joués). Les `None` sont convertis en NaN via `.astype(float)` avant `predict_proba` (sinon colonne `object` → XGBoost lève une erreur, bug corrigé en session d'audit).
4. **Changement de saison** : la saison du match est déduite de SA date (`features_lib.season_for_date()`, frontière en août), pas de `team_state.json`. Si elle est postérieure à la saison de `team_state.json` (ex : match d'octobre 2026 avec un état arrêté en avril 2026), on reproduit le pipeline d'entraînement : régression Elo de 25 % vers 1500 ET `win_pct`/`win_pct_context` remis à zéro (→ NaN). `win_pct_last10` et les moyennes glissantes ne sont pas remis à zéro, comme à l'entraînement. Avant la session d'audit, ni la régression ni la remise à zéro n'étaient appliquées (décalage entraînement/prédiction qui aurait touché tous les matchs 2026-27).
5. Retourne probabilités de victoire + un score approximatif (**heuristique simple** basée sur forme offensive/défensive récente — le modèle ne fait QUE classifier victoire/défaite, il n'est pas entraîné à prédire un score).

- Erreurs distinguées : `PredictorNotReadyError` (modèle/fichiers absents) et `UnknownTeamError` (équipe absente de `team_state.json`, ex. nouvelle franchise) — jamais un plantage silencieux.
- Testé avec de vrais `team_id` (Boston Celtics vs Denver Nuggets) : 64,3 % de proba pour Boston à domicile, cohérent avec un Elo plus élevé (1710 vs 1604) et une meilleure forme récente.

### `api/app/services/schedule_service.py` (nouveau)

Récupère le calendrier des matchs du jour (ce qui manquait totalement : aucun collecteur ne s'occupait des matchs À VENIR, seulement de l'historique). Utilise `ScoreboardV3` de `stats.nba.com`, PAS l'endpoint "live" de `cdn.nba.com` :

- `cdn.nba.com` s'est révélé **géobloqué depuis le réseau de l'IUT** de l'utilisateur (Access Denied Akamai) — probablement une restriction liée aux droits de diffusion.
- `stats.nba.com` est le domaine déjà utilisé avec succès par `fetch_games.py` sur ce même réseau — pas de nouvelle dépendance réseau fragile.
- `ScoreboardV2` a été écarté également : explicitement déprécié par `nba_api` pour la saison 2025-26 (bug connu sur les scores en début de saison, signalé par les mainteneurs eux-mêmes), qui recommandent `ScoreboardV3`.
- Lit le JSON brut (`get_dict()`) plutôt que les DataFrames de l'endpoint : `homeTeam`/`awayTeam` y sont labellisés explicitement, plus robuste qu'une déduction par ordre d'insertion depuis le dataset `LineScore`.

### `api/app/routers/games.py` + `api/app/main.py` (nouveaux)

`main.py` : squelette FastAPI minimal (CORS pour `localhost:3000`, inclusion du routeur). `games.py` : combine `schedule_service` + `predictor_service`, avec gestion d'erreurs distinguée (400 date mal formée, 502 calendrier NBA indisponible, 503 modèle non prêt) — un match avec une équipe inconnue de `team_state.json` renvoie `prediction: null` pour ce match plutôt que de faire échouer tout l'endpoint.

Réponse de `/games/today` :

```json
{
  "date": "2026-09-15",
  "count": 1,
  "games": [
    {
      "game_id": "...",
      "game_time_utc": "...",
      "status": "...",
      "season_type": "regular_season",
      "home_team": {"id": ..., "name": "...", "tricode": "..."},
      "away_team": {"id": ..., "name": "...", "tricode": "..."},
      "prediction": {
        "home_win_probability": 0.643,
        "away_win_probability": 0.357,
        "predicted_winner": "home",
        "predicted_home_score": 115.9,
        "predicted_away_score": 112.3
      }
    }
  ]
}
```

**Aucune donnée fake n'a été vue/documentée pour l'ancien `main.py`** (le fichier n'existait plus/n'a pas pu être fourni en session) : ce schéma de réponse est donc une conception NOUVELLE, pas une reprise de l'existant.

### Adaptation front (FAIT, session suivante)

Le front utilisait encore l'ancien format de données fake (`game.home_team` en string, `game.win_probability`/`game.predicted_home_score` à la racine, `game.predicted_winner` = nom d'équipe). Adapté au nouveau schéma ci-dessus :

- **`api.js`** : `getTodayGames(date)` accepte désormais un paramètre `date` optionnel (`YYYY-MM-DD`), transmis en query string à `/games/today?date=...` s'il est fourni. Comportement inchangé si absent (matchs du jour).
- **`GameCard.js`** : déstructure `game.home_team.name`/`game.away_team.name` (objets, plus des strings) et `game.prediction.*`. `predicted_winner` ("home"/"away") est traduit en nom d'équipe réel, avec la probabilité correspondante (`home_win_probability` ou `away_win_probability` selon le vainqueur prédit — il n'y a plus de `win_probability` unique).
- **`page.js`** : réécrit en Client Component (`"use client"`, avant : Server Component `async`) pour supporter un **sélecteur de date** (`input type="date"`, état `useState`, refetch en `useEffect` à chaque changement). Gère 3 états explicites : chargement, erreur, liste vide (`data.games` extrait de l'objet `{date, count, games}` renvoyé par l'API — avant : `games.map` plantait car le front attendait un tableau brut).

⚠️ **Limite connue et acceptée** : le sélecteur de date interroge `team_state.json`, qui ne contient que l'état ACTUEL des équipes (voir limite déjà documentée dans "Intégration modèle → API" plus haut). Une date passée affiche donc les matchs réels de cette date mais avec l'Elo/la forme d'AUJOURD'HUI, pas ceux de l'époque — utile pour tester le pipeline visuellement (intersaison, pas de matchs avant mi-octobre 2026), pas un vrai backtest historique.

Validé visuellement côté utilisateur : cartes affichées correctement avec noms d'équipes, scores prédits, vainqueur et probabilité, sur une date de saison passée (`count: 0` confirmé par ailleurs en date du jour réelle, intersaison).

### Validation effectuée

- Non-régression Elo (refactoring `features_lib.py`) : testée sur données synthétiques ET sur les 8 279 matchs réels (accuracies identiques : 68,6 %/69,6 %).
- `team_state.py` : testé sur données synthétiques puis confirmé sur les vraies données (30 équipes, saison détectée correctement).
- `predictor_service.py` : testé end-to-end (modèle jouet + `team_state.json` synthétique), PUIS confirmé avec un vrai match (Celtics vs Nuggets, résultat cohérent).
- Routeur `/games/today` : 6 scénarios testés via `TestClient` + mocks (match normal, équipe inconnue, panne réseau simulée, modèle non chargeable, date invalide, date valide) — tous passent.
- Test réel côté utilisateur : API démarrée avec `uvicorn`, `/games/today` répond correctement (`count: 0` car intersaison — normal, la saison régulière démarre mi-octobre).
- **Non testé par Claude** : l'appel réseau réel à `nba_api`/`stats.nba.com` (sandbox Claude sans accès à ce domaine) — validé uniquement côté utilisateur.
- **Non testé du tout** : `/games/today` avec de VRAIS matchs programmés (impossible avant le retour de la saison régulière, mi-octobre 2026). Le paramètre `?date=` permet un test partiel avec des team_id réels dès maintenant, mais avec la limite de `team_state.json` évoquée plus haut (pas d'historique par date).

## Ce qui a été fait jusqu'à présent (chronologie)

1. Création de l'arborescence complète du projet (dossiers + fichiers vides) via script PowerShell.
2. Setup de l'API FastAPI (CORS, `/`, `/games/today` avec données fake) — testé et fonctionnel.
3. Setup du Front Next.js (`create-next-app`, `api.js`, `GameCard.jsx`, `page.js` Server Component) — testé et fonctionnel, tuyau Front ↔ API validé bout en bout avec données mockées.
4. **Collecte réelle** : `fetch_games.py` → 7 saisons, 8 279 matchs via `nba_api`.
5. **Nettoyage** : `clean_data.py` → `games_clean.csv`, cible `home_win`, tri chronologique.
6. **Feature engineering v1** : `avg_pts_last5`, `win_pct`, `rest_days`, anti-fuite via `.shift(1)`.
7. **Premier modèle (baseline)** : Régression Logistique, 66,75% accuracy vs 55,33% baseline.
8. **Session ML approfondie (cette session)** :
   - Boxscore complet conservé dès la collecte (`fetch_games.py`, `clean_data.py`).
   - Feature engineering v2 : Pace/Net Rating estimés, fenêtres 5 et 10, features différentielles.
   - Comparaison XGBoost vs Régression Logistique intégrée à `train_model.py`, avec détection automatique des features par liste blanche (protection anti-fuite).
   - Expérimentation win_pct contextuel (domicile/extérieur) : remplacement pur testé puis ajusté en ajout complémentaire après légère régression observée.
   - **Ajout d'un système de rating Elo par équipe** (le levier le plus efficace de la session) : +2 points d'accuracy XGBoost.
   - StandardScaler sur la régression logistique pour corriger le déséquilibre d'échelle introduit par l'Elo.
   - Résultat final : **68,6% (LogReg) / 69,6% (XGBoost)**, contre 66,75% en début de session.
9. **Session intégration modèle → API (cette session)** :
   - `features_lib.py` créé : Elo + rolling stats/win% extraits en module partagé entraînement/prédiction, non-régression vérifiée (écart 0.0 avec l'ancienne implémentation, données réelles incluses).
   - `team_state.py` créé : génère `team_state.json` (état courant des 30 équipes) à partir de `games_clean.csv`.
   - `train_model.py` modifié : sauvegarde `feature_columns.json` (ordre des colonnes) à côté des `.pkl`.
   - `predictor_service.py` créé : prédictions XGBoost réelles à partir de `team_state.json`, testé avec de vrais `team_id` (Celtics vs Nuggets).
   - `schedule_service.py` créé : récupération du calendrier du jour via `nba_api` — deux itérations (endpoint "live" cdn.nba.com abandonné car géobloqué depuis le réseau IUT ; `ScoreboardV2` abandonné car déprécié par `nba_api` pour 2025-26 ; `ScoreboardV3` retenu).
   - `main.py` + `routers/games.py` créés : `/games/today` sert désormais de vraies prédictions, avec gestion d'erreurs distinguée (400/502/503) et paramètre `?date=` optionnel.
   - Plusieurs allers-retours de correction de faux positifs Pylance (typage `Scalar`/`ExtensionArray` de pandas, attribut `Optional` mal inféré) — aucun n'était un bug d'exécution réel.
   - Validé en conditions réelles côté utilisateur : API démarrée, `/games/today` répond correctement (intersaison → `count: 0`, normal).
   - **Non fait cette session** : adaptation du front au nouveau format de réponse, BDD, `nba_sync_service.py` (régénération auto de `team_state.json`), `requirements.txt` de `api/` pas confirmé regénéré.
10. **Session front (session suivante, hors sandbox Claude — bug local + adaptation)** :
    - Bug de démarrage API résolu : `uvicorn app.main:app --reload` était lancé depuis `api/venv/Scripts` au lieu de `api/` → `ModuleNotFoundError: No module named 'app'`. Pas un bug de code, erreur de répertoire de travail après activation du venv.
    - `api.js`, `GameCard.jsx`, `page.js` adaptés au nouveau format de réponse de `/games/today` (voir "Adaptation front" plus haut) — tuyau front↔API↔modèle complet et validé visuellement côté utilisateur.
    - Sélecteur de date ajouté dans `page.js` (passé de Server Component à Client Component) pour tester l'interface en intersaison, sans attendre la reprise de la saison régulière mi-octobre 2026.
    - **Non fait cette session** : BDD, `nba_sync_service.py`, confirmation de `api/requirements.txt`, `routers/predictions.py`/`teams.py`/`players.py`, `data/requirements.txt`.
11. **Session d'audit d'architecture (première session Claude Code)** :
    - Bug corrigé : changement de saison non géré en prédiction live (Elo non régressé, `win_pct` de la saison précédente réutilisé) — voir point 4 de `predictor_service.py` plus haut.
    - Bug corrigé : `predictor_service.py` plantait dès qu'une feature valait `None` (colonne `object` refusée par XGBoost).
    - Bug corrigé : `GameCard.js` plantait sur `prediction: null` (cas prévu par l'API pour une équipe inconnue).
    - `page.js` : erreur ESLint `react-hooks/set-state-in-effect` corrigée (état de chargement dérivé de la date du dernier résultat) + réponses obsolètes ignorées lors de changements de date en rafale ; date du jour calculée en heure locale (plus en UTC).
    - `api/requirements.txt` reconverti d'UTF-16 en UTF-8 ; `data/requirements.txt` rempli (versions alignées sur l'API).
    - `.gitignore` : `data/raw/`, `data/processed/`, `data/notebooks/*.csv` ajoutés (documentés comme non versionnés mais ne l'étaient pas).
    - `front/tailwind.config.js` (vide, inutile en Tailwind v4) supprimé ; métadonnées `layout.js` (titre, `lang="fr"`) mises à jour.
    - Vérifié : prédiction réelle BOS–DEN sur 2026-01-15 (même saison, inchangée) et 2026-10-20 (Elo 1710 → 1657, win_pct → NaN) ; routeur appelé directement (équipe inconnue → `prediction: null`, date invalide → 400) ; `eslint` et `next build` OK. `TestClient` non utilisable dans le venv API (nécessite `httpx2`, non installé).
12. **Session préparation du déploiement VPS (2026-09-28)** :
    - Cible d'hébergement documentée : VPS OVH Starter de l'utilisateur (voir "Déploiement" ci-dessous).
    - Tests réseau depuis le VPS : `stats.nba.com` (timeout silencieux) ET `cdn.nba.com` (403 Akamai) bloqués → décision : synchro depuis le PC, poussée vers le VPS.
    - Implémenté : `fetch_games.py --refresh` + saisons calculées à partir de la date, `fetch_schedule.py` (calendrier → `schedule.json`), `schedule_service.py` qui lit ce fichier (appel direct à la NBA seulement en dev, `NBA_LIVE_SCHEDULE_FALLBACK`), rechargement auto de `team_state.json` dans `predictor_service.py`, `infra/sync_to_vps.ps1`.
    - Champ `season_type` ajouté à `/games/today` + badge "Présaison" dans `GameCard.js` (choix de l'utilisateur : garder les matchs de présaison, signalés).
    - Vérifié : synchro complète en local (`-NoPush`), 6 scénarios du routeur, connexion SSH par clé vers `ubuntu@remicoimbra.fr`, `eslint` sur `GameCard.js`. **Pas testé** : envoi réel vers le VPS, tâche planifiée, badge vu dans le navigateur.

13. **Refonte UI (2026-09-28)** : voir "Refonte UI" ci-dessous.
14. **Logos des équipes (2026-09-29)** : voir "Logos des équipes" dans "Refonte UI" ci-dessous.

## Refonte UI (2026-09-28)

Demande de l'utilisateur : palette NBA bleu/blanc/rouge + mode sombre bleu/noir/rouge, contrastes **WCAG AAA**, calendrier fait maison (pas l'`<input type="date">`), animations fluides, vrais graphiques (pas juste « 60 % »). Choix de l'utilisateur : fiche match sur une **page dédiée** `/match/[id]`, et une page **« Le modèle »**.

### Côté API / données
- `team_state.py` : chaque match de `recent_games` a en plus `opponent_id` et `elo_before` ; nouveau `elo_history` (`{date, elo}` après chacun des 30 derniers matchs, Elo après match recalculé avec `features_lib.elo_update`). Affichage uniquement, pas des features. L'API reste compatible avec un ancien `team_state.json` (champs manquants → `null`/`[]`).
- `train_model.py` écrit `model_metrics.json` (accuracy/log loss des 2 modèles, baseline, importances XGBoost, courbe de calibration en 10 tranches). Réentraîné le 2026-09-28 : modèle identique (écart de prédiction 0,0), 68,6 % / 69,6 %.
- `predictor_service.py` : `GamePrediction` gagne `elo_home_probability` (Elo seul), `factors` et `base_value`. Les facteurs sont les contributions XGBoost de type SHAP (`get_booster().predict(DMatrix, pred_contribs=True)`, en log-odds) sommées par famille (`FACTOR_FAMILIES` : Elo, bilan dom./ext., forme, bilan saison, Net Rating, attaque, défense, repos). Vérifié : sigmoïde(biais + somme) = `home_win_probability`. Nouveau `team_snapshot()` (stats + bilans V-D + derniers matchs + `elo_history`) ; la logique de changement de saison est factorisée dans `_season_context()`.
- `schedule_service.py` : `team_ref` (public), `find_game()`, `covered_dates()`.
- `routers/games.py` réécrit (même contrat pour `/games/today`, enrichi de `elo`, `record`, `last_results` par équipe et `prediction.elo_home_probability`) ; `routers/model.py` nouveau.

### Côté front
- **Palette** : chaque couleur déclarée une fois en `light-dark(clair, sombre)` dans `globals.css` ; `color-scheme` suit le système sans JS, le bouton Clair/Sombre/Auto pose `data-theme` sur `<html>` (localStorage `theme`, script inline anti-flash dans `layout.js`). Le rouge NBA `#C8102E` (5,9:1) n'est jamais utilisé pour du texte (`--danger-text` pour ça). **Toute modification de couleur → `npm run check:contrast`.** Séries des graphiques (domicile bleu `--home`, extérieur rouge `--away`) validées avec le validateur de palette du skill dataviz (daltonisme, bande de luminosité) dans les 2 thèmes.
- **Polices** : Inter (texte) + Barlow Condensed (titres, tricodes, scores).
- **Animations** : `motion` (dépendance ajoutée). `<MotionConfig reducedMotion="user">` + `useChartTransition()` (les attributs SVG animés ne sont pas couverts par MotionConfig) + media query CSS.
- **Graphiques** : SVG/HTML maison, pas de bibliothèque. Chacun a un `aria-label`, une vue « Voir les données » (tableau) et, pour les courbes/colonnes, une infobulle au survol.
- **Calendrier** : bandeau de 14 jours (pastille = jour avec matchs, via `/games/calendar`) + grille mensuelle (flèches, Début/Fin, Page préc./suiv., Entrée, Échap, focus rendu au bouton). Jours non synchronisés désactivés si `live_fallback` est faux (VPS). Date dans l'URL (`/?date=`).
- Vérifié : `eslint`, `next build`, contrastes, captures Chrome headless des 3 pages (clair, sombre, 375 px sans débordement), navigation clavier du calendrier pilotée par CDP. **Non vérifié** : lecteur d'écran réel, Safari/Firefox.
- ⚠️ Captures Chrome headless (`--screenshot --virtual-time-budget`) : elles ressortent parfois vides ou sans les cartes (animations `motion` et fondu de `template.js` pas terminés), alors que l'API a bien répondu 200. Relancer, ou vérifier dans un vrai navigateur, avant de conclure à un bug.

### Logos des équipes (2026-09-29)
- **Source** : `cdn.nba.com/logos/nba/{team_id}/global/{L|D}/logo.svg` (ni `nba_api` ni notre API ne fournissent d'images). Variante `L` pour fond clair, `D` pour fond sombre (différente pour 10 équipes sur 30).
- **Servis par le front, pas pris directement sur le site NBA** : `cdn.nba.com` est bloqué depuis le réseau de l'IUT et depuis le VPS, le navigateur du visiteur ne doit donc pas en dépendre. Les 60 SVG (~740 Ko) sont **versionnés** dans `front/public/logos/` pour qu'un `git clone` suffise au déploiement. `npm run fetch:logos` les retélécharge (depuis le PC, le VPS étant bloqué).
- `TeamBadge` (`TeamBits.js`) : logo dans la pastille colorée domicile/extérieur, les deux variantes rendues avec `dark:hidden` / `hidden dark:block`. Les ID des 30 franchises se suivent (`1610612737` → `1610612766`) : hors de cette plage (club étranger en présaison) ou si le fichier manque (`onError`), le tricode s'affiche comme avant. `next/image` avec `unoptimized` (SVG, pas d'optimisation ni de `dangerouslyAllowSVG`).
- `globals.css` : variante Tailwind `dark:` (`@custom-variant`) qui suit `data-theme` ET le thème système quand aucun choix manuel n'est fait. À utiliser seulement pour ce que `light-dark()` ne couvre pas (choix d'image).
- Contour clair autour des logos foncés en mode sombre (Memphis, Orlando) : essayé puis **écarté**, il rend flou le texte des logos ; sans lui ils restent lisibles.
- Vérifié : `eslint`, `next build`, CSS `dark:` généré, les 30 logos sur les 4 fonds de pastille, page d'accueil et fiche match en sombre avec les vraies données. **Non vérifié** : page complète en thème clair dans le navigateur (captures headless vides, voir ci-dessus).

## Déploiement (cible : VPS personnel)

L'utilisateur dispose d'un **VPS personnel** et y hébergera le projet complet (front + API + pipeline data). Pas de Vercel ni de PaaS : tout tourne sur la même machine. Rien n'est encore déployé.

### Caractéristiques du VPS (fournies par l'utilisateur, 2026-09-28)

| Élément        | Valeur                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------- |
| Offre          | OVH VPS Starter                                                                                                  |
| OS             | Ubuntu (utilisateur `ubuntu`, hôte `vps-468382b1`)                                                               |
| RAM            | 3,7 Gio au total, environ 0,9 Gio utilisés et 2,8 Gio disponibles (déjà d'autres services dessus)                |
| Stockage       | 40 Go                                                                                                            |
| Docker         | Installé                                                                                                         |
| Serveurs web   | **Caddy actif sur 80/443** (vérifié avec `ss -tlnp`) ; Apache installé mais n'écoute pas sur ces ports           |
| Python système | `python3` uniquement (pas de `python`), `nba_api` non installé                                                   |
| Accès SSH      | `ubuntu@remicoimbra.fr`, clé SSH du PC déjà autorisée (vérifié le 2026-09-28 avec `ssh -o BatchMode=yes`)        |
| Nom de domaine | `remicoimbra.fr` pointe sur le VPS ; domaine/sous-domaine du projet pas encore choisi (ex. `nba.remicoimbra.fr`) |
| CPU            | Pas encore connu                                                                                                 |

Conséquences :

- **Reverse proxy = Caddy** (actif sur 80/443). Le projet doit **s'ajouter** au `Caddyfile` existant (nouveau bloc de site, HTTPS automatique), sans lancer de deuxième reverse proxy ni toucher aux sites déjà servis. Apache est présent mais inactif sur 80/443 : ne pas s'en servir. Les conteneurs du projet n'exposent leurs ports que sur `127.0.0.1` (ex. `127.0.0.1:8000:8000`), jamais publiquement.
- **RAM (≈ 2,8 Gio libres, partagés avec l'existant)** : suffisant pour faire tourner le front (`next start`, ~150-250 Mo), l'API (pandas + xgboost chargés, ~300-400 Mo) et plus tard un Postgres léger. En revanche `next build` et le pipeline ML complet (`feature_engineering.py` sur 8 000+ matchs) sont des pics plus lourds : les lancer plutôt en local ou dans une image Docker construite ailleurs, ou au minimum vérifier qu'un swap existe (`swapon --show`).
- **Stockage (40 Go)** : largement suffisant pour les données (quelques dizaines de Mo) ; surveiller surtout les images Docker accumulées (`docker system prune` de temps en temps).
- **OVH = IP de datacenter** : `stats.nba.com` ET `cdn.nba.com` bloquent le VPS (voir plus bas). La synchro des données se fait donc depuis le PC de l'utilisateur.

### Architecture visée

```
PC de l'utilisateur (IP résidentielle, tâche planifiée Windows quotidienne)
  infra/sync_to_vps.ps1 : fetch_games --refresh → clean_data → team_state → fetch_schedule
        │  scp + mv atomique (SSH par clé)
        ▼
VPS : data/processed/team_state.json + schedule.json
        ▲ relus automatiquement par l'API quand leur date de modification change
Internet ──HTTPS──> Caddy (déjà présent et actif sur le VPS)
                      ├── /      → front Next.js (next start, port 3000)
                      └── /api   → API FastAPI (uvicorn, port 8000, NBA_LIVE_SCHEDULE_FALLBACK=0)
                                     ├── lit data/processed/team_state.json + schedule.json
                                     └── lit data/models/saved_models/
(plus tard) PostgreSQL sur le VPS lui-même
```

L'API sur le VPS ne fait **aucun** appel réseau vers la NBA.

Un seul domaine avec l'API sous `/api` évite de gérer le CORS entre deux origines. Des sous-domaines séparés (`api.xxx`) marchent aussi, mais dans ce cas les origines CORS doivent être configurées.

### Conséquences pour le code

- **Config en dur à sortir** (bloquant) : `http://127.0.0.1:8000` dans `front/src/services/api.js` → `NEXT_PUBLIC_API_URL` ; origines CORS dans `api/app/main.py` → variable d'environnement via `api/app/core/config.py`. ⚠️ `NEXT_PUBLIC_*` est injecté **au build** Next.js, pas au démarrage : il faut rebuilder le front si l'URL change.
- **Docker Compose** devient la voie naturelle (`infra/` est déjà prévu) : ça règle aussi la version de Python (le venv local est en 3.14, la distribution du VPS risque d'avoir une version plus ancienne). L'image API doit embarquer ou monter `data/preprocessing/features_lib.py`, `data/processed/` et `data/models/saved_models/` (import via `sys.path`, voir plus bas).
- **Artefacts non versionnés** : `.pkl`, `feature_columns.json`, `team_state.json` et les CSV sont dans `.gitignore`. Un `git clone` sur le VPS ne suffit donc pas : il faut soit les copier (`scp`/`rsync`), soit relancer le pipeline sur le VPS.
- **BDD** : avec un VPS, un PostgreSQL local (conteneur Docker) devient l'option la plus simple, plutôt que Supabase.
- **Sécurité** : `POST /admin/*` ne doit jamais être exposé publiquement sans protection (token ou accès limité au réseau local du VPS). Désactiver `--reload` d'uvicorn en production.

### ❌ `stats.nba.com` est BLOQUÉ depuis le VPS (testé le 2026-09-28)

`stats.nba.com` est connu pour laisser pendre sans réponse les requêtes venant d'IP de datacenters/hébergeurs cloud. **Confirmé sur le VPS OVH** avec les deux tests ci-dessous : curl avec en-têtes navigateur complets → `000` après 20 s (0 octet reçu) ; `nba_api` dans un conteneur → `ReadTimeout` après 15 s. La connexion TLS s'établit mais le serveur ne répond jamais : blocage silencieux côté NBA, pas un problème de config du VPS. Les en-têtes navigateur n'y changent rien.

**`cdn.nba.com` aussi bloqué depuis le VPS (testé le 2026-09-28)** : `403 Access Denied` (page Akamai, référence `#18.d5831002.1`) immédiat sur `static/json/staticData/scheduleLeagueV2.json` et `static/json/liveData/boxscore/boxscore_0022500001.json`, avec OU sans User-Agent navigateur. La piste "synchro 100 % autonome sur le VPS via le CDN" est donc écartée.

### ✅ Décision : synchro depuis le PC, poussée vers le VPS (implémentée le 2026-09-28)

Dépend du PC allumé une fois par jour (la tâche planifiée rattrape l'exécution manquée au prochain démarrage). Si le PC reste éteint plusieurs jours, les prédictions utilisent un `team_state.json` de plus en plus périmé et le calendrier finit par ne plus couvrir la date du jour (404) — dégradé, pas en panne.

Fichiers ajoutés/modifiés :

- **`data/collectors/fetch_games.py`** : option `--refresh` (re-télécharge la saison en cours + la dernière saison présente dans le cache + les saisons manquantes, garde le reste du cache). Sans l'option, comportement inchangé (cache réutilisé tel quel, aucun appel réseau). La liste des saisons est désormais **calculée à partir de la date** (`all_seasons()`, via `features_lib.season_for_date`) : l'ancienne liste en dur s'arrêtait à 2025-26 et aurait ignoré 2026-27 en silence.
- **`data/collectors/fetch_schedule.py`** (nouveau) : `ScoreboardV3` jour par jour sur J-3 → J+14 (ou `--from/--to`) → `data/processed/schedule.json`. Une date absente du fichier = non synchronisée ; une liste vide = synchronisée, aucun match. Tout ou rien : si un jour échoue, le fichier existant n'est pas touché. Écriture atomique. Les dates déjà présentes hors fenêtre sont conservées.
- **`api/app/services/schedule_service.py`** : lit d'abord `schedule.json` (cache invalidé par date de modification). Si la date n'y est pas : appel direct à la NBA (même code `fetch_schedule.fetch_day`, timeout 15 s) **seulement si** `NBA_LIVE_SCHEDULE_FALLBACK` est actif, sinon `ScheduleNotCoveredError` → **404** immédiat dans `routers/games.py`. Limite : le statut ("7:00 pm ET", "Final") est celui du moment de la synchro.
- **`api/app/core/config.py`** (était vide) : `NBA_LIVE_SCHEDULE_FALLBACK`, `True` par défaut (dev local : le sélecteur de date marche pour toute date). **À mettre à `0` sur le VPS**, sinon chaque date non synchronisée fait attendre le visiteur 15 s avant une erreur 502.
- **`api/app/services/predictor_service.py`** : `team_state.json` relu automatiquement quand sa date de modification change (`reload_team_state_if_changed()`, appelé par `get_predictor_service()`). Fichier illisible → état précédent conservé + log d'erreur.
- **`infra/sync_to_vps.ps1`** (nouveau) : enchaîne `fetch_games --refresh` → `clean_data` → `team_state` → `fetch_schedule` avec le Python de `api/venv`, puis `scp` des deux JSON vers `<fichier>.tmp` sur le VPS et `mv` atomique via `ssh`. `-NoPush` pour tester en local. Journal dans `infra/logs/sync_AAAA-MM-JJ.log` (ignoré par git via `*.log`). Enregistré en UTF-8 **avec BOM** (sinon PowerShell 5.1 abîme les accents) : garder ce BOM si le fichier est réécrit. Pas de réentraînement du modèle au quotidien (inutile).

Validé le 2026-09-28 : `sync_to_vps.ps1 -NoPush` complet sur le PC (2025-26 re-téléchargée, 1 230 matchs, identique au cache ; 2026-27 → 0 match joué ; calendrier 2026-09-25 → 2026-10-12, 41 matchs). Routeur appelé directement : date couverte → vraies prédictions ; date couverte sans match → `count: 0` ; date non couverte avec repli désactivé → 404 en 0,00 s ; avec repli → appel direct OK ; `team_state.json` remplacé → rechargé sans redémarrage ; fichier corrompu → état précédent conservé. **Pas encore testé** : l'envoi `scp`/`ssh` vers le VPS (pas de clé SSH ni de dossier cible configurés) et la tâche planifiée.

⚠️ **Présaison dans le calendrier** : `ScoreboardV3` renvoie aussi les matchs de présaison (ID en `001…`, ex. `0012600009` le 2026-10-03 ; saison régulière = `002…`, playoffs = `004…`). Le modèle n'est entraîné que sur la saison régulière et les rotations de présaison n'ont rien à voir. **Décision de l'utilisateur : les garder, avec un badge.** L'API renvoie un champ `season_type` par match (déduit du préfixe du `game_id` : `preseason`, `regular_season`, `all_star`, `playoffs`, `play_in`, sinon `other`), et `GameCard.js` affiche un badge "Présaison" (avec infobulle "prédiction indicative") quand `season_type === "preseason"`. Les éventuels adversaires hors NBA (clubs étrangers) donnent `prediction: null` (équipe inconnue, déjà géré).

**Reste à faire côté utilisateur pour activer l'envoi** :

1. ~~Clé SSH~~ : déjà en place, `-VpsHost ubuntu@remicoimbra.fr`.
2. Dossier cible sur le VPS (celui monté comme `data/processed/` dans le conteneur API).
3. Tâche planifiée Windows (heure à choisir après la fin des matchs de la nuit, ~6 h heure de Paris) :

```powershell
$script = "C:\Users\Rémi\Documents\projets_perso\nba-predictor\infra\sync_to_vps.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -VpsHost ubuntu@remicoimbra.fr -RemoteDir <dossier>"
$trigger = New-ScheduledTaskTrigger -Daily -At 10:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable   # rattrape si le PC était éteint à 10:00
Register-ScheduledTask -TaskName "NBA Predictor - synchro VPS" -Action $action -Trigger $trigger -Settings $settings
```

Commandes de test réseau utilisées (pour mémoire). `nba_api` n'est pas installé sur le système (et Ubuntu récent refuse `pip install` hors venv, PEP 668) :

```bash
# 1) curl seul, rien à installer (affiche code HTTP + durée ; 000 = timeout/blocage)
curl -sS -m 20 -o /dev/null -w "%{http_code} %{time_total}s\n" \
  -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" \
  -H "Referer: https://www.nba.com/" -H "Origin: https://www.nba.com" -H "Accept: application/json" \
  "https://stats.nba.com/stats/scoreboardv3?GameDate=2026-01-15&LeagueID=00"

# 2) test via nba_api dans un conteneur jetable (même code que l'API)
docker run --rm python:3.12-slim sh -c "pip install -q nba_api && python -c \"from nba_api.stats.endpoints import scoreboardv3; print(scoreboardv3.ScoreboardV3(game_date='2026-01-15', timeout=15).get_dict().keys())\""
```

## Prochaines étapes (à faire)

0. **Déploiement sur le VPS** (voir "Déploiement" plus haut) : la synchro PC → VPS est codée et testée en local. Restent : Docker Compose + bloc `Caddyfile` (étape 6), config sortie du code (étape 7), puis dossier cible + tâche planifiée pour activer l'envoi (clé SSH déjà OK). **Propositions faites, pas encore validées par l'utilisateur** (reportées à une prochaine session) :
   - sous-domaine `nba.remicoimbra.fr` (enregistrement DNS CNAME vers `remicoimbra.fr` ou A vers l'IP du VPS), pour ne pas toucher à ce que sert déjà `remicoimbra.fr` ;
   - projet cloné dans `/home/ubuntu/nba-predictor` sur le VPS → `-RemoteDir /home/ubuntu/nba-predictor/data/processed` pour la synchro ;
   - lire le `Caddyfile` existant par SSH (lecture seule) avant d'y ajouter le bloc du projet, pour ne pas casser les sites déjà servis.
1. ~~`nba_sync_service.py`~~ : **remplacé** par `infra/sync_to_vps.ps1` (la synchro ne peut pas tourner dans l'API sur le VPS, NBA bloquée). `api/app/services/nba_sync_service.py` reste un fichier vide, à supprimer. `POST /admin/sync/games` n'a plus lieu d'être sous cette forme.
2. **Finir le refactor API** : `routers/predictions.py`, `routers/teams.py`, `routers/players.py` sont toujours des fichiers vides — seul `routers/games.py` a une vraie logique pour l'instant.
3. **Choix et setup de la BDD** : décider entre PostgreSQL local ou Supabase, créer les tables du schéma (à mettre à jour avec les vraies colonnes : Elo, Net Rating, win_pct_context...). `team_state.json` fait office de solution transitoire correcte pour l'instant.
4. Tester `/games/today` (et le sélecteur de date du front) avec de VRAIS matchs programmés dès le retour de la saison régulière NBA (mi-octobre 2026) — jusque-là, le paramètre `?date=` ne permet qu'un test partiel (Elo/forme actuels appliqués à une date passée, voir limite documentée dans "Adaptation front" plus haut).
5. (Optionnel) Poursuivre l'optimisation ML si le temps le permet : tuning XGBoost (GridSearch/early stopping), validation croisée temporelle multi-saisons pour vérifier la robustesse du gain Elo (actuellement mesuré sur une seule saison de test), affiner le calcul du Net Rating (ratio de sommes plutôt que moyenne de ratios, pour réduire le bruit). Amélioration possible aussi sur le score approximatif prédit (actuellement une heuristique simple, pas un vrai modèle de régression).
6. (Prérequis du déploiement VPS) Dockeriser l'API et le service data (`infra/docker-compose.yml`). Prérequis : l'API importe `data/preprocessing/features_lib.py` via `sys.path` et lit `data/processed/` + `data/models/saved_models/` → l'image API doit embarquer (ou monter) ces dossiers de `data/`.
7. (Prérequis du déploiement VPS) Sortir la config en dur : ~~URL de l'API dans `front/src/services/api.js`~~ (FAIT : `NEXT_PUBLIC_API_URL`, à fixer AU BUILD) ; reste les origines CORS dans `api/app/main.py` (→ `api/app/core/config.py`). Indispensable avant tout déploiement.
8. ~~(Optionnel, UX) skeleton de chargement~~ : fait avec la refonte UI.
10. (Modèle) `rest_days` hors distribution : sur une date passée interrogée avec l'état actuel, il est NÉGATIF ; en début de saison il vaut ~170 jours (intersaison) alors qu'à l'entraînement il dépasse rarement quelques jours. Le front masque les valeurs négatives (« n/d »), mais le modèle les reçoit telles quelles. À traiter (écrêtage identique entraînement/prédiction dans features_lib.py) avant la reprise de la saison régulière.
9. Écrire `README.md` et `docs/architecture.md` (tous deux vides) — important pour un projet portfolio.

## Comment relancer le projet en local

**Terminal 1 — API :**

```powershell
cd api
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

→ `http://127.0.0.1:8000/docs`

⚠️ Le venv doit être ACTIVÉ (`(venv)` visible dans le prompt) avant de lancer `uvicorn`, sinon `uvicorn` n'est pas trouvé (confondu avec un Python global qui n'a pas les dépendances). Si `Activate.ps1` est bloqué par la politique d'exécution PowerShell : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

⚠️ Le venv de `api/` a SES PROPRES dépendances, distinctes de `data/` : `predictor_service.py` et `schedule_service.py` tournent dans le processus API et ont donc besoin de `pandas`, `numpy`, `joblib`, `xgboost`, `scikit-learn`, `nba_api` installés DANS `api/venv`, pas seulement dans l'environnement utilisé pour `data/`.

**Terminal 2 — Front :**

```powershell
cd front
npm run dev
```

→ `http://localhost:3000` (ex. `/?date=2026-10-05` pour un jour avec matchs, `/modele`)

⚠️ **Port 8000 occupé** sur le PC de l'utilisateur par un serveur Symfony (autre projet, constaté le 2026-09-28). Dans ce cas, lancer l'API sur un autre port et le dire au front :

```powershell
uvicorn app.main:app --reload --port 8001          # dans api/, venv activé
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8001"; npm run dev   # dans front/
```

`NEXT_PUBLIC_API_URL` est lu au démarrage de `next dev` et figé AU BUILD pour `next build`. CORS : le front doit rester sur `localhost:3000`.

**Contrôles front :** `npm run lint`, `npm run build`, `npm run check:contrast` (à relancer après toute modification de couleur dans `globals.css`). `npm run fetch:logos` retélécharge les logos des équipes (rarement utile).

**Pipeline ML (à relancer si les CSV sources changent) :**

```powershell
cd data
python collectors\fetch_games.py          # réutilise le brut existant si présent
python preprocessing\clean_data.py
python preprocessing\feature_engineering.py
python models\train_model.py
python preprocessing\team_state.py        # génère team_state.json — À REFAIRE après tout changement de games_clean.csv, l'API le lit directement
python collectors\fetch_schedule.py       # génère schedule.json (calendrier J-3 → J+14)
```

**Synchro quotidienne (nouveaux matchs + calendrier), sans réentraînement :**

```powershell
.\infra\sync_to_vps.ps1 -NoPush           # en local ; sans -NoPush : envoi vers le VPS (voir "Déploiement")
```
