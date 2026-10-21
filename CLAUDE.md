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
| Hébergement     | VPS personnel de l'utilisateur, **déployé le 2026-10-14** sur https://nba.remicoimbra.fr (voir "Déploiement") |

## Arborescence du projet

```
nba-predictor/
│
├── data/                          # Scripts Python : collecte & ML
│   ├── collectors/
│   │   ├── fetch_games.py          # FAIT — voir "État du pipeline ML" ci-dessous ; saisons depuis 2016-17 (2026-10-19, avant 2019-20)
│   │   ├── fetch_schedule.py       # FAIT (2026-09-28) — calendrier J-3 → J+14 → processed/schedule.json (lu par l'API)
│   │   ├── fetch_player_logs.py    # FAIT (2026-10-19, étape 6) — feuilles de match par joueur → raw/player_games_raw.csv (cache + --refresh)
│   │   ├── fetch_players_stats.py  # pas encore développé
│   │   └── fetch_teams_stats.py    # pas encore développé
│   ├── preprocessing/
│   │   ├── clean_data.py           # FAIT
│   │   ├── feature_engineering.py  # FAIT — v3 (Pace/Net Rating, win_pct contexte, Elo), refactoré cette session
│   │   ├── features_lib.py         # FAIT (nouveau, cette session) — logique Elo/rolling stats PARTAGÉE entre feature_engineering.py (batch) et predictor_service.py (live) ; EloParams + DEFAULT_ELO_PARAMS (réglés le 2026-10-19) ; FACTOR_FAMILIES + linear_contributions() (explication, étape 3) ; formules du score (expected_margin, split_score, rounded_score, étape 5 ; ratings de score ScoreRatingParams/compute_score_rating_timeline/expected_game + score_range, score v2 du 2026-10-21) ; statistiques des joueurs (PlayerProjectionParams, player_form_timeline, player_projection_features, compute_player_rows, MINUTES_FEATURES, stat_ranges, 2026-10-21)
│   │   ├── player_games.py         # FAIT (2026-10-19, étape 6) — lecture/typage des feuilles joueur, restriction aux matchs de games_clean
│   │   ├── team_state.py           # FAIT — construit team_state.json (état courant par équipe) ; + opponent_id/elo_before par match récent et elo_history (30 pts) depuis la refonte UI ; as_of.elo_params depuis le 2026-10-19 ; score_ratings (params, league) + score_rating par équipe depuis le 2026-10-21 ; players + opponent_allowed par équipe, player_league et as_of.player_params depuis le 2026-10-21 (statistiques des joueurs)
│   │   └── check_features.py       # FAIT (2026-10-19) — contrôles sur les 31 features calculées : game_features.csv à jour, cohérence entraînement/API (+ biais + facteurs = logit(p)), absence de fuite (voir "Amélioration des prédictions") ; + projections des joueurs (contrôles 1, 2 et 4) depuis le 2026-10-21, ~4 min
│   ├── models/
│   │   ├── train_model.py          # FAIT — LogReg (SERVIE depuis l'étape 3) + XGBoost (arrêt précoce) + Elo seul (EloOnly) évalués en walk-forward sur 12 features (diff_* + home/away_elo ; 11 avant l'effectif de l'étape 6), réentraînés sur toutes les saisons ; écrit feature_columns.json ET model_metrics.json (lu par GET /model/metrics)
│   │   ├── evaluate_model.py       # FAIT (2026-10-19) — walk-forward, métriques, comparaison appariée par bootstrap (paired_bootstrap générique), poids par saison (season_weights, SEASON_HALF_LIFE = 2), mode --legacy
│   │   ├── score_model.py          # FAIT (2026-10-19, étape 5 ; score v2 le 2026-10-21) — score prédit : sigma, fourchette à 80 %, walk-forward des erreurs (MAE) et de la couverture, comparaison à la méthode précédente (moyennes des 10 derniers matchs) et au naïf
│   │   ├── tune_elo.py             # FAIT (2026-10-19) — grille K × avantage du terrain × régression, notée sur la log loss de l'Elo seul
│   │   ├── tune_score_ratings.py   # FAIT (2026-10-21) — grille des ratings de score (rythme, points, régression, ligue), notée sur la MAE du total
│   │   ├── player_model.py         # FAIT (2026-10-21) — statistiques des joueurs : modèle de minutes (XGBoost), walk-forward depuis 2018-19, fourchettes, synthèse (model_metrics.json, clé player_model)
│   │   ├── tune_player_projections.py  # FAIT (2026-10-21) — grille des demi-vies / lissages des projections des joueurs, ne modifie rien (~2 min)
│   │   └── saved_models/           # logreg.pkl (servi), player_minutes.pkl (modèle de minutes, SERVI), xgboost.pkl (XGBClassifier nu, comparaisons locales), feature_columns.json, model_metrics.json (dont score_model et player_model, LUS PAR L'API), eval/oof_<variante>.csv (référence actuelle : oof_history_hl2.csv), eval/tune_elo_grid.csv (non versionnés)
│   ├── raw/                        # games_raw_all_seasons.csv, games_history.csv, player_games_raw.csv (non versionnés)
│   ├── processed/                  # games_clean.csv, game_features.csv, team_state.json (non versionnés)
│   ├── notebooks/
│   └── requirements.txt            # FAIT — versions de scikit-learn/xgboost/joblib/scipy alignées sur api/requirements.txt (les .pkl sont rechargés par l'API)
│
├── api/                            # Back-end FastAPI — vraies prédictions branchées cette session
│   ├── venv/                       # environnement virtuel Python (non versionné) — Python 3.14
│   ├── app/
│   │   ├── main.py                 # FAIT (réécrit cette session) — squelette FastAPI + CORS, logique déportée dans routers/
│   │   ├── core/
│   │   │   ├── config.py           # FAIT — NBA_LIVE_SCHEDULE_FALLBACK, NBA_CORS_ORIGINS, NBA_API_DOCS (variables d'environnement, défauts = dev local)
│   │   │   └── database.py         # à créer (connexion Supabase/Postgres)
│   │   ├── models/                 # à créer (ORM SQLAlchemy)
│   │   ├── schemas/                # à créer (schémas Pydantic)
│   │   ├── routers/
│   │   │   ├── games.py            # FAIT — /games/today (+ résumé équipes), /games/calendar, /games/{game_id} (fiche complète)
│   │   │   └── model.py            # FAIT (refonte UI) — /model/metrics (lit model_metrics.json)
│   │   │                           # (predictions.py, teams.py, players.py : fichiers vides supprimés le 2026-10-19, à recréer si besoin)
│   │   └── services/
│   │       ├── predictor_service.py    # FAIT — charge logreg.pkl (xgboost_v1.pkl avant l'étape 3) + player_minutes.pkl + team_state.json, construit les features live, prédit et explique (contributions linéaires), projette les stats des joueurs (2026-10-21)
│   │       └── schedule_service.py     # FAIT — lit processed/schedule.json ; appel direct ScoreboardV3 en repli (dev local uniquement)
│   │                                   # (nba_sync_service.py, vide, supprimé le 2026-10-19 : remplacé par infra/sync_to_vps.ps1)
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
│   │   │   ├── match/page.js       # fiche match (/match?id=…&date=…) → components/MatchView.js
│   │   │   └── modele/page.js      # page « Le modèle » → components/ModelView.js
│   │   ├── components/
│   │   │   ├── calendar/           # DatePicker (bandeau de jours) + CalendarPopover (grille mensuelle clavier) + dayStatus
│   │   │   ├── charts/             # graphiques SVG/HTML maison animés (motion) : ProbabilityBar, WinGauge, FactorsChart, ComparisonBars, FormChart, EloLine, ModelCharts, DataTable, Tooltip
│   │   │   ├── TeamBits.js         # TeamBadge (logo, sinon tricode), FormPills, PreseasonBadge
│   │   ├── PlayerProjections.js # tableau des stats projetées des joueurs d'une équipe (fiche match, 2026-10-21)
│   │   │   ├── GameCard.js, DaySummary.js, States.js, Header.js, ThemeToggle.js, Providers.js
│   │   ├── lib/                    # format.js (dates/nombres fr-FR), useApi.js (fetch + états), charts.js (échelles, useWidth, transitions), teamColors.js (couleurs des 30 équipes)
│   │   └── services/
│   │       └── api.js              # getGames, getGame, getCalendar, getModelMetrics ; base = NEXT_PUBLIC_API_URL (défaut http://127.0.0.1:8000)
│   ├── public/team-logos/               # logos des 30 équipes, {team_id}-L.svg (fond clair) / -D.svg (fond sombre, seule variante affichée depuis le 2026-10-16), VERSIONNÉS
│   ├── scripts/check-contrast.mjs  # `npm run check:contrast` : toutes les paires de couleurs >= 7:1 (texte) / 3:1 (graphiques)
│   ├── scripts/fetch-logos.mjs     # `npm run fetch:logos` : (re)télécharge public/team-logos/ depuis cdn.nba.com
│   ├── package.json
│   └── postcss.config.mjs          # Tailwind v4 : config "CSS-first" (@import "tailwindcss" dans globals.css), PAS de tailwind.config.js
│
├── infra/                          # (pas de Docker : fichiers vides supprimés le 2026-10-14)
│   ├── deploy_to_vps.ps1           # FAIT (2026-10-14) — déploiement PC → VPS (build front, code API, modèle), voir "Déploiement"
│   ├── sync_to_vps.ps1             # FAIT (2026-09-28) — synchro quotidienne PC → VPS (team_state.json + schedule.json)
│   └── vps/                        # fichiers installés sur le VPS (LF forcé par .gitattributes)
│       ├── deploy_remote.sh        # partie serveur du déploiement (lancée par deploy_to_vps.ps1)
│       ├── nba-predictor-api.service  # unité systemd de l'API (bac à sable + filtrage IP)
│       └── nba.remicoimbra.fr.caddy   # copie de référence du bloc ajouté au Caddyfile du VPS
│
├── docs/
│   ├── architecture.md             # FAIT (2026-10-14) — pipeline, choix techniques, pièges, déploiement
│   ├── deploiement.md              # FAIT (2026-10-14) — guide d'exploitation : redéployer, retour arrière, surveiller
│   └── screenshots/                # captures du README (accueil, fiche match, modèle ; clair/sombre), VERSIONNÉES
│
└── README.md                       # FAIT (2026-10-14)
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

**État actuel du flux : les étapes 1 → 2 → 3 → 4 → 5 sont fonctionnelles et validées de bout en bout (1→2→3 sur données réelles, 10 saisons depuis le 2026-10-19 (2016-17 → 2025-26), 11 969 matchs ; 4 testée via TestClient + mocks côté Claude, et confirmée fonctionnelle en conditions réelles côté utilisateur ; 5 adaptée au nouveau format de réponse et testée visuellement côté utilisateur avec de vraies données, voir "Adaptation front" plus bas). `/games/today` sert désormais de VRAIES prédictions (XGBoost jusqu'à l'étape 3 du 2026-10-19, **régression logistique depuis**), affichées correctement dans le dashboard. Depuis la refonte UI du 2026-09-28, le front a 3 pages (matchs du jour, fiche match avec graphiques explicatifs, page « Le modèle »), voir "Refonte UI". Aucune base de données n'est encore connectée (les CSV + `team_state.json` font office de BDD pour l'instant). L'étape 6 (automatisation) : `infra/sync_to_vps.ps1` enchaîne la mise à jour des données et l'envoi au VPS ; reste à le planifier côté utilisateur. Le site est en ligne depuis le 2026-10-14 (voir "Déploiement").**

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

- 10 saisons régulières (`2016-17` → `2025-26`, `FIRST_SEASON_START_YEAR = 2016` depuis le 2026-10-19 ; avant : 7 saisons depuis `2019-20`) via `nba_api` (`LeagueGameFinder`). Voir "Historique étendu".
- `games_raw_all_seasons.csv` (23 958 lignes, 1 ligne par équipe par match) → recombiné en `games_history.csv` (11 969 matchs, 1 ligne par match ; 8 279 avant l'extension).
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
   - **`elo` (`home_elo`/`away_elo`/`diff_elo`)** : rating Elo par équipe, calculé séquentiellement match par match sur tout l'historique chronologique (pas équipe par équipe comme le reste — un match met à jour les 2 équipes en même temps). Formule classique + multiplicateur d'écart de score façon FiveThirtyEight + régression vers la moyenne entre saisons (25 % à l'origine, 50 % depuis le réglage du 2026-10-19, avantage du terrain 100 → 40 : voir « Étape 2 »). **C'est de loin la feature la plus forte du pipeline** (voir résultats ci-dessous) : contrairement à win_pct, elle pondère par la force de l'adversaire.
   - Détection automatique des colonnes disponibles (`has_box_score()`) : le script fonctionne même sans les colonnes de boxscore (désactive juste Pace/Net Rating), pas de crash si `games_clean.csv` est une ancienne version.
   - ⚠️ Piège corrigé en route : `game_features.csv` transporte aussi les colonnes de boxscore BRUT du match courant (`home_fga`, `away_oreb`...), connues seulement APRÈS le match → fuite temporelle si utilisées comme features. `train_model.py` s'en protège avec une liste blanche de motifs (`FEATURE_PATTERNS`), jamais une liste noire.

### Modèles (`data/models/train_model.py`)

- **Depuis le 2026-10-19 : évaluation walk-forward** (voir "Amélioration des prédictions") : chaque saison de 2021-22 à 2025-26 est prédite par un modèle entraîné sur les saisons précédentes ; les modèles servis sont ensuite réentraînés sur toutes les saisons. Avant : split unique, train sur saisons ≠ `2025-26`, test sur `2025-26` (toujours calculé, `evaluation.legacy` de `model_metrics.json`).
- Entraînement sur les matchs complets uniquement (`evaluate_model.complete_rows()`) ; l'évaluation garde tous les matchs, y compris ceux où `win_pct` manque.
- Détection automatique des colonnes de features via liste blanche de motifs (`win_pct`, `rest_days`, `avg_pts_scored_last`, `avg_pts_allowed_last`, `net_rtg_last`, `elo`) — pas de liste figée à maintenir à la main.
- **Deux modèles entraînés et comparés systématiquement** : Régression Logistique (dans un `Pipeline` avec `StandardScaler` — nécessaire depuis l'ajout de l'Elo, qui vit sur une échelle ~1300-1700 très différente de `win_pct` en 0-1 ; corrige aussi le `ConvergenceWarning`) et XGBoost (`n_estimators=300, max_depth=3, learning_rate=0.05`).
- Les deux modèles sont sauvegardés dans `saved_models/` (`logreg_baseline.pkl` est maintenant le pipeline complet scaler+modèle, `xgboost_v1.pkl`).
- **Depuis l'étape 3 (2026-10-19)** : 11 features seulement (`detect_feature_cols()` = diff_* + `MODEL_LEVEL_COLS` parmi `candidate_feature_cols()`, la liste blanche), XGBoost = `XGBoostEarlyStopping` (profondeur 2, lr 0,03, arrêt précoce sur les 20 % de matchs les plus récents, patience 100, puis réentraînement sur tout avec ce nombre d'arbres ; 171 arbres pour le modèle sauvegardé), **modèle servi = régression logistique** (`SERVED_MODEL = "logreg"`, `logreg.pkl`). Fichiers renommés : `logreg.pkl` / `xgboost.pkl` (le XGBClassifier nu, pas l'enveloppe, pour ne pas dépendre de `train_model` au chargement). Voir "Étape 3".

### Résultats obtenus (progression au fil de la session, saison test 2025-26, 1 186 matchs)

| Étape                                             | Accuracy LogReg | Accuracy XGBoost | Log loss (meilleur) |
| ------------------------------------------------- | --------------- | ---------------- | ------------------- |
| Baseline ("home gagne toujours")                  | 55,1%           | —                | —                   |
| v1 (win_pct global, avg_pts_last5, rest_days)     | 66,75%          | 67,4%            | 0,605               |
| v3 sans Elo (+ Pace/Net Rating + win_pct_context) | 66,4%           | 67,6%            | 0,606               |
| v3 + Elo + scaling LogReg                         | 68,6%           | 69,6%            | 0,596               |
| **+ `rest_days` plafonné à 7 j (état final, 2026-10-14)** | **69,0%** | **68,9%**   | **0,595** (LogReg)  |

(Le passage de 69,6 % à 68,9 % pour XGBoost n'est pas une régression réelle : sur 5 graines, XGBoost fait 69,1 % sans plafond contre 68,6 % avec, écart-type 0,4 pt ; l'IC à 95 % d'une accuracy sur 1 186 matchs est d'environ ±2,7 pts.)

**Walk-forward (2026-10-19, protocole de référence désormais)**, 6 140 matchs (2021-22 → 2025-26, matchs incomplets compris), XGBoost en moyenne sur 5 graines :

| Modèle                  | Accuracy | Log loss | Brier  | ECE    |
| ----------------------- | -------- | -------- | ------ | ------ |
| Baseline (taux domicile de l'entraînement) | 55,3 % | 0,6876 | 0,2472 | 0,002 |
| XGBoost (servi)         | 64,6 %   | 0,6346   | 0,2217 | 0,030  |
| **Régression logistique** | **65,5 %** | **0,6199** | **0,2153** | **0,011** |

Comparaison appariée XGBoost − LogReg : log loss **+0,0147** (IC95 [+0,0096 ; +0,0195]), Brier +0,0064 [+0,0041 ; +0,0084], même sens sur les 5 saisons → **la LogReg est significativement meilleure en probabilités** ; accuracy −0,85 pt [−1,70 ; +0,02], dans le bruit. Par saison (XGBoost / LogReg, log loss) : 2021-22 0,658/0,639 ; 2022-23 0,662/0,646 ; 2023-24 0,628/0,609 ; 2024-25 0,620/0,609 ; 2025-26 0,605/0,597. **2025-26 est la saison la plus facile à prédire de la période** : le 69 % du split unique surestimait la performance attendue (≈ 65 % en moyenne sur 5 saisons). Dispersion due aux graines (XGBoost) : 0,0013 de log loss, 0,3 pt d'accuracy.

**Après le réglage de l'Elo (étape 2, 2026-10-19, état actuel)**, même protocole :

| Modèle                  | Accuracy | Log loss | Brier  | ECE    |
| ----------------------- | -------- | -------- | ------ | ------ |
| Baseline                | 55,3 %   | 0,6876   | 0,2472 | 0,002  |
| XGBoost (servi)         | 64,6 %   | 0,6330   | 0,2211 | 0,027  |
| Régression logistique   | 65,6 %   | 0,6191   | 0,2150 | 0,012  |
| **Elo seul** (sans apprentissage) | **65,9 %** | **0,6184** | **0,2148** | 0,015 |

XGBoost − LogReg : log loss +0,0139 [+0,0090 ; +0,0185]. **XGBoost − Elo seul : +0,0146 [+0,0091 ; +0,0197]**, Brier +0,0063 [+0,0040 ; +0,0085], 5/5 saisons ; sur les seules saisons jamais vues par le réglage (2024-25 + 2025-26) : +0,0095 [+0,0021 ; +0,0169]. LogReg − Elo seul : +0,0007 [−0,0016 ; +0,0028], dans le bruit. Autrement dit, les 30 autres features n'apportent rien de mesurable aux probabilités, et XGBoost les dégrade. Par saison (XGBoost / LogReg / Elo seul, log loss) : 2021-22 0,655/0,638/0,632 ; 2022-23 0,659/0,645/0,647 ; 2023-24 0,631/0,610/0,612 ; 2024-25 0,616/0,607/0,607 ; 2025-26 0,604/0,596/0,594 (2021-22 → 2023-24 recoupent les saisons de réglage de l'Elo : légèrement optimiste pour l'Elo seul sur ces 3 saisons). Dispersion XGBoost entre graines : 0,0007 de log loss.

**Après l'étape 3 (2026-10-19, état actuel : 11 features, XGBoost avec arrêt précoce, LogReg servie)**, même protocole :

| Modèle                  | Accuracy | Log loss | Brier  | ECE    |
| ----------------------- | -------- | -------- | ------ | ------ |
| Baseline                | 55,3 %   | 0,6876   | 0,2472 | 0,002  |
| **Régression logistique (servie)** | **65,5 %** | **0,6174** | **0,2143** | 0,015 |
| XGBoost                 | 65,7 %   | 0,6214   | 0,2160 | 0,014  |
| Elo seul                | 65,9 %   | 0,6184   | 0,2148 | 0,015  |

LogReg − XGBoost : −0,0040 [−0,0064 ; −0,0015], Brier −0,0018, 5/5 saisons. LogReg − Elo seul : −0,0010 [−0,0030 ; +0,0011], dans le bruit. Servi avant/après (LogReg étape 3 − XGBoost d'avant) : **−0,0156 [−0,0202 ; −0,0106]**, Brier −0,0068, accuracy +0,9 pt [+0,05 ; +1,8], 5/5 saisons ; sur 2024-25 + 2025-26 : −0,0096 [−0,0164 ; −0,0031]. Par saison (LogReg / XGBoost / Elo seul) : 2021-22 0,635/0,639/0,632 ; 2022-23 0,643/0,647/0,647 ; 2023-24 0,609/0,616/0,612 ; 2024-25 0,605/0,610/0,607 ; 2025-26 0,595/0,596/0,594. `--legacy` : LogReg 0,6906/0,5931, XGBoost 0,6948/0,5961, Elo seul 0,6922/0,5920. Dispersion XGBoost entre graines : 0,0005.

**Après l'effectif (étape 6, 2026-10-19, état actuel, 12 features)**, même protocole (référence : `oof_lineup.csv`) :

| Modèle                  | Accuracy | Log loss | Brier  | ECE    |
| ----------------------- | -------- | -------- | ------ | ------ |
| Baseline                | 55,3 %   | 0,6879   | 0,2474 | 0,014  |
| **Régression logistique (servie)** | **66,1 %** | **0,6141** | **0,2128** | 0,010 |
| XGBoost                 | 65,7 %   | 0,6180   | 0,2146 | 0,014  |
| Elo seul                | 65,9 %   | 0,6184   | 0,2148 | 0,015  |

Face à `history_hl2` : LogReg −0,0023 [−0,0040 ; −0,0006], XGBoost −0,0015 [−0,0028 ; −0,0002]. LogReg − XGBoost −0,0039 [−0,0061 ; −0,0015] ; **LogReg − Elo seul −0,0042 [−0,0070 ; −0,0015], 5/5**. Détails : "Étape 6".

**Après l'historique étendu + poids par saison (2026-10-19)**, même protocole (référence : `oof_history_hl2.csv`) :

| Modèle                  | Accuracy | Log loss | Brier  | ECE    |
| ----------------------- | -------- | -------- | ------ | ------ |
| Baseline                | 55,3 %   | 0,6879   | 0,2474 | 0,014  |
| **Régression logistique (servie)** | **65,7 %** | **0,6164** | **0,2138** | 0,014 |
| XGBoost                 | 65,6 %   | 0,6195   | 0,2153 | 0,015  |
| Elo seul                | 65,9 %   | 0,6184   | 0,2148 | 0,015  |

Face à l'étape 3 (`oof_step3.csv`) : LogReg −0,0010 [−0,0017 ; −0,0003], Brier −0,0005, 4/5 saisons ; XGBoost −0,0019 [−0,0034 ; −0,0005], 5/5 ; Elo seul identique. LogReg − XGBoost −0,0031 [−0,0052 ; −0,0009] ; LogReg − Elo seul −0,0020 [−0,0042 ; +0,0002]. Par saison (LogReg / XGBoost / Elo seul) : 2021-22 0,633/0,636/0,632 ; 2022-23 0,641/0,643/0,647 ; 2023-24 0,610/0,616/0,612 ; 2024-25 0,605/0,608/0,607 ; 2025-26 0,594/0,594/0,594. `--legacy` : LogReg 0,6838/0,5922, XGBoost 0,6838/0,5912, Elo seul 0,6922/0,5920 (baseline des saisons ≠ 2025-26 changée : 10 406 matchs d'entraînement).

**Feature la plus importante dans les deux modèles : `diff_elo`** (loin devant `diff_win_pct`, qui était dominante avant l'ajout de l'Elo). `diff_net_rtg_last10` reste un signal solide en 2ᵉ position. `win_pct_context` apporte un complément modeste mais réel. Poids par famille dans la LogReg servie (|contribution| moyenne, `factor_importances` de `model_metrics.json`) : à l'étape 3, Elo 68 %, forme récente 11 %, bilan dom./ext. 8 %, repos 8 %, attaque 3 %, bilan saison 1 % ; depuis l'historique étendu, Elo 65 %, forme récente 15 %, repos 9 %, attaque 6 %, bilan dom./ext. 4 %, bilan saison 1 %.

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
4. **Changement de saison** : la saison du match est déduite de SA date (`features_lib.season_for_date()`, frontière en août), pas de `team_state.json`. Si elle est postérieure à la saison de `team_state.json` (ex : match d'octobre 2026 avec un état arrêté en avril 2026), on reproduit le pipeline d'entraînement : régression Elo vers 1500 (`season_regression`, 50 % depuis le 2026-10-19) ET `win_pct`/`win_pct_context` remis à zéro (→ NaN). `win_pct_last10` et les moyennes glissantes ne sont pas remis à zéro, comme à l'entraînement. Avant la session d'audit, ni la régression ni la remise à zéro n'étaient appliquées (décalage entraînement/prédiction qui aurait touché tous les matchs 2026-27).
5. Retourne probabilités de victoire + un score estimé : jusqu'au 2026-10-19, **heuristique simple** (forme offensive/défensive récente, jamais évaluée, contredisait le favori sur 22 % des matchs) ; depuis l'étape 5, total d'après les moyennes récentes + écart σ·Φ⁻¹(p), en entiers sans égalité (voir "Étape 5"). σ et le total de repli sont lus dans `model_metrics.json` (`score_model`), devenu **obligatoire** (`PredictorNotReadyError` sinon).

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
        "predicted_home_score": 116,
        "predicted_away_score": 112,
        "predicted_home_score_range": [101, 131],   // depuis le 2026-10-21 (fourchette à 80 %)
        "predicted_away_score_range": [97, 127]
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
15. **Plafond `rest_days` + documentation (2026-10-14)** : voir "Plafond des jours de repos" ci-dessous.
16. **Déploiement sur le VPS (2026-10-14)** : en ligne sur https://nba.remicoimbra.fr, voir "Déploiement" ci-dessous.
17. **Plan d'amélioration des prédictions + étape 1, évaluation walk-forward (2026-10-19)** : voir "Amélioration des prédictions" ci-dessous.
18. **Étape 2, réglage de l'Elo (2026-10-19)** : K 20 / avantage 40 / régression 0,5 adoptés, « Elo seul » ajouté aux modèles évalués, voir "Étape 2" ci-dessous. Déployé le 2026-10-19 avec l'étape 1 (commit `de884dc`), synchro dans la foulée.
19. **Étape 3, modèle (2026-10-19)** : élagage à 11 features et arrêt précoce d'XGBoost adoptés, lissage du bilan / modèle de marge / ensembles écartés, **régression logistique servie** (choix de l'utilisateur), explication par contributions linéaires, voir "Étape 3" ci-dessous. Déployé le 2026-10-19 (commit `7903758`).
20. **Historique étendu à 2016-17 + poids par saison, et étape 5, score prédit (2026-10-19)** : voir "Historique étendu" et "Étape 5" ci-dessous. Déployé le 2026-10-19 (20:21, commit `3b64e1a`), `deploy_to_vps.ps1` seul.
21. **Étape 4, calendrier (2026-10-19)** : essayée, non adoptée (aucun gain qui tienne sur 2024-25 + 2025-26), voir "Étape 4" ci-dessous. Aucun code fusionné.
22. **Étape 6, blessures (2026-10-19)** : essai concluant (indicateur d'effectif tiré des feuilles de match par joueur, −0,0023 de log loss, tient sur 2024-25 + 2025-26), puis intégré avec l'accord de l'utilisateur (nouveau collecteur, `team_state.json`, API, front, `sync_to_vps.ps1`), voir "Étape 6" ci-dessous. Déployé le 2026-10-19 (21:20, commit `513470a`) + synchro.
23. **Score v2 (2026-10-21)** : demande « IA complète » (meilleur score + statistiques des joueurs, en deux fois avec l'accord de l'utilisateur). Total du score par ratings de rythme et de points ajustés aux adversaires, fourchette à 80 % (choix de l'utilisateur), voir "Score v2" ci-dessous. Déployé le 2026-10-21 (12:56, commit `dc7927e`) + synchro. Statistiques des joueurs : prochaine session.
24. **Statistiques des joueurs (2026-10-21, session suivante)** : minutes, points, rebonds, passes, interceptions, contres de chaque joueur attendu, avec fourchette à 80 %, sur la fiche match (choix de l'utilisateur : ces 6 stats, fourchette, fiche match seule). Minutes par XGBoost, autres stats = minutes × taux par minute. Voir "Statistiques des joueurs" ci-dessous. Déployé le 2026-10-21 (14:58, commit `4faf800`) + synchro.

## Refonte UI (2026-09-28)

Demande de l'utilisateur : palette NBA bleu/blanc/rouge + mode sombre bleu/noir/rouge, contrastes **WCAG AAA**, calendrier fait maison (pas l'`<input type="date">`), animations fluides, vrais graphiques (pas juste « 60 % »). Choix de l'utilisateur : fiche match sur une **page dédiée** `/match/[id]` (devenue `/match?id=` avec l'export statique, 2026-10-14), et une page **« Le modèle »**.

### Côté API / données
- `team_state.py` : chaque match de `recent_games` a en plus `opponent_id` et `elo_before` ; nouveau `elo_history` (`{date, elo}` après chacun des 30 derniers matchs, Elo après match recalculé avec `features_lib.elo_update`). Affichage uniquement, pas des features. L'API reste compatible avec un ancien `team_state.json` (champs manquants → `null`/`[]`).
- `train_model.py` écrit `model_metrics.json` (accuracy/log loss des 2 modèles, baseline, importances XGBoost — remplacées le 2026-10-19 par `factor_importances` du modèle servi —, courbe de calibration en 10 tranches). Réentraîné le 2026-09-28 : modèle identique (écart de prédiction 0,0), 68,6 % / 69,6 % ; réentraîné le 2026-10-14 avec le plafond de repos : 69,0 % / 68,9 %.
- `predictor_service.py` : `GamePrediction` gagne `elo_home_probability` (Elo seul), `factors` et `base_value`. Les facteurs étaient les contributions XGBoost de type SHAP (`pred_contribs`, en log-odds) ; **depuis l'étape 3, contributions exactes de la régression logistique** (`features_lib.linear_contributions` : coef × valeur standardisée, biais = intercept), même format d'API. Sommées par famille (`features_lib.FACTOR_FAMILIES` : Elo, bilan dom./ext., forme récente = bilan 10 derniers + Net Rating, bilan saison, attaque, défense, repos). Vérifié : sigmoïde(biais + somme) = `home_win_probability` (contrôlé aussi par `check_features.py`). Nouveau `team_snapshot()` (stats + bilans V-D + derniers matchs + `elo_history`) ; la logique de changement de saison est factorisée dans `_season_context()`.
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
- **Servis par le front, pas pris directement sur le site NBA** : `cdn.nba.com` est bloqué depuis le réseau de l'IUT et depuis le VPS, le navigateur du visiteur ne doit donc pas en dépendre. Les 60 SVG (~740 Ko) sont **versionnés** dans `front/public/team-logos/` et partent avec l'export statique du front. `npm run fetch:logos` les retélécharge (depuis le PC, le VPS étant bloqué).
- `TeamBadge` (`TeamBits.js`) : **depuis le 2026-10-16**, logo (variante `D` seulement, dans les deux thèmes) sur une pastille à la couleur de l'équipe, sans le liseré bas domicile/extérieur d'avant ; voir « Couleurs des équipes » ci-dessous. Équipe absente de `TEAM_COLORS` (club étranger en présaison) ou fichier manquant (`onError`) : tricode sur fond teinté domicile/extérieur. `next/image` avec `unoptimized` (SVG, pas d'optimisation ni de `dangerouslyAllowSVG`).
- `globals.css` : variante Tailwind `dark:` (`@custom-variant`) qui suit `data-theme` ET le thème système quand aucun choix manuel n'est fait. À utiliser seulement pour ce que `light-dark()` ne couvre pas (choix d'image). Plus utilisée depuis le 2026-10-16 (les logos `L` non plus), gardée.
- Contour clair autour des logos foncés en mode sombre (Memphis, Orlando) : essayé puis **écarté**, il rend flou le texte des logos ; sans lui ils restent lisibles.
- Vérifié : `eslint`, `next build`, CSS `dark:` généré, les 30 logos sur les 4 fonds de pastille, page d'accueil et fiche match en sombre avec les vraies données. **Non vérifié** : page complète en thème clair dans le navigateur (captures headless vides, voir ci-dessus).

### Couleurs des équipes (2026-10-16)
- `front/src/lib/teamColors.js` : `TEAM_COLORS[team_id] = { color, alt }`. `color` = couleur la plus présente du logo (variante L rendue avec `sharp`, pixels comptés), hors blanc/noir/gris (S < 20 %, L < 10 % ou > 80 % en HSL) ; Nets et Spurs (logos noir et blanc) → noir. `alt` = la couleur suivante. Valeurs exactes des SVG (ex. Celtics `#008348`). Choix discutables mais conformes à la règle : Cleveland en or (`#bc945c`, le bordeaux `#860038` est 2ᵉ), Oklahoma City en marine (`#002d62`, le bleu Thunder `#007ac1` n'est que 4ᵉ).
- Utilisées par la pastille du logo (`TeamBadge`) et par `ProbabilityBar` (cartes de l'accueil). **Depuis le 2026-10-19, aussi dans la fiche match** : `MatchView` calcule `matchupColors(home, away)` une fois et le passe en prop `colors` (ou `color`) à `WinGauge` (couleur du favori), `FactorsChart`/`ModelVsElo`, `ComparisonBars`, `FormChart`, `EloLine`, au bandeau du haut et à la liste des absents. `fillStyle(color)` (`teamColors.js`) = aplat + liseré `lowContrastEdge` ; pour les traits SVG (jauge, courbes Elo, colonnes de forme), un liseré est posé sous le trait ou autour. Seuls les graphiques de la page « Le modèle » restent en `--home` bleu (`--away` n'est plus utilisé que par `TeamBadge` en repli).
- `matchupColors(home, away)` : si les deux couleurs sont à moins de 0,12 en distance OKLab (66 paires d'équipes, ex. ATL/DET, PHI/GSW, DEN/WAS), l'extérieur passe à son `alt`, sinon le domicile. Vérifié sur les 870 affiches possibles : aucune ne retombe sur le repli neutre.
- `lowContrastEdge(color)` : liseré `--line-strong` (1 px, intérieur) autour d'un segment sous 3:1 face au `--bg` des cartes, par thème via `light-dark()` (marine/noir/violet en sombre, or/orange en clair). `CARD_BG` y recopie `--bg` : à mettre à jour si `--bg` change.
- Vérifié : `eslint`, `next build`, captures CDP (Chrome headless piloté par le protocole DevTools, `prefers-reduced-motion: reduce` pour éviter les captures vides) de l'accueil clair/sombre et d'une fiche match.

### Plafond des jours de repos (2026-10-14)
- `features_lib.REST_DAYS_CAP = 7`, appliqué à l'entraînement (`feature_engineering.py`, `.clip(upper=...)`) ET en prédiction (`features_lib.rest_days()`). Avant : les ~100 matchs d'ouverture de saison de l'historique avaient 130-288 jours de repos (intersaison, bulle 2020) et un match de reprise de saison était prédit avec ~170 jours.
- `rest_days()` renvoie `None` (→ NaN) quand la date demandée précède le dernier match connu (avant : nombre négatif). Le `valid: v >= 0` de `ComparisonBars.js` devient inutile mais reste inoffensif.
- Réentraîné : voir tableau de résultats (68,9 % XGBoost / 69,0 % LogReg). Mesure de sensibilité (5 graines, plafonds none/14/7/5/3, et sans la variable) : toutes les variantes sont dans le bruit ; le plafond ne coûte rien de mesurable.
- Vérifié : prédiction BOS–DEN le 2026-10-22 → repos 7/7 ; sur 2026-01-15 → repos `None`. `team_state.json` n'est pas affecté (il stocke `last_game_date`).
- Observation, NON traitée : `train_model.prepare()` supprime 265 lignes (premiers matchs de chaque équipe/saison, `win_pct` ou `win_pct_context` manquants), donc le modèle n'a presque jamais vu `win_pct` = NaN, ce qui est le cas des 1ers matchs de la saison en live. Test : entraîner AVEC ces lignes n'aide pas (265 lignes seulement) ; sur les 39 matchs concernés de 2025-26, le modèle fait 58 % contre 62 % pour l'Elo seul (échantillon trop petit pour conclure). Piste si les premiers matchs de 2026-27 paraissent trop tranchés : rapprocher la proba de `elo_home_probability` tant que `win_pct` est manquant.
- `schedule.json` régénéré le 2026-10-14 (J-3 → J+14) : la saison régulière 2026-27 commence le **2026-10-22** (2 matchs), 12 matchs le 2026-10-23. Vrais matchs de saison régulière prédits et affichés dans le navigateur (accueil clair/sombre, fiche match, page modèle).

## Amélioration des prédictions (plan validé le 2026-10-19)

Objectif de l'utilisateur : des probabilités mieux calibrées (log loss, Brier), mesurées honnêtement ; un gain dans le bruit n'est pas un gain. Plan en 6 étapes, dans cet ordre (l'utilisateur proposait calendrier avant Elo ; Elo avancé car c'est le seul signal déjà mesuré) :

**Avancement : étapes 1, 2, 3 et 5 (score) + question ouverte 1 (historique) faites et déployées le 2026-10-19 ; étape 4 (calendrier) essayée le 2026-10-19, non adoptée (rien à déployer) ; étape 6 (blessures) essayée le 2026-10-19, concluante (−0,0023 de log loss) et intégrée et déployée le même jour (voir "Étape 6"). Le plan est terminé.**

1. **Évaluation walk-forward** : FAIT le 2026-10-19 (détails ci-dessous).
2. **Elo** : FAIT le 2026-10-19 (détails dans "Étape 2" ci-dessous). Plan d'origine : rendre K, avantage du terrain et régression inter-saison paramétrables (`EloParams` dans `features_lib.py`, valeurs par défaut = constantes actuelles), grille réglée sur le log loss de l'Elo seul des saisons 2020-21 → 2023-24, confirmation sur 2024-25/2025-26. Ajouter « Elo seul » comme modèle dans `model_metrics.json`, `elo_params` dans `model_metrics.json` ET `team_state.json` (avertissement de l'API s'ils diffèrent), et lire le « 25 % » de `ModelView.js` depuis les métriques. Indice mesuré avant le plan (optimiste, réglage vu sur 2025-26) : avantage du terrain 100 → proba domicile moyenne 61,7 % contre 55,4 % réel ; à 50, log loss de l'Elo seul 0,6075 → 0,5969 sur 2025-26.
3. **Modèle** : FAIT le 2026-10-19 (détails dans "Étape 3" ci-dessous). Plan d'origine : sous-essais un par un — 3a `win_pct` lissé `(V + k·0,5)/(N + k)` dans `features_lib` (préféré au rapprochement a posteriori vers l'Elo, qui casserait l'additivité des facteurs SHAP) ; 3b élagage (diff_* + Elo) ; 3c XGBoost avec arrêt précoce ; 3d modèle de marge (Ridge, P = Φ(μ/σ)) ; 3e ensemble en log-odds ; 3f Platt seulement si ECE > 0,02 (pas d'isotonique). Modèle servi = meilleur log loss walk-forward → **régression logistique, choix confirmé par l'utilisateur le 2026-10-19**.
4. **Calendrier** : essayé le 2026-10-19, **NON ADOPTÉ** (détails dans "Étape 4" ci-dessous). Plan d'origine : `games_last_4d`, `travel_km` (table `ARENAS` statique), `tz_shift` ; tout se déduit de `recent_games`, pas de changement de `team_state.json`. Gain attendu dans le bruit (`rest_days` = 1 code déjà le back-to-back).
5. **Score prédit** : FAIT le 2026-10-19 (détails dans "Étape 5" ci-dessous ; la régression du total n'a pas été adoptée, voir pourquoi). Plan d'origine : régression du total + marge affichée = σ·Φ⁻¹(p servie) (signe toujours cohérent avec le vainqueur prédit) ; `score_model` dans `model_metrics.json`, tuile et texte dans `ModelView.js`.
6. **Blessures** : essai court (1-2 j) puis go/no-go. Seul un indicateur calculé sur les matchs passés est cohérent avec la production (savoir qui joue CE match n'est possible que ~30 min avant, la synchro tourne à 10 h). Version complète = modifier `sync_to_vps.ps1` (accord de l'utilisateur requis).

**Règles communes** : décision sur le Δ log loss regroupé du walk-forward, IC à 95 % par bootstrap apparié rééchantillonné par journée (2 000 tirages) entièrement < 0 et Brier non dégradé ; à égalité, adopter seulement si ça simplifie ; sinon résultat consigné ici et code non fusionné. Réglages d'hyperparamètres sur saisons ≤ 2023-24. Toujours rapporter aussi l'ancien découpage (`--legacy`). Rejouer `check_features.py` à chaque étape. Quand une étape change des features ou l'Elo : `deploy_to_vps.ps1` puis **immédiatement** `sync_to_vps.ps1` (côté utilisateur).

**Questions ouvertes** : (1) ~~étendre l'historique à 2016-17 → 2018-19~~ : **fait le 2026-10-19 à la demande de l'utilisateur**, adopté avec un poids par saison (voir "Historique étendu") ; (2) ~~modèle servi~~ : **tranché le 2026-10-19, régression logistique** (proposée face à XGBoost amélioré et à l'Elo seul, choisie par l'utilisateur) ; (3) ~~modifier `sync_to_vps.ps1` si l'essai blessures est concluant~~ : **accepté par l'utilisateur le 2026-10-19, fait**.

### Étape 1 : évaluation walk-forward (FAIT le 2026-10-19)

- `data/models/evaluate_model.py` : `walk_forward()` (pli = saison, entraînement sur toutes les saisons antérieures), `summarize()` (log loss, Brier, accuracy, ECE à 10 tranches ; moyenne et écart-type sur les graines 0-4 pour XGBoost), `paired_compare()` (Δ, IC bootstrap par journée, signe par saison), `legacy_split()`, `calibration_curve()` (prédictions des 5 graines empilées). CLI : `--variant NOM` sauvegarde `saved_models/eval/oof_NOM.csv`, `--compare REF` compare à une variante sauvegardée, `--legacy` pour l'ancien découpage. `train_model.py` importe ce module (l'inverse se fait dans `main()`, pour éviter un import croisé).
- `train_model.py` : modèles servis réentraînés sur TOUTES les saisons (8 014 matchs complets, 2025-26 compris ; avant : 6 828, sans 2025-26). LogReg avec `SimpleImputer(median)` en tête du pipeline (sert seulement à la prédiction d'un match incomplet). XGBoost servi : graine 42, comme avant. Toujours XGBoost servi (`SERVED_MODEL`).
- `model_metrics.json` : ajouts seulement. `models[]` (accuracy, log_loss = walk-forward regroupé ; + brier, ece, n_games, seed_sd), `baseline` (constante = taux de victoire à domicile des saisons d'entraînement), `served_model`, `evaluation` (`test_seasons`, `per_season[]`, `early_season`, `comparisons.xgboost_vs_logreg`, `legacy`). `test_games` = 6 140 matchs du walk-forward, `train_games` = matchs du modèle servi. `calibration` vient du walk-forward.
- Front : tuiles Accuracy (modèle servi) / Gain vs baseline / Log loss / **Brier** (avec la valeur naïve), carte « Saison par saison » (`SeasonTable` dans `ModelCharts.js` ; colonne Matchs masquée sous `sm`), textes « Sans tricher » et « Les limites » mis à jour, `dec()` ajouté à `format.js`. La page affiche donc **64,6 %** (moyenne de 5 saisons) au lieu de 68,9 %.
- `data/preprocessing/check_features.py` (nouveau) + `feature_engineering.build_features()` (pipeline en mémoire, extrait de `main()`) : (0) `game_features.csv` = features recalculées ; (1) 50 matchs de 2025-26 dont les 10 premiers (changement de saison) : `team_state` reconstruit avec les seuls matchs antérieurs → `PredictorService.build_feature_row()` identique au pipeline d'entraînement à 1e-9 ; (2) 5 matchs de 2022-23 au résultat inversé : aucune feature du match ni des matchs antérieurs ne bouge, les suivantes si. Les deux contrôles ont été vérifiés capables d'échouer (repos décalé d'un jour côté API, résultat injecté dans `diff_elo`).

Résultats (tableau dans "Résultats obtenus" plus haut) et observations :
- `--legacy` reproduit exactement les chiffres historiques (XGBoost 0,6889 / 0,6022, LogReg 0,6897 / 0,5954).
- **Résultat négatif, non adopté** : entraîner aussi sur les ~265 matchs incomplets (XGBoost natif, LogReg avec imputation + indicateurs) : Δ log loss −0,0012 [−0,0030 ; +0,0006] pour XGBoost, +0,0008 [−0,0004 ; +0,0024] pour la LogReg, dans le bruit ; et la LogReg passe de 0,627 à 0,662 de log loss sur ces matchs. Entraînement sur matchs complets conservé.
- Matchs d'ouverture (au moins une feature manquante, 191 dans le walk-forward) : Elo seul 0,634 de log loss, XGBoost 0,652-0,660, LogReg (imputation médiane) 0,627 ; les modèles y sous-estiment le domicile (≈ 50-52 % prédits pour 55,5 % réels). C'est la cible de l'étape 3a.
- Modèle servi réentraîné : sur les 107 matchs de saison régulière du calendrier (2026-10-20 → 2026-11-02), écart moyen de 4,3 pts avec l'ancien XGBoost, sans dérive systématique (corrélation avec l'Elo seul : 0,94 contre 0,93 avant ; LogReg : 0,98). Cas extrême : BOS–DEN le 2026-10-22, 66 % → 50 % pour Boston (Elo seul 68 %) — les prédictions XGBoost des matchs d'ouverture sont instables d'un entraînement à l'autre.
- Vérifié : `eslint`, `next build`, page « Le modèle » capturée par CDP en 375 px (clair/sombre, pas de débordement) et 1280 px, `/games/today` et `/games/{id}` en local, facteurs + base = logit(p). Déployé le 2026-10-19 avec l'étape 2.
- Capture headless « simple » (`--screenshot --window-size=375,...`) : Chrome n'émule pas vraiment 375 px, la page paraît coupée à droite. Utiliser CDP (`Emulation.setDeviceMetricsOverride`).

### Étape 2 : Elo (FAIT et déployé le 2026-10-19)

- `features_lib.py` : `EloParams(k, home_advantage, season_regression)` (dataclass figée, `as_dict()`), `DEFAULT_ELO_PARAMS`, paramètre `params` (défaut `DEFAULT_ELO_PARAMS`) sur `elo_expected_home` (accepte aussi des tableaux numpy), `elo_update`, `regress_elo_between_seasons`, `compute_elo_timeline` (et `EloTimeline.params`), `elo_for_target_season` ; nouveau `regress_rating()` (une seule formule de régression). Les constantes `ELO_K`/`ELO_HOME_ADVANTAGE`/`ELO_SEASON_REGRESSION` n'existent plus. Non-régression avant changement des valeurs : `check_features.py` OK, `--legacy` identique (0,6889/0,6022 et 0,6897/0,5954), walk-forward XGBoost/LogReg identique à `oof_baseline.csv` à l'octet près (variante `elo_default`).
- `data/models/tune_elo.py` : 396 combinaisons (K {10…35} × avantage {0…100} × régression {0 ; 0,1 ; 0,2 ; 0,25 ; 0,33 ; 0,5}), ~30 s. Objectif : log loss de l'Elo seul sur 2020-21 → 2023-24 (2019-20 = rodage). Contrôle sur 2024-25 + 2025-26 : `paired_compare` contre `REFERENCE_PARAMS` (20/100/0,25, constante du script) + écart proba moyenne / taux réel. Grille dans `saved_models/eval/tune_elo_grid.csv`.
- **Résultat : K 20, avantage 40, régression 0,5 (adoptés)**. Profil : K plat entre 15 et 20 ; avantage optimal 40-50 ; régression 0,5 au bord de la grille, mais sondage hors grille (informatif, non adopté) : 0,6 → 0,6330, 0,75 → 0,6338, 1,0 → 0,6371 contre 0,6328 à 0,5, plateau 0,4-0,67. Les paramètres d'origine étaient 335ᵉ sur 396.
  - Saisons de réglage : log loss Elo seul 0,6470 → 0,6328, Δ −0,0142 [−0,0191 ; −0,0092] ; proba domicile moyenne − taux réel +6,8 → −0,3 pt.
  - **Saisons jamais vues (2024-25 + 2025-26)** : 0,6163 → 0,6002, **Δ −0,0161 [−0,0230 ; −0,0096]**, 2/2 saisons ; Brier −0,0061 [−0,0089 ; −0,0035] ; accuracy +0,9 pt, dans le bruit ; écart de calibration +6,8 → **+0,0 pt** (55,0 % prédit pour 54,9 % réel). Les deux critères (IC < 0, écart < 1,5 pt) sont remplis.
- **Modèle complet (walk-forward `elo_tuned` contre `baseline`)** : XGBoost log loss **−0,0016 [−0,0035 ; +0,0001]**, Brier −0,0006 [−0,0014 ; +0,0002], accuracy −0,05 pt ; LogReg −0,0009 [−0,0018 ; +0,0001], Brier −0,0003 [−0,0007 ; +0,0001]. Critère (Δ ≤ 0, borne haute < +0,001, Brier non dégradé) rempli → adopté. Sur 2024-25 + 2025-26 seules : XGBoost −0,0024 [−0,0045 ; −0,0003], LogReg −0,0014 [−0,0032 ; +0,0003]. `--legacy` : XGBoost 0,6889/0,6022 (inchangé), LogReg 0,6897/0,5937, Elo seul 0,6922/0,5920. Début de saison (< 10 matchs, 782) : XGBoost 0,646, LogReg 0,633, Elo seul 0,631.
- **« Elo seul » parmi les modèles évalués** : `train_model.EloOnly` (`fit` ne fait rien, `predict_proba` = `elo_expected_home(home_elo, away_elo, DEFAULT_ELO_PARAMS)`, donc la même proba que `elo_home_probability` de l'API), `ModelSpec("elo", "Elo seul", …)`. Apparaît dans `models[]`, `per_season`, `early_season`, `legacy` de `model_metrics.json` et dans `AccuracyBars`. `evaluation.comparisons` contient maintenant `xgboost_vs_logreg` ET `xgboost_vs_elo`. Pas de `.pkl` pour lui.
- `model_metrics.json` : `elo_params` (= `DEFAULT_ELO_PARAMS` au moment de l'entraînement ; `check_features.py` garantit que `game_features.csv` est calculé avec). `team_state.json` : `as_of.elo_params`. `predictor_service.py` : `_check_elo_params()` à chaque (re)chargement de `team_state.json` compare code / `model_metrics.json` / `team_state.json` et journalise un `WARNING` s'ils diffèrent (un `team_state.json` sans le champ compte comme différent ; `model_metrics.json` absent ou ancien : ignoré). Vérifié : aucun avertissement avec les fichiers régénérés, avertissement avec l'ancien `team_state.json`.
- `ModelView.js` : avantage du terrain et régression lus dans `metrics.elo_params` (phrase omise si absent), « Elo seul » mentionné dans l'en-tête.
- Prédictions réelles (107 matchs de saison régulière du calendrier, 2026-10-20 → 2026-11-02, ancien modèle + ancien `team_state.json` + anciens paramètres contre nouveaux) : XGBoost proba domicile moyenne 57,4 % → 53,5 %, écart moyen 4,9 pts (max 24,6) ; Elo seul 63,0 % → 56,1 %, plage 0,18-0,94 → 0,26-0,82. BOS–DEN le 2026-10-22 : XGBoost 49,8 % → 43,3 %, Elo seul 68,2 % → 57,7 % (Elo BOS 1657 → 1598, DEN 1625 → 1584 après régression de 50 %). Facteurs + base = logit(p) à 1e-4. XGBoost prédit en moyenne 53,5 % au domicile sur ces matchs d'ouverture (taux historique ≈ 55 %) : même sous-estimation que dans le walk-forward, cible de l'étape 3a.
- Vérifié aussi : `check_features.py`, `/model/metrics`, `/games/today?date=2026-10-22`, `/games/{id}` en local, `eslint`, `next build`, page « Le modèle » par CDP (375 px clair/sombre, 1280 px clair/sombre : pas de débordement, aucune erreur console, toutes les cartes rendues).
- Piège CDP : les cartes animées par `whileInView` restent vides sur une capture pleine page si on saute directement en bas ; faire défiler par pas (~300 px) avant de capturer.
- **Déployé le 2026-10-19** : `deploy_to_vps.ps1` (commit `de884dc`, 15:15) puis `sync_to_vps.ps1` 50 s après. Vérifié en production : `/api/model/metrics` renvoie `elo_params` et l'Elo seul, `team_state.json` du VPS a `as_of.elo_params` à 20/40/0,5, prédictions du 2026-10-22 identiques au local (PHI–CLE 40,2 %, OKC–DEN 48,7 %), pages en 200. Aucun avertissement dans le journal : le service de prédiction n'est chargé qu'à la première requête de prédiction, et aucune n'est arrivée entre les deux scripts (avertissement vérifié en local seulement).
- **Ordre à respecter pour tout futur changement de paramètres Elo** : `deploy_to_vps.ps1` PUIS immédiatement `sync_to_vps.ps1`.

### Étape 3 : modèle (FAIT et déployé le 2026-10-19)

Banc d'essai hors dépôt (scratchpad), sous-essais un par un sur l'état cumulé ; réglages sur 2021-22 → 2023-24 (walk-forward restreint à ces 3 saisons), confirmation sur les 5 saisons et sur 2024-25 + 2025-26. Référence de départ : `oof_step3_ref.csv` (identique à `oof_elo_tuned.csv`).

- **3a, `win_pct`/`win_pct_context` lissés, NON ADOPTÉ.** Grille k ∈ {0, 1, 2, 3, 5, 8, 12, 20, 30} sur les saisons de réglage : LogReg plate (0,6306-0,6311), XGBoost 0,6484 → ~0,645 (plateau k 3-30). k = 8 (meilleure moyenne des deux) sur les 5 saisons : XGBoost −0,0011 [−0,0039 ; +0,0019] mais **+0,0028 sur 2024-25 + 2025-26** (le gain de réglage ne se généralise pas), LogReg −0,0002 [−0,0009 ; +0,0006]. Code non fusionné. Vérifié au passage : k = 0 reproduit exactement les features actuelles.
- **3b, élagage, ADOPTÉ.** Jeux testés : complet (31), diff_* + home/away_elo (11), diff_* seul (9), diff_* sans repos + repos home/away + Elo (12). LogReg (11) : **−0,0016 [−0,0026 ; −0,0006]**, Brier −0,0007 [−0,0012 ; −0,0003], 5/5 saisons, significatif aussi sur réglage et confirmation ; meilleur des jeux sur les saisons de réglage. XGBoost : tous dans le bruit (−0,0003 [−0,0044 ; +0,0042] pour 11). Adopté pour les deux modèles (une seule liste de features, plus simple). Conséquence : la famille « Défense récente » disparaît des facteurs (pas de diff_avg_pts_allowed).
- **3c, XGBoost avec arrêt précoce, ADOPTÉ pour XGBoost.** Grille profondeur {1, 2, 3} × min_child_weight {1, 20}, lr 0,03, validation = 20 % derniers matchs : 0,6338-0,6349 sur les saisons de réglage (contre 0,6498 avant), profondeur 2 / mcw 1 retenue (~135 arbres en médiane). 5 saisons : **−0,0113 [−0,0152 ; −0,0076]**, Brier −0,0044, 5/5 saisons ; reste derrière la LogReg (+0,0040 [+0,0015 ; +0,0063]) et l'Elo seul (+0,0030 [+0,0001 ; +0,0059]).
- **3d, modèle de marge, NON ADOPTÉ (égalité).** Ridge (imputer + scaler) sur `home_score − away_score`, σ = écart-type des résidus d'entraînement (ddl corrigés), P = Φ(μ/σ). α ∈ {0,1 … 10 000} : 11 features α 0,1 → 0,6292 sur réglage (31 features : 0,6308 au mieux). 5 saisons : **+0,0000 [−0,0011 ; +0,0011] face à la LogReg**, −0,0039 face à XGBoost. Plus complexe (lien probit, facteurs en points de marge) pour rien. **À reprendre pour l'étape 5** (score) : marge moyenne 1,85 pt, écart-type 15,2.
- **3e, ensembles en log-odds (poids égaux), NON ADOPTÉS.** LogReg+Elo, LogReg+XGBoost, LogReg+Elo+XGBoost, les 4, Elo+XGBoost : tous dans le bruit face à la LogReg seule (meilleur : LogReg+Elo −0,0004 [−0,0014 ; +0,0006]).
- **3f, Platt : sans objet.** ECE regroupé : LogReg 0,015, XGBoost 0,014, Elo 0,015, tous < 0,02 (et proches du plancher de bruit d'une ECE à 10 tranches sur 6 140 matchs).

**Choix du modèle servi** : proposé à l'utilisateur (LogReg recommandée / XGBoost amélioré / Elo seul), **régression logistique choisie**. Implémentation :
- `features_lib.py` : `FACTOR_FAMILIES` et `factor_family()` (déplacés de `predictor_service.py`), `linear_contributions(pipeline, X)`. Familles : « Forme (10 derniers) » et « Net Rating récent » **fusionnées en « Forme récente »** : `diff_win_pct_last10` et `diff_net_rtg_last10` corrélés à 0,88, coefficient de la forme instable selon les saisons d'entraînement (+0,04 en 2021-22 → −0,13 sur tout l'historique, en miroir du Net Rating 0,07 → 0,24 ; sans Net Rating il est ≈ 0). Séparées, la fiche match aurait affiché « Forme : avantage à l'adversaire » pour une équipe en bonne série.
- `train_model.py` : voir "Modèles" plus haut ; `model_metrics.json` : `served_model: "logreg"`, `models[]` dans l'ordre logreg/xgboost/elo, `evaluation.comparisons` = `logreg_vs_xgboost` + `logreg_vs_elo`, `factor_importances` (liste `{key, label, share, mean_abs_contribution}`, |somme des contributions de la famille| moyenne sur l'entraînement) **remplace `feature_importances`**, `calibration` = celle de la LogReg.
- `evaluate_model.py` : compare le modèle servi à chacun des autres. `check_features.py` : contrôle les 31 features (`train_model.candidate_feature_cols`), vérifie `feature_columns.json` ⊆ features, et biais + facteurs = logit(p) sur les 50 matchs (tolérance 1e-3, arrondis de l'API ; mesuré 4,5e-4).
- `predictor_service.py` : `MODEL_PATH = logreg.pkl`, plus d'import xgboost, `_factors()` via `linear_contributions`. Les NaN (bilans en début de saison) sont remplacés par la médiane par le `SimpleImputer` du pipeline → contribution ≈ 0.
- Front : `ImportanceBars` lit `factor_importances` (6 familles), `featureLabel()` supprimé (plus utilisé), textes de `ModelView.js`, `MatchView.js` (sous-titre des facteurs), `FactorsChart.js` (« Modèle » au lieu de « Modèle XGBoost »), `layout.js` (description, pied de page). Corrigé au passage : « Elo ramené de 25 % » codé en dur dans « À savoir » de `MatchView.js` (faux depuis l'étape 2) → sans chiffre, renvoi à la page « Le modèle ».
- Déploiement : `logreg.pkl` au lieu de `xgboost_v1.pkl` dans `deploy_to_vps.ps1` et `deploy_remote.sh` ; `deploy_remote.sh` copie l'ancien `saved_models` dans `saved_models.prev` avant d'installer le nouveau (un retour arrière via `app.prev` doit restaurer le modèle de la même version, voir `docs/deploiement.md`), et vérifie `/games/today` après redémarrage (200 ou 404 admis : charge le modèle et `team_state.json`). L'ancien `xgboost_v1.pkl` reste sur le VPS, inutilisé.
- Prédictions réelles (107 matchs de saison régulière du calendrier, 2026-10-20 → 2026-11-02), ancien XGBoost contre LogReg : proba domicile moyenne 53,5 % → 56,0 % (Elo seul 56,1 %), écart moyen 7,8 pts (max 23,7), corrélation avec l'Elo seul 0,91 → 0,98, plage 0,09-0,89 → 0,23-0,84. PHI–CLE le 2026-10-22 : 40,2 % → 49,7 % ; OKC–DEN : 48,7 % → 63,7 % (Elo seul 59,5 %).
- Vérifié : `train_model.py`, `check_features.py` (tout OK), `evaluate_model.py --variant step3 --compare step3_ref`, API locale (`/model/metrics`, `/games/today?date=2026-10-22` et `2026-11-01`, `/games/{id}` : biais + facteurs = logit(p)), `eslint`, `next build`, captures CDP (page modèle 375/1280 px clair/sombre, fiche match 375 sombre / 1280 clair, accueil 1280 sombre : aucun débordement, aucune erreur console). Captures `docs/screenshots/modele-*.png` remplacées ; `match-*.png` pas refaites (seul le sous-titre des facteurs a changé, et la capture headless ne trace pas les courbes Elo animées).
- **Déployé le 2026-10-19** (16:00, commit `7903758`) avec `deploy_to_vps.ps1` seul : `team_state.json` ne dépend ni des features du modèle ni de `FACTOR_FAMILIES`, pas de synchro nécessaire. Vérifié en production : `/api/model/metrics` → `served_model: logreg` et `factor_importances` ; prédictions du 2026-10-22 identiques au local (PHI–CLE 49,7 %, OKC–DEN 63,7 %) ; fiche match : biais + facteurs = logit(p) ; pages en 200 ; `saved_models.prev` contient l'ancien modèle XGBoost ; journal sans avertissement ; contrôle `/games/today` du script distant passé ; mémoire du service 188 Mo. Entre les deux, l'API du VPS sert le nouveau modèle avec l'ancien `team_state.json` et journalise l'avertissement. Inversement, **aucune synchro avant le déploiement** : `sync_to_vps.ps1` calcule `team_state.json` avec le code local, donc avec les nouveaux paramètres, et l'ancien modèle du VPS recevrait des Elo sur la nouvelle échelle (l'API du VPS le signale désormais dans son journal, mais sert quand même les prédictions).

### Historique étendu à 2016-17 + poids par saison (FAIT et déployé le 2026-10-19)

- `fetch_games.py --refresh` a téléchargé 2016-17, 2017-18 et 2018-19 (3 × 1 230 matchs, aucune ligne orpheline en plus). Mêmes 30 `team_id`.
- **Dérive de l'époque** : victoires à domicile 58-59 % en 2016-19 contre 54-55 % depuis 2019-20 ; total moyen de points 211 (2016-17) → 231 (2025-26).
- Historique seul (`oof_history.csv` contre `oof_step3.csv`) : LogReg −0,0006 [−0,0015 ; +0,0003], XGBoost −0,0013 [−0,0031 ; +0,0005] **et −0,5 pt d'accuracy (significatif)**, Elo seul identique à 4 décimales (régression de 50 % : le point de départ est oublié en 2 saisons, donc pas de « meilleur rodage » mesurable). La LogReg surestime le domicile : proba moyenne − taux réel +1,3 pt (contre −0,05 avant).
- **Poids par saison** `0,5 ** (âge / SEASON_HALF_LIFE)` (âge compté depuis la saison la plus récente de l'entraînement), `evaluate_model.season_weights()` / `fit_spec()`, appliqués à la LogReg (`clf__sample_weight`) et à XGBoost (`sample_weight`, aussi dans l'arrêt précoce), en walk-forward, en `--legacy` et pour les modèles servis (`ModelSpec.weight_param`). Banc d'essai hors dépôt, LogReg, réglage sur 2021-22 → 2023-24 : demi-vie 1/2/3/5 → 0,6280/0,6277/0,6278/0,6279 (plat), fenêtres glissantes 3 et 5 saisons moins bonnes. **Demi-vie 2 adoptée** : −0,0010 [−0,0017 ; −0,0003] contre `step3`, Brier −0,0005 [−0,0008 ; −0,0001], 4/5 saisons ; 2024-25 + 2025-26 : −0,0007 [−0,0016 ; +0,0001] ; biais domicile +0,6 pt ; ECE 0,015 → 0,014. **Contrôle d'attribution** : les poids sur l'ancien historique 2019+ seul restent dans le bruit (−0,0004 [−0,0009 ; +0,0000]) ; c'est la combinaison qui passe la règle.
- `tune_elo.py` rejoué sur l'historique étendu : mêmes paramètres optimaux (20/40/0,5), log loss de réglage 0,6328 → 0,6329.
- `team_state.json` régénéré : identique à l'ancien à 1e-6 point d'Elo près (même `recent_games`, `season_results`, `elo_history`). **Un déploiement suffit, pas de synchro nécessaire.** Inversement, une synchro faite avant le déploiement est sans danger (Elo identiques, mêmes `elo_params`).
- Prédictions réelles (107 matchs de saison régulière du calendrier, 2026-10-20 → 2026-11-02) : proba domicile moyenne 56,0 % → 56,3 %, écart moyen 1,3 pt, max 3,3. PHI–CLE le 2026-10-22 : 49,7 % → 49,3 % ; OKC–DEN 63,7 % → 65,8 %.
- `model_metrics.json` : `train_seasons`, `season_half_life` (ajouts) ; `train_games` = 11 592 matchs complets. Pied de page du front : « 10 saisons ».

### Étape 5 : score prédit (FAIT et déployé le 2026-10-19)

- Méthode servie (formules dans `features_lib.py`, partagées API/évaluation) : **total** = `expected_total()` = (attaque domicile + défense extérieur)/2 + (attaque extérieur + défense domicile)/2 sur `SCORE_WINDOW` = 10 matchs (même total que l'ancienne heuristique) ; **écart** = `expected_margin()` = σ·Φ⁻¹(p) (`scipy.special.ndtri`, p écrêtée à 1e-6) ; `split_score()` ; `rounded_score()` : entiers, et si les arrondis sont égaux le favori prend +1 (jamais d'égalité, vainqueur du score = `predicted_winner`). Total manquant (équipe sans match récent) → `fallback_total` = total moyen de la dernière saison (231,2).
- σ = moindres carrés pondérés (poids par saison) de l'écart réel sur Φ⁻¹(p) des matchs d'entraînement, p = probabilités du modèle servi (`score_model.fit_sigma`). Modèle servi : **σ = 12,77** ; par pli 11,9 → 12,7. Alternative mesurée : σ = écart-type résiduel d'une Ridge de l'écart (étape 3d) = 13,5, MAE de l'écart identique à 0,01 pt près ; moindres carrés gardés (pas de second modèle).
- **Régression du total NON adoptée** : linéaire ou Ridge sur les moyennes 10 matchs, 5 + 10 matchs, + jours de repos, poids demi-vie 1/2/4 ou aucun, historique 2016 ou 2019 : toutes de −0,004 à +0,025 pt de MAE face à la formule (dans le bruit ou pire). Formule gardée (plus simple). Piste non essayée : un rythme (pace) glissant.
- **Walk-forward** (`score_model.walk_forward_scores`, scores affichés = entiers), 6 140 matchs, MAE par équipe / sur l'écart / sur le total : servi 9,29 / 10,89 / 14,91 ; ancienne heuristique 9,44 / 11,41 / 14,92 ; naïf (total et écart moyens de la saison précédente) 10,18 / 12,29 / 16,15. Servi − ancienne : **par équipe −0,145 [−0,175 ; −0,114], écart −0,525 [−0,617 ; −0,430], 5/5 saisons** ; total −0,009 [−0,026 ; +0,008]. L'ancienne heuristique contredisait le favori sur **21,6 % des matchs** (21 sur les 107 matchs réels du calendrier), la nouvelle méthode jamais.
- `model_metrics.json` : `score_model` = `{sigma, fallback_total, window, evaluation: {methods[] (model/previous/naive : mae_team, mae_margin, mae_total, winner_mismatch), per_season[] (avec sigma du pli), comparisons: {model_vs_previous, model_vs_naive}}}`. **L'API en a besoin** : un `model_metrics.json` sans `score_model` → 503 (`PredictorNotReadyError`). Il est déjà dans les fichiers envoyés par `deploy_to_vps.ps1`.
- API : `predicted_home_score` / `predicted_away_score` sont maintenant des **entiers** (avant : flottants à 0,1). Le front les affichait déjà avec `int()`.
- `check_features.py` : nouveau contrôle, sur les 50 matchs : score servi = formules de `features_lib` appliquées aux features d'entraînement, et vainqueur du score = vainqueur prédit. Vérifié capable d'échouer (σ multiplié par 1,15 côté API → échec, code 1).
- Front (`ModelView.js`) : carte « Score estimé » (3 mesures : par équipe, sur l'écart, sur le total, chacune avec l'ancienne méthode et le naïf, + texte sur la méthode et les 22 %), textes « Sans tricher » (historique depuis 2016-17, poids) et « Les limites » (erreur moyenne du score).
- Vérifié : `train_model.py`, `check_features.py` (tout OK), API locale (`/model/metrics`, `/games/today?date=2026-10-22` : PHI–CLE 49,3 % → 118–119, départage du favori ; `/games/{id}` : biais + facteurs = logit(p) à 2e-4), `eslint`, `next build`, captures CDP de la page modèle (1280 clair, 375 sombre : pas de débordement, aucune erreur console). Docs (`README.md`, `docs/architecture.md`) mises à jour. Captures `docs/screenshots/modele-*.png` **pas refaites**.
- **Déployé le 2026-10-19** (20:21, commits `c13c5fa` + `3b64e1a`) avec `deploy_to_vps.ps1` seul, sans synchro (team_state.json inchangé). Vérifié en production : `/api/model/metrics` → `train_seasons` 2016-17 → 2025-26, `score_model.sigma` 12,77 ; prédictions du 2026-10-22 identiques au local (PHI–CLE 49,3 % → 118–119, OKC–DEN 65,8 % → 124–118) ; fiche match : biais + facteurs = logit(p) ; pages `/`, `/modele`, `/match` en 200 ; journal du service sans avertissement (`sudo journalctl`, sans sudo on ne voit rien) ; mémoire 142 Mo.

### Étape 4 : calendrier (essayée le 2026-10-19, NON ADOPTÉE)

Banc d'essai hors dépôt (scratchpad, non conservé), même protocole que l'étape 3 : variantes choisies sur 2021-22 → 2023-24 (LogReg), confirmation sur les 5 saisons et sur 2024-25 + 2025-26, référence `oof_history_hl2.csv` (reproduite à l'identique par le banc). **Aucun code fusionné, rien à déployer.**

- Variables, par équipe, ajoutées en `diff_*` aux 11 features : `games_last_4d` = matchs joués de J-3 à J-1 (0 à 2) ; `travel_km` = distance (haversine) entre la salle du match précédent et celle du match ; `tz_shift` = écart de fuseau (heures, > 0 vers l'est) entre ces deux salles, offsets `zoneinfo` pris à la date du match (Phoenix sans heure d'été). Table `ARENAS` : salle actuelle des 30 équipes (lat/lon/fuseau), exceptions bulle d'Orlando (été 2020) et Toronto à Tampa (2020-21) ; matchs délocalisés (Mexico, Paris, Londres, demi-finales NBA Cup à Las Vegas, ~30 matchs) ignorés. Premier match connu d'une équipe : 0. Distributions (lignes équipe-match) : `games_last_4d` 0 / 1 / 2 pour 6 % / 68 % / 26 % ; décalage nul pour 65 %, de −3 à +3 h ; distance médiane 690 km.
- **Saisons de réglage** (Δ log loss LogReg contre la référence) : `games_last_4d` −0,0001 [−0,0005 ; +0,0004] ; `travel_km` +0,0004 [−0,0000 ; +0,0009] (identique en remettant à 0 après plus de 7 jours sans match) ; `tz_shift` −0,0002 [−0,0008 ; +0,0003] (−0,0003 avec remise à 0 après 7 jours) ; |`tz_shift`| +0,0003 ; les 3 ensemble +0,0001 (avec niveaux home/away en plus : −0,0001 ; remise à 0 après 3 jours : +0,0002). **Seule variante significative** : déplacement et décalage comptés uniquement en back-to-back (0 sinon), `travel_km` seul −0,0012 [−0,0024 ; −0,0001], les 3 −0,0010 [−0,0024 ; +0,0003] ; `tz_shift` seul en back-to-back +0,0002.
- **Confirmation (5 saisons)**, meilleure variante (`travel_km` en back-to-back) : LogReg −0,0005 [−0,0015 ; +0,0005], Brier −0,0003 [−0,0007 ; +0,0002], accuracy +0,5 pt [+0,2 ; +0,8] (5/5 saisons) mais **2024-25 + 2025-26 : +0,0004 [−0,0014 ; +0,0021]** ; XGBoost +0,0002. Le gain de réglage ne se généralise pas (meilleure de 14 variantes : effet de sélection probable). Version du plan (les 3 variables brutes) : LogReg +0,0004 [−0,0002 ; +0,0011], 2024-26 +0,0008 ; XGBoost +0,0000. Les 3 en back-to-back : LogReg −0,0003 [−0,0014 ; +0,0008], XGBoost +0,0015 sur 2024-26 [+0,0000 ; +0,0029].
- `--legacy` (log loss / accuracy LogReg) : référence 0,5922 / 68,4 % ; 3 variables 0,5918 / 68,6 % ; `travel_km` en back-to-back 0,5936 / 69,1 %.
- Coefficients de la LogReg entraînée sur tout (features standardisées) : `diff_travel_km` −0,047 (−0,062 en back-to-back), `diff_games_last_4d` −0,022, `diff_tz_shift` +0,009 : signes attendus (plus de route ou de matchs que l'adversaire = moins de chances), effets minimes devant `diff_rest_days`, qui code déjà le back-to-back.
- Si la piste est reprise un jour : ne pas regarder de nouveau les saisons 2024-25/2025-26 pour choisir la forme (déjà « consommées » ici) ; attendre 2026-27 comme confirmation.

### Étape 6 : blessures / effectif (essai concluant, intégré et DÉPLOYÉ le 2026-10-19, accord de l'utilisateur)

Essai d'abord sur un banc hors dépôt (scratchpad de la session), même protocole que les étapes 3-4, référence `oof_history_hl2.csv` reproduite à 3e-8 près ; intégration ensuite (voir « Intégration » en fin de section).

- **Données** : `LeagueGameLog(player_or_team_abbreviation="P", season_type_all_star="Regular Season")`, une requête par saison (~2 s), 2016-17 → 2025-26 : 254 885 lignes joueur-match, 100 % des 11 969 matchs couverts. Seuls les joueurs entrés en jeu y figurent (absence = pas de ligne). Accessible depuis le PC (même domaine `stats.nba.com` que `fetch_games.py`), donc pas depuis le VPS.
- **Indicateur retenu `lineup_delta`** (calculable à 10 h avec le seul passé) : valeur d'un joueur = Game Score moyen (Hollinger : PTS + 0,4 FGM − 0,7 FGA − 0,4 (FTA − FTM) + 0,7 OREB + 0,3 DREB + STL + 0,7 AST + 0,7 BLK − 0,4 PF − TOV) sur ses 15 dernières apparitions strictement avant la date du match (toutes équipes, saisons précédentes comprises) ; force d'un effectif = somme des 8 meilleures valeurs parmi les joueurs d'un match (le top 8 neutralise les fins de match pliées) ; `lineup_delta` = force de l'effectif du DERNIER match de l'équipe − force moyenne de ses 10 derniers matchs de la saison (valeurs prises à la date du match cible). Hypothèse : les absents du dernier match le restent ; l'Elo et la forme reflètent l'effectif habituel. Premier match de la saison : 0. Ajouté en `diff_lineup_delta` aux 11 features. Distribution par équipe-match : médiane +0,1, écart-type 7,6 (unités de Game Score), de −64 à +40.
- **Oracle** (borne haute, inutilisable : effectif réel du match lui-même, connu ~30 min avant) : même calcul avec les joueurs entrés en jeu. Corrélation proxy/oracle 0,42.
- **Saisons de réglage** (LogReg, Δ log loss) : `gmsc:delta` −0,0022 [−0,0045 ; +0,0003] ; `gmsc:missing` (valeur des titulaires habituels, présents dans ≥ 50 % des 10 derniers matchs, absents du dernier, pondérée par leur taux de présence) −0,0022 [−0,0049 ; +0,0005] ; les deux −0,0023 ; mêmes calculs en minutes au lieu du Game Score −0,0019 / −0,0019 ; oracle Game Score −0,0091 [−0,0138 ; −0,0043], oracle minutes −0,0076. 7 variantes seulement, fixées à l'avance ; `gmsc:delta` choisi AVANT la confirmation (à égalité : une seule variable, symétrique).
- **Confirmation (5 saisons)**, `gmsc:delta` : **LogReg −0,0023 [−0,0041 ; −0,0007], Brier −0,0011 [−0,0019 ; −0,0003], 4/5 saisons ; 2024-25 + 2025-26 −0,0025 [−0,0050 ; −0,0002], 2/2** ; accuracy +0,4 pt (bruit) ; ECE 0,014 → 0,010. XGBoost −0,0014 [−0,0027 ; −0,0002], 5/5 saisons. `--legacy` : LogReg 0,5922 → 0,5887 (accuracy 68,4 → 68,6 %), XGBoost 0,5912 → 0,5905. Pour mémoire : `gmsc:missing` −0,0026 [−0,0046 ; −0,0007] (2024-26 −0,0031) ; oracle −0,0082 [−0,0117 ; −0,0049] (2024-26 −0,0070). Coefficient LogReg (standardisé, tout l'historique) : +0,14, signe attendu.
- **Contrôles** : fuite (retirer les joueurs d'un match ne change pas l'indicateur de ce match, change celui du suivant, 5 matchs de 2022-23 : OK) ; gain réparti sur la saison (oct-déc −0,0021, jan-fév −0,0023, mars-avr −0,0026 : pas seulement les mises au repos de fin de saison) ; concentré sur les matchs où l'effectif a changé (|diff| ≥ 10, 1/3 des matchs : −0,0075 [−0,0121 ; −0,0029] ; ailleurs +0,0004, bruit). Limite visible : DEN le 2026-01-07, delta −48,6 (absents au match précédent) mais oracle −0,8 (ils sont revenus).
- **Verdict : GO pour la version « matchs passés »** (règle remplie, gain qui tient sur les saisons non vues). La version « rapport officiel des blessures » viserait au mieux l'écart à l'oracle (−0,006 de plus), mais ce rapport (PDF, mis à jour jusqu'au soir du match) n'est pas exploitable à 10 h de façon fiable : non essayée.
- **Intégration (FAITE le 2026-10-19, accord de l'utilisateur)** :
  - `data/collectors/fetch_player_logs.py` (nouveau) : `LeagueGameLog` par saison → `data/raw/player_games_raw.csv` (23 Mo, 255 086 lignes, colonnes utiles seulement, non versionné), cache + `--refresh` (même règle que `fetch_games.py`, dont il importe `all_seasons`), écriture atomique. Saison sans match (2026-27 aujourd'hui) : 0 ligne, pas d'erreur.
  - `data/preprocessing/player_games.py` (nouveau) : `load_player_games()` (colonnes en minuscules, `nba_game_id` entier), `restrict_to_games()` (lignes des seuls matchs de `games_clean.csv`, date du match : un historique tronqué à une date ne voit rien après), `coverage_warning()`.
  - `features_lib.py` : `PLAYER_VALUE_GAMES` 15, `LINEUP_WINDOW` 10, `LINEUP_TOP` 8, `game_score()`, `player_value_timeline()`, `lineup_strength()`, `lineup_delta(strengths)` (version une équipe, API) et `compute_lineup_deltas()` (batch vectorisé, ~5 s) ; famille « Effectif » (`lineup`) dans `FACTOR_FAMILIES`.
  - `feature_engineering.build_features(df_games, player_games)` (2ᵉ argument obligatoire) : `home/away/diff_lineup_delta` ; `train_model.FEATURE_PATTERNS` + `lineup_delta` → **12 features**.
  - `team_state.json` : `teams[id].lineup = {strengths (10 derniers effectifs de la saison, valeurs des joueurs à ce jour, flottants non arrondis), absent [{player_id, name, value, games, of}]}` ; `build_team_state(df_games, player_games)`.
  - `predictor_service.py` : `lineup_delta` = `features_lib.lineup_delta(strengths)`, 0 pour un match d'une saison postérieure (comme à l'entraînement) ou si `team_state.json` n'a pas le champ (avertissement dans le journal : « relancer sync_to_vps.ps1 ») ; `team_snapshot()` → `absent_players` (vide en nouvelle saison) ; `/games/{id}` le renvoie par équipe.
  - `check_features.py` : feuilles joueur passées partout ; nouveau contrôle 3 (meilleur joueur retiré de la feuille de 5 matchs : rien ne change avant/pendant, `lineup_delta` du match suivant change). Vérifié capable d'échouer (fuite injectée : valeur des joueurs incluant le match lui-même → ÉCHEC). Le contrôle 1 a attrapé un vrai bug en cours de route (effectifs regroupés par match et non par (équipe, match) dans `team_state.py`). Durée totale ~75 s (au lieu de ~15 s).
  - `sync_to_vps.ps1` : étape « Feuilles de match joueur » (`fetch_player_logs.py --refresh`) après `fetch_games.py --refresh` (BOM conservé).
  - Front : ligne « Effectif au dernier match, vs habituel » dans `ComparisonBars` (domaine −30 → +10) ; section « Absents au dernier match » dans `MatchView` (masquée si les deux listes sont vides) ; puce « À savoir » sur la limite (retours de blessure, forfaits de dernière minute) ; textes de `ModelView` (en-tête, « Ce qui compte le plus », « Les limites »). README et `docs/architecture.md` (section « Les absences ») à jour.
- **Résultats de la version intégrée** (`oof_lineup.csv`, désormais LA RÉFÉRENCE) : LogReg 66,1 % / log loss 0,6141 / Brier 0,2128 / ECE 0,010 ; XGBoost 65,7 % / 0,6180 ; Elo seul inchangé 0,6184. Contre `history_hl2` : LogReg −0,0023 [−0,0040 ; −0,0006], Brier −0,0011, 4/5 ; XGBoost −0,0015 [−0,0028 ; −0,0002], 5/5. **LogReg − Elo seul : −0,0042 [−0,0070 ; −0,0015], 5/5 saisons** (dans le bruit avant). `--legacy` : LogReg 0,5888 / 68,6 %, XGBoost 0,5908. Poids par famille : Elo 58,9 %, forme 12,8 %, **effectif 10,9 %**, repos 7,9 %, attaque 4,3 %, dom./ext. 4,0 %, bilan 1,3 %. Coefficient standardisé de `diff_lineup_delta` +0,139. σ du score 12,74 ; MAE par équipe 9,27.
- Écarts avec le banc d'essai (491 matchs, expliqués) : une ligne joueur ajoutée par la NBA depuis l'essai (Schröder, 0 min, 2022-11-06), et les 10 matchs sur terrain neutre absents de `games_clean.csv` (voir ci-dessous), que l'essai comptait dans la valeur des joueurs et plus la version intégrée (cohérent avec l'Elo).
- **Découverte, NON traitée** : `fetch_games.recombine_games()` écarte les matchs sur terrain neutre (NBA Cup à Las Vegas, Paris, Mexico, Berlin, Londres : `MATCHUP` en « @ » pour les deux équipes) : 10 matchs depuis 2024-25 absents de tout l'historique (Elo, forme, entraînement). Piste : un domicile conventionnel.
- Prédictions réelles (calendrier, matchs du 2026-10-22, nouvelle saison donc `lineup_delta` = 0) : PHI–CLE 49,3 % → 50,4 %, OKC–DEN 65,8 % → 66,0 % (changement du seul réentraînement). L'indicateur n'agira qu'à partir du 2ᵉ match de chaque équipe en 2026-27.
- Vérifié : `check_features.py` (tout OK), routeurs appelés directement (`/games/today` 2026-10-22 et 2026-04-08, `/games/{id}` : biais + facteurs = logit(p) à 2e-4, facteur « Effectif » présent), `npm run lint`, `npm run build`, captures Chrome headless 1280 px de la fiche match CLE–ATL du 2026-04-08 (absents des deux côtés) et de la page modèle (sombre). **Non vérifié** : affichage 375 px et thème clair de la nouvelle section, `sync_to_vps.ps1` réel (seul le collecteur a été lancé à la main, avec et sans `--refresh`).
- **Déployé le 2026-10-19** (21:20, commit `513470a`) avec `deploy_to_vps.ps1`, puis `sync_to_vps.ps1` 1 min après (première synchro réelle avec l'étape « Feuilles de match joueur » : OK ; scp affiche « close - IO is still pending on closed socket », sans effet, code de sortie 0). Vérifié en production : `/api/model/metrics` (LogReg 0,6141, « Effectif » 10,9 %), prédictions du 2026-10-22 identiques au local (PHI–CLE 50,4 % → 119–118, OKC–DEN 66,0 % → 124–118), `/games/{id}` avec `absent_players` et biais + facteurs = logit(p), pages `/`, `/modele`, `/match` en 200, `team_state.json` du VPS avec `lineup` pour les 30 équipes, mémoire 139 Mo. Journal : un seul avertissement « team_state.json sans données d'effectif » au redémarrage (avant la synchro, attendu), plus rien après.
- Ordre pour tout futur changement de l'effectif : `deploy_to_vps.ps1` puis immédiatement `sync_to_vps.ps1` (entre les deux, `lineup_delta` = 0 et avertissement, pas de plantage).
- **Commits** : `669d6f4` (pipeline + API + synchro), `44840b5` (front), `21ff3b1` (docs), `513470a` : suppression des 4 fichiers vides de l'API (`routers/players.py`, `predictions.py`, `teams.py`, `services/nba_sync_service.py`, supprimés par l'utilisateur avant la session et jamais importés) — nécessaire, car `deploy_to_vps.ps1` copie les fichiers de `git ls-files` et échouait sur des fichiers suivis mais absents.

## Score v2 : rythme × efficacité + fourchette à 80 % (2026-10-21)

Demande de l'utilisateur : « une IA complète », meilleur score prédit ET statistiques des joueurs ; accord pour le faire en deux fois. Cette partie = le score. Plan validé (fichier de plan de la session), même protocole que les étapes du plan d'amélioration : réglage sur 2021-22 → 2023-24 (rodage 2016-17 → 2020-21), confirmation sur les 5 saisons et sur 2024-25 + 2025-26, bootstrap apparié par journée, sous-essais sur un banc hors dépôt.

- **Méthode** (`features_lib.py`, section « Ratings de score ») : total = possessions attendues × points par possession attendus. Par équipe, deux notes mises à jour après chaque match comme l'Elo (donc ajustées aux adversaires) : `pace` (possessions au-dessus de la ligue) et `scoring` (points pour 100 possessions des deux équipes réunies, au-dessus de la ligue). Niveau de la ligue (`pace`, `efficiency` par équipe) suivi lui aussi, **mis à jour en fin de journée seulement**. Régression des notes d'équipe vers 0 entre saisons. Possessions d'un match = moyenne des deux équipes de FGA + 0,44 FTA − OREB + TOV (`game_possessions`, différente de `net_rtg_from_boxscore`, feature du modèle, inchangée). Départ fixe `LEAGUE_INITIAL` (100 possessions, 106 pts/100) en 2016-17. **Écart inchangé** (σ·Φ⁻¹(p)), vainqueur du score = favori. **Fourchette** : ± demi-largeur fixe = quantile 80 % pondéré par saison de |erreur par équipe| sur l'entraînement (`score_model.fit_score_params`), 14,9 pts pour le modèle servi.
- **Une seule note de points au lieu d'attaque + défense** : pour le total, seule compte attaque − défense par équipe ; vérifié, les deux versions donnent les mêmes totaux à 1e-13, et l'avantage du terrain n'a alors aucun effet sur le total (paramètre retiré).
- **Paramètres** `DEFAULT_SCORE_RATING_PARAMS` : k_pace 0,04, k_scoring 0,03, régression 0,5, league_rate 0,01 (`tune_score_ratings.py`, 500 combinaisons, profil plat autour). Réglage : MAE du total 14,79 (moyennes des 10 derniers matchs) → 14,56 ; contrôle 2024-26 : 15,10 → 14,77, Δ −0,34 [−0,51 ; −0,17], 2/2.
- **Walk-forward** (scores affichés, 6 140 matchs), servi / précédente / naïf : par équipe **9,19** / 9,27 / 10,18 ; écart 10,85 / 10,84 / 12,29 ; total **14,64** / 14,93 / 16,15. Servi − précédente : total **−0,28 [−0,39 ; −0,18], 5/5** ; par équipe −0,09 [−0,13 ; −0,05], 5/5 ; écart +0,008, bruit. Sur 2024-25 + 2025-26 seules : total −0,35 [−0,52 ; −0,18], par équipe −0,12 [−0,18 ; −0,06], 2/2. Début de saison (782 matchs) : total 14,72 contre 14,83. **Couverture de la fourchette : 80,9 %** (79,6 à 82,5 % par saison, 81,5 % en début de saison ; légèrement au-dessus de 80 % à cause des bornes entières incluses).
- **Sous-essais NON adoptés** (saisons de réglage) : (a) niveau de la ligue remis à la moyenne de la saison précédente à chaque nouvelle saison −0,008 [−0,049 ; +0,033] (vise un biais de −4,6 pts en octobre, dû surtout à 2021-22 et au changement des règles sur les fautes provoquées) ; (c) mélange avec les moyennes des 10 derniers matchs, au mieux −0,03 (0,7/0,3 : [−0,07 ; +0,01]), garderait deux méthodes ; (d) efficacité nette ajustée (attaque + défense, avec avantage du terrain 2 pts/100) comme feature de la LogReg : log loss +0,0000 [−0,0007 ; +0,0007], corrélée à 0,989 avec `diff_elo`. Sondage avant le plan : écart tiré des ratings 10,79 contre 10,68 pour σ·Φ⁻¹(p) (mélanges aussi moins bons), back-to-back et `lineup_delta` sans effet notable sur le total, σ proportionnel au rythme sans effet.
- **Fuite trouvée par `check_features.py`** : première version, niveau de la ligue mis à jour après chaque match → un match du soir changeait la prévision d'un autre match du même soir (le contrôle 2 et le contrôle 1, écart API/pipeline de 0,1-0,5 pt, l'ont attrapée). Corrigé (mise à jour par journée), grille rejouée : mêmes paramètres optimaux.
- `check_features.py` : colonnes `exp_possessions`/`exp_total` contrôlées partout (`SCORE_COLS`) ; contrôle 1 compare aussi `PredictorService.expected_total` à `exp_total` (1e-9) et le score + les fourchettes servis ; contrôle 2 ajoute +10 points à chaque équipe du match inversé (l'inversion seule laisse le total intact) et exige que le total attendu des matchs suivants change. Vérifié capable d'échouer : total d'avant-match qui « voit » 1 % du résultat → ÉCHEC ; API sans régression de saison → ÉCHEC sur les matchs du soir d'ouverture. Durée ~110 s.
- **Pipeline** : `feature_engineering.build_features` ajoute `exp_possessions`, `exp_total` (pas des features du modèle : pas de préfixe home_/away_/diff_). `score_model.py` : méthodes `model` (servie), `previous` (= méthode du 2026-10-19), `naive` ; l'heuristique d'avant le 2026-10-19 n'est plus évaluée ; `latest_season_total` prend la dernière saison COMPLÈTE (≥ 1 000 matchs : 2026-27 n'en avait que 3). `model_metrics.json` → `score_model = {sigma, fallback_total, rating_params, interval: {level, half_width}, evaluation: {..., interval: {level, coverage, coverage_early_season, mean_half_width}, per_season[].half_width/coverage}}` ; `window` retiré. Modèle de victoire inchangé (LogReg identique à 4e-16, `--variant score_v2 --compare lineup` : Δ log loss 0).
- **`team_state.json`** : `score_ratings = {params, league}` + `teams[id].score_rating = {pace, scoring}`, reste du fichier identique. **API** : `PredictorService.expected_total()` (régression de saison comme l'Elo) et `scores()` (public, utilisés par `check_features.py`) ; `predicted_home_score_range` / `predicted_away_score_range` dans `/games/today` et `/games/{id}` ; `_check_score_rating_params()` (avertissement si code / `model_metrics.json` / `team_state.json` diffèrent) ; `team_state.json` sans `score_ratings` → avertissement et total = `fallback_total` (231,2) ; `model_metrics.json` sans `rating_params`/`interval` → 503 (`PredictorNotReadyError`). Vérifié : les deux cas dégradés.
- **Front** : fiche match, ligne « Fourchette à 80 % » sous le score (pastille de couleur de chaque équipe, texte `sr-only` « entre X et Y points »), puce « À savoir » réécrite ; page modèle, `ScoreCard` avec 4ᵉ tuile (couverture de la fourchette, ±demi-largeur), « méthode précédente », texte sur le rythme et les notes. Cartes de l'accueil inchangées.
- Prédictions réelles (calendrier, avec le `team_state.json` synchronisé le 2026-10-21) : PHI–CLE le 2026-10-22 38,2 % → 115–119, fourchettes [100, 130] / [104, 134] ; BOS–NYK le 2026-10-23 63,3 % → 114–110.
- Vérifié : `tune_score_ratings.py`, `feature_engineering.py`, `train_model.py`, `team_state.py`, `check_features.py` (tout OK), routeurs appelés directement (`/games/today` 2026-10-22 et 2026-10-23 : fourchettes présentes, vainqueur cohérent ; `/games/{id}` : biais + facteurs = logit(p) à 1e-4 ; `/model/metrics`), `npm run lint`, `npm run build`, captures CDP (API locale sur 8011, export statique servi sur 3011) : fiche match 375 px sombre / 1280 px clair, page modèle 1280 px sombre / 375 px clair, aucun débordement, aucune erreur console.
- **Déployé le 2026-10-21** (12:56, commit `dc7927e` ; commits `952e608` pipeline + API, `d76634b` front, `dc7927e` docs) avec `deploy_to_vps.ps1`, puis `sync_to_vps.ps1` 1 min après. Vérifié en production : `/api/model/metrics` (`score_model.rating_params`, `interval` ±14,9, couverture 80,9 %), `/api/games/today?date=2026-10-23` identique au local (12 matchs, fourchettes présentes), `/api/games/{id}` : biais + facteurs = logit(p) à 1e-4, pages `/`, `/modele`, `/match` en 200, `team_state.json` du VPS avec `score_ratings` et `score_rating` pour les 30 équipes, mémoire 193 Mo. Journal : un seul avertissement « team_state.json sans ratings de score » au redémarrage (avant la synchro, attendu), plus rien après.
- Ordre pour tout futur changement des ratings de score : `deploy_to_vps.ps1` PUIS immédiatement `sync_to_vps.ps1` (entre les deux : total de repli + avertissement, pas de panne). Une synchro AVANT le déploiement est sans danger (l'ancienne API ignore les nouveaux champs).

## Statistiques des joueurs (2026-10-21, « IA complète » partie 2)

Demande de l'utilisateur : « un système de prédiction des statistiques des joueurs ». Choix de l'utilisateur : **minutes, points, rebonds, passes décisives, interceptions, contres** (pas les 3 pts, qui auraient demandé de re-télécharger les feuilles joueur), **fourchette à 80 %**, affichage sur la **fiche match seule** (pas l'accueil, pas de page joueur). Données : `player_games_raw.csv` existant, rien à télécharger, `sync_to_vps.ps1` inchangé. Modèle de victoire et score inchangés (vérifié : `logreg.pkl` identique à 0,0 près, métriques et `score_model` identiques).

- **Méthode servie** (formules dans `features_lib.py`, section « Statistiques des joueurs ») : joueurs attendus = joueurs du dernier match de l'équipe dans la saison (aucun au premier match de saison) ; **minutes = XGBoost** (`player_model.fit_minutes_model`, profondeur 4, lr 0,05, min_child_weight 50, arrêt précoce sur les 20 % de lignes les plus récentes, patience 50, poids par saison ; 314 arbres pour le modèle servi, entraîné sur 196 468 lignes 2017-18 → 2025-26) sur `MINUTES_FEATURES` (28 entrées : moyennes de minutes h 2 et h 10, dernier match, saison, 10 derniers, nombre de matchs, matchs de l'équipe dans la saison, repos, domicile, possessions et total attendus, proba Elo de l'équipe, nombre de joueurs du dernier match et leurs minutes, taux, projections de la formule, facteurs adversaire) ; **autres stats = minutes × per_min** : taux du joueur = (somme pondérée stat + k · taux ligue saison précédente) / (somme pondérée minutes + k), demi-vie et k par stat (`DEFAULT_PLAYER_PARAMS` : pts 10/25, reb 20/25, ast 10/25, stl 40/200, blk 40/100), × possessions attendues (`exp_possessions`) / possessions moyennes des matchs du joueur, × (ce que concède l'adversaire, moyenne pondérée h 20 / moyenne ligue saison précédente)^0,5 pour reb/ast/stl/blk. Valeurs = espérances « s'il joue ».
- **Fourchette à 80 %** (`player_model.fit_interval_table`) : tranches de projection (20 quantiles + 0,975/0,99/0,995), dans chaque tranche l'intervalle entier de masse la plus proche de 80 % (pour chaque largeur l'intervalle le plus chargé). q10/q90 avec bornes incluses donnaient 85 à 95 % (stats discrètes) ; « intervalle le plus proche de 80 % » sans contrainte de largeur donnait des intervalles absurdes (1–12 passes). **Apprise sur des projections hors échantillon** (walk-forward depuis `OOF_START` = 2018-19), les minutes in-sample étant trop bonnes. Tables dans `model_metrics.json` → `player_model.intervals.tables`, lookup `features_lib.stat_ranges`.
- **Protocole** : banc d'essai hors dépôt (scratchpad, non conservé), réglage sur 2021-22 → 2023-24 (rodage 2016-17 → 2020-21), confirmation sur 5 saisons et 2024-25 + 2025-26, bootstrap apparié par journée (`paired_bootstrap`). Lignes évaluées : joueurs attendus ET entrés en jeu (111 136). **Décision sur l'erreur quadratique** (valeur affichée = espérance), MAE affichée.
- **Réglage** (`tune_player_projections.py`, reproduit à l'identique) : demi-vie des minutes 2 (MSE 41,87 ; 1 → 43,55, 5 → 43,40, 15 → 48,30) ; remise à l'échelle des minutes à 241,6 par équipe (moyenne de la ligue, prolongations comprises) **moins bonne** quel que soit le total visé (h 3 : 42,15 → 44,18), écartée. Taux : grille demi-vie {5…160} × k {0…800}, profils plats ; contrôle 2024-26 contre la moyenne de la saison : 6/6 stats, IC < 0.
- **Sous-essais** (MSE, Δ face à l'état précédent, 5 saisons ; tous fixés à l'avance sauf la variante « minutes XGBoost × taux », issue de (e)) :
  - (a) rythme du match : **adopté**, pts −0,080 [−0,100 ; −0,060], reb −0,008, ast −0,004, stl −0,0005, blk −0,0004, 2024-26 aussi, MAE meilleure partout ;
  - (b) points ramenés au score servi de l'équipe : **écarté**, +0,50 [+0,37 ; +0,62] (sur base + a), 5/5 ;
  - (c) minutes réduites selon l'écart attendu |σΦ⁻¹(p)| (β ajusté sur le réglage) : minutes −0,70 en MSE mais **MAE dégradée** sur toutes les stats (+0,013 minutes) → **écarté** ;
  - (d) adversaire : exposant 0,5 choisi sur le réglage (meilleur que 1 sur 3 stats sur 4), **adopté** : reb −0,013 [−0,016 ; −0,009], ast −0,012, stl −0,003, blk −0,004, 5/5, 2024-26 aussi, MAE meilleure ;
  - (e) XGBoost par stat (profondeur 2/4/6 sur le réglage → 4) : bat la formule (minutes −2,88, pts −0,65, reb −0,09, ast −0,02 ; stl MAE +0,004, blk dans le bruit), **mais « minutes XGBoost × taux de la formule » fait mieux** : XGBoost direct − variante : pts −0,04 (bruit), reb +0,022*, ast +0,008*, stl +0,002*, blk +0,0015*. **Adopté : un seul modèle appris (minutes)**, plus simple et meilleur. Profondeur 6 pour les minutes : −0,05 (2/3 saisons), gardé 4 (plus petit).
- **Résultats servis** (walk-forward, `player_model.player_summary`, MAE ; RMSE) : minutes 4,77 ; 6,20 · points 4,68 ; 6,06 · rebonds 1,91 ; 2,51 · passes 1,35 ; 1,82 · interceptions 0,72 ; 0,92 · contres 0,53 ; 0,73. Moyenne de saison : 5,31 / 4,80 / 1,97 / 1,38 / 0,73 / 0,53 ; 10 derniers : 5,14 / 4,81 / 1,98 / 1,39 / 0,74 / 0,53 ; formule sans modèle de minutes : 4,89 / 4,70 / 1,92 / 1,35 / 0,71 / 0,53. MSE servi − moyenne de saison : minutes −10,3, pts −2,97, reb −0,55, ast −0,26, stl −0,058, blk −0,035, tous 5/5 ; servi − formule : minutes −2,88 [−3,37 ; −2,51], pts −0,61 [−0,73 ; −0,51], reb −0,11, ast −0,030, stl −0,0039, blk −0,0022, 5/5. Couverture des fourchettes : min 79,6 %, pts 80,2 %, reb 80,8 %, ast 81,0 %, stl 81,7 %, blk 82,7 % (largeurs 13,8 / 13,1 / 4,9 / 3,1 / 1,0 / 0,5). 14,2 % des joueurs attendus ne jouent pas ; 9,0 % des minutes vont à des joueurs non attendus.
- ⚠️ **Exception à la règle « MAE non dégradée »** (décision de Claude, signalée à l'utilisateur) : face à la formule seule, MAE des interceptions +0,0021 et des contres +0,0024 (significatif), alors que la MSE baisse (5/5). La MAE récompense la médiane (0 pour la plupart des joueurs), la valeur affichée est une espérance ; garder les minutes de la formule pour ces deux stats aurait donné deux projections de minutes pour un même joueur. Contres : MAE aussi légèrement moins bonne que la moyenne de saison (+0,004). Si l'utilisateur préfère respecter la règle à la lettre : `projected_stats` avec `features["minutes"]` pour stl/blk.
- **Pipeline** : `player_games.py` type aussi `min` ; `train_model.py` appelle `player_model` (walk-forward ~1 min en plus), écrit `player_minutes.pkl` et `model_metrics.json["player_model"] = {params, minutes_features, minutes_model, intervals: {level, tables}, evaluation}`. `team_state.py` : `teams[id].players` (état de chaque joueur du dernier match de la saison, `player_state_cols`, + `name`), `teams[id].opponent_allowed`, `player_league` (`rates`, `allowed` de la saison précédente), `as_of.player_params`.
- **API** : `PredictorService.player_inputs()` / `player_rows()` / `player_projections()` ; `predict(..., with_players=True)` → `GamePrediction.players = {home: [...], away: [...]}` (chaque joueur : `player_id, name, games, stats: {min|pts|reb|ast|stl|blk: {value (0,1 près), range [bas, haut]}}`, tri par minutes) ; `/games/today` appelle `with_players=False` (0,07 s pour 12 matchs) ; `/games/{id}` renvoie `prediction.players`. `player_minutes.pkl` absent ou `model_metrics.json` sans `player_model` → 503 ; `MINUTES_FEATURES` différent de `player_model.minutes_features` → 503 ; `team_state.json` sans `player_league` → listes vides + avertissement ; `_check_player_params()` comme pour l'Elo. Nouvelle saison → listes vides. `expected_game()` (possessions, total) ajouté, `expected_total()` s'appuie dessus.
- **check_features.py** : contrôle 1 étendu (79 équipes avec joueurs, 21 sans sur l'échantillon : mêmes joueurs, `PLAYER_COLS` à 1e-9, mêmes stats/fourchettes servies) ; contrôle 2 étendu (inverser un match ne change aucune entrée de joueur antérieure, en change après) ; **contrôle 4** (ligne du joueur le plus utilisé gonflée : rien ne bouge jusqu'au match inclus, ses entrées au match suivant changent). Vérifiés capables d'échouer : minutes du match lui-même injectées dans l'état (contrôles 4 et 1 en échec), `rest_days` + 1 côté API (contrôle 1 en échec).
- **Front** : `PlayerProjections.js` (tableau par équipe, MIN PTS REB PD INT CTR, valeur + fourchette en dessous, « 0 » si bornes égales, `sr-only` « fourchette de X à Y », colonne des noms figée, conteneur `relative overflow-x-auto` : sans `relative`, les `sr-only` en position absolue élargissaient la page de 103 px à 375 px) ; section « Statistiques des joueurs » de `MatchView.js` (côte à côte à partir de `xl`, message si liste vide) + puce « À savoir » ; `PlayerCard` de `ModelView.js` (MAE à 2 décimales : à 1 décimale les petites stats se confondaient avec les références), en-tête et « Les limites ».
- **Déploiement** : `player_minutes.pkl` ajouté à `$ModelFiles` (`deploy_to_vps.ps1`) et `MODEL_FILES` (`deploy_remote.sh`). Ordre : `deploy_to_vps.ps1` PUIS immédiatement `sync_to_vps.ps1` (entre les deux : aucune projection + avertissement, pas de panne). Une synchro avant le déploiement est sans danger.
- Vérifié : `train_model.py`, `team_state.py`, `check_features.py` (tout OK, ~4 min), `tune_player_projections.py`, routeurs appelés directement et API locale (8011) : `/games/today` et `/games/{id}` 2026-10-23 (BOS–NYK : 11 et 10 joueurs ; MIA–DET : 0 et 11, premier match de MIA), cas dégradés (ancien `team_state.json`, `player_model` absent, modèle absent, saison suivante) ; `npm run lint`, `npm run build`, captures CDP (export statique servi sur 3011) : fiche match 375 sombre / 375 clair / 1280 clair / 1280 sombre, page modèle 1280 sombre / 375 clair, aucun débordement, aucune erreur console hors 404 `__PAGE__.txt` connu (corrigé par Caddy en production). Mémoire du processus API en local : ~313 Mo (XGBoost chargé). **Non vérifié** : lecteur d'écran réel, production.
- **Déployé le 2026-10-21** (14:58, commits `7f02b43` pipeline + API, `ebe2e6a` front, `4faf800` docs) avec `deploy_to_vps.ps1`, puis `sync_to_vps.ps1` dans la foulée. Vérifié en production : `/api/model/metrics` → `player_model` (314 arbres, MAE pts 4,68) ; `/api/games/0022600008?date=2026-10-23` → 11 et 10 joueurs, valeurs identiques au local ; MIA–DET → 0 et 11 ; `/api/games/today` sans `players` ; pages `/`, `/modele`, `/match` en 200 ; `team_state.json` du VPS avec `player_league` (6 équipes avec joueurs) ; `player_minutes.pkl` présent dans `saved_models` ; mémoire du service 193 Mo. Journal : un seul avertissement « team_state.json sans données joueur » au redémarrage (avant la synchro, attendu), plus rien après. Le message ne contient pas « warn » : chercher dans `sudo journalctl -u nba-predictor-api` sans filtre.
- Exemple (2026-10-23, BOS–NYK) : Tatum 35,8 min (32–41), 23,6 pts (13–32), 8,7 reb (4–12), 6,4 pd (3–9). La somme des points des joueurs attendus dépasse le score de l'équipe (118 contre 114 ; 133 contre 116 pour SAS, 14 joueurs) : attendu, expliqué dans le sous-titre de la section.

## Déploiement (VPS personnel, en ligne depuis le 2026-10-14)

**En ligne : https://nba.remicoimbra.fr.** Guide d'exploitation (redéployer, retour arrière, surveiller, installation initiale) : `docs/deploiement.md`. Le dépôt GitHub est **public** : ne rien y écrire sur les autres services/projets hébergés sur le VPS.

| Élément        | Valeur                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| Machine        | OVH VPS Starter, Ubuntu 26.04, 2 vCPU, 3,7 Gio de RAM + 2 Gio de swap, 40 Go — partagée avec d'autres projets |
| Accès SSH      | `ubuntu@remicoimbra.fr`, clé SSH du PC autorisée (`ssh -o BatchMode=yes` OK)                             |
| Python         | 3.14 du système (même version que le venv local) + `python3.14-venv` (installé le 2026-10-14)            |
| Reverse proxy  | Caddy (systemd), un seul `/etc/caddy/Caddyfile` partagé avec les autres sites                            |
| DNS / HTTPS    | `nba.remicoimbra.fr` → IP du VPS (déjà en place) ; certificat Let's Encrypt obtenu par Caddy             |

### Architecture déployée

```
PC de l'utilisateur
  infra/sync_to_vps.ps1   : données du jour -> ~/nba-predictor/data/processed/ (scp, chmod 644, mv atomique)
  infra/deploy_to_vps.ps1 : build du front + code API + modèle -> infra/vps/deploy_remote.sh sur le VPS
Internet ──HTTPS──> Caddy (bloc nba.remicoimbra.fr)
                      ├── /api/* (préfixe retiré) -> service systemd nba-predictor-api (uvicorn, 127.0.0.4:8000)
                      └── le reste -> /var/www/nba-predictor/current -> releases/<horodatage> (export statique Next)
```

Choix faits le 2026-10-14 (plan validé par l'utilisateur ; consigne : ne pas casser les autres projets, priorité à la sécurité) :

- **Front = export statique** (`output: "export"` dans `next.config.mjs`), construit sur le PC : aucun processus Node ni `npm install` sur le VPS, donc aucune faille côté serveur de Next exploitable, et 0 Mo de RAM. Conséquence : `/match/[id]` est devenu **`/match?id=…&date=…`** (l'export refuse une route dynamique sans liste d'ID connue au build). `NEXT_PUBLIC_API_URL=/api` au build : même origine, pas de CORS en production. `next start` ne fonctionne plus (inutile) ; `next dev` marche comme avant.
- **API = service systemd isolé, pas Docker** (les fichiers Docker vides ont été supprimés), sur le modèle des services existants du VPS : `DynamicUser`, `ProtectSystem=strict`, `ProtectHome=tmpfs` + `BindReadOnlyPaths=/home/ubuntu/nba-predictor`, aucune capacité, `MemoryDenyWriteExecute=yes` (fonctionne avec XGBoost), `MemoryMax=800M`, `CPUQuota=100%`. Réseau : `IPAddressDeny=any` + `IPAddressAllow=127.0.0.5` ; Caddy se connecte depuis 127.0.0.5 (`local_address`) ; l'API ne peut joindre personne (ni Internet, ni DNS, ni les autres services locaux). Score `systemd-analyze security` : 1,4 « OK ». Mémoire réelle : ~140 Mo avec le modèle chargé.
- Variables en production (dans l'unité) : `NBA_LIVE_SCHEDULE_FALLBACK=0`, `NBA_CORS_ORIGINS=` (vide : pas de middleware CORS), `NBA_API_DOCS=0` (pas de `/docs` ni `/openapi.json`), `TZ=Europe/Paris`. En local, les valeurs par défaut de `config.py` gardent le comportement de dev (CORS `localhost:3000`, docs actives).
- **Déploiement poussé depuis le PC**, pas de clone git sur le VPS : `deploy_to_vps.ps1` envoie les fichiers **suivis par git** (`git ls-files`) de `api/app`, `api/requirements.txt` et `features_lib.py`, dans leur état local. `~/nba-predictor/VERSION` donne le commit déployé (avec « modifications non commitées » le cas échéant). `pip install --only-binary=:all:` seulement si `requirements.txt` a changé. Les 3 dernières versions du front sont gardées ; l'ancien code de l'API reste dans `api/app.prev`.
- Caddy : HSTS (sans `includeSubDomains`), CSP (`'unsafe-inline'` requis par les scripts inline de l'export Next et le script de thème), `nosniff`, `X-Frame-Options`, COOP, `Permissions-Policy`, corps ≤ 16 Ko, `/_next/static/*` en cache immuable. **Modifier le Caddyfile** : toujours sur une copie, `sudo -u caddy caddy validate --config <copie>`, sauvegarde `Caddyfile.bak-AAAA-MM-JJ-HHMMSS`, puis `sudo systemctl reload caddy`. Contrôler les codes HTTP des autres sites avant et après.

Pièges rencontrés :

- **Git Bash convertit `/api`** en `C:/Program Files/Git/api` dans `NEXT_PUBLIC_API_URL=/api npm run build` (conversion de chemins MSYS). Builder depuis PowerShell, ce que fait `deploy_to_vps.ps1` (qui vérifie aussi que l'URL de dev n'est pas restée dans le bundle), ou préfixer `MSYS_NO_PATHCONV=1`.
- **Next 16, export statique** : le préchargement demande `modele/__next.modele.__PAGE__.txt` mais l'export écrit `modele/__next.modele/__PAGE__.txt` (404 puis requête de repli). Corrigé par une réécriture Caddy (`@segment` avec `not file` : sans effet si Next corrige le bug).
- `try_files` de Caddy n'accepte pas de matcher nommé (il le prend pour un nom de fichier) : vérifier l'ordre réel des routes avec `caddy adapt`.
- Dans Chrome, des `HEAD` en 503 et des requêtes vers `static-lib.com` / Google Analytics viennent d'une extension du navigateur, pas du site (`curl -I` → 200).

Vérifié le 2026-10-14 : déploiement et synchro réels (premier envoi `scp`/`ssh` réussi), pages et API publiques (200 ; `/api/docs` 404 ; POST 405 ; date non synchronisée → 404 immédiat ; http → https), en-têtes de sécurité, filtre IP (refus depuis 127.0.0.1, accès depuis 127.0.0.5), autres sites inchangés avant/après chaque rechargement de Caddy, navigateur (accueil sombre, fiche match en clair, page modèle, console sans erreur, logos). **Reste côté utilisateur** : la tâche planifiée de la synchro (voir plus bas).

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

**Reste à faire côté utilisateur : la tâche planifiée Windows** (non créée par Claude, à la demande de l'utilisateur). Elle tourne sous le compte de l'utilisateur, seulement quand il est connecté (accès à la clé SSH), et rattrape une exécution manquée :

```powershell
$script = "C:\Users\Rémi\Documents\projets_perso\nba-predictor\infra\sync_to_vps.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -VpsHost ubuntu@remicoimbra.fr -RemoteDir /home/ubuntu/nba-predictor/data/processed"
$trigger = New-ScheduledTaskTrigger -Daily -At 10:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
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

0. ~~**Déploiement sur le VPS**~~ : FAIT le 2026-10-14 (voir "Déploiement"). Reste côté utilisateur : la tâche planifiée de la synchro.
1. ~~`nba_sync_service.py`~~ : **remplacé** par `infra/sync_to_vps.ps1` (la synchro ne peut pas tourner dans l'API sur le VPS, NBA bloquée). `api/app/services/nba_sync_service.py` (vide) supprimé le 2026-10-19. `POST /admin/sync/games` n'a plus lieu d'être sous cette forme.
2. **Finir le refactor API** : `routers/predictions.py`, `routers/teams.py`, `routers/players.py` n'existent plus (fichiers vides supprimés le 2026-10-19) ; seuls `routers/games.py` et `routers/model.py` existent. À créer si les endpoints cibles deviennent utiles.
3. **Choix et setup de la BDD** : décider entre PostgreSQL local ou Supabase, créer les tables du schéma (à mettre à jour avec les vraies colonnes : Elo, Net Rating, win_pct_context...). `team_state.json` fait office de solution transitoire correcte pour l'instant.
4. Tester `/games/today` (et le sélecteur de date du front) avec de VRAIS matchs programmés dès le retour de la saison régulière NBA (mi-octobre 2026) — jusque-là, le paramètre `?date=` ne permet qu'un test partiel (Elo/forme actuels appliqués à une date passée, voir limite documentée dans "Adaptation front" plus haut).
5. **Amélioration des prédictions** : plan en 6 étapes, étapes 1 (walk-forward) et 2 (Elo) FAITES et déployées le 2026-10-19, étape 3 (modèle, LogReg servie) FAITE et déployée le 2026-10-19, étape 5 (score) et historique étendu FAITS et déployés le 2026-10-19, étape 4 (calendrier) essayée et NON ADOPTÉE le 2026-10-19 ; étape 6 (blessures) essayée, concluante, intégrée et déployée le 2026-10-19 ; plan terminé. À surveiller dès le 2ᵉ match de chaque équipe en 2026-27 (premier usage réel de l'effectif). Nouvelle référence pour de futurs essais : `oof_lineup.csv`. Piste repérée : matchs sur terrain neutre absents de l'historique (voir "Étape 6"), voir "Amélioration des prédictions". Hors plan, toujours possible : affiner le Net Rating (ratio de sommes plutôt que moyenne de ratios).
6. ~~Dockeriser l'API~~ : abandonné, l'API tourne en service systemd isolé (voir "Déploiement").
7. ~~Sortir la config en dur~~ : FAIT (`NEXT_PUBLIC_API_URL` au build, `NBA_CORS_ORIGINS` et `NBA_API_DOCS` dans `config.py`).
8. ~~(Optionnel, UX) skeleton de chargement~~ : fait avec la refonte UI.
10. ~~(Modèle) `rest_days` hors distribution~~ : FAIT le 2026-10-14 (plafond à 7 jours, voir "Plafond des jours de repos"). Reste à surveiller : premiers matchs de 2026-27 avec `win_pct` manquant (voir l'observation dans cette même section).
9. ~~Écrire `README.md` et `docs/architecture.md`~~ : FAIT le 2026-10-14. À relire/ajuster à la main (le README parle à la première personne du projet). Lien de démo ajouté le 2026-10-14.
11. **« IA complète » (demande du 2026-10-21)** : partie 1, meilleur score prédit, FAITE (voir "Score v2") ; partie 2, statistiques des joueurs, FAITE et déployée le 2026-10-21 (voir "Statistiques des joueurs"). « Points de l'équipe = score servi » a été essayé et écarté. Pistes non essayées : 3 pts (re-téléchargement des feuilles joueur), rapport officiel des blessures, titulaires (absents de `LeagueGameLog`).

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

→ `http://localhost:3000` (ex. `/?date=2026-10-23` pour un jour avec matchs, `/match?id=0022600008&date=2026-10-23`, `/modele`)

⚠️ **Port 8000 occupé** sur le PC de l'utilisateur par un serveur Symfony (autre projet, constaté le 2026-09-28). Dans ce cas, lancer l'API sur un autre port et le dire au front :

```powershell
uvicorn app.main:app --reload --port 8001          # dans api/, venv activé
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8001"; npm run dev   # dans front/
```

`NEXT_PUBLIC_API_URL` est lu au démarrage de `next dev` et figé AU BUILD pour `next build`. CORS : le front doit rester sur `localhost:3000`.

**Contrôles front :** `npm run lint`, `npm run build` (produit l'export statique `front/out/`), `npm run check:contrast` (à relancer après toute modification de couleur dans `globals.css`). `npm run fetch:logos` retélécharge les logos des équipes (rarement utile).

**Pipeline ML (à relancer si les CSV sources changent) :**

```powershell
cd data
python collectors\fetch_games.py          # réutilise le brut existant si présent
python collectors\fetch_player_logs.py    # feuilles de match par joueur (effectif), idem
python preprocessing\clean_data.py
python preprocessing\feature_engineering.py
python models\train_model.py             # walk-forward (~20 s) + score prédit + modèles servis + model_metrics.json
python preprocessing\team_state.py        # génère team_state.json — À REFAIRE après tout changement de games_clean.csv, l'API le lit directement
python collectors\fetch_schedule.py       # génère schedule.json (calendrier J-3 → J+14)
```

**Expérimenter / contrôler (sans toucher aux modèles servis) :**

```powershell
cd data
python models\evaluate_model.py --variant baseline              # walk-forward, prédictions dans saved_models\eval\
python models\evaluate_model.py --variant essai --compare baseline   # après modification des features : comparaison appariée
python models\evaluate_model.py --legacy                        # ancien découpage (test 2025-26)
python models\tune_elo.py                                       # grille des paramètres Elo (~30 s), ne modifie rien
python models\tune_score_ratings.py                             # grille des ratings de score (~70 s), ne modifie rien
python models\tune_player_projections.py                        # grille des projections des joueurs (~2 min), ne modifie rien
# --variant baseline (aussi la valeur par défaut) réécrit oof_baseline.csv (état d'avant l'étape 2) : nommer autrement un essai
# référence de l'état actuel (étape 6, effectif) : oof_lineup.csv -> --variant essai --compare lineup
python preprocessing\check_features.py                          # cohérence entraînement/API + absence de fuite (~4 min depuis les joueurs)
```

Sous Git Bash, préfixer par `PYTHONIOENCODING=utf-8` (sinon accents illisibles dans la console).

**Synchro quotidienne (nouveaux matchs + calendrier), sans réentraînement :**

```powershell
.\infra\sync_to_vps.ps1 -NoPush           # en local ; sans -NoPush : envoi vers le VPS (voir "Déploiement")
```

**Déployer une nouvelle version (code, front ou modèle réentraîné) :**

```powershell
.\infra\deploy_to_vps.ps1                # voir docs/deploiement.md
```
