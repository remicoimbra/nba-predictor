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
| --------------- | -------------------------------------------------------------------------- |
| Data & ML       | Python, `nba_api`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib` |
| Back-End / API  | Python, FastAPI, uvicorn                                                  |
| Base de données | PostgreSQL ou Supabase (pas encore branchée)                              |
| Front-End       | Next.js (App Router) + Tailwind CSS, JavaScript (pas TypeScript)          |

## Arborescence du projet

```
nba-predictor/
│
├── data/                          # Scripts Python : collecte & ML
│   ├── collectors/
│   │   ├── fetch_games.py          # FAIT — voir "État du pipeline ML" ci-dessous
│   │   ├── fetch_players_stats.py  # pas encore développé
│   │   └── fetch_teams_stats.py    # pas encore développé
│   ├── preprocessing/
│   │   ├── clean_data.py           # FAIT
│   │   └── feature_engineering.py  # FAIT — v3 (Pace/Net Rating, win_pct contexte, Elo)
│   ├── models/
│   │   ├── train_model.py          # FAIT — LogReg + XGBoost comparés
│   │   ├── evaluate_model.py       # pas encore développé (séparé de train_model.py)
│   │   └── saved_models/           # logreg_baseline.pkl, xgboost_v1.pkl (non versionnés)
│   ├── raw/                        # games_raw_all_seasons.csv, games_history.csv (non versionnés)
│   ├── processed/                  # games_clean.csv, game_features.csv (non versionnés)
│   ├── notebooks/
│   └── requirements.txt            # à mettre à jour : ajouter xgboost, joblib
│
├── api/                            # Back-end FastAPI — EN COURS (inchangé cette session)
│   ├── venv/                       # environnement virtuel Python (non versionné)
│   ├── app/
│   │   ├── main.py                 # FAIT — endpoint /games/today avec données fake
│   │   ├── core/
│   │   │   ├── config.py           # à créer
│   │   │   └── database.py         # à créer (connexion Supabase/Postgres)
│   │   ├── models/                 # à créer (ORM SQLAlchemy)
│   │   ├── schemas/                # à créer (schémas Pydantic)
│   │   ├── routers/                # à créer (actuellement tout dans main.py)
│   │   │   ├── games.py
│   │   │   ├── predictions.py
│   │   │   ├── teams.py
│   │   │   └── players.py
│   │   └── services/
│   │       ├── predictor_service.py    # à créer — PROCHAINE ÉTAPE PRIORITAIRE
│   │       └── nba_sync_service.py
│   ├── tests/
│   └── requirements.txt            # FAIT — fastapi, uvicorn installés
│
├── front/                          # Next.js + Tailwind — EN COURS (inchangé cette session)
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.js
│   │   │   ├── page.js             # FAIT — affiche les matchs du jour (Server Component)
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   └── GameCard.jsx        # FAIT — carte d'affichage d'un match
│   │   ├── services/
│   │   │   └── api.js              # FAIT — appel à l'API FastAPI (getTodayGames)
│   │   └── styles/
│   ├── package.json
│   └── tailwind.config.js
│
├── infra/
│   ├── docker-compose.yml          # pas encore créé
│   ├── Dockerfile.api              # pas encore créé
│   └── Dockerfile.data              # pas encore créé
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

**État actuel du flux : les étapes 1 → 2 → 3 sont fonctionnelles et validées sur données réelles (7 saisons, 8 279 matchs). L'étape 3 → 4 (brancher le modèle entraîné dans l'API) N'EST PAS ENCORE FAITE : l'API sert toujours des données fake. Aucune base de données n'est encore connectée (les CSV font office de BDD pour l'instant).**

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
GET  /games/{game_id}
GET  /games/today                # FAIT (données fake)

# Équipes
GET  /teams
GET  /teams/{team_id}/stats

# Prédictions
GET  /predictions
GET  /predictions/{game_id}
POST /predictions/generate

# Modèle
GET  /model/metrics
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

| Étape                                          | Accuracy LogReg | Accuracy XGBoost | Log loss (meilleur) |
| ----------------------------------------------- | --------------- | ----------------- | -------------------- |
| Baseline ("home gagne toujours")                | 55,1%           | —                  | —                     |
| v1 (win_pct global, avg_pts_last5, rest_days)   | 66,75%          | 67,4%             | 0,605                 |
| v3 sans Elo (+ Pace/Net Rating + win_pct_context)| 66,4%           | 67,6%             | 0,606                 |
| **v3 + Elo + scaling LogReg (état final)**      | **68,6%**       | **69,6%**         | **0,596**             |

**Feature la plus importante dans les deux modèles : `diff_elo`** (loin devant `diff_win_pct`, qui était dominante avant l'ajout de l'Elo). `diff_net_rtg_last10` reste un signal solide en 2ᵉ position. `win_pct_context` apporte un complément modeste mais réel.

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

## Prochaines étapes (à faire)

1. **PRIORITAIRE — Intégration modèle → API** : créer `predictor_service.py`, charger `xgboost_v1.pkl` (meilleur modèle actuellement), remplacer les données fake de `/games/today` par de vraies prédictions. Nécessite de recalculer les features en direct pour les matchs à venir (pas seulement à partir de l'historique déjà en CSV) — probablement le point le plus délicat de cette étape.
2. **Choix et setup de la BDD** : décider entre PostgreSQL local ou Supabase, créer les tables du schéma (à mettre à jour avec les vraies colonnes : Elo, Net Rating, win_pct_context...).
3. **Refactor API** : sortir la logique de `main.py` vers les `routers/` dédiés (`games.py`, `predictions.py`, etc.).
4. (Optionnel) Poursuivre l'optimisation ML si le temps le permet : tuning XGBoost (GridSearch/early stopping), validation croisée temporelle multi-saisons pour vérifier la robustesse du gain Elo (actuellement mesuré sur une seule saison de test), affiner le calcul du Net Rating (ratio de sommes plutôt que moyenne de ratios, pour réduire le bruit).
5. (Optionnel) Dockeriser l'API et le service data (`infra/docker-compose.yml`).
6. (Optionnel) Scheduler quotidien pour automatiser la collecte + régénération des prédictions.
7. Mettre à jour `data/requirements.txt` avec `xgboost` et `joblib` (installés en session mais pas encore figés dans le fichier).

## Comment relancer le projet en local

**Terminal 1 — API :**

```powershell
cd api
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --app-dir .
```

→ `http://127.0.0.1:8000/docs`

**Terminal 2 — Front :**

```powershell
cd front
npm run dev
```

→ `http://localhost:3000`

**Pipeline ML (à relancer si les CSV sources changent) :**

```powershell
cd data
python collectors\fetch_games.py          # réutilise le brut existant si présent
python preprocessing\clean_data.py
python preprocessing\feature_engineering.py
python models\train_model.py
```