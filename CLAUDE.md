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

| Domaine         | Techno                                                           |
| --------------- | ---------------------------------------------------------------- |
| Data & ML       | Python, `nba_api`, `pandas`, `scikit-learn` / `XGBoost`          |
| Back-End / API  | Python, FastAPI, uvicorn                                         |
| Base de données | PostgreSQL ou Supabase (pas encore branchée)                     |
| Front-End       | Next.js (App Router) + Tailwind CSS, JavaScript (pas TypeScript) |

## Arborescence du projet

```
nba-predictor/
│
├── data/                          # Scripts Python : collecte & ML (pas encore développé)
│   ├── collectors/
│   │   ├── fetch_games.py
│   │   ├── fetch_players_stats.py
│   │   └── fetch_teams_stats.py
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── saved_models/
│   ├── notebooks/
│   └── requirements.txt
│
├── api/                            # Back-end FastAPI — EN COURS
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
│   │       ├── predictor_service.py
│   │       └── nba_sync_service.py
│   ├── tests/
│   └── requirements.txt            # FAIT — fastapi, uvicorn installés
│
├── front/                          # Next.js + Tailwind — EN COURS
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
│   └── Dockerfile.data             # pas encore créé
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

**État actuel du flux : seule l'étape 5 → 4 est fonctionnelle, avec des données fake côté API (pas encore de vraie collecte ni de modèle ML).**

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

## Ce qui a été fait jusqu'à présent (chronologie)

1. Création de l'arborescence complète du projet (dossiers + fichiers vides) via script PowerShell.
2. Setup de l'API FastAPI :
   - Environnement virtuel Python créé dans `api/venv/`
   - `fastapi` + `uvicorn` installés, `requirements.txt` généré
   - `api/app/main.py` créé avec :
     - Middleware CORS autorisant `http://localhost:3000`
     - `GET /` → healthcheck
     - `GET /games/today` → retourne 2 matchs fake avec prédictions simulées
   - Testé et fonctionnel : `http://127.0.0.1:8000/docs` accessible (Swagger UI)
3. Setup du Front Next.js :
   - `create-next-app` lancé (avec les choix listés ci-dessus)
   - Contenu généré déplacé dans `front/` via `robocopy` (erreur initiale : sous-dossier `front/nba-predictor/` créé par erreur, corrigé)
   - `front/src/services/api.js` créé avec `getTodayGames()`
   - `front/src/components/GameCard.jsx` créé (affichage d'un match : équipes, score prédit, probabilité)
   - `front/src/app/page.js` modifié : Server Component async qui fetch `/games/today` et affiche les `GameCard`
   - Testé et fonctionnel : `http://localhost:3000` affiche les 2 matchs fake

**État global : le tuyau Front ↔ API est validé de bout en bout avec des données mockées. Aucune donnée NBA réelle n'est encore branchée. Aucun modèle ML n'est encore entraîné. Aucune base de données n'est encore connectée.**

## Prochaines étapes (à faire)

1. **Collecte de données réelle** (`data/collectors/fetch_games.py`) : utiliser `nba_api` pour récupérer le calendrier et les résultats historiques. Attention aux rate limits de `nba_api` (pas officiel).
2. **Choix et setup de la BDD** : décider entre PostgreSQL local ou Supabase, créer les tables du schéma ci-dessus.
3. **Feature engineering** : calculer les features prédictives (forme récente, repos, home/away, etc.) à partir des données brutes.
4. **Entraînement du modèle** : XGBoost (classification victoire/défaite + régression score), en faisant attention à ne pas mélanger les saisons entre train/test (fuite temporelle).
5. **Intégration modèle → API** : remplacer les données fake de `/games/today` par de vraies prédictions issues du modèle entraîné, via `predictor_service.py`.
6. **Refactor API** : sortir la logique de `main.py` vers les `routers/` dédiés (`games.py`, `predictions.py`, etc.) au fur et à mesure que le nombre d'endpoints augmente.
7. (Optionnel) Dockeriser l'API et le service data (`infra/docker-compose.yml`).
8. (Optionnel) Scheduler quotidien pour automatiser la collecte + régénération des prédictions.

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
