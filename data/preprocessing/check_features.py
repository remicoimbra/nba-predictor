"""
check_features.py

Contrôles automatiques du pipeline de features, à relancer après toute
modification de features_lib.py, feature_engineering.py, team_state.py ou
predictor_service.py :

Les contrôles portent sur TOUTES les features calculées
(train_model.candidate_feature_cols : celles que l'API affiche sur la fiche
match), pas seulement sur celles du modèle (feature_columns.json).

0. game_features.csv est à jour : identique aux features recalculées à
   partir de games_clean.csv avec le code actuel (sinon train_model.py
   s'entraîne sur d'anciennes features).
1. Cohérence entraînement / prédiction : pour un échantillon de matchs de
   la saison la plus récente, on reconstruit team_state.json à partir des
   seuls matchs antérieurs, puis on calcule les features comme l'API
   (PredictorService.build_feature_row). Elles doivent être identiques à
   celles du pipeline d'entraînement. Les 10 premiers matchs de la saison
   sont toujours inclus : ils passent par le changement de saison
   (régression Elo, bilans remis à zéro). Pour ces mêmes matchs,
   l'explication servie doit être exacte : biais + somme des facteurs =
   logit de la probabilité (à l'arrondi de l'API près), et le score servi
   doit être celui des formules de features_lib appliquées aux features
   d'entraînement, avec le vainqueur prédit en tête.
2. Absence de fuite temporelle : on inverse le résultat d'un match (scores
   et boxscores échangés) et on recalcule toutes les features. Celles de ce
   match et des matchs antérieurs ne doivent pas bouger ; celles des matchs
   suivants des deux équipes, si.

Usage :
    cd data
    python preprocessing/check_features.py
Code de sortie 1 si un contrôle échoue.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import build_features
from features_lib import SCORE_WINDOW, expected_margin, expected_total, rounded_score, split_score
from team_state import build_team_state

DATA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_DIR.parent / "api"))
sys.path.insert(0, str(DATA_DIR / "models"))
from app.services.predictor_service import FEATURE_COLUMNS_PATH, PredictorService  # noqa: E402
from train_model import candidate_feature_cols  # noqa: E402

CLEAN_PATH = DATA_DIR / "processed" / "games_clean.csv"
FEATURES_PATH = DATA_DIR / "processed" / "game_features.csv"

TOLERANCE = 1e-9
# Facteurs, biais et probabilité sont arrondis à 4 décimales par l'API.
EXPLANATION_TOLERANCE = 1e-3
SEASON_START_GAMES = 10  # premiers matchs de la saison, toujours contrôlés
RANDOM_GAMES = 40
LEAK_GAMES = 5
MIN_SEASON_GAMES = 200  # saison contrôlée : la plus récente avec au moins autant de matchs


def same_value(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return abs(float(a) - float(b)) <= TOLERANCE


def differing_columns(row_a, row_b, columns) -> list[str]:
    return [c for c in columns if not same_value(row_a[c], row_b[c])]


def check_features_file_up_to_date(batch: pd.DataFrame, columns: list[str], model_columns: list[str]) -> bool:
    saved = pd.read_csv(FEATURES_PATH, parse_dates=["game_date"]).set_index("nba_game_id")
    fresh = batch.set_index("nba_game_id")
    if set(saved.index) != set(fresh.index):
        print(f"  ÉCHEC : {len(saved)} matchs dans game_features.csv, {len(fresh)} recalculés")
        return False
    missing = [c for c in columns if c not in saved.columns]
    if missing:
        print(f"  ÉCHEC : colonnes absentes de game_features.csv : {missing}")
        return False
    unknown = [c for c in model_columns if c not in columns]
    if unknown:
        print(f"  ÉCHEC : colonnes de feature_columns.json inconnues du pipeline : {unknown} (relancer train_model.py)")
        return False
    a = saved.loc[fresh.index, columns].to_numpy(dtype=float)
    b = fresh[columns].to_numpy(dtype=float)
    mismatch = ~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b)))
    if mismatch.any():
        cols = [columns[j] for j in np.where(mismatch.any(axis=0))[0]]
        print(f"  ÉCHEC : {int(mismatch.any(axis=1).sum())} matchs diffèrent ({cols[:5]}...) : relancer feature_engineering.py")
        return False
    print(f"  OK : {len(fresh)} matchs, {len(columns)} features identiques (dont les {len(model_columns)} du modèle)")
    return True


def batch_score(service: PredictorService, batch_row: pd.Series) -> tuple[int, int]:
    """Score attendu d'après les features d'entraînement (modèle et sigma du service)."""
    model_cols = service._feature_columns
    p = float(service._model.predict_proba(batch_row[model_cols].to_frame().T.astype(float))[0, 1])
    total = expected_total(*(
        batch_row[f"{side}_avg_pts_{kind}_last{SCORE_WINDOW}"]
        for side, kind in (("home", "scored"), ("home", "allowed"), ("away", "scored"), ("away", "allowed"))
    ))
    if pd.isna(total):
        total = service._score_model["fallback_total"]
    margin = float(expected_margin(p, service._score_model["sigma"]))
    return rounded_score(*split_score(total, margin), p >= 0.5)


def check_train_live_consistency(clean: pd.DataFrame, batch: pd.DataFrame, columns: list[str]) -> bool:
    counts = clean["season"].value_counts()
    season = max(s for s, n in counts.items() if n >= MIN_SEASON_GAMES)
    games = clean[clean["season"] == season].sort_values(["game_date", "nba_game_id"])
    rest = games.iloc[SEASON_START_GAMES:]
    sample = pd.concat([
        games.iloc[:SEASON_START_GAMES],
        rest.sample(n=min(RANDOM_GAMES, len(rest)), random_state=0),
    ])
    batch_rows = batch.set_index("nba_game_id")

    failures = []
    score_failures = []
    worst_gap = 0.0  # écart max |biais + somme des facteurs - logit(p)|
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "team_state.json"
        service = None
        for game_date, day_games in sample.groupby("game_date"):
            state = build_team_state(clean[clean["game_date"] < game_date])
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
            if service is None:
                service = PredictorService(team_state_path=state_path)
            else:
                service.reload_team_state()

            for game in day_games.itertuples():
                teams = (int(game.home_team_id), int(game.away_team_id))
                as_of = game_date.strftime("%Y-%m-%d")
                live = service.build_feature_row(*teams, as_of_date=as_of)
                absent = [c for c in columns if c not in live]
                diff = absent or differing_columns(live, batch_rows.loc[game.nba_game_id], columns)
                if diff:
                    failures.append((game.nba_game_id, game_date.date(), diff))

                pred = service.predict(*teams, as_of_date=as_of)
                p = pred.home_win_probability
                explained = pred.base_value + sum(f["contribution"] for f in pred.factors)
                worst_gap = max(worst_gap, abs(explained - math.log(p / (1 - p))))

                served = (pred.predicted_home_score, pred.predicted_away_score)
                home_ahead = served[0] > served[1]
                if served != batch_score(service, batch_rows.loc[game.nba_game_id]) or home_ahead != (pred.predicted_winner == "home"):
                    score_failures.append((game.nba_game_id, served, pred.predicted_winner))

    ok = True
    if failures:
        print(f"  ÉCHEC : {len(failures)} matchs sur {len(sample)} diffèrent entre entraînement et API")
        for game_id, day, cols in failures[:5]:
            print(f"    {game_id} ({day}) : {cols}")
        ok = False
    else:
        print(f"  OK : {len(sample)} matchs de {season} ({sample['game_date'].nunique()} journées), features identiques à {TOLERANCE}")
    if worst_gap > EXPLANATION_TOLERANCE:
        print(f"  ÉCHEC : biais + facteurs s'écartent du logit de la probabilité servie (jusqu'à {worst_gap:.4f})")
        ok = False
    else:
        print(f"  OK : biais + facteurs = logit(p) pour ces matchs (écart max {worst_gap:.1e}, arrondis de l'API)")
    if score_failures:
        print(f"  ÉCHEC : score servi différent du calcul d'entraînement ou du vainqueur prédit : {score_failures[:5]}")
        ok = False
    else:
        print("  OK : score servi = formules de features_lib sur les features d'entraînement, vainqueur prédit en tête")
    return ok


def flipped_result(clean: pd.DataFrame, game_id) -> pd.DataFrame:
    """Copie de l'historique où le match `game_id` a le résultat inverse (stats domicile/extérieur échangées)."""
    modified = clean.copy()
    row = modified["nba_game_id"] == game_id
    for home_col in clean.columns:
        away_col = "away_" + home_col.removeprefix("home_")
        if not home_col.startswith("home_") or home_col in ("home_team_id", "home_team_abbr", "home_win"):
            continue
        if away_col in clean.columns:
            modified.loc[row, [home_col, away_col]] = clean.loc[row, [away_col, home_col]].to_numpy()
    modified.loc[row, "home_win"] = 1 - clean.loc[row, "home_win"]
    return modified


def check_no_leakage(clean: pd.DataFrame, batch: pd.DataFrame, columns: list[str]) -> bool:
    seasons = sorted(clean["season"].unique())
    season = seasons[len(seasons) // 2]
    games = clean[clean["season"] == season]
    candidates = games[games["game_date"] < games["game_date"].max() - pd.Timedelta(days=30)]
    picked = candidates.sample(n=LEAK_GAMES, random_state=1)
    base = batch.set_index("nba_game_id")

    ok = True
    for game in picked.itertuples():
        feats = build_features(flipped_result(clean, game.nba_game_id)).set_index("nba_game_id")
        dates = feats["game_date"]
        teams = (game.home_team_id, game.away_team_id)

        before = feats.index[dates <= game.game_date]  # le match lui-même, les matchs antérieurs et ceux du même soir
        a = base.loc[before, columns].to_numpy(dtype=float)
        b = feats.loc[before, columns].to_numpy(dtype=float)
        leaked = ~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b)))

        later = feats.index[(dates > game.game_date) & (feats["home_team_id"].isin(teams) | feats["away_team_id"].isin(teams))]
        a = base.loc[later, columns].to_numpy(dtype=float)
        b = feats.loc[later, columns].to_numpy(dtype=float)
        propagated = (~(np.isclose(a, b, rtol=0, atol=TOLERANCE) | (np.isnan(a) & np.isnan(b)))).any()

        if leaked.any():
            cols = [columns[j] for j in np.where(leaked.any(axis=0))[0]]
            print(f"  ÉCHEC : inverser {game.nba_game_id} change les features de matchs antérieurs ou de lui-même : {cols}")
            ok = False
        elif not propagated:
            print(f"  ÉCHEC : inverser {game.nba_game_id} ne change aucun match suivant (contrôle inopérant ?)")
            ok = False
    if ok:
        print(f"  OK : {LEAK_GAMES} matchs de {season} inversés, aucune feature antérieure modifiée, matchs suivants modifiés")
    return ok


def main():
    clean = pd.read_csv(CLEAN_PATH, parse_dates=["game_date"])
    with open(FEATURE_COLUMNS_PATH, encoding="utf-8") as f:
        model_columns = json.load(f)
    batch = build_features(clean)
    columns = candidate_feature_cols(batch)

    print("0. game_features.csv à jour")
    results = [check_features_file_up_to_date(batch, columns, model_columns)]
    print("1. Cohérence entraînement / prédiction")
    results.append(check_train_live_consistency(clean, batch, columns))
    print("2. Absence de fuite temporelle")
    results.append(check_no_leakage(clean, batch, columns))

    if not all(results):
        sys.exit(1)
    print("Tous les contrôles passent.")


if __name__ == "__main__":
    main()
