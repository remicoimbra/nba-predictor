"""
tune_score_ratings.py

Réglage des ratings de score (features_lib.ScoreRatingParams : vitesses
d'ajustement du rythme et des points, régression inter-saison, suivi du
niveau de la ligue) sur l'erreur du TOTAL de points prédit.

Protocole (même principe que tune_elo.py) :
- grille de 500 combinaisons ; les ratings sont rejoués sur tout
  l'historique pour chacune (features_lib.compute_score_rating_timeline, le
  même code que l'entraînement et l'API) ;
- objectif : MAE du total d'avant-match sur TUNING_SEASONS. Les saisons
  antérieures (2016-17 -> 2020-21) servent de rodage et ne sont pas notées ;
- confirmation sur HOLDOUT_SEASONS, jamais utilisées pour choisir : écart
  apparié de MAE avec le total de la méthode précédente (moyennes des 10
  derniers matchs, score_model.previous_points), IC à 95 % par bootstrap par
  journée.

Résultat du 2026-10-21 : k_pace 0,04, k_scoring 0,03, régression 0,5,
league_rate 0,01 (profil plat autour) ; saisons de réglage, MAE du total
14,79 -> 14,56 ; saisons de contrôle 15,10 -> 14,77, Δ -0,34
[-0,51 ; -0,17], 2/2 saisons.

Usage :
    cd data
    python models/tune_score_ratings.py
Grille complète sauvegardée dans saved_models/eval/tune_score_grid.csv.
"""

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model import EVAL_DIR, format_comparison, paired_bootstrap
from score_model import PREVIOUS_INPUTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_SCORE_RATING_PARAMS,
    ScoreRatingParams,
    compute_score_rating_timeline,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
GRID_PATH = EVAL_DIR / "tune_score_grid.csv"

K_PACE_GRID = (0.02, 0.03, 0.04, 0.06, 0.08)
K_SCORING_GRID = (0.015, 0.02, 0.03, 0.04, 0.05)
REGRESSION_GRID = (0.25, 0.33, 0.5, 0.67, 0.75)
LEAGUE_RATE_GRID = (0.003, 0.005, 0.01, 0.02)

TUNING_SEASONS = ("2021-22", "2022-23", "2023-24")
HOLDOUT_SEASONS = ("2024-25", "2025-26")
PARAM_NAMES = ("k_pace", "k_scoring", "season_regression", "league_rate")


def load_games() -> pd.DataFrame:
    """Matchs joués + total de la méthode précédente (moyennes des derniers matchs, game_features.csv)."""
    games = pd.read_csv(PROCESSED_DIR / "games_clean.csv", parse_dates=["game_date"])
    features = pd.read_csv(PROCESSED_DIR / "game_features.csv", usecols=["nba_game_id", *PREVIOUS_INPUTS])
    games = games.dropna(subset=["home_score", "away_score"]).merge(features, on="nba_game_id", how="left")
    home_scored, home_allowed, away_scored, away_allowed = (games[c] for c in PREVIOUS_INPUTS)
    games["previous_total"] = (home_scored + away_allowed) / 2 + (away_scored + home_allowed) / 2
    games["total"] = games["home_score"] + games["away_score"]
    return games


def rating_errors(games: pd.DataFrame, params: ScoreRatingParams) -> pd.Series:
    """Erreur absolue du total d'avant-match des ratings, alignée sur `games`."""
    per_game = compute_score_rating_timeline(games, params).per_game
    expected = games[["nba_game_id"]].merge(per_game, on="nba_game_id", how="left")["exp_total"]
    return (expected.to_numpy() - games["total"]).abs()


def run_grid(games: pd.DataFrame) -> pd.DataFrame:
    tuning = games["season"].isin(TUNING_SEASONS)
    holdout = games["season"].isin(HOLDOUT_SEASONS)
    rows = []
    for values in product(K_PACE_GRID, K_SCORING_GRID, REGRESSION_GRID, LEAGUE_RATE_GRID):
        params = ScoreRatingParams(*values)
        errors = rating_errors(games, params)
        rows.append({
            **params.as_dict(),
            "tuning_mae_total": float(errors[tuning].mean()),
            "holdout_mae_total": float(errors[holdout].mean()),
        })
    return pd.DataFrame(rows).sort_values("tuning_mae_total").reset_index(drop=True)


def describe(params: ScoreRatingParams) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in params.as_dict().items())


def main():
    games = load_games()
    n_combos = len(K_PACE_GRID) * len(K_SCORING_GRID) * len(REGRESSION_GRID) * len(LEAGUE_RATE_GRID)
    print(f"{len(games)} matchs ; réglage sur {', '.join(TUNING_SEASONS)} ; contrôle sur {', '.join(HOLDOUT_SEASONS)}")
    print(f"Grille : {n_combos} combinaisons...")

    start = time.perf_counter()
    grid = run_grid(games)
    print(f"  terminé en {time.perf_counter() - start:.0f} s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_PATH, index=False)

    print("\n10 meilleures combinaisons (MAE du total sur les saisons de réglage) :")
    print(grid.head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nMeilleure MAE de réglage par valeur de chaque paramètre (les autres optimisés) :")
    for col in PARAM_NAMES:
        profile = grid.groupby(col)["tuning_mae_total"].min()
        print(f"  {col:<18} " + "  ".join(f"{v:g}: {mae:.4f}" for v, mae in profile.items()))

    best = ScoreRatingParams(**{k: float(grid.iloc[0][k]) for k in PARAM_NAMES})
    errors = rating_errors(games, best)
    previous = (games["previous_total"] - games["total"]).abs()
    for label, seasons in (("Saisons de réglage", TUNING_SEASONS), ("Saisons de contrôle (jamais vues)", HOLDOUT_SEASONS)):
        mask = games["season"].isin(seasons) & previous.notna()
        sub = games[mask]
        comparison = paired_bootstrap((errors - previous)[mask].to_numpy(), sub["game_date"], sub["season"], "mae_total")
        print(f"\n{label}, A = ratings ({describe(best)}), B = moyennes des derniers matchs :")
        print(f"  MAE du total A {errors[mask].mean():.3f} / B {previous[mask].mean():.3f}")
        print("  " + format_comparison(comparison))

    if best != DEFAULT_SCORE_RATING_PARAMS:
        print(f"\nParamètres actuels de features_lib.DEFAULT_SCORE_RATING_PARAMS : {describe(DEFAULT_SCORE_RATING_PARAMS)}")
    print(f"Grille complète : {GRID_PATH}")


if __name__ == "__main__":
    main()
