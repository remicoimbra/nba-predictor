"""
tune_player_projections.py

Réglage des paramètres des projections des joueurs
(features_lib.PlayerProjectionParams) : demi-vie des minutes, et pour
chaque stat la demi-vie du taux par minute et le lissage vers la ligue
(minutes fictives).

Protocole (même principe que tune_elo.py et tune_score_ratings.py) :
- les lignes sont celles de l'évaluation (joueurs attendus ET entrés en
  jeu, features_lib.compute_player_rows, le même code que l'entraînement et
  l'API) ; objectif : erreur quadratique moyenne sur TUNING_SEASONS, les
  saisons antérieures servant de rodage ;
- minutes et taux sont réglés séparément : la demi-vie des minutes sur
  l'erreur des minutes moyennes, chaque taux sur l'erreur de « minutes
  RÉELLES × taux » (pour ne pas mélanger les deux erreurs), avant rythme et
  adversaire ;
- confirmation sur HOLDOUT_SEASONS, jamais utilisées pour choisir : écart
  apparié avec la moyenne de la saison du joueur, IC à 95 % par bootstrap
  par journée.

Résultat du 2026-10-21 : minutes demi-vie 2 matchs ; points h 10 / k 25,
rebonds h 20 / k 25, passes h 10 / k 25, interceptions h 40 / k 200,
contres h 40 / k 100 (profils plats autour). Essai écarté au même moment :
remettre les minutes de l'équipe à 241,6 au total (moyenne de la ligue,
prolongations comprises), moins bon quel que soit le total visé.

Usage :
    cd data
    python models/tune_player_projections.py
Grille des taux sauvegardée dans saved_models/eval/tune_player_rates.csv.
"""

import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from evaluate_model import EVAL_DIR, format_comparison, paired_bootstrap
from player_model import load_players, reference_columns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    COUNT_STATS,
    DEFAULT_PLAYER_PARAMS,
    PLAYER_STATS,
    compute_player_rows,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
GRID_PATH = EVAL_DIR / "tune_player_rates.csv"

MINUTES_HALFLIFE_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0)
RATE_HALFLIFE_GRID = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
PRIOR_MINUTES_GRID = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0)

TUNING_SEASONS = ("2021-22", "2022-23", "2023-24")
HOLDOUT_SEASONS = ("2024-25", "2025-26")


def played_rows(games: pd.DataFrame, players: pd.DataFrame, params) -> pd.DataFrame:
    """Joueurs attendus et entrés en jeu des saisons de réglage et de contrôle, avec leurs stats réelles."""
    rows = compute_player_rows(games, players, params)
    rows = rows[rows["season"].isin(TUNING_SEASONS + HOLDOUT_SEASONS)]
    actual = players[["player_id", "team_id", "nba_game_id", *PLAYER_STATS]].rename(columns={s: f"actual_{s}" for s in PLAYER_STATS})
    return rows.merge(actual, on=["player_id", "team_id", "nba_game_id"])


def with_rates(halflife: float, prior_minutes: float):
    """Paramètres par défaut, avec la même demi-vie et le même lissage pour tous les taux."""
    rates = {s: {"halflife": halflife, "prior_minutes": prior_minutes} for s in COUNT_STATS}
    return replace(DEFAULT_PLAYER_PARAMS, rates=rates)


def main():
    games = pd.read_csv(PROCESSED_DIR / "game_features.csv", parse_dates=["game_date"])
    players = load_players(games)
    print(f"Réglage sur {', '.join(TUNING_SEASONS)} ; contrôle sur {', '.join(HOLDOUT_SEASONS)}")
    start = time.perf_counter()

    print("\nMinutes : erreur quadratique des minutes moyennes pondérées (saisons de réglage)")
    for halflife in MINUTES_HALFLIFE_GRID:
        rows = played_rows(games, players, replace(DEFAULT_PLAYER_PARAMS, minutes_halflife=halflife))
        tuning = rows[rows["season"].isin(TUNING_SEASONS)]
        mse = float(((tuning["minutes"] - tuning["actual_min"]) ** 2).mean())
        print(f"  demi-vie {halflife:>4g} : {mse:.3f}")

    print(f"\nTaux : {len(RATE_HALFLIFE_GRID) * len(PRIOR_MINUTES_GRID)} combinaisons (minutes réelles × taux)...")
    grid = []
    for halflife, prior_minutes in product(RATE_HALFLIFE_GRID, PRIOR_MINUTES_GRID):
        rows = played_rows(games, players, with_rates(halflife, prior_minutes))
        tuning = rows[rows["season"].isin(TUNING_SEASONS)]
        grid.append({
            "halflife": halflife, "prior_minutes": prior_minutes,
            **{s: float(((tuning["actual_min"] * tuning[f"rate_{s}"] - tuning[f"actual_{s}"]) ** 2).mean()) for s in COUNT_STATS},
        })
    grid = pd.DataFrame(grid)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_PATH, index=False)
    print(f"  terminé en {time.perf_counter() - start:.0f} s")
    for s in COUNT_STATS:
        best = grid.loc[grid[s].idxmin()]
        profile = grid.groupby("halflife")[s].min()
        print(
            f"  {s} : meilleur h {best['halflife']:g} / k {best['prior_minutes']:g} ({best[s]:.4f}) ; "
            "par demi-vie : " + "  ".join(f"{h:g}: {v:.4f}" for h, v in profile.items())
        )

    rows = reference_columns(played_rows(games, players, DEFAULT_PLAYER_PARAMS), players)
    holdout = rows[rows["season"].isin(HOLDOUT_SEASONS)]
    print(f"\nSaisons de contrôle, A = paramètres actuels (minutes moyennes × taux, sans rythme ni adversaire), B = moyenne de la saison :")
    projected = {"min": holdout["minutes"], **{s: holdout["minutes"] * holdout[f"rate_{s}"] for s in COUNT_STATS}}
    for s in PLAYER_STATS:
        diff = (projected[s] - holdout[f"actual_{s}"]) ** 2 - (holdout[f"season_{s}"] - holdout[f"actual_{s}"]) ** 2
        print("  " + format_comparison(paired_bootstrap(diff.to_numpy(), holdout["game_date"], holdout["season"], f"mse_{s}")))
    print(f"\nParamètres actuels : {DEFAULT_PLAYER_PARAMS.as_dict()}")
    print(f"Grille des taux : {GRID_PATH}")


if __name__ == "__main__":
    main()
