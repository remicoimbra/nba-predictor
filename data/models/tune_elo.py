"""
tune_elo.py

Réglage des paramètres de l'Elo (features_lib.EloParams : K, avantage du
terrain, régression inter-saison) sur la qualité des probabilités de l'Elo
SEUL, sans modèle par-dessus.

Protocole :
- grille K × avantage du terrain × régression (396 combinaisons) ; l'Elo
  est rejoué sur tout l'historique pour chacune (features_lib.compute_elo_timeline,
  le même code que l'entraînement et l'API) ;
- objectif : log loss de la probabilité Elo d'avant-match sur
  TUNING_SEASONS. Les saisons antérieures servent de rodage (toutes les
  équipes partent de 1500) et ne sont pas notées : 2019-20 lors du réglage
  du 2026-10-19, 2016-17 -> 2019-20 depuis l'extension de l'historique
  (grille rejouée le même jour : mêmes paramètres optimaux, log loss de
  réglage 0,6328 -> 0,6329 ; avec une régression de 50 %, le point de
  départ est oublié en deux saisons) ;
- confirmation sur HOLDOUT_SEASONS, jamais utilisées pour choisir : écart
  apparié de log loss / Brier / accuracy avec les paramètres d'origine
  (IC à 95 % par bootstrap par journée, evaluate_model.paired_compare), et
  écart entre la probabilité moyenne de victoire à domicile et le taux réel.

Critères d'adoption (plan d'amélioration, CLAUDE.md) : IC du Δ log loss sur
HOLDOUT_SEASONS entièrement < 0 et écart de calibration < MAX_HOME_GAP.

Usage :
    cd data
    python models/tune_elo.py
Grille complète sauvegardée dans saved_models/eval/tune_elo_grid.csv.
"""

import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model import EVAL_DIR, TARGET_COL, format_comparison, paired_compare, per_game_scores

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_ELO_PARAMS,
    EloParams,
    compute_elo_timeline,
    elo_expected_home,
)

CLEAN_PATH = Path(__file__).resolve().parent.parent / "processed" / "games_clean.csv"
GRID_PATH = EVAL_DIR / "tune_elo_grid.csv"

K_GRID = (10, 15, 20, 25, 30, 35)
HOME_ADVANTAGE_GRID = tuple(range(0, 101, 10))
REGRESSION_GRID = (0.0, 0.1, 0.2, 0.25, 0.33, 0.5)

TUNING_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24")
HOLDOUT_SEASONS = ("2024-25", "2025-26")

# Paramètres d'origine du projet (avant ce réglage), référence de la comparaison.
REFERENCE_PARAMS = EloParams(k=20.0, home_advantage=100.0, season_regression=0.25)
MAX_HOME_GAP = 0.015  # écart toléré entre proba moyenne à domicile et taux réel


def elo_predictions(games: pd.DataFrame, params: EloParams) -> pd.DataFrame:
    """Probabilité de victoire à domicile donnée par l'Elo seul (ratings d'avant-match), match par match."""
    per_game = compute_elo_timeline(games, params).per_game
    out = games[["nba_game_id", "game_date", "season", TARGET_COL]].merge(per_game, on="nba_game_id")
    out["p_elo"] = elo_expected_home(out["home_elo"].to_numpy(), out["away_elo"].to_numpy(), params)
    return out


def scores(pred: pd.DataFrame, seasons) -> dict:
    sub = pred[pred["season"].isin(seasons)]
    y = sub[TARGET_COL].to_numpy(dtype=float)
    p = sub["p_elo"].to_numpy(dtype=float)
    result = {m: float(v.mean()) for m, v in per_game_scores(y, p).items()}
    result["home_gap"] = float(p.mean() - y.mean())  # > 0 : domicile surestimé
    return result


def run_grid(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, home_advantage, regression in product(K_GRID, HOME_ADVANTAGE_GRID, REGRESSION_GRID):
        params = EloParams(k=float(k), home_advantage=float(home_advantage), season_regression=regression)
        pred = elo_predictions(games, params)
        tuning, holdout = scores(pred, TUNING_SEASONS), scores(pred, HOLDOUT_SEASONS)
        rows.append({
            "k": params.k, "home_advantage": params.home_advantage, "season_regression": regression,
            **{f"tuning_{m}": v for m, v in tuning.items()},
            **{f"holdout_{m}": v for m, v in holdout.items()},
        })
    return pd.DataFrame(rows).sort_values("tuning_log_loss").reset_index(drop=True)


def params_of(row) -> EloParams:
    return EloParams(k=float(row["k"]), home_advantage=float(row["home_advantage"]),
                     season_regression=float(row["season_regression"]))


def describe(params: EloParams) -> str:
    return f"K {params.k:g}, avantage {params.home_advantage:g}, régression {params.season_regression:g}"


def calibration_by_season(pred: pd.DataFrame, seasons) -> str:
    parts = []
    for s in seasons:
        sub = pred[pred["season"] == s]
        parts.append(f"{s} {sub['p_elo'].mean():.3f}/{sub[TARGET_COL].mean():.3f}")
    return ", ".join(parts)


def main():
    games = pd.read_csv(CLEAN_PATH, parse_dates=["game_date"])
    games = games.dropna(subset=[TARGET_COL])
    n_combos = len(K_GRID) * len(HOME_ADVANTAGE_GRID) * len(REGRESSION_GRID)
    print(f"{len(games)} matchs ; réglage sur {', '.join(TUNING_SEASONS)} ; contrôle sur {', '.join(HOLDOUT_SEASONS)}")
    print(f"Grille : {n_combos} combinaisons...")

    start = time.perf_counter()
    grid = run_grid(games)
    print(f"  terminé en {time.perf_counter() - start:.0f} s")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_PATH, index=False)

    cols = ["k", "home_advantage", "season_regression", "tuning_log_loss", "tuning_brier",
            "tuning_home_gap", "holdout_log_loss", "holdout_home_gap"]
    is_ref = (
        (grid["k"] == REFERENCE_PARAMS.k)
        & (grid["home_advantage"] == REFERENCE_PARAMS.home_advantage)
        & (grid["season_regression"] == REFERENCE_PARAMS.season_regression)
    )
    print("\n10 meilleures combinaisons (log loss sur les saisons de réglage) :")
    print(grid[cols].head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nRéférence ({describe(REFERENCE_PARAMS)}), rang {int(grid.index[is_ref][0]) + 1}/{len(grid)} :")
    print(grid.loc[is_ref, cols].to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nMeilleur log loss de réglage par valeur de chaque paramètre (les deux autres optimisés) :")
    for col in ("k", "home_advantage", "season_regression"):
        profile = grid.groupby(col)["tuning_log_loss"].min()
        print(f"  {col:<18} " + "  ".join(f"{v:g}: {ll:.4f}" for v, ll in profile.items()))

    best = params_of(grid.iloc[0])
    pred_best = elo_predictions(games, best)
    pred_ref = elo_predictions(games, REFERENCE_PARAMS)

    for label, seasons in (("Saisons de réglage", TUNING_SEASONS), ("Saisons de contrôle (jamais vues)", HOLDOUT_SEASONS)):
        a = pred_best[pred_best["season"].isin(seasons)]
        b = pred_ref[pred_ref["season"].isin(seasons)]
        print(f"\n{label}, A = meilleurs ({describe(best)}), B = référence :")
        for metric in ("log_loss", "brier", "accuracy"):
            print("  " + format_comparison(paired_compare(a, "elo", b, "elo", metric)))
        sa, sb = scores(pred_best, seasons), scores(pred_ref, seasons)
        print(f"  log loss A {sa['log_loss']:.4f} / B {sb['log_loss']:.4f} ; "
              f"Brier A {sa['brier']:.4f} / B {sb['brier']:.4f} ; "
              f"accuracy A {sa['accuracy']:.4f} / B {sb['accuracy']:.4f}")
        print(f"  proba domicile moyenne − taux réel : A {sa['home_gap'] * 100:+.1f} pts, B {sb['home_gap'] * 100:+.1f} pts")
        print(f"  par saison (proba moyenne / taux réel) : A {calibration_by_season(pred_best, seasons)}")
        print(f"                                          B {calibration_by_season(pred_ref, seasons)}")

    holdout = pred_best[pred_best["season"].isin(HOLDOUT_SEASONS)]
    reference = pred_ref[pred_ref["season"].isin(HOLDOUT_SEASONS)]
    ll = paired_compare(holdout, "elo", reference, "elo", "log_loss")
    gap = abs(scores(pred_best, HOLDOUT_SEASONS)["home_gap"])
    adopt = ll["ci_high"] < 0 and gap < MAX_HOME_GAP
    print(
        f"\nCritères sur {', '.join(HOLDOUT_SEASONS)} : IC du Δ log loss < 0 "
        f"({'oui' if ll['ci_high'] < 0 else 'non'}), écart de calibration {gap * 100:.1f} pts < "
        f"{MAX_HOME_GAP * 100:.1f} ({'oui' if gap < MAX_HOME_GAP else 'non'}) -> "
        f"{'ADOPTER' if adopt else 'NE PAS ADOPTER'} {describe(best)}"
    )
    if best != DEFAULT_ELO_PARAMS:
        print(f"Paramètres actuels de features_lib.DEFAULT_ELO_PARAMS : {describe(DEFAULT_ELO_PARAMS)}")
    print(f"Grille complète : {GRID_PATH}")


if __name__ == "__main__":
    main()
