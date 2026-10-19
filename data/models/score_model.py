"""
score_model.py

Score prédit (étape 5 du plan d'amélioration, 2026-10-19) : évaluation
walk-forward et réglage de sigma. Les formules elles-mêmes sont dans
features_lib.py (expected_total, expected_margin, split_score,
rounded_score), partagées avec l'API.

Méthode servie :
- total des deux équipes = attaque récente de chacune et défense récente de
  l'adversaire (features_lib.expected_total, SCORE_WINDOW derniers matchs) ;
- écart = sigma * Φ⁻¹(p), p = probabilité de victoire à domicile du modèle
  servi. sigma est ajusté par moindres carrés (pondérés par saison, comme le
  modèle) de l'écart réel sur Φ⁻¹(p) des matchs d'entraînement : c'est la
  valeur qui fait de sigma * Φ⁻¹(p) la meilleure prédiction linéaire de
  l'écart à partir de p.

Comparée à :
- "previous" : l'ancienne heuristique de l'API (avant le 2026-10-19), même
  total mais chaque équipe prédite séparément, donc un écart sans lien avec
  la probabilité (le score désignait un autre vainqueur que le modèle sur
  22 % des matchs) ;
- "naive" : total moyen et écart moyen de la saison précédente, pour tous
  les matchs.

Mesures du 2026-10-19 (walk-forward 2021-22 -> 2025-26, 6 140 matchs) :
- écart : MAE 10,9 pts contre 11,4 pour l'ancienne méthode, -0,5 pt
  [-0,6 ; -0,4], même sens sur les 5 saisons ;
- total : une régression (linéaire ou Ridge, moyennes sur 5 et 10 matchs,
  jours de repos, poids par saison) ne fait pas mieux que la formule
  ci-dessus (écarts de -0,004 à +0,02 pt de MAE, dans le bruit) : formule
  gardée, plus simple ;
- sigma estimé par l'écart-type résiduel d'une régression Ridge de l'écart
  (étape 3d) plutôt que par moindres carrés : 13,5 au lieu de 12,3, MAE de
  l'écart identique à 0,01 pt près. Moindres carrés gardés : pas de second
  modèle à entraîner.

Usage : appelé par train_model.py (métriques dans model_metrics.json,
clé "score_model").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_model import (
    WALK_FORWARD_SEASONS,
    ModelSpec,
    complete_rows,
    fit_spec,
    format_comparison,
    paired_bootstrap,
    season_weights,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    SCORE_WINDOW,
    expected_margin,
    expected_total,
    rounded_score,
    split_score,
)

SCORE_INPUTS = tuple(
    f"{side}_avg_pts_{kind}_last{SCORE_WINDOW}"
    for side, kind in (("home", "scored"), ("home", "allowed"), ("away", "scored"), ("away", "allowed"))
)
METHODS = {
    "model": "Écart déduit de la probabilité",
    "previous": "Ancienne méthode",
    "naive": "Moyennes de la saison précédente",
}
SCORE_METRICS = ("mae_team", "mae_margin", "mae_total")
PREVIOUS_FALLBACK = (110.0, 108.0)  # valeurs par défaut de l'ancienne heuristique


# ---------------------------------------------------------------------------
# Méthodes
# ---------------------------------------------------------------------------

def fit_sigma(model, train: pd.DataFrame, feature_cols: list[str]) -> float:
    """sigma par moindres carrés pondérés : écart réel ≈ sigma * Φ⁻¹(p) (voir l'en-tête)."""
    z = expected_margin(model.predict_proba(train[feature_cols])[:, 1], 1.0)
    margin = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    w = season_weights(train["season"])
    return float(np.sum(w * margin * z) / np.sum(w * z ** 2))


def latest_season_total(games: pd.DataFrame) -> float:
    """Total moyen de la saison la plus récente : repli si une moyenne récente manque."""
    last = games[games["season"] == games["season"].max()]
    return float((last["home_score"] + last["away_score"]).mean())


def model_scores(df: pd.DataFrame, p: np.ndarray, sigma: float, fallback_total: float):
    """Points domicile / extérieur de la méthode servie (non arrondis)."""
    total = expected_total(*(df[c].to_numpy(dtype=float) for c in SCORE_INPUTS))
    total = np.where(np.isnan(total), fallback_total, total)
    return split_score(total, expected_margin(p, sigma))


def previous_scores(df: pd.DataFrame):
    """Ancienne heuristique de l'API : chaque équipe = (son attaque + défense adverse) / 2."""
    home_scored, home_allowed, away_scored, away_allowed = (df[c] for c in SCORE_INPUTS)
    home = ((home_scored + away_allowed) / 2).fillna(PREVIOUS_FALLBACK[0])
    away = ((away_scored + home_allowed) / 2).fillna(PREVIOUS_FALLBACK[1])
    return home.to_numpy(dtype=float), away.to_numpy(dtype=float)


def naive_scores(history: pd.DataFrame, n: int):
    """Total et écart moyens de la saison précédente, identiques pour tous les matchs."""
    last = history[history["season"] == history["season"].max()]
    home, away = split_score(
        (last["home_score"] + last["away_score"]).mean(), (last["home_score"] - last["away_score"]).mean()
    )
    return np.full(n, home), np.full(n, away)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_scores(
    df: pd.DataFrame, feature_cols: list[str], served: ModelSpec, test_seasons=WALK_FORWARD_SEASONS,
) -> pd.DataFrame:
    """
    Scores prédits hors échantillon, une ligne par match des saisons
    testées, pour chaque méthode : colonnes <méthode>_home / <méthode>_away
    (scores affichés, arrondis par features_lib.rounded_score), plus p
    (probabilité du modèle servi) et sigma (celui du pli).
    """
    folds = []
    for season in test_seasons:
        history = df[df["season"] < season]
        train = complete_rows(history, feature_cols)
        test = df[df["season"] == season]
        model = fit_spec(served, 0, train, feature_cols)
        sigma = fit_sigma(model, train, feature_cols)
        p = model.predict_proba(test[feature_cols])[:, 1]

        out = test[["nba_game_id", "game_date", "season", "home_score", "away_score"]].copy()
        out["p"] = p
        out["sigma"] = sigma
        predictions = {
            "model": model_scores(test, p, sigma, latest_season_total(history)),
            "previous": previous_scores(test),
            "naive": naive_scores(history, len(test)),
        }
        for method, (home, away) in predictions.items():
            shown = [rounded_score(h, a, h >= a) for h, a in zip(home, away)]
            out[f"{method}_home"] = [h for h, _ in shown]
            out[f"{method}_away"] = [a for _, a in shown]
        folds.append(out)
    return pd.concat(folds, ignore_index=True)


def per_game_errors(oof: pd.DataFrame, method: str) -> dict:
    """Erreurs absolues de chaque match : par équipe (moyenne des deux), sur l'écart, sur le total."""
    home, away = oof[f"{method}_home"], oof[f"{method}_away"]
    return {
        "mae_team": ((home - oof["home_score"]).abs() + (away - oof["away_score"]).abs()).to_numpy() / 2,
        "mae_margin": ((home - away) - (oof["home_score"] - oof["away_score"])).abs().to_numpy(),
        "mae_total": ((home + away) - (oof["home_score"] + oof["away_score"])).abs().to_numpy(),
    }


def method_summary(oof: pd.DataFrame, method: str) -> dict:
    errors = per_game_errors(oof, method)
    winner_home = oof[f"{method}_home"] > oof[f"{method}_away"]
    return {
        **{m: float(v.mean()) for m, v in errors.items()},
        # Part des matchs où le score désigne un autre vainqueur que le modèle
        "winner_mismatch": float((winner_home != (oof["p"] >= 0.5)).mean()),
    }


def score_summary(oof: pd.DataFrame) -> dict:
    """Résultats au format de model_metrics.json (clé score_model.evaluation)."""
    seasons = sorted(oof["season"].unique())
    model_errors = per_game_errors(oof, "model")
    comparisons = {}
    for other in ("previous", "naive"):
        other_errors = per_game_errors(oof, other)
        comparisons[f"model_vs_{other}"] = {
            m: paired_bootstrap(model_errors[m] - other_errors[m], oof["game_date"], oof["season"], m)
            for m in SCORE_METRICS
        }
    return {
        "method": "walk_forward",
        "test_seasons": [str(s) for s in seasons],
        "n_games": int(len(oof)),
        "methods": [{"key": k, "name": name, **method_summary(oof, k)} for k, name in METHODS.items()],
        "per_season": [
            {
                "season": str(s),
                "n_games": int((oof["season"] == s).sum()),
                "sigma": float(oof.loc[oof["season"] == s, "sigma"].iloc[0]),
                "methods": {k: method_summary(oof[oof["season"] == s], k) for k in METHODS},
            }
            for s in seasons
        ],
        "comparisons": comparisons,
    }


def print_score_report(summary: dict) -> None:
    rows = [{"méthode": m["key"], **{k: m[k] for k in (*SCORE_METRICS, "winner_mismatch")}} for m in summary["methods"]]
    print(pd.DataFrame(rows).set_index("méthode").to_string(float_format=lambda v: f"{v:.3f}"))
    sigmas = ", ".join(f"{s['season']} {s['sigma']:.2f}" for s in summary["per_season"])
    print(f"sigma par pli : {sigmas}")
    for name, comparison in summary["comparisons"].items():
        print(f"\nComparaison appariée {name.replace('_vs_', ' (A) contre ')} (B) :")
        for c in comparison.values():
            print("  " + format_comparison(c))
