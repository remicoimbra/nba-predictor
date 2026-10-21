"""
score_model.py

Score prédit : évaluation walk-forward, réglage de sigma et de la
fourchette. Les formules elles-mêmes sont dans features_lib.py (ratings de
score, expected_margin, split_score, rounded_score, score_range), partagées
avec l'API.

Méthode servie (score v2, 2026-10-21) :
- total des deux équipes = ratings de score d'avant-match (colonne
  exp_total de game_features.csv : rythme × points par possession, notes
  mises à jour après chaque match comme l'Elo, donc ajustées aux
  adversaires) ;
- écart = sigma * Φ⁻¹(p), p = probabilité de victoire à domicile du modèle
  servi (inchangé depuis le 2026-10-19). sigma est ajusté par moindres
  carrés (pondérés par saison, comme le modèle) de l'écart réel sur Φ⁻¹(p)
  des matchs d'entraînement : c'est la valeur qui fait de sigma * Φ⁻¹(p)
  la meilleure prédiction linéaire de l'écart à partir de p ;
- fourchette à INTERVAL_LEVEL : points de chaque équipe ± une demi-largeur
  fixe, quantile (pondéré par saison) de l'erreur absolue par équipe sur
  les matchs d'entraînement. Sa couverture réelle est mesurée sur la
  saison prédite.

Comparée à :
- "previous" : la méthode du 2026-10-19, même écart mais total = moyenne
  des points marqués et encaissés par les deux équipes sur leurs
  PREVIOUS_WINDOW derniers matchs, sans ajustement à l'adversaire ;
- "naive" : total moyen et écart moyen de la saison précédente, pour tous
  les matchs.
L'heuristique d'avant le 2026-10-19 (chaque équipe estimée séparément, le
score désignait un autre vainqueur que le modèle sur 22 % des matchs)
n'est plus évaluée.

Mesures du 2026-10-21 (walk-forward 2021-22 -> 2025-26, 6 140 matchs,
détails dans CLAUDE.md, "Score v2") :
- total : MAE 14,93 -> 14,64 pts face à la méthode précédente, Δ -0,28
  [-0,39 ; -0,18], même sens sur les 5 saisons ; sur 2024-25 + 2025-26
  seules (jamais vues pendant le réglage) -0,35 [-0,52 ; -0,18] ; par
  équipe 9,27 -> 9,19, -0,09 [-0,13 ; -0,05] ; écart inchangé ;
- non adoptés (banc d'essai) : remettre le niveau de la ligue à la moyenne
  de la saison précédente à chaque nouvelle saison, mélanger avec la
  méthode précédente, écart tiré des ratings, efficacité nette ajustée
  comme feature de la régression logistique : dans le bruit ;
- fourchette à 80 % : demi-largeur ~14,6 pts par pli, couverture 80,9 %
  (79,6 à 82,5 % selon la saison, 81,5 % en début de saison).

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
    early_season_mask,
    fit_spec,
    format_comparison,
    paired_bootstrap,
    season_weights,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    expected_margin,
    rounded_score,
    score_range,
    split_score,
)

INTERVAL_LEVEL = 0.8
COMPLETE_SEASON_GAMES = 1000  # 1 230 matchs par saison, 1 059 et 1 080 pour 2019-20 et 2020-21 (Covid)
PREVIOUS_WINDOW = 10
PREVIOUS_INPUTS = tuple(
    f"{side}_avg_pts_{kind}_last{PREVIOUS_WINDOW}"
    for side, kind in (("home", "scored"), ("home", "allowed"), ("away", "scored"), ("away", "allowed"))
)
METHODS = {
    "model": "Rythme × efficacité",
    "previous": f"Moyennes des {PREVIOUS_WINDOW} derniers matchs",
    "naive": "Moyennes de la saison précédente",
}
SCORE_METRICS = ("mae_team", "mae_margin", "mae_total")


# ---------------------------------------------------------------------------
# Méthodes
# ---------------------------------------------------------------------------

def model_points(df: pd.DataFrame, p: np.ndarray, sigma: float):
    """Points domicile / extérieur de la méthode servie (non arrondis)."""
    return split_score(df["exp_total"].to_numpy(dtype=float), expected_margin(p, sigma))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(cumulative, q)])


def fit_score_params(model, train: pd.DataFrame, feature_cols: list[str]) -> tuple[float, float]:
    """
    (sigma, demi-largeur de la fourchette), appris sur les matchs
    d'entraînement avec les probabilités du modèle servi (voir l'en-tête).
    """
    p = model.predict_proba(train[feature_cols])[:, 1]
    w = season_weights(train["season"])
    z = expected_margin(p, 1.0)
    margin = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    sigma = float(np.sum(w * margin * z) / np.sum(w * z ** 2))

    home, away = model_points(train, p, sigma)
    errors = np.concatenate([np.abs(home - train["home_score"]), np.abs(away - train["away_score"])])
    return sigma, weighted_quantile(errors, np.concatenate([w, w]), INTERVAL_LEVEL)


def latest_season_total(games: pd.DataFrame) -> float:
    """
    Total moyen de la dernière saison complète (une saison qui vient de
    commencer ne compte que quelques matchs) : repli si un total attendu
    manque.
    """
    counts = games["season"].value_counts()
    season = max(s for s, n in counts.items() if n >= COMPLETE_SEASON_GAMES)
    last = games[games["season"] == season]
    return float((last["home_score"] + last["away_score"]).mean())


def previous_points(df: pd.DataFrame, p: np.ndarray, sigma: float, fallback_total: float):
    """Méthode du 2026-10-19 : même écart, total d'après les moyennes des derniers matchs."""
    home_scored, home_allowed, away_scored, away_allowed = (df[c].to_numpy(dtype=float) for c in PREVIOUS_INPUTS)
    total = (home_scored + away_allowed) / 2 + (away_scored + home_allowed) / 2
    total = np.where(np.isnan(total), fallback_total, total)
    return split_score(total, expected_margin(p, sigma))


def naive_points(history: pd.DataFrame, n: int):
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
    (probabilité du modèle servi), sigma et half_width (ceux du pli),
    early_season, et la fourchette affichée de la méthode servie
    (model_home_low / _high, model_away_low / _high).
    """
    early = early_season_mask(df)
    folds = []
    for season in test_seasons:
        history = df[df["season"] < season]
        train = complete_rows(history, feature_cols)
        test = df[df["season"] == season]
        model = fit_spec(served, 0, train, feature_cols)
        sigma, half_width = fit_score_params(model, train, feature_cols)
        p = model.predict_proba(test[feature_cols])[:, 1]

        out = test[["nba_game_id", "game_date", "season", "home_score", "away_score"]].copy()
        out["p"] = p
        out["sigma"] = sigma
        out["half_width"] = half_width
        out["early_season"] = early.loc[test.index]
        points = {
            "model": model_points(test, p, sigma),
            "previous": previous_points(test, p, sigma, latest_season_total(history)),
            "naive": naive_points(history, len(test)),
        }
        for method, (home, away) in points.items():
            shown = [rounded_score(h, a, h >= a) for h, a in zip(home, away)]
            out[f"{method}_home"] = [h for h, _ in shown]
            out[f"{method}_away"] = [a for _, a in shown]
        for side, values in zip(("home", "away"), points["model"]):
            ranges = [score_range(v, half_width) for v in values]
            out[f"model_{side}_low"] = [low for low, _ in ranges]
            out[f"model_{side}_high"] = [high for _, high in ranges]
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


def coverage(oof: pd.DataFrame) -> float:
    """Part des scores d'équipe réels compris dans la fourchette affichée (bornes incluses)."""
    inside = [
        oof[f"{side}_score"].between(oof[f"model_{side}_low"], oof[f"model_{side}_high"])
        for side in ("home", "away")
    ]
    return float(pd.concat(inside).mean())


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
        "interval": {
            "level": INTERVAL_LEVEL,
            "coverage": coverage(oof),
            "coverage_early_season": coverage(oof[oof["early_season"]]),
            "mean_half_width": float(oof.groupby("season")["half_width"].first().mean()),
        },
        "per_season": [
            {
                "season": str(s),
                "n_games": int((oof["season"] == s).sum()),
                "sigma": float(oof.loc[oof["season"] == s, "sigma"].iloc[0]),
                "half_width": float(oof.loc[oof["season"] == s, "half_width"].iloc[0]),
                "coverage": coverage(oof[oof["season"] == s]),
                "methods": {k: method_summary(oof[oof["season"] == s], k) for k in METHODS},
            }
            for s in seasons
        ],
        "comparisons": comparisons,
    }


def print_score_report(summary: dict) -> None:
    rows = [{"méthode": m["key"], **{k: m[k] for k in (*SCORE_METRICS, "winner_mismatch")}} for m in summary["methods"]]
    print(pd.DataFrame(rows).set_index("méthode").to_string(float_format=lambda v: f"{v:.3f}"))
    per_fold = ", ".join(
        f"{s['season']} {s['sigma']:.2f} / ±{s['half_width']:.1f} / {s['coverage']:.1%}" for s in summary["per_season"]
    )
    print(f"sigma / demi-largeur / couverture par pli : {per_fold}")
    interval = summary["interval"]
    print(
        f"fourchette à {interval['level']:.0%} : couverture {interval['coverage']:.1%} "
        f"(début de saison {interval['coverage_early_season']:.1%})"
    )
    for name, comparison in summary["comparisons"].items():
        print(f"\nComparaison appariée {name.replace('_vs_', ' (A) contre ')} (B) :")
        for c in comparison.values():
            print("  " + format_comparison(c))
