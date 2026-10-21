"""
player_model.py

Statistiques des joueurs : modèle de minutes, évaluation walk-forward,
fourchettes. Les formules (état des joueurs, taux attendus, entrées du
modèle, joueurs attendus) sont dans features_lib.py, partagées avec l'API.

Méthode servie (2026-10-21) :
- joueurs attendus : ceux du dernier match de l'équipe dans la saison
  (projection « s'il joue ») ;
- minutes : XGBoost (fit_minutes_model) sur features_lib.MINUTES_FEATURES
  (moyennes de minutes, repos, nombre de joueurs du dernier match, rythme,
  Elo, taux...) ;
- points, rebonds, passes, interceptions, contres : minutes × taux par
  minute attendus pour le match (features_lib.player_projection_features :
  taux du joueur lissé vers la ligue, rythme du match, adversaire) ;
- fourchette à INTERVAL_LEVEL : par stat, tranches de projection, et dans
  chaque tranche l'intervalle entier le plus proche de 80 % des valeurs
  réelles (fit_interval_table). Apprise sur des projections HORS
  échantillon (walk-forward depuis OOF_START) : sur ses propres matchs
  d'entraînement, le modèle de minutes est trop bon et les fourchettes
  seraient trop étroites.

Comparée à :
- "formula" : même chose avec les minutes moyennes pondérées (demi-vie 2
  matchs) au lieu du modèle appris ;
- "season" : moyenne de la saison du joueur (à date) ;
- "last10" : moyenne de ses 10 derniers matchs.
Erreurs mesurées sur les joueurs attendus ET entrés en jeu (la projection
est conditionnelle à la participation), décision sur l'erreur quadratique
(la valeur affichée est une espérance), MAE affichée.

Choix et mesures du 2026-10-21 (banc d'essai, détails dans CLAUDE.md,
« Statistiques des joueurs ») :
- réglage (saisons 2021-22 -> 2023-24) : demi-vie des minutes 2 matchs ;
  remise à l'échelle des minutes à 240 par équipe : moins bonne, écartée ;
  taux : voir features_lib.DEFAULT_PLAYER_PARAMS (tune_player_projections.py) ;
- adoptés : rythme du match, adversaire (exposant 0,5), minutes apprises
  (XGBoost, profondeur 4) ; XGBoost appliqué directement à chaque stat fait
  moins bien que « minutes apprises × taux » (sauf points, à égalité) ;
- écartés : points des joueurs ramenés au score servi de l'équipe (nettement
  moins bon), minutes réduites selon l'écart attendu (MAE dégradée).
Walk-forward 2021-22 -> 2025-26, 111 136 joueurs attendus entrés en jeu,
MAE servie (moyenne de la saison / 10 derniers matchs) : minutes 4,77
(5,31 / 5,14), points 4,68 (4,80 / 4,81), rebonds 1,91 (1,97 / 1,98),
passes 1,35 (1,38 / 1,39), interceptions 0,72 (0,73 / 0,74), contres 0,53
(0,53 / 0,53). Erreur quadratique plus basse que les deux références sur
les 6 stats, 5 saisons sur 5. Face à la formule seule : minutes -2,88
[-3,37 ; -2,51], points -0,61 [-0,73 ; -0,51] ; MAE des interceptions et
des contres +0,002 (la MAE récompense la médiane, 0 pour la plupart des
joueurs ; une seule projection de minutes gardée pour toutes les stats).
Fourchettes : couverture 79,6 à 82,7 % selon la stat. 14,2 % des joueurs
attendus ne jouent pas ; 9,0 % des minutes vont à des joueurs non attendus.

Usage : appelé par train_model.py (métriques dans model_metrics.json, clé
"player_model" ; modèle de minutes dans saved_models/player_minutes.pkl).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from evaluate_model import WALK_FORWARD_SEASONS, format_comparison, paired_bootstrap, season_weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_PLAYER_PARAMS,
    MINUTES_FEATURES,
    PLAYER_STATS,
    compute_player_rows,
    projected_stats,
    stat_ranges,
)
from player_games import load_player_games, restrict_to_games  # type: ignore[import-not-found]  # noqa: E402

PLAYER_TRAIN_START = "2017-18"  # 2016-17 : rodage des moyennes pondérées (aucun historique avant)
OOF_START = "2018-19"           # première saison prédite hors échantillon (fourchettes)
EARLY_SEASON_GAMES = 10         # « début de saison » : l'équipe a joué moins de 10 matchs

# Modèle de minutes : nombre d'arbres choisi par arrêt précoce sur les
# VALIDATION_FRACTION lignes les plus récentes (comme train_model.XGBoostEarlyStopping).
MINUTES_XGB = {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50}
VALIDATION_FRACTION = 0.2
MAX_TREES = 3000
PATIENCE = 50

INTERVAL_LEVEL = 0.8
INTERVAL_BINS = 20                   # tranches de projection (quantiles)...
INTERVAL_TAIL = (0.975, 0.99, 0.995)  # ... plus des tranches fines pour les plus grosses projections

METHODS = {
    "model": "Minutes apprises × taux",
    "formula": "Formule seule",
    "season": "Moyenne de la saison",
    "last10": "10 derniers matchs",
}


# ---------------------------------------------------------------------------
# Lignes : joueurs attendus, valeurs réelles, références
# ---------------------------------------------------------------------------

def load_players(df_games: pd.DataFrame) -> pd.DataFrame:
    players = restrict_to_games(load_player_games(), df_games)
    players["reb"] = players["oreb"] + players["dreb"]
    return players


def reference_columns(rows: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Moyennes de la saison (à date) et des 10 derniers matchs du joueur, avant la date du match."""
    p = players.sort_values(["player_id", "game_date", "nba_game_id"]).reset_index(drop=True)
    by_season = p.groupby(["player_id", "season"])
    by_player = p.groupby("player_id")
    ref = p[["player_id", "game_date"]].rename(columns={"game_date": "ref_date"})
    for s in PLAYER_STATS:
        ref[f"season_{s}"] = by_season[s].cumsum() / (by_season.cumcount() + 1)
        ref[f"last10_{s}"] = by_player[s].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    out = pd.merge_asof(
        rows.sort_values("game_date"), ref.sort_values("ref_date"),
        left_on="game_date", right_on="ref_date", by="player_id", allow_exact_matches=False,
    ).drop(columns="ref_date")
    return out.sort_values(["game_date", "nba_game_id", "team_id", "player_id"]).reset_index(drop=True)


def player_rows(df_games: pd.DataFrame, players: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Joueurs attendus de chaque match (features_lib.compute_player_rows), avec
    leurs stats réelles (NaN s'ils n'ont pas joué, colonne `played`) et les
    références. `df_games` : game_features.csv (boxscore, ratings de score, Elo).
    """
    players = load_players(df_games) if players is None else players
    rows = compute_player_rows(df_games, players)
    actual = players[["player_id", "team_id", "nba_game_id", *PLAYER_STATS]].rename(columns={s: f"actual_{s}" for s in PLAYER_STATS})
    rows = rows.merge(actual, on=["player_id", "team_id", "nba_game_id"], how="left")
    rows["played"] = rows["actual_min"].notna()
    return reference_columns(rows, players)


def uncovered_minutes(rows: pd.DataFrame, players: pd.DataFrame, seasons) -> float:
    """Part des minutes réellement jouées, dans les matchs projetés, par des joueurs NON attendus."""
    games = rows.loc[rows["season"].isin(seasons), ["team_id", "nba_game_id"]].drop_duplicates()
    played = players.merge(games, on=["team_id", "nba_game_id"])
    expected = rows[["team_id", "nba_game_id", "player_id"]].assign(expected=True)
    played = played.merge(expected, on=["team_id", "nba_game_id", "player_id"], how="left")
    return float(played.loc[played["expected"].isna(), "min"].sum() / played["min"].sum())


# ---------------------------------------------------------------------------
# Modèle de minutes
# ---------------------------------------------------------------------------

def fit_minutes_model(train: pd.DataFrame) -> XGBRegressor:
    """
    XGBoost des minutes (joueurs attendus ET entrés en jeu, lignes dans
    l'ordre chronologique), pondéré par saison comme les autres modèles :
    nombre d'arbres par arrêt précoce sur les lignes les plus récentes, puis
    réentraînement sur toutes avec ce nombre.
    """
    X, y = train[MINUTES_FEATURES], train["actual_min"]
    w = season_weights(train["season"])
    n_val = int(len(train) * VALIDATION_FRACTION)
    probe = XGBRegressor(n_estimators=MAX_TREES, early_stopping_rounds=PATIENCE, n_jobs=-1, **MINUTES_XGB)
    probe.fit(
        X.iloc[:-n_val], y.iloc[:-n_val], sample_weight=w[:-n_val],
        eval_set=[(X.iloc[-n_val:], y.iloc[-n_val:])], verbose=False,
    )
    model = XGBRegressor(n_estimators=probe.best_iteration + 1, n_jobs=-1, **MINUTES_XGB)
    return model.fit(X, y, sample_weight=w)


def training_rows(rows: pd.DataFrame, before_season=None) -> pd.DataFrame:
    keep = rows["played"] & (rows["season"] >= PLAYER_TRAIN_START)
    if before_season is not None:
        keep &= rows["season"] < before_season
    return rows[keep]


def walk_forward_players(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Projections hors échantillon de chaque saison depuis OOF_START (modèle
    de minutes entraîné sur les seules saisons antérieures) : colonnes
    model_<stat> et formula_<stat> (minutes moyennes × taux), n_trees.
    """
    folds = []
    for season in sorted(s for s in rows["season"].unique() if s >= OOF_START):
        test = rows[rows["season"] == season]
        train = training_rows(rows, season)
        model = fit_minutes_model(train)
        out = test.copy()
        model_stats = projected_stats(test, model.predict(test[MINUTES_FEATURES]))
        formula_stats = projected_stats(test, test["minutes"])
        for s in PLAYER_STATS:
            out[f"model_{s}"] = model_stats[s]
            out[f"formula_{s}"] = formula_stats[s]
        out["n_trees"] = model.n_estimators
        print(f"  {season} : {len(train)} lignes d'entraînement, {model.n_estimators} arbres, {len(test)} joueurs attendus")
        folds.append(out)
    return pd.concat(folds, ignore_index=True)


# ---------------------------------------------------------------------------
# Fourchettes
# ---------------------------------------------------------------------------

def best_interval(values: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    """
    Intervalle entier [bas, haut] dont la masse (pondérée) est la plus
    proche de INTERVAL_LEVEL : pour chaque largeur, l'intervalle le plus
    chargé, et la largeur la plus proche du niveau visé. Les stats sont des
    entiers : des quantiles 10 % / 90 % avec bornes incluses couvriraient
    bien plus que 80 % (94 % pour les interceptions).
    """
    low = int(values.min())
    mass = np.bincount((values - low).astype(int), weights=weights) / weights.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(mass)])
    best = None
    for width in range(len(mass)):
        masses = cumulative[width + 1:] - cumulative[: len(mass) - width]
        i = int(np.argmax(masses))
        gap = abs(masses[i] - INTERVAL_LEVEL)
        if best is None or gap < best[0] - 1e-12:
            best = (gap, low + i, low + i + width)
        if masses[i] >= INTERVAL_LEVEL:
            break
    return int(best[1]), int(best[2])


def fit_interval_table(projected: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> dict:
    """Table d'une stat pour features_lib.stat_ranges : bornes des tranches et intervalle de chacune."""
    qs = sorted({*np.linspace(0, 1, INTERVAL_BINS + 1)[1:-1], *INTERVAL_TAIL})
    edges = np.unique(np.quantile(projected, qs))
    idx = np.searchsorted(edges, projected, side="right")
    actual = np.round(actual).astype(int)
    bounds = [list(best_interval(actual[idx == b], weights[idx == b])) for b in range(len(edges) + 1)]
    return {"edges": [float(e) for e in edges], "bounds": bounds}


def fit_interval_tables(oof: pd.DataFrame) -> dict:
    """Tables des PLAYER_STATS, apprises sur des projections hors échantillon (joueurs entrés en jeu)."""
    played = oof[oof["played"]]
    w = season_weights(played["season"])
    return {
        s: fit_interval_table(played[f"model_{s}"].to_numpy(dtype=float), played[f"actual_{s}"].to_numpy(dtype=float), w)
        for s in PLAYER_STATS
    }


def add_interval_columns(oof: pd.DataFrame, test_seasons=WALK_FORWARD_SEASONS) -> pd.DataFrame:
    """Fourchettes de chaque saison testée, apprises sur les saisons hors échantillon qui la précèdent."""
    parts = []
    for season in test_seasons:
        tables = fit_interval_tables(oof[oof["season"] < season])
        part = oof[oof["season"] == season].copy()
        for s in PLAYER_STATS:
            part[f"low_{s}"], part[f"high_{s}"] = stat_ranges(tables[s], part[f"model_{s}"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Synthèse (model_metrics.json, clé player_model.evaluation)
# ---------------------------------------------------------------------------

def errors(ev: pd.DataFrame, method: str, stat: str) -> np.ndarray:
    return (ev[f"{method}_{stat}"] - ev[f"actual_{stat}"]).to_numpy(dtype=float)


def coverage(ev: pd.DataFrame, stat: str) -> float:
    return float(ev[f"actual_{stat}"].between(ev[f"low_{stat}"], ev[f"high_{stat}"]).mean())


def player_summary(oof: pd.DataFrame, rows: pd.DataFrame, players: pd.DataFrame, test_seasons=WALK_FORWARD_SEASONS) -> dict:
    expected = oof[oof["season"].isin(test_seasons)]
    ev = add_interval_columns(oof, test_seasons)
    ev = ev[ev["played"]]
    early = ev["team_games"] < EARLY_SEASON_GAMES

    def stats_summary(method: str, sub: pd.DataFrame) -> dict:
        return {
            s: {"mae": float(np.abs(errors(sub, method, s)).mean()), "rmse": float(np.sqrt((errors(sub, method, s) ** 2).mean()))}
            for s in PLAYER_STATS
        }

    comparisons = {}
    for other in ("formula", "season", "last10"):
        comparisons[f"model_vs_{other}"] = {
            s: {
                kind: paired_bootstrap(
                    (np.abs(errors(ev, "model", s)) - np.abs(errors(ev, other, s))) if kind == "mae"
                    else (errors(ev, "model", s) ** 2 - errors(ev, other, s) ** 2),
                    ev["game_date"], ev["season"], f"{kind}_{s}",
                )
                for kind in ("mse", "mae")
            }
            for s in PLAYER_STATS
        }
    return {
        "method": "walk_forward",
        "test_seasons": [str(s) for s in test_seasons],
        "n_rows": int(len(ev)),
        "participation": {
            "expected_players": int(len(expected)),
            "did_not_play": float(1 - expected["played"].mean()),
            "uncovered_minutes": uncovered_minutes(rows, players, test_seasons),
        },
        "methods": [{"key": k, "name": name, "stats": stats_summary(k, ev)} for k, name in METHODS.items()],
        "interval": {
            "level": INTERVAL_LEVEL,
            "coverage": {s: coverage(ev, s) for s in PLAYER_STATS},
            "coverage_early_season": {s: coverage(ev[early], s) for s in PLAYER_STATS},
            "mean_width": {s: float((ev[f"high_{s}"] - ev[f"low_{s}"]).mean()) for s in PLAYER_STATS},
        },
        "early_season": {"max_team_games": EARLY_SEASON_GAMES, "n_rows": int(early.sum()), "model": stats_summary("model", ev[early])},
        "per_season": [
            {
                "season": str(season),
                "n_rows": int((ev["season"] == season).sum()),
                "minutes_trees": int(ev.loc[ev["season"] == season, "n_trees"].iloc[0]),
                "mae": {k: {s: float(np.abs(errors(ev[ev["season"] == season], k, s)).mean()) for s in PLAYER_STATS} for k in METHODS},
                "coverage": {s: coverage(ev[ev["season"] == season], s) for s in PLAYER_STATS},
            }
            for season in test_seasons
        ],
        "comparisons": comparisons,
    }


def print_player_report(summary: dict) -> None:
    rows = []
    for m in summary["methods"]:
        rows.append({"méthode": m["key"], **{f"{s} MAE": m["stats"][s]["mae"] for s in PLAYER_STATS}})
        rows.append({"méthode": f"{m['key']} (RMSE)", **{f"{s} MAE": m["stats"][s]["rmse"] for s in PLAYER_STATS}})
    print(pd.DataFrame(rows).set_index("méthode").rename(columns=lambda c: c.replace(" MAE", "")).to_string(float_format=lambda v: f"{v:.3f}"))
    p = summary["participation"]
    print(
        f"{summary['n_rows']} lignes évaluées ; joueurs attendus qui n'ont pas joué : {p['did_not_play']:.1%} ; "
        f"minutes jouées par des joueurs non attendus : {p['uncovered_minutes']:.1%}"
    )
    interval = summary["interval"]
    print(f"fourchette à {interval['level']:.0%}, couverture : " + ", ".join(
        f"{s} {interval['coverage'][s]:.1%} (largeur {interval['mean_width'][s]:.1f})" for s in PLAYER_STATS
    ))
    for name, comparison in summary["comparisons"].items():
        print(f"\nComparaison appariée {name.replace('_vs_', ' (A) contre ')} (B) :")
        for s in PLAYER_STATS:
            print("  " + format_comparison(comparison[s]["mse"]) + "   | MAE " + f"{comparison[s]['mae']['delta']:+.4f}"
                  + (" (significatif)" if comparison[s]["mae"]["significant"] else ""))


def served_player_model(rows: pd.DataFrame, oof: pd.DataFrame, evaluation: dict) -> tuple[XGBRegressor, dict]:
    """Modèle de minutes réentraîné sur toutes les saisons, et bloc player_model de model_metrics.json."""
    model = fit_minutes_model(training_rows(rows))
    block = {
        # Comparés par l'API à ceux de team_state.json (as_of.player_params).
        "params": DEFAULT_PLAYER_PARAMS.as_dict(),
        "minutes_features": list(MINUTES_FEATURES),
        "minutes_model": {**MINUTES_XGB, "n_trees": int(model.n_estimators), "train_rows": int(len(training_rows(rows)))},
        "intervals": {"level": INTERVAL_LEVEL, "tables": fit_interval_tables(oof)},
        "evaluation": evaluation,
    }
    return model, block

