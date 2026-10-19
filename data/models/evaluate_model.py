"""
evaluate_model.py

Évaluation "walk-forward" (validation croisée temporelle à fenêtre
croissante) des modèles définis dans train_model.py :

- chaque saison de WALK_FORWARD_SEASONS est prédite par un modèle entraîné
  sur TOUTES les saisons qui la précèdent, et seulement celles-là (la
  situation du modèle en production au début de la saison suivante) ;
- tous les matchs de la saison testée sont gardés, y compris ceux dont une
  feature manque (début de saison) : le modèle en production les rencontre
  (l'entraînement, lui, ne garde que les matchs complets, voir
  complete_rows()) ;
- les modèles non déterministes (XGBoost) tournent sur plusieurs graines
  fixes (SEEDS) et les métriques sont la moyenne sur ces graines.

Métriques principales : log loss et Brier (qualité des probabilités), puis
accuracy et ECE (erreur de calibration moyenne, en points de probabilité).

Pour comparer deux variantes, paired_compare() mesure l'écart d'une
métrique sur les MÊMES matchs, avec un intervalle de confiance par
bootstrap rééchantillonné par journée de matchs. C'est bien plus sensible
que de comparer deux intervalles de confiance séparés : sur la seule saison
2025-26, l'erreur standard de l'écart d'accuracy entre XGBoost et la
régression logistique est d'environ 1 pt, contre ±2,7 pts pour l'IC de
l'accuracy elle-même.

Usage :
    cd data
    python models/evaluate_model.py                                # variante "baseline"
    python models/evaluate_model.py --variant elo_k25 --compare baseline
    python models/evaluate_model.py --legacy                       # ancien découpage (test 2025-26)

Les prédictions hors échantillon sont sauvegardées dans
saved_models/eval/oof_<variante>.csv (non versionné) : une variante se
compare ensuite à une autre sans tout recalculer (par exemple : prédictions
"baseline" calculées, puis features modifiées et game_features.csv
régénéré, puis --variant nouvelle --compare baseline).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

TARGET_COL = "home_win"

# Saisons prédites une à une. 2019-20 et 2020-21 ne servent qu'à
# l'entraînement : avant 2021-22, le premier pli aurait moins de deux
# saisons d'historique (et un Elo encore en rodage, parti de 1500).
WALK_FORWARD_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
SEEDS = (0, 1, 2, 3, 4)

# Ancien protocole (un seul découpage), gardé pour comparer avec les
# chiffres historiques du projet (69,0 % / 0,595).
LEGACY_TEST_SEASON = "2025-26"
LEGACY_SEED = 42

EARLY_SEASON_GAMES = 10  # "début de saison" : une des deux équipes a joué moins de 10 matchs
CALIBRATION_BINS = 10
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
METRICS = ("log_loss", "brier", "accuracy", "ece")
EPS = 1e-15  # même écrêtage que sklearn.metrics.log_loss

EVAL_DIR = Path(__file__).resolve().parent / "saved_models" / "eval"


@dataclass(frozen=True)
class ModelSpec:
    """Un modèle à évaluer : `factory(seed)` renvoie un estimateur scikit-learn neuf."""
    key: str
    name: str
    factory: Callable
    stochastic: bool = False  # True : une passe par graine de SEEDS


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def early_season_mask(df: pd.DataFrame) -> pd.Series:
    """
    True si l'une des deux équipes a joué moins de EARLY_SEASON_GAMES matchs
    de la saison avant celui-ci : la période où win_pct manque ou repose
    sur trop peu de matchs.
    """
    cols = ["nba_game_id", "game_date", "season"]
    long_df = pd.concat([
        df[cols + ["home_team_id"]].rename(columns={"home_team_id": "team_id"}).assign(side="home"),
        df[cols + ["away_team_id"]].rename(columns={"away_team_id": "team_id"}).assign(side="away"),
    ])
    long_df = long_df.sort_values(["team_id", "game_date", "nba_game_id"])
    long_df["played_before"] = long_df.groupby(["team_id", "season"]).cumcount()
    played = long_df.pivot(index="nba_game_id", columns="side", values="played_before")
    fewest = played[["home", "away"]].min(axis=1)
    return (df["nba_game_id"].map(fewest) < EARLY_SEASON_GAMES).set_axis(df.index)


def complete_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Matchs d'entraînement : ceux dont toutes les features sont connues.
    L'évaluation, elle, garde TOUS les matchs (le modèle en production
    prédit aussi le premier match de chaque équipe dans la saison, où
    win_pct manque). Mesuré le 2026-10-19 en walk-forward : entraîner aussi
    sur les ~265 matchs incomplets ne change rien de significatif au total
    (IC à 95 % du Δ log loss contenant 0 pour les deux modèles), et dégrade
    nettement la régression logistique sur ces matchs-là (0,662 contre
    0,627 de log loss).
    """
    return df.dropna(subset=feature_cols + [TARGET_COL])


def prob_columns(oof: pd.DataFrame, key: str) -> list[str]:
    """Colonnes de probabilité d'un modèle : p_<key> (déterministe) ou p_<key>_s<graine>."""
    return [c for c in oof.columns if c == f"p_{key}" or c.startswith(f"p_{key}_s")]


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    specs: list[ModelSpec],
    test_seasons=WALK_FORWARD_SEASONS,
    seeds=SEEDS,
) -> pd.DataFrame:
    """
    Prédictions hors échantillon, une ligne par match des saisons testées.

    Colonnes : nba_game_id, game_date, season, home_win, early_season,
    train_games, p_baseline (taux de victoire à domicile des saisons
    d'entraînement : la baseline naïve, sans fuite), puis une colonne de
    probabilité par modèle et par graine.
    """
    early = early_season_mask(df)
    folds = []
    for season in test_seasons:
        train = complete_rows(df[df["season"] < season], feature_cols)
        test = df[df["season"] == season]
        if train.empty or test.empty:
            raise ValueError(f"Saison {season} : entraînement ou test vide, game_features.csv incomplet ?")

        out = test[["nba_game_id", "game_date", "season", TARGET_COL]].copy()
        out["early_season"] = early.loc[test.index]
        out["train_games"] = len(train)
        out["p_baseline"] = train[TARGET_COL].mean()

        for spec in specs:
            for seed in (seeds if spec.stochastic else (None,)):
                model = spec.factory(0 if seed is None else seed)
                model.fit(train[feature_cols], train[TARGET_COL])
                col = f"p_{spec.key}" if seed is None else f"p_{spec.key}_s{seed}"
                out[col] = model.predict_proba(test[feature_cols])[:, 1]

        print(f"  {season} : {len(train)} matchs d'entraînement, {len(test)} prédits")
        folds.append(out)
    return pd.concat(folds, ignore_index=True)


def legacy_split(df: pd.DataFrame, feature_cols: list[str], specs: list[ModelSpec]) -> dict:
    """
    Ancien protocole, pour la continuité avec les chiffres historiques :
    lignes avec une feature manquante retirées, entraînement sur toutes les
    saisons sauf LEGACY_TEST_SEASON, test sur celle-ci, graine LEGACY_SEED.
    """
    complete = complete_rows(df, feature_cols)
    train = complete[complete["season"] != LEGACY_TEST_SEASON]
    test = complete[complete["season"] == LEGACY_TEST_SEASON]
    y = test[TARGET_COL].to_numpy(dtype=float)

    models = {}
    for spec in specs:
        model = spec.factory(LEGACY_SEED)
        model.fit(train[feature_cols], train[TARGET_COL])
        p = model.predict_proba(test[feature_cols])[:, 1]
        models[spec.key] = {m: float(v.mean()) for m, v in per_game_scores(y, p).items()}
    return {
        "test_season": LEGACY_TEST_SEASON,
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "models": models,
    }


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def per_game_scores(y: np.ndarray, p: np.ndarray) -> dict:
    """Log loss, Brier et bonne réponse (0/1) de chaque match."""
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
        "brier": (p - y) ** 2,
        # > 0,5 comme predict() de scikit-learn et XGBoost
        "accuracy": ((p > 0.5) == (y == 1)).astype(float),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = CALIBRATION_BINS) -> float:
    """Écart moyen |proba prédite - taux réel| par tranche, pondéré par le nombre de matchs."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(p[mask].mean() - y[mask].mean())
    return float(total / len(p))


def summarize(oof: pd.DataFrame, key: str, mask=None) -> dict:
    """Métriques d'un modèle (moyenne sur ses graines), sur tous les matchs ou sur `mask`."""
    sub = oof if mask is None else oof[mask]
    y = sub[TARGET_COL].to_numpy(dtype=float)
    per_seed = []
    for col in prob_columns(oof, key):
        p = sub[col].to_numpy(dtype=float)
        scores = {m: float(v.mean()) for m, v in per_game_scores(y, p).items()}
        scores["ece"] = expected_calibration_error(y, p)
        per_seed.append(scores)

    result = {m: float(np.mean([s[m] for s in per_seed])) for m in METRICS}
    result["n_games"] = int(len(sub))
    if len(per_seed) > 1:
        result["seed_sd"] = {m: float(np.std([s[m] for s in per_seed], ddof=1)) for m in METRICS}
    return result


def calibration_curve(oof: pd.DataFrame, key: str, n_bins: int = CALIBRATION_BINS) -> list[dict]:
    """
    Pour chaque tranche de probabilité prédite (0-10 %, 10-20 %...) : proba
    moyenne prédite, taux de victoire réel à domicile, nombre de matchs.
    Les prédictions de toutes les graines sont empilées (calibration
    attendue d'un modèle à une graine) ; `count` est ramené en nombre de
    matchs. Tranches vides omises.
    """
    cols = prob_columns(oof, key)
    y = np.tile(oof[TARGET_COL].to_numpy(dtype=float), len(cols))
    p = np.concatenate([oof[c].to_numpy(dtype=float) for c in cols])
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    curve = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        curve.append({
            "bin_start": b / n_bins,
            "bin_end": (b + 1) / n_bins,
            "mean_predicted": float(p[mask].mean()),
            "actual_rate": float(y[mask].mean()),
            "count": int(round(mask.sum() / len(cols))),
        })
    return curve


def evaluation_summary(oof: pd.DataFrame, keys: list[str]) -> dict:
    """Résultats par saison et sur le début de saison, au format de model_metrics.json."""
    seasons = sorted(oof["season"].unique())
    early = oof["early_season"].astype(bool)
    return {
        "method": "walk_forward",
        "test_seasons": [str(s) for s in seasons],
        "n_games": int(len(oof)),
        "seeds": list(SEEDS),
        "per_season": [
            {
                "season": str(s),
                "n_games": int((oof["season"] == s).sum()),
                "train_games": int(oof.loc[oof["season"] == s, "train_games"].iloc[0]),
                "models": {k: summarize(oof, k, oof["season"] == s) for k in keys},
            }
            for s in seasons
        ],
        "early_season": {
            "max_games_played": EARLY_SEASON_GAMES,
            "n_games": int(early.sum()),
            "models": {k: summarize(oof, k, early) for k in keys},
        },
    }


# ---------------------------------------------------------------------------
# Comparaison appariée
# ---------------------------------------------------------------------------

def per_game_metric(oof: pd.DataFrame, key: str, metric: str) -> pd.Series:
    """Métrique de chaque match, moyennée sur les graines, indexée par nba_game_id."""
    y = oof[TARGET_COL].to_numpy(dtype=float)
    values = np.mean(
        [per_game_scores(y, oof[c].to_numpy(dtype=float))[metric] for c in prob_columns(oof, key)],
        axis=0,
    )
    return pd.Series(values, index=oof["nba_game_id"].to_numpy())


def paired_compare(
    oof_a: pd.DataFrame, key_a: str, oof_b: pd.DataFrame, key_b: str, metric: str = "log_loss",
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Écart moyen (a - b) d'une métrique sur les matchs communs aux deux
    jeux de prédictions, avec un intervalle de confiance à 95 % par
    bootstrap. On rééchantillonne des JOURNÉES entières plutôt que des
    matchs : les matchs d'un même soir partagent le contexte (calendrier,
    blessures annoncées), l'IC reste ainsi honnête s'ils sont corrélés.

    Log loss et Brier : négatif = a meilleur. Accuracy : positif = a meilleur.
    `significant` : l'IC exclut 0.
    """
    a = per_game_metric(oof_a, key_a, metric)
    b = per_game_metric(oof_b, key_b, metric)
    common = a.index.intersection(b.index)
    meta = oof_a.set_index("nba_game_id").loc[common, ["game_date", "season"]]
    diff = (a.loc[common] - b.loc[common]).to_numpy()

    _, day_idx = np.unique(meta["game_date"].to_numpy(), return_inverse=True)
    day_sums = np.bincount(day_idx, weights=diff)
    day_counts = np.bincount(day_idx)
    picks = np.random.default_rng(seed).integers(0, len(day_sums), size=(draws, len(day_sums)))
    boot = day_sums[picks].sum(axis=1) / day_counts[picks].sum(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    seasons = meta["season"].to_numpy()
    per_season = {str(s): float(diff[seasons == s].mean()) for s in sorted(np.unique(seasons))}
    return {
        "metric": metric,
        "delta": float(diff.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(ci_low > 0 or ci_high < 0),
        "n_games": int(len(common)),
        "per_season": per_season,
    }


def format_comparison(c: dict) -> str:
    lower_is_better = c["metric"] != "accuracy"
    if not c["significant"]:
        verdict = "dans le bruit"
    elif (c["delta"] < 0) == lower_is_better:
        verdict = "A meilleur"
    else:
        verdict = "B meilleur"
    same_sign = sum(np.sign(v) == np.sign(c["delta"]) for v in c["per_season"].values())
    return (
        f"{c['metric']:<8} Δ(A−B) = {c['delta']:+.4f}  IC95 [{c['ci_low']:+.4f} ; {c['ci_high']:+.4f}]"
        f"  -> {verdict} (même signe sur {same_sign}/{len(c['per_season'])} saisons, {c['n_games']} matchs)"
    )


# ---------------------------------------------------------------------------
# Affichage / sauvegarde
# ---------------------------------------------------------------------------

def print_report(oof: pd.DataFrame, keys: list[str]) -> None:
    segments = [(str(s), oof["season"] == s) for s in sorted(oof["season"].unique())]
    segments.append(("Total", None))
    segments.append((f"Début de saison (< {EARLY_SEASON_GAMES} m.)", oof["early_season"].astype(bool)))

    rows = []
    for label, mask in segments:
        for key in keys:
            s = summarize(oof, key, mask)
            rows.append({"segment": label, "modèle": key, "matchs": s["n_games"], **{m: s[m] for m in METRICS}})
    table = pd.DataFrame(rows).set_index(["segment", "modèle"])
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    for key in keys:
        sd = summarize(oof, key).get("seed_sd")
        if sd:
            print(
                f"Dispersion entre graines ({key}, écart-type) : log loss {sd['log_loss']:.4f}, "
                f"Brier {sd['brier']:.4f}, accuracy {sd['accuracy']:.4f}"
            )


def print_legacy(legacy: dict) -> None:
    print(
        f"Ancien découpage : test {legacy['test_season']} ({legacy['test_games']} matchs complets), "
        f"entraînement {legacy['train_games']} matchs, graine {LEGACY_SEED}"
    )
    for key, s in legacy["models"].items():
        print(f"  {key:<10} accuracy {s['accuracy']:.4f}  log loss {s['log_loss']:.4f}  Brier {s['brier']:.4f}")


def oof_path(variant: str) -> Path:
    return EVAL_DIR / f"oof_{variant}.csv"


def save_oof(oof: pd.DataFrame, variant: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = oof_path(variant)
    oof.to_csv(path, index=False)
    return path


def load_oof(variant: str) -> pd.DataFrame:
    path = oof_path(variant)
    if not path.exists():
        raise SystemExit(f"Prédictions introuvables : {path}. Lancer d'abord --variant {variant}.")
    return pd.read_csv(path, parse_dates=["game_date"])


def main():
    parser = argparse.ArgumentParser(description="Évaluation walk-forward des modèles")
    parser.add_argument("--variant", default="baseline", help="nom des prédictions sauvegardées")
    parser.add_argument("--compare", help="variante de référence à laquelle comparer celle-ci")
    parser.add_argument("--legacy", action="store_true", help="ancien découpage seulement (test 2025-26)")
    args = parser.parse_args()

    # Import ici et pas en tête de module : train_model importe ce module
    # au chargement, un import croisé en tête échouerait.
    import train_model

    df = train_model.prepare(train_model.load_features(train_model.INPUT_PATH))
    feature_cols = train_model.detect_feature_cols(df)
    specs = train_model.model_specs()
    keys = [s.key for s in specs] + ["baseline"]

    if args.legacy:
        print_legacy(legacy_split(df, feature_cols, specs))
        return

    print(f"\nWalk-forward ({len(SEEDS)} graines pour les modèles non déterministes) :")
    oof = walk_forward(df, feature_cols, specs)
    print(f"Prédictions sauvegardées : {save_oof(oof, args.variant)}\n")
    print_report(oof, keys)

    print("\nComparaison appariée, A = xgboost, B = logreg :")
    for metric in ("log_loss", "brier", "accuracy"):
        print("  " + format_comparison(paired_compare(oof, "xgboost", oof, "logreg", metric)))

    if args.compare:
        reference = load_oof(args.compare)
        for key in keys:
            if key == "baseline" or not prob_columns(reference, key):
                continue
            print(f"\nComparaison appariée, A = {args.variant}/{key}, B = {args.compare}/{key} :")
            for metric in ("log_loss", "brier", "accuracy"):
                print("  " + format_comparison(paired_compare(oof, key, reference, key, metric)))


if __name__ == "__main__":
    main()
