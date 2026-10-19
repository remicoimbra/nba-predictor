"""
train_model.py

Entraîne et compare trois modèles sur game_features.csv :
- Régression Logistique (le modèle servi par l'API)
- XGBoost, avec arrêt précoce (voir XGBoostEarlyStopping)
- Elo seul (référence sans apprentissage : la probabilité donnée par le
  rating Elo d'avant-match, voir EloOnly)

Évaluation walk-forward (voir evaluate_model.py) : chaque saison de 2021-22
à 2025-26 est prédite par un modèle entraîné sur les seules saisons qui la
précèdent. Les métriques exposées par l'API (model_metrics.json) en
viennent. Jamais de découpage aléatoire : entraîner sur le futur pour
prédire le passé serait une fuite temporelle.

Modèle servi (étape 3 du plan d'amélioration, 2026-10-19) : celui qui a le
meilleur log loss en walk-forward, la régression logistique. Elle bat
XGBoost (écart significatif, même sens sur les 5 saisons), fait jeu égal
avec l'Elo seul, et chacune de ses prédictions s'explique exactement
(features_lib.linear_contributions).

Les modèles sauvegardés sont ensuite réentraînés sur TOUTES les saisons
disponibles (2016-17 -> dernière saison), les plus anciennes pondérées à
la baisse (evaluate_model.SEASON_HALF_LIFE). C'est la situation que simule
chaque pli du walk-forward (prédire la saison suivante avec tout le passé
connu), et le modèle servi profite de la saison la plus récente.

Score prédit : déduit de la probabilité du modèle servi (score_model.py),
évalué lui aussi en walk-forward ; son paramètre sigma est écrit dans
model_metrics.json (clé score_model), où l'API le lit.

Usage :
    cd data
    python models/train_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate_model import (
    LEGACY_SEED,
    LEGACY_TEST_SEASON,
    TARGET_COL,
    WALK_FORWARD_SEASONS,
    ModelSpec,
    SEASON_HALF_LIFE,
    calibration_curve,
    complete_rows,
    evaluation_summary,
    fit_spec,
    format_comparison,
    legacy_split,
    paired_compare,
    print_legacy,
    print_report,
    summarize,
    walk_forward,
)
from score_model import (
    fit_sigma,
    latest_season_total,
    print_score_report,
    score_summary,
    walk_forward_scores,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import (  # type: ignore[import-not-found]  # noqa: E402
    DEFAULT_ELO_PARAMS,
    SCORE_WINDOW,
    elo_expected_home,
    factor_family,
    linear_contributions,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "game_features.csv"

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SERVED_MODEL = "logreg"  # modèle chargé par predictor_service.py
# Fichiers des modèles entraînés sur toutes les saisons. Seul celui du
# modèle servi part sur le VPS (deploy_to_vps.ps1) ; XGBoost est gardé pour
# comparer ses prédictions en local.
MODEL_FILES = {"logreg": "logreg.pkl", "xgboost": "xgboost.pkl"}
SAVED_XGBOOST_SEED = 42  # graine du XGBoost sauvegardé (celle de l'ancien protocole)

# Motifs des VRAIES features générées par feature_engineering.py (toutes
# calculées avec shift(1), donc sans fuite temporelle).
#
# IMPORTANT : on utilise une liste BLANCHE plutôt qu'une liste noire.
# game_features.csv contient aussi les colonnes de boxscore brut du match
# À PRÉDIRE lui-même (home_fga, away_oreb, ... propagées depuis
# games_clean.csv par le merge de back_to_wide()). Ce ne sont pas des
# features valides : ce sont des stats connues seulement APRÈS le match,
# donc les utiliser serait une fuite temporelle. Une liste noire oublierait
# facilement une nouvelle colonne brute ajoutée en amont ; la liste blanche
# est plus sûre par construction.
FEATURE_PATTERNS = (
    "win_pct",       # win_pct, win_pct_last10, diff_win_pct, diff_win_pct_last10
    "rest_days",     # home/away/diff_rest_days
    "avg_pts_scored_last",
    "avg_pts_allowed_last",
    "net_rtg_last",
    "elo",           # home_elo, away_elo, diff_elo
    "lineup_delta",  # home/away/diff_lineup_delta (effectif, depuis le 2026-10-19)
)

# Le modèle n'utilise que les écarts domicile - extérieur (diff_*) et les
# deux Elo. Élagage mesuré le 2026-10-19 en walk-forward : régression
# logistique -0,0016 de log loss [-0,0026 ; -0,0006], même sens sur les 5
# saisons ; XGBoost inchangé (dans le bruit). Les deux Elo restent : EloOnly
# en a besoin, et ils faisaient mieux que diff_* seul sur les saisons de
# réglage. Les autres colonnes home_/away_ sont toujours calculées (l'API
# les affiche sur la fiche match). diff_lineup_delta (étape 6) porte le
# total à 12 features.
MODEL_LEVEL_COLS = ("home_elo", "away_elo")

# XGBoost : nombre d'arbres choisi par arrêt précoce (voir XGBoostEarlyStopping).
XGB_VALIDATION_FRACTION = 0.2
XGB_MAX_TREES = 3000
XGB_PATIENCE = 100


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    print(f"Chargé : {len(df)} matchs")
    return df


def candidate_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Toutes les colonnes de features valides générées par feature_engineering.py
    (home_*, away_*, diff_* qui correspondent à FEATURE_PATTERNS), détectées
    automatiquement pour ne pas avoir à maintenir une liste en dur.
    preprocessing/check_features.py les contrôle toutes.
    """
    return [
        c for c in df.columns
        if (c.startswith("home_") or c.startswith("away_") or c.startswith("diff_"))
        and any(pattern in c for pattern in FEATURE_PATTERNS)
    ]


def detect_feature_cols(df: pd.DataFrame) -> list[str]:
    """Features du modèle : parmi les candidates, les diff_* et MODEL_LEVEL_COLS."""
    cols = [c for c in candidate_feature_cols(df) if c.startswith("diff_") or c in MODEL_LEVEL_COLS]
    print(f"Features du modèle ({len(cols)}) : {cols}")
    return cols


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ne retire que les matchs sans résultat. Les matchs dont une feature
    manque (premier match de chaque équipe dans la saison : win_pct pas
    encore défini) restent dans le jeu de données : l'évaluation les prédit,
    comme le fera le modèle en production. L'entraînement les écarte
    (evaluate_model.complete_rows()).
    """
    df = df.dropna(subset=[TARGET_COL])
    print(f"Dataset utilisable : {len(df)} matchs")
    return df


def make_logreg() -> Pipeline:
    # SimpleImputer : l'entraînement ne voit que des matchs complets, il ne
    # sert qu'à la prédiction d'un match où une feature manque (win_pct en
    # tout début de saison), remplacée par la médiane de l'entraînement.
    #
    # StandardScaler avant la régression logistique : sans ça, les features
    # à grande échelle (l'Elo tourne autour de 1300-1700) se retrouvent avec
    # des coefficients écrasés par la régularisation, même si elles sont
    # très informatives. Les coefficients redeviennent aussi directement
    # comparables entre eux, et le ConvergenceWarning disparaît.
    #
    # features_lib.linear_contributions() dépend de ces trois étapes et de
    # leurs noms.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


class XGBoostEarlyStopping:
    """
    XGBoost dont le nombre d'arbres est choisi par arrêt précoce : un premier
    modèle est entraîné sans les XGB_VALIDATION_FRACTION derniers matchs et
    s'arrête quand son log loss sur ces matchs ne baisse plus depuis
    XGB_PATIENCE arbres ; le modèle final est réentraîné sur tous les matchs
    avec ce nombre d'arbres. Les lignes de X doivent être dans l'ordre
    chronologique (celui de game_features.csv), pour que la validation
    porte sur les matchs les plus récents.

    Avant le 2026-10-19 : 300 arbres fixes de profondeur 3, qui
    surapprenaient. Arrêt précoce + profondeur 2 (réglés sur 2021-22 ->
    2023-24, ~150 arbres) : -0,0113 de log loss [-0,0152 ; -0,0076] en
    walk-forward, même sens sur les 5 saisons. XGBoost reste moins bon que
    la régression logistique.
    """

    def __init__(self, seed: int):
        self.seed = seed

    def _new_model(self, **kwargs) -> XGBClassifier:
        return XGBClassifier(
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.seed,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        n_val = int(len(X) * XGB_VALIDATION_FRACTION)
        w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight)
        probe = self._new_model(n_estimators=XGB_MAX_TREES, early_stopping_rounds=XGB_PATIENCE)
        probe.fit(
            X.iloc[:-n_val], y.iloc[:-n_val], sample_weight=w[:-n_val],
            eval_set=[(X.iloc[-n_val:], y.iloc[-n_val:])], verbose=False,
        )
        self.n_trees_ = probe.best_iteration + 1
        self.model_ = self._new_model(n_estimators=self.n_trees_)
        self.model_.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)


class EloOnly:
    """
    « Modèle » Elo seul : la probabilité features_lib.elo_expected_home des
    ratings d'avant-match (home_elo, away_elo), avec les paramètres Elo qui
    ont servi à calculer game_features.csv. Rien à apprendre (fit ne fait
    rien) : c'est la référence que la régression logistique et XGBoost, qui
    reçoivent l'Elo parmi leurs features, doivent battre. Même probabilité
    que elo_home_probability dans l'API.
    """

    def __init__(self, params=DEFAULT_ELO_PARAMS):
        self.params = params

    def fit(self, X, y):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = elo_expected_home(
            X["home_elo"].to_numpy(dtype=float), X["away_elo"].to_numpy(dtype=float), self.params
        )
        return np.column_stack([1 - p, p])


def model_specs() -> list[ModelSpec]:
    """Modèles évalués (walk-forward, ancien découpage), le modèle servi en premier."""
    return [
        ModelSpec("logreg", "Régression logistique", lambda seed: make_logreg(), weight_param="clf__sample_weight"),
        ModelSpec("xgboost", "XGBoost", XGBoostEarlyStopping, stochastic=True, weight_param="sample_weight"),
        ModelSpec("elo", "Elo seul", lambda seed: EloOnly()),
    ]


def factor_importances(model: Pipeline, X: pd.DataFrame) -> list[dict]:
    """
    Poids de chaque famille de features dans les prédictions de la régression
    logistique : moyenne, sur les matchs X, de |somme des contributions de la
    famille| (en log-odds), et part du total. Mêmes familles que les facteurs
    de la fiche match (features_lib.FACTOR_FAMILIES) : sommer par famille
    neutralise aussi les colonnes redondantes (diff_elo = home_elo - away_elo).
    """
    contribs, _ = linear_contributions(model, X)
    totals: dict = {}
    for column, values in zip(X.columns, contribs.T):
        key, label = factor_family(column)
        entry = totals.setdefault(key, {"key": key, "label": label, "values": 0.0})
        entry["values"] = entry["values"] + values
    weights = {key: float(np.mean(np.abs(e["values"]))) for key, e in totals.items()}
    total = sum(weights.values())
    return sorted(
        (
            {"key": key, "label": e["label"], "mean_abs_contribution": weights[key], "share": weights[key] / total}
            for key, e in totals.items()
        ),
        key=lambda f: f["share"],
        reverse=True,
    )


def print_logreg(model: Pipeline, feature_cols: list[str]) -> None:
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_cols).sort_values()
    print("\nRégression Logistique — coefficients (features standardisées, donc comparables) :")
    print(coefs)


def print_xgboost(model: XGBoostEarlyStopping, feature_cols: list[str]) -> None:
    importances = pd.Series(model.model_.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nXGBoost — {model.n_trees_} arbres (arrêt précoce), importance des features :")
    print(importances)


def main():
    df = prepare(load_features(INPUT_PATH))
    feature_cols = detect_feature_cols(df)
    incomplete = int(df[feature_cols].isna().any(axis=1).sum())
    print(f"Dont {incomplete} avec au moins une feature manquante (prédits par l'évaluation, exclus de l'entraînement)")

    specs = model_specs()
    spec_of = {s.key: s for s in specs}
    keys = [s.key for s in specs] + ["baseline"]

    print(f"\n=== Évaluation walk-forward ({WALK_FORWARD_SEASONS[0]} -> {WALK_FORWARD_SEASONS[-1]}) ===")
    oof = walk_forward(df, feature_cols, specs)
    print()
    print_report(oof, keys)
    comparisons = {}
    for other in (s.key for s in specs if s.key != SERVED_MODEL):
        comparisons[f"{SERVED_MODEL}_vs_{other}"] = {
            m: paired_compare(oof, SERVED_MODEL, oof, other, m) for m in ("log_loss", "brier", "accuracy")
        }
        print(f"\nComparaison appariée, A = {SERVED_MODEL}, B = {other} :")
        for c in comparisons[f"{SERVED_MODEL}_vs_{other}"].values():
            print("  " + format_comparison(c))

    print(f"\n=== Score prédit, walk-forward (probabilités du modèle servi : {SERVED_MODEL}) ===")
    scores = score_summary(walk_forward_scores(df, feature_cols, spec_of[SERVED_MODEL]))
    print_score_report(scores)

    print(f"\n=== Ancien découpage (test {LEGACY_TEST_SEASON}, matchs complets, graine {LEGACY_SEED}) ===")
    legacy = legacy_split(df, feature_cols, specs)
    print_legacy(legacy)

    train = complete_rows(df, feature_cols)
    print(
        f"\n=== Modèles réentraînés sur toutes les saisons ({len(train)} matchs complets, "
        f"{train['season'].min()} -> {train['season'].max()}, demi-vie {SEASON_HALF_LIFE:g} saisons) ==="
    )
    logreg = fit_spec(spec_of["logreg"], 0, train, feature_cols)
    print_logreg(logreg, feature_cols)
    xgb = fit_spec(spec_of["xgboost"], SAVED_XGBOOST_SEED, train, feature_cols)
    print_xgboost(xgb, feature_cols)
    importances = factor_importances(logreg, train[feature_cols])
    print("\nRégression Logistique — poids de chaque famille de features :")
    for f in importances:
        print(f"  {f['label']:<28} {f['share']:6.1%}  (|contribution| moyenne {f['mean_abs_contribution']:.3f})")
    served_model = {"logreg": logreg, "xgboost": xgb}[SERVED_MODEL]
    score_model = {
        "sigma": fit_sigma(served_model, train, feature_cols),
        "fallback_total": latest_season_total(df),
        "window": SCORE_WINDOW,
        "evaluation": scores,
    }
    print(f"\nScore prédit : sigma {score_model['sigma']:.2f}, total de repli {score_model['fallback_total']:.1f}")

    joblib.dump(logreg, MODEL_DIR / MODEL_FILES["logreg"])
    # Le XGBClassifier lui-même, pas l'enveloppe : le recharger ne doit pas
    # dépendre de ce module.
    joblib.dump(xgb.model_, MODEL_DIR / MODEL_FILES["xgboost"])

    # Sauvegarde de l'ordre exact des features utilisées à l'entraînement.
    # Les modèles sont sensibles à l'ordre des colonnes : un DataFrame de
    # prédiction construit dans un ordre différent peut donner un résultat
    # silencieusement faux. En figeant cette liste ici, predictor_service.py
    # n'a jamais à deviner ou à recopier la sélection à la main.
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Métriques exposées par l'API (GET /model/metrics) pour la page
    # « Le modèle » du front : régénérées à chaque entraînement, jamais
    # recopiées à la main. Toutes viennent du walk-forward, sauf
    # factor_importances (modèle servi) et evaluation.legacy.
    pooled = {k: summarize(oof, k) for k in keys}
    evaluation = evaluation_summary(oof, keys)
    evaluation["comparisons"] = comparisons
    evaluation["legacy"] = legacy
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_season": evaluation["test_seasons"][-1],
        "train_games": int(len(train)),    # matchs d'entraînement du modèle servi
        "train_seasons": [str(train["season"].min()), str(train["season"].max())],
        "season_half_life": SEASON_HALF_LIFE,
        "test_games": evaluation["n_games"],  # matchs prédits par le walk-forward
        "baseline_accuracy": pooled["baseline"]["accuracy"],
        "baseline": pooled["baseline"],
        "served_model": SERVED_MODEL,
        # Paramètres Elo de game_features.csv (ceux du code au moment de
        # l'entraînement : check_features.py vérifie que le fichier est à
        # jour). predictor_service.py les compare à ceux de team_state.json.
        "elo_params": DEFAULT_ELO_PARAMS.as_dict(),
        "models": [{"key": s.key, "name": s.name, **pooled[s.key]} for s in specs],
        "evaluation": evaluation,
        "factor_importances": importances,
        "calibration": calibration_curve(oof, SERVED_MODEL),
        # Lu par predictor_service.py (sigma, fallback_total) : obligatoire.
        "score_model": score_model,
    }
    with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModèles sauvegardés dans : {MODEL_DIR} (servi : {MODEL_FILES[SERVED_MODEL]})")
    print(f"Ordre des features sauvegardé : {MODEL_DIR / 'feature_columns.json'}")


if __name__ == "__main__":
    main()
