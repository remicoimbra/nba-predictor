"""
train_model.py

Entraîne et compare deux modèles sur game_features.csv :
- Régression Logistique
- XGBoost (le modèle servi par l'API)

Évaluation walk-forward (voir evaluate_model.py) : chaque saison de 2021-22
à 2025-26 est prédite par un modèle entraîné sur les seules saisons qui la
précèdent. Les métriques exposées par l'API (model_metrics.json) en
viennent. Jamais de découpage aléatoire : entraîner sur le futur pour
prédire le passé serait une fuite temporelle.

Les modèles sauvegardés pour l'API sont ensuite réentraînés sur TOUTES les
saisons disponibles. C'est la situation que simule chaque pli du
walk-forward (prédire la saison suivante avec tout le passé connu), et le
modèle servi profite de la saison la plus récente, qu'il ignorait avant.

Usage :
    cd data
    python models/train_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
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
    calibration_curve,
    complete_rows,
    evaluation_summary,
    format_comparison,
    legacy_split,
    paired_compare,
    print_legacy,
    print_report,
    summarize,
    walk_forward,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "game_features.csv"

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SERVED_SEED = 42  # graine du XGBoost servi par l'API (celle de l'ancien protocole)
SERVED_MODEL = "xgboost"  # modèle chargé par predictor_service.py

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
)


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    print(f"Chargé : {len(df)} matchs")
    return df


def detect_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Détecte automatiquement les colonnes de features générées par
    feature_engineering.py (home_*, away_*, diff_*), pour ne pas avoir à
    maintenir une liste en dur qui casse dès qu'on ajoute une feature.
    """
    cols = [
        c for c in df.columns
        if (c.startswith("home_") or c.startswith("away_") or c.startswith("diff_"))
        and any(pattern in c for pattern in FEATURE_PATTERNS)
    ]
    print(f"Features détectées ({len(cols)}) : {cols}")
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
    # XGBoost gère ces NaN nativement.
    #
    # StandardScaler avant la régression logistique : sans ça, les features
    # à grande échelle (l'Elo tourne autour de 1300-1700) se retrouvent avec
    # des coefficients écrasés par la régularisation, même si elles sont
    # très informatives. Les coefficients redeviennent aussi directement
    # comparables entre eux, et le ConvergenceWarning disparaît.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def make_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )


def model_specs() -> list[ModelSpec]:
    """Modèles évalués (walk-forward, ancien découpage) puis entraînés pour l'API."""
    return [
        ModelSpec("xgboost", "XGBoost", make_xgboost, stochastic=True),
        ModelSpec("logreg", "Régression logistique", lambda seed: make_logreg()),
    ]


def train_logreg(X, y) -> Pipeline:
    model = make_logreg()
    model.fit(X, y)
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns).sort_values()
    print("\nRégression Logistique — coefficients (features standardisées, donc comparables) :")
    print(coefs)
    return model


def train_xgboost(X, y, feature_cols) -> XGBClassifier:
    model = make_xgboost(SERVED_SEED)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nXGBoost — importance des features :")
    print(importances)
    return model


def main():
    df = prepare(load_features(INPUT_PATH))
    feature_cols = detect_feature_cols(df)
    incomplete = int(df[feature_cols].isna().any(axis=1).sum())
    print(f"Dont {incomplete} avec au moins une feature manquante (prédits par l'évaluation, exclus de l'entraînement)")

    specs = model_specs()
    keys = [s.key for s in specs] + ["baseline"]

    print(f"\n=== Évaluation walk-forward ({WALK_FORWARD_SEASONS[0]} -> {WALK_FORWARD_SEASONS[-1]}) ===")
    oof = walk_forward(df, feature_cols, specs)
    print()
    print_report(oof, keys)
    comparisons = {m: paired_compare(oof, "xgboost", oof, "logreg", m) for m in ("log_loss", "brier", "accuracy")}
    print("\nComparaison appariée, A = xgboost, B = logreg :")
    for c in comparisons.values():
        print("  " + format_comparison(c))

    print(f"\n=== Ancien découpage (test {LEGACY_TEST_SEASON}, matchs complets, graine {LEGACY_SEED}) ===")
    legacy = legacy_split(df, feature_cols, specs)
    print_legacy(legacy)

    train = complete_rows(df, feature_cols)
    print(f"\n=== Modèles servis : réentraînés sur toutes les saisons ({len(train)} matchs complets) ===")
    X, y = train[feature_cols], train[TARGET_COL]
    logreg = train_logreg(X, y)
    xgb = train_xgboost(X, y, feature_cols)

    joblib.dump(logreg, MODEL_DIR / "logreg_baseline.pkl")
    joblib.dump(xgb, MODEL_DIR / "xgboost_v1.pkl")

    # Sauvegarde de l'ordre exact des features utilisées à l'entraînement.
    # XGBoost (et le pipeline de la régression logistique) sont sensibles à
    # l'ordre des colonnes : un DataFrame de prédiction construit dans un
    # ordre différent peut donner un résultat silencieusement faux. En
    # figeant cette liste ici, predictor_service.py n'a jamais à deviner ou
    # à recopier `FEATURE_PATTERNS` à la main.
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Métriques exposées par l'API (GET /model/metrics) pour la page
    # « Le modèle » du front : régénérées à chaque entraînement, jamais
    # recopiées à la main. Toutes viennent du walk-forward, sauf
    # feature_importances (modèle servi) et evaluation.legacy.
    pooled = {k: summarize(oof, k) for k in keys}
    evaluation = evaluation_summary(oof, keys)
    evaluation["comparisons"] = {"xgboost_vs_logreg": comparisons}
    evaluation["legacy"] = legacy
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_season": evaluation["test_seasons"][-1],
        "train_games": int(len(train)),    # matchs d'entraînement du modèle servi
        "test_games": evaluation["n_games"],  # matchs prédits par le walk-forward
        "baseline_accuracy": pooled["baseline"]["accuracy"],
        "baseline": pooled["baseline"],
        "served_model": SERVED_MODEL,
        "models": [{"key": s.key, "name": s.name, **pooled[s.key]} for s in specs],
        "evaluation": evaluation,
        "feature_importances": {
            col: float(v) for col, v in sorted(
                zip(feature_cols, xgb.feature_importances_), key=lambda kv: kv[1], reverse=True
            )
        },
        "calibration": calibration_curve(oof, SERVED_MODEL),
    }
    with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModèles sauvegardés dans : {MODEL_DIR}")
    print(f"Ordre des features sauvegardé : {MODEL_DIR / 'feature_columns.json'}")


if __name__ == "__main__":
    main()
