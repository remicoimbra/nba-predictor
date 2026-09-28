"""
train_model.py

Entraîne et compare deux modèles sur game_features.csv :
- Régression Logistique (baseline, déjà validée : ~66,75% accuracy)
- XGBoost (attendu : mieux capturer les interactions non-linéaires entre
  features, notamment maintenant qu'on a Net Rating / forme sur 10 matchs)

Split train/test PAR SAISON (pas aléatoire) : entraîner sur des saisons
passées et tester sur une saison plus récente, jamais l'inverse, sinon
c'est de la fuite temporelle (le modèle "connaît" déjà le futur).

Usage :
    cd data
    pip install scikit-learn xgboost joblib
    python models/train_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "processed"
INPUT_PATH = PROCESSED_DIR / "game_features.csv"

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "home_win"
CALIBRATION_BINS = 10  # tranches de probabilité pour la courbe de calibration
TEST_SEASON = "2025-26"  # saison la plus récente et complète -> jeu de test

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


def prepare(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=feature_cols + [TARGET_COL])
    dropped = before - len(df)
    print(f"Lignes avec features manquantes supprimées : {dropped}")
    print(f"Dataset final utilisable : {len(df)} matchs")
    return df


def split_by_season(df: pd.DataFrame):
    train = df[df["season"] != TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    print(f"Train : {len(train)} matchs (saisons != {TEST_SEASON})")
    print(f"Test  : {len(test)} matchs (saison {TEST_SEASON})")
    return train, test


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} : résultats sur le jeu de test ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Log loss : {ll:.4f}")
    print("Matrice de confusion :")
    print(cm)

    return {"name": name, "accuracy": acc, "log_loss": ll}


def calibration_curve(y_true, y_proba, n_bins: int = CALIBRATION_BINS) -> list[dict]:
    """
    Pour chaque tranche de probabilité prédite (0-10 %, 10-20 %...) : proba
    moyenne prédite, taux de victoire réel à domicile, nombre de matchs.
    Un modèle bien calibré a des points proches de la diagonale (quand il
    annonce 70 %, l'équipe à domicile gagne ~70 % du temps). Tranches vides
    omises.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bins = np.minimum((y_proba * n_bins).astype(int), n_bins - 1)
    curve = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        curve.append({
            "bin_start": b / n_bins,
            "bin_end": (b + 1) / n_bins,
            "mean_predicted": float(y_proba[mask].mean()),
            "actual_rate": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return curve


def train_logreg(X_train, y_train, feature_cols):
    # StandardScaler avant la régression logistique : sans ça, les features
    # à grande échelle (l'Elo tourne autour de 1300-1700) se retrouvent avec
    # des coefficients écrasés par la régularisation, même si elles sont
    # très informatives - ce n'était pas visible avant l'Elo car toutes les
    # autres features étaient déjà sur des échelles proches (0-1 ou 0-40).
    # Bénéfice supplémentaire : les coefficients redeviennent directement
    # comparables entre eux (même échelle), et ça corrige aussi le
    # ConvergenceWarning qu'on traînait depuis le début.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_cols).sort_values()
    print("\nRégression Logistique — coefficients (features standardisées, donc comparables) :")
    print(coefs)

    return model


def train_xgboost(X_train, y_train, feature_cols):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nXGBoost — importance des features :")
    print(importances)

    return model


def main():
    df = load_features(INPUT_PATH)
    feature_cols = detect_feature_cols(df)
    df = prepare(df, feature_cols)
    train, test = split_by_season(df)

    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    # Point de repère : une baseline "on prédit toujours l'équipe à
    # domicile gagnante" tourne historiquement autour de 55-60% en NBA.
    home_baseline_acc = y_test.mean()
    print(f"\nBaseline 'home gagne toujours' : {home_baseline_acc:.4f}")

    logreg = train_logreg(X_train, y_train, feature_cols)
    logreg_results = evaluate("Régression Logistique", logreg, X_test, y_test)

    xgb = train_xgboost(X_train, y_train, feature_cols)
    xgb_results = evaluate("XGBoost", xgb, X_test, y_test)

    print("\n=== Comparaison finale ===")
    comparison = pd.DataFrame([logreg_results, xgb_results]).set_index("name")
    comparison.loc["Baseline (home gagne toujours)"] = [home_baseline_acc, None]
    print(comparison)

    joblib.dump(logreg, MODEL_DIR / "logreg_baseline.pkl")
    joblib.dump(xgb, MODEL_DIR / "xgboost_v1.pkl")

    # Sauvegarde de l'ordre exact des features utilisées à l'entraînement.
    # XGBoost (et le pipeline scaler+logreg) sont sensibles à l'ordre des
    # colonnes : un DataFrame de prédiction construit dans un ordre
    # différent peut donner un résultat silencieusement faux. En figeant
    # cette liste ici, predictor_service.py n'a jamais à deviner ou à
    # recopier `FEATURE_PATTERNS` à la main (source de désynchronisation
    # si on ajoute une feature plus tard sans mettre à jour les deux côtés).
    with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Métriques exposées par l'API (GET /model/metrics) pour la page
    # « Le modèle » du front : régénérées à chaque entraînement, jamais
    # recopiées à la main.
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_season": TEST_SEASON,
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "baseline_accuracy": float(home_baseline_acc),
        "models": [
            {"key": "xgboost", "name": "XGBoost", "accuracy": float(xgb_results["accuracy"]),
             "log_loss": float(xgb_results["log_loss"])},
            {"key": "logreg", "name": "Régression logistique", "accuracy": float(logreg_results["accuracy"]),
             "log_loss": float(logreg_results["log_loss"])},
        ],
        "feature_importances": {
            col: float(v) for col, v in sorted(
                zip(feature_cols, xgb.feature_importances_), key=lambda kv: kv[1], reverse=True
            )
        },
        "calibration": calibration_curve(y_test, xgb.predict_proba(X_test)[:, 1]),
    }
    with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModèles sauvegardés dans : {MODEL_DIR}")
    print(f"Ordre des features sauvegardé : {MODEL_DIR / 'feature_columns.json'}")


if __name__ == "__main__":
    main()