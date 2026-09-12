"""
clean_data.py

Nettoie games_history.csv (sortie de fetch_games.py) :
- typage correct des colonnes (dates, scores en int)
- gestion des valeurs manquantes / matchs incomplets
- tri chronologique (important : feature_engineering.py aura besoin
  d'un ordre strict par date pour calculer les moyennes glissantes
  sans fuite temporelle)
- dédoublonnage de sécurité

Usage :
    cd data
    python preprocessing/clean_data.py
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "raw" / "games_history.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "games_clean.csv"

# Colonnes de boxscore ajoutées par fetch_games.py (facultatives : si le
# CSV brut a été généré avant cette mise à jour, elles seront absentes et
# on continue sans, feature_engineering.py s'adaptera).
BOX_SCORE_COLS = [
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus",
]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Chargé : {len(df)} lignes, colonnes = {df.columns.tolist()}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Dates ---
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    n_bad_dates = df["game_date"].isna().sum()
    if n_bad_dates:
        print(f"[WARN] {n_bad_dates} dates invalides -> lignes supprimées")
        df = df.dropna(subset=["game_date"])

    # --- Scores ---
    # Certains matchs peuvent avoir un score manquant (reportés, annulés,
    # ou pas encore joués si jamais un scrap partiel de saison en cours).
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] {dropped} matchs sans score complet -> supprimés")

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # --- IDs ---
    df["home_team_id"] = df["home_team_id"].astype(int)
    df["away_team_id"] = df["away_team_id"].astype(int)

    # --- Boxscore avancé (FGA, FTA, OREB, TOV...) ---
    # Utilisé plus tard pour estimer Pace / Net Rating. On garde ces colonnes
    # en float (pas int) : une valeur manquante isolée ne doit pas faire
    # tomber tout le match, feature_engineering.py gère les NaN via shift/rolling.
    present_box_cols = [
        c for c in BOX_SCORE_COLS
        if f"home_{c}" in df.columns and f"away_{c}" in df.columns
    ]
    missing_box_cols = [c for c in BOX_SCORE_COLS if c not in present_box_cols]
    if missing_box_cols:
        print(f"[INFO] Colonnes boxscore absentes du brut (ancien fetch ?) : {missing_box_cols}")

    for c in present_box_cols:
        df[f"home_{c}"] = pd.to_numeric(df[f"home_{c}"], errors="coerce")
        df[f"away_{c}"] = pd.to_numeric(df[f"away_{c}"], errors="coerce")

    # --- Dédoublonnage de sécurité ---
    before = len(df)
    df = df.drop_duplicates(subset=["nba_game_id"])
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] {dropped} doublons sur nba_game_id -> supprimés")

    # --- Colonne cible pour la classification (utile pour l'entraînement) ---
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

    # --- Tri chronologique ---
    # Critique : feature_engineering.py doit calculer les stats "avant le
    # match" en respectant l'ordre du temps, sinon fuite temporelle
    # (le modèle "voit" des infos du futur pendant l'entraînement).
    df = df.sort_values("game_date").reset_index(drop=True)

    return df


def main():
    df_raw = load_raw(RAW_PATH)
    df_clean = clean(df_raw)

    print(f"Résultat : {len(df_clean)} matchs propres")
    print(f"Période : {df_clean['game_date'].min()} -> {df_clean['game_date'].max()}")
    print(df_clean.head())
    print(df_clean.dtypes)

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Sauvegardé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()