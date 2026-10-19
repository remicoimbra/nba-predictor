"""
player_games.py

Lecture des feuilles de match par joueur (collectors/fetch_player_logs.py)
au format attendu par features_lib (colonnes en minuscules, identifiants
entiers comme dans games_clean.csv). Partagé par feature_engineering.py et
team_state.py.
"""

from pathlib import Path

import pandas as pd

from features_lib import PLAYER_STAT_COLS

RAW_PATH = Path(__file__).resolve().parent.parent / "raw" / "player_games_raw.csv"


def load_player_games(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Feuilles de match joueur introuvables : {path}. Lancer collectors/fetch_player_logs.py.")
    df = pd.read_csv(path, parse_dates=["GAME_DATE"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"game_id": "nba_game_id"})
    for col in ("nba_game_id", "team_id", "player_id"):
        df[col] = df[col].astype("int64")
    for col in PLAYER_STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.drop_duplicates(subset=["nba_game_id", "player_id"])


def restrict_to_games(player_games: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Lignes joueur des seuls matchs de df_games (même date que le match).
    Calculer les valeurs des joueurs sur ce sous-ensemble garantit qu'un
    historique tronqué à une date (check_features.py) ne voit rien après.
    """
    games = df_games[["nba_game_id", "game_date"]]
    return player_games.drop(columns=["game_date"]).merge(games, on="nba_game_id")


def coverage_warning(player_games: pd.DataFrame, df_games: pd.DataFrame, season=None) -> str | None:
    """Message si des matchs (de `season`, ou de tout l'historique) n'ont aucune ligne joueur."""
    games = df_games if season is None else df_games[df_games["season"] == season]
    missing = games.loc[~games["nba_game_id"].isin(player_games["nba_game_id"]), "nba_game_id"]
    if missing.empty:
        return None
    return (
        f"[WARN] {len(missing)} match(s) sans feuille joueur (ex. {missing.iloc[:3].tolist()}) : "
        "effectif inconnu, lineup_delta = 0 si c'est le dernier match d'une équipe. "
        "Relancer collectors/fetch_player_logs.py --refresh."
    )
