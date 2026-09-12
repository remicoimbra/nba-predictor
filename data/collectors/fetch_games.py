"""
fetch_games.py

Collecte l'historique des matchs NBA sur plusieurs saisons via nba_api,
et exporte un CSV "une ligne par match" (home + away sur la même ligne),
prêt à être repris par preprocessing/clean_data.py.

Usage :
    cd data
    pip install nba_api pandas
    python collectors/fetch_games.py

nba_api tape sur stats.nba.com (non officiel) : on met des pauses entre
les appels et on retry en cas d'erreur réseau/timeout, ce qui arrive
régulièrement et ne veut pas dire que le script est cassé.
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# --- Config -----------------------------------------------------------

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SEASON_TYPE = "Regular Season"  # ou "Playoffs" si tu veux les inclure séparément
SLEEP_BETWEEN_CALLS = 1.5  # secondes, pour ne pas se faire bloquer
MAX_RETRIES = 3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Collecte brute -----------------------------------------------------

def fetch_season_raw(season: str, season_type: str) -> pd.DataFrame:
    """Récupère les données brutes (1 ligne par équipe par match) pour une saison."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",
            )
            df = gamefinder.get_data_frames()[0]
            print(f"[OK] {season} ({season_type}) : {len(df)} lignes")
            return df
        except Exception as e:
            last_error = e
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {season} a échoué : {e}")
            time.sleep(SLEEP_BETWEEN_CALLS * attempt)  # backoff simple
    raise RuntimeError(f"Échec définitif pour {season} : {last_error}")


def fetch_all_seasons(seasons: list[str], season_type: str) -> pd.DataFrame:
    frames = []
    for season in seasons:
        df = fetch_season_raw(season, season_type)
        df["SEASON"] = season  # au cas où la colonne native ne suffit pas
        frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return pd.concat(frames, ignore_index=True)


# --- Recombinaison en 1 ligne par match ---------------------------------

# Colonnes de boxscore fournies par LeagueGameFinder qu'on veut conserver
# en plus du score, pour pouvoir calculer Pace / Net Rating plus tard
# dans feature_engineering.py.
BOX_SCORE_COLS = [
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
]


def recombine_games(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    LeagueGameFinder renvoie 2 lignes par match (une par équipe).
    On les recombine en 1 ligne : home_team, away_team, home_score, away_score,
    + les stats de boxscore de chaque équipe (préfixées home_/away_), qui
    serviront à estimer Pace et Net Rating dans feature_engineering.py.
    La colonne MATCHUP contient "vs." pour l'équipe qui joue à domicile
    et "@" pour l'équipe à l'extérieur.
    """
    rows = []
    skipped = 0

    for game_id, group in df_raw.groupby("GAME_ID"):
        if len(group) != 2:
            # Cas rare : ligne orpheline (match annulé, données incomplètes...)
            skipped += 1
            continue

        home_row = group[group["MATCHUP"].str.contains("vs.", regex=False)]
        away_row = group[group["MATCHUP"].str.contains("@", regex=False)]

        if home_row.empty or away_row.empty:
            skipped += 1
            continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        row = {
            "nba_game_id": game_id,
            "game_date": home_row["GAME_DATE"],
            "season": home_row.get("SEASON", None),
            "home_team_id": home_row["TEAM_ID"],
            "home_team_abbr": home_row["TEAM_ABBREVIATION"],
            "away_team_id": away_row["TEAM_ID"],
            "away_team_abbr": away_row["TEAM_ABBREVIATION"],
            "home_score": home_row["PTS"],
            "away_score": away_row["PTS"],
            "status": "Final",  # LeagueGameFinder ne renvoie que des matchs joués
        }

        for col in BOX_SCORE_COLS:
            row[f"home_{col.lower()}"] = home_row.get(col, None)
            row[f"away_{col.lower()}"] = away_row.get(col, None)

        rows.append(row)

    print(f"Matchs recombinés : {len(rows)} | lignes ignorées (orphelines) : {skipped}")
    return pd.DataFrame(rows)


# --- Main -----------------------------------------------------------------

def load_or_fetch_raw() -> pd.DataFrame:
    """
    Réutilise le CSV brut déjà collecté s'il existe, pour ne pas re-taper
    inutilement stats.nba.com (rate limits) quand on ne fait que retravailler
    la recombinaison / les features.
    """
    raw_path = OUTPUT_DIR / "games_raw_all_seasons.csv"
    if raw_path.exists():
        print(f"Brut déjà présent, réutilisation : {raw_path}")
        return pd.read_csv(raw_path)

    print(f"Collecte de {len(SEASONS)} saisons : {SEASONS}")
    df_raw = fetch_all_seasons(SEASONS, SEASON_TYPE)
    df_raw.to_csv(raw_path, index=False)
    print(f"Brut sauvegardé : {raw_path} ({len(df_raw)} lignes)")
    return df_raw


def main():
    df_raw = load_or_fetch_raw()

    df_games = recombine_games(df_raw)
    games_path = OUTPUT_DIR / "games_history.csv"
    df_games.to_csv(games_path, index=False)
    print(f"Historique des matchs sauvegardé : {games_path} ({len(df_games)} lignes)")


if __name__ == "__main__":
    main()