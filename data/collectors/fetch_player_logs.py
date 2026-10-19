"""
fetch_player_logs.py

Collecte les feuilles de match PAR JOUEUR (saison régulière) via nba_api
(LeagueGameLog, une requête par saison) : une ligne par joueur entré en jeu
et par match. Un joueur absent (blessé, mis au repos) n'a pas de ligne.
Sert à l'indicateur d'effectif lineup_delta (features_lib.py), calculé par
preprocessing/feature_engineering.py et preprocessing/team_state.py.

Usage :
    cd data
    python collectors/fetch_player_logs.py             # réutilise le cache s'il existe
    python collectors/fetch_player_logs.py --refresh   # re-télécharge la saison en cours (synchro quotidienne)

Même domaine que fetch_games.py (stats.nba.com), donc mêmes contraintes :
bloqué depuis le VPS, tourne sur le PC via infra/sync_to_vps.ps1.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from fetch_games import MAX_RETRIES, SEASON_TYPE, SLEEP_BETWEEN_CALLS, all_seasons

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "raw" / "player_games_raw.csv"

# Colonnes conservées (le reste : pourcentages, fantasy, vidéo, inutiles ici).
KEEP_COLS = [
    "SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "MIN",
    "PTS", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF",
]


def fetch_season(season: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=SEASON_TYPE,
                player_or_team_abbreviation="P",
                league_id="00",
                timeout=60,
            ).get_data_frames()[0]
            df["SEASON"] = season
            print(f"[OK] {season} : {len(df)} lignes joueur-match, {df['GAME_ID'].nunique()} matchs")
            return df[KEEP_COLS]
        except Exception as e:
            last_error = e
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {season} a échoué : {e}")
            time.sleep(SLEEP_BETWEEN_CALLS * attempt)
    raise RuntimeError(f"Échec définitif pour {season} : {last_error}")


def fetch_seasons(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        frames.append(fetch_season(season))
        time.sleep(SLEEP_BETWEEN_CALLS)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Collecte des feuilles de match par joueur.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-télécharge la saison en cours même si un cache existe (synchro quotidienne).",
    )
    args = parser.parse_args()

    seasons = all_seasons()
    if not OUTPUT_PATH.exists():
        print(f"Collecte de {len(seasons)} saisons : {seasons}")
        df = fetch_seasons(seasons)
    elif not args.refresh:
        print(f"Cache déjà présent, rien à faire : {OUTPUT_PATH}")
        return
    else:
        # Même règle que fetch_games.load_or_fetch_raw : saison en cours,
        # dernière saison du cache (peut-être mise en cache avant sa fin) et
        # saisons manquantes.
        cached = pd.read_csv(OUTPUT_PATH, dtype={"GAME_ID": str})
        known = set(cached["SEASON"].astype(str))
        to_refresh = {seasons[-1], max(known)} | (set(seasons) - known)
        to_fetch = [s for s in seasons if s in to_refresh]
        print(f"Cache présent, rafraîchissement de : {to_fetch}")
        df = pd.concat([cached[~cached["SEASON"].isin(to_fetch)], fetch_seasons(to_fetch)], ignore_index=True)

    df["GAME_ID"] = df["GAME_ID"].astype(str)
    # Écriture atomique : un fichier à moitié écrit casserait la synchro suivante.
    tmp = OUTPUT_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT_PATH)
    print(f"Sauvegardé : {OUTPUT_PATH} ({len(df)} lignes, {df['GAME_ID'].nunique()} matchs)")


if __name__ == "__main__":
    main()
