"""
fetch_games.py

Collecte l'historique des matchs NBA sur plusieurs saisons via nba_api,
et exporte un CSV "une ligne par match" (home + away sur la même ligne),
prêt à être repris par preprocessing/clean_data.py.

Usage :
    cd data
    pip install nba_api pandas
    python collectors/fetch_games.py             # réutilise le brut en cache s'il existe
    python collectors/fetch_games.py --refresh   # re-télécharge la saison en cours (synchro quotidienne)

nba_api tape sur stats.nba.com (non officiel) : on met des pauses entre
les appels et on retry en cas d'erreur réseau/timeout, ce qui arrive
régulièrement et ne veut pas dire que le script est cassé.

⚠️ stats.nba.com bloque l'IP du VPS de production (cf. CLAUDE.md,
"Déploiement") : ce script tourne sur une machine à IP résidentielle,
via infra/sync_to_vps.ps1.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Même règle de rattachement date -> saison que le reste du pipeline
# (frontière en août), plutôt qu'une deuxième implémentation à maintenir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from features_lib import season_for_date  # type: ignore[import-not-found]  # noqa: E402

# --- Config -----------------------------------------------------------

FIRST_SEASON_START_YEAR = 2019  # 2019-20
SEASON_TYPE = "Regular Season"  # ou "Playoffs" si tu veux les inclure séparément
SLEEP_BETWEEN_CALLS = 1.5  # secondes, pour ne pas se faire bloquer
MAX_RETRIES = 3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def all_seasons(today: date | None = None) -> list[str]:
    """
    De 2019-20 jusqu'à la saison en cours INCLUSE (calculée à partir de la
    date, pas figée dans le code : une liste en dur aurait silencieusement
    ignoré 2026-27 au changement de saison).
    """
    current = season_for_date(today or date.today())
    last_start_year = int(current[:4])
    return [
        f"{y}-{(y + 1) % 100:02d}"
        for y in range(FIRST_SEASON_START_YEAR, last_start_year + 1)
    ]


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

def load_or_fetch_raw(refresh: bool = False) -> pd.DataFrame:
    """
    Sans `refresh` : réutilise le CSV brut déjà collecté s'il existe, pour
    ne pas re-taper inutilement stats.nba.com (rate limits) quand on ne
    fait que retravailler la recombinaison / les features.

    Avec `refresh` (synchro quotidienne) : re-télécharge uniquement
      - la saison en cours (nouveaux matchs joués depuis la dernière fois),
      - la dernière saison présente dans le cache (elle a pu être mise en
        cache avant sa fin : sans ça, ses derniers matchs manqueraient
        pour toujours une fois la saison suivante commencée),
      - les saisons absentes du cache,
    et garde le reste du cache tel quel.
    """
    raw_path = OUTPUT_DIR / "games_raw_all_seasons.csv"
    seasons = all_seasons()

    if not raw_path.exists():
        print(f"Collecte de {len(seasons)} saisons : {seasons}")
        df_raw = fetch_all_seasons(seasons, SEASON_TYPE)
    elif not refresh:
        print(f"Brut déjà présent, réutilisation : {raw_path}")
        return pd.read_csv(raw_path)
    else:
        # GAME_ID lu en str : les matchs fraîchement téléchargés ont des ID
        # en str ("0022600001"), un mélange int/str ferait planter le
        # groupby trié de recombine_games().
        cached = pd.read_csv(raw_path, dtype={"GAME_ID": str})
        known = set(cached["SEASON"].astype(str))
        to_refresh = {seasons[-1], max(known)} | (set(seasons) - known)
        to_fetch = [s for s in seasons if s in to_refresh]
        print(f"Brut présent, rafraîchissement de : {to_fetch}")
        fresh = fetch_all_seasons(to_fetch, SEASON_TYPE)
        fresh["GAME_ID"] = fresh["GAME_ID"].astype(str)
        df_raw = pd.concat([cached[~cached["SEASON"].isin(to_fetch)], fresh], ignore_index=True)

    df_raw.to_csv(raw_path, index=False)
    print(f"Brut sauvegardé : {raw_path} ({len(df_raw)} lignes)")
    return df_raw


def main():
    parser = argparse.ArgumentParser(description="Collecte de l'historique des matchs NBA.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-télécharge la saison en cours même si un brut est en cache (synchro quotidienne).",
    )
    args = parser.parse_args()

    df_raw = load_or_fetch_raw(refresh=args.refresh)

    df_games = recombine_games(df_raw)
    games_path = OUTPUT_DIR / "games_history.csv"
    df_games.to_csv(games_path, index=False)
    print(f"Historique des matchs sauvegardé : {games_path} ({len(df_games)} lignes)")


if __name__ == "__main__":
    main()