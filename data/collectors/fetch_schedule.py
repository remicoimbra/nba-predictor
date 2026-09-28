"""
fetch_schedule.py

Récupère le calendrier NBA (matchs programmés, PAS l'historique — c'est le
rôle de fetch_games.py) sur une fenêtre de dates autour d'aujourd'hui, et
l'écrit dans data/processed/schedule.json. C'est ce fichier que lit
api/app/services/schedule_service.py.

Pourquoi un fichier plutôt qu'un appel NBA à chaque requête de l'API :
stats.nba.com ET cdn.nba.com bloquent l'IP du VPS de production (testé le
2026-09-28, cf. CLAUDE.md "Déploiement"). Ce script tourne donc sur une
machine à IP résidentielle (infra/sync_to_vps.ps1), et l'API sur le VPS ne
fait plus que lire le fichier produit.

Usage :
    cd data
    python collectors/fetch_schedule.py                        # J-3 -> J+14
    python collectors/fetch_schedule.py --from 2026-01-10 --to 2026-01-20

Format de sortie :
    {
      "generated_at": "...",
      "dates": {
        "2026-10-20": [
          {"game_id": "...", "game_time_utc": "...", "status": "7:00 pm ET",
           "home_team_id": 1610612738, "away_team_id": 1610612743}
        ],
        "2026-10-21": []          # date couverte, mais aucun match
      }
    }
Une date ABSENTE de "dates" veut dire "pas synchronisée", à ne pas
confondre avec une liste vide ("synchronisée, aucun match ce jour-là").
Les dates déjà présentes dans le fichier et hors de la fenêtre demandée
sont conservées (utile pour garder un --from/--to de test).
"""

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from nba_api.stats.endpoints import scoreboardv3

PAST_DAYS = 3      # statut "Final" des derniers matchs
FUTURE_DAYS = 14   # matchs à venir affichables sans nouvelle synchro
SLEEP_BETWEEN_CALLS = 0.8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "processed" / "schedule.json"


def parse_scoreboard(raw: dict) -> list[dict]:
    """
    Extrait les matchs du JSON brut de ScoreboardV3. Lecture volontaire du
    JSON (get_dict()) plutôt que des DataFrames de l'endpoint : `homeTeam`/
    `awayTeam` y sont labellisés explicitement, alors que le dataset
    `LineScore` ne les distingue que par l'ordre d'insertion.
    """
    games = []
    for g in raw.get("scoreboard", {}).get("games", []):
        games.append(
            {
                "game_id": str(g.get("gameId")),
                "game_time_utc": g.get("gameTimeUTC") or "",
                "status": str(g.get("gameStatusText", "")),
                "home_team_id": int(g["homeTeam"]["teamId"]),
                "away_team_id": int(g["awayTeam"]["teamId"]),
            }
        )
    return games


def fetch_day(target_date: date, timeout: int = REQUEST_TIMEOUT) -> list[dict]:
    """Matchs programmés à `target_date` (liste vide si aucun). Un seul essai."""
    board = scoreboardv3.ScoreboardV3(
        game_date=target_date.strftime("%Y-%m-%d"), league_id="00", timeout=timeout
    )
    return parse_scoreboard(board.get_dict())


def fetch_day_with_retry(target_date: date) -> list[dict]:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_day(target_date)
        except Exception as e:
            last_error = e
            print(f"[RETRY {attempt}/{MAX_RETRIES}] {target_date} a échoué : {e}")
            time.sleep(SLEEP_BETWEEN_CALLS * 2 * attempt)
    raise RuntimeError(f"Échec définitif pour {target_date} : {last_error}")


def load_existing() -> dict:
    if not OUTPUT_PATH.exists():
        return {}
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        return json.load(f).get("dates", {})


def main():
    today = date.today()
    parser = argparse.ArgumentParser(description="Calendrier NBA -> data/processed/schedule.json")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, default=today - timedelta(days=PAST_DAYS))
    parser.add_argument("--to", dest="end", type=date.fromisoformat, default=today + timedelta(days=FUTURE_DAYS))
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--to doit être postérieur ou égal à --from")

    # Toute la fenêtre doit réussir, sinon on ne touche pas au fichier : un
    # échec (réseau, rate limit) ne doit pas écraser un calendrier valide
    # par un calendrier troué, qui serait ensuite poussé sur le VPS.
    fetched = {}
    d = args.start
    while d <= args.end:
        games = fetch_day_with_retry(d)
        fetched[d.isoformat()] = games
        print(f"[OK] {d} : {len(games)} match(s)")
        d += timedelta(days=1)
        time.sleep(SLEEP_BETWEEN_CALLS)

    dates = load_existing()
    dates.update(fetched)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dates": dict(sorted(dates.items())),
    }

    # Écriture atomique : l'API peut relire le fichier à tout moment, elle
    # ne doit jamais tomber sur un JSON à moitié écrit.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)

    n_games = sum(len(g) for g in fetched.values())
    print(f"Sauvegardé : {OUTPUT_PATH} ({len(fetched)} jours synchronisés, {n_games} matchs)")


if __name__ == "__main__":
    main()
