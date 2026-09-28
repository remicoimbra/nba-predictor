"""
schedule_service.py

Récupère les matchs du jour (calendrier), PAS l'historique. Différent de
data/collectors/fetch_games.py, qui ne récupère que des matchs déjà joués.

Source principale : data/processed/schedule.json, produit par
data/collectors/fetch_schedule.py lors de la synchro quotidienne. L'API ne
contacte donc PAS la NBA à chaque requête, parce que stats.nba.com et
cdn.nba.com bloquent l'IP du VPS de production (testé le 2026-09-28, cf.
CLAUDE.md "Déploiement") : un appel direct y pend jusqu'au timeout.

Repli (développement local uniquement, cf. LIVE_SCHEDULE_FALLBACK dans
app/core/config.py) : si la date demandée n'est pas dans le fichier, appel
direct à ScoreboardV3 via le même code que fetch_schedule.py.

Pourquoi ScoreboardV3 (stats.nba.com) plutôt que l'endpoint "live" de
cdn.nba.com ou ScoreboardV2 :
- cdn.nba.com s'est révélé géobloqué depuis le réseau de l'IUT (accès
  refusé par le CDN Akamai de nba.com), alors que stats.nba.com y répond.
- ScoreboardV2 est explicitement déprécié par nba_api pour la saison
  2025-26 (bug connu sur les données de score en début de saison) ; le
  package recommande ScoreboardV3 en remplacement direct.

Les team_id du calendrier sont les MÊMES ID nba_api que ceux déjà utilisés
dans tout le pipeline (games_clean.csv, team_state.json, etc.) — pas de
mapping à faire.

Limite : le statut ("7:00 pm ET", "Final"...) est celui du moment de la
synchro, pas du temps réel.
"""

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from nba_api.stats.static import teams as static_teams

from app.core.config import LIVE_SCHEDULE_FALLBACK

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SCHEDULE_PATH = DATA_DIR / "processed" / "schedule.json"
LIVE_TIMEOUT = 15  # secondes : un visiteur attend la réponse

# Chargé une fois : mapping team_id -> nom complet / tricode, pour l'affichage
# côté front. Ne nécessite AUCUN appel réseau (dataset statique embarqué
# dans nba_api).
_TEAM_INFO_BY_ID = {t["id"]: t for t in static_teams.get_teams()}


class ScheduleNotCoveredError(LookupError):
    """Date absente de schedule.json, et repli en direct désactivé."""


@dataclass
class TeamRef:
    id: int
    name: str
    tricode: str


@dataclass
class ScheduledGame:
    game_id: str
    game_time_utc: str
    status: str          # texte statut NBA, ex: "7:00 pm ET", "Final", "Q2 05:12"
    season_type: str     # cf. _SEASON_TYPE_BY_PREFIX
    home_team: TeamRef
    away_team: TeamRef


# Les 3 premiers chiffres du game_id NBA donnent le type de match. Utile au
# front : le modèle n'est entraîné que sur la saison régulière, une
# prédiction de présaison (rotations expérimentales) est à prendre avec
# beaucoup plus de recul.
_SEASON_TYPE_BY_PREFIX = {
    "001": "preseason",
    "002": "regular_season",
    "003": "all_star",
    "004": "playoffs",
    "005": "play_in",
}


def _season_type(game_id: str) -> str:
    return _SEASON_TYPE_BY_PREFIX.get(game_id[:3], "other")


def team_ref(team_id: int) -> TeamRef:
    info = _TEAM_INFO_BY_ID.get(team_id)
    return TeamRef(
        id=team_id,
        name=info["full_name"] if info else f"Équipe {team_id}",
        tricode=info["abbreviation"] if info else "???",
    )


# Cache du fichier, invalidé quand sa date de modification change : la
# synchro remplace le fichier sans redémarrer l'API.
_schedule_cache: dict = {"mtime": None, "dates": {}}


def _load_schedule_dates() -> dict:
    try:
        mtime = SCHEDULE_PATH.stat().st_mtime
    except FileNotFoundError:
        return {}
    if _schedule_cache["mtime"] != mtime:
        with open(SCHEDULE_PATH, encoding="utf-8") as f:
            _schedule_cache["dates"] = json.load(f).get("dates", {})
        _schedule_cache["mtime"] = mtime
    return _schedule_cache["dates"]


def _fetch_live(target_date: date) -> list[dict]:
    # Import tardif : data/collectors/ n'a pas besoin d'exister là où le
    # repli est désactivé (VPS). Même principe de sys.path que
    # predictor_service.py pour features_lib.
    collectors_dir = str(DATA_DIR / "collectors")
    if collectors_dir not in sys.path:
        sys.path.insert(0, collectors_dir)
    from fetch_schedule import fetch_day  # type: ignore[import-not-found]

    return fetch_day(target_date, timeout=LIVE_TIMEOUT)


def get_todays_games(target_date: date | None = None) -> list[ScheduledGame]:
    """
    Renvoie les matchs programmés pour `target_date` (aujourd'hui si non
    précisé) — peut être une liste vide en dehors de la saison régulière/
    playoffs, ou pendant l'intersaison. Ce n'est pas une erreur.

    Lève ScheduleNotCoveredError si la date n'a pas été synchronisée et
    que le repli en direct est désactivé. En repli, peut lever une
    exception réseau si stats.nba.com est injoignable ou bloque la requête :
    à laisser remonter au routeur, qui décide de la réponse HTTP appropriée
    plutôt que de masquer le problème ici.
    """
    target_date = target_date or date.today()
    key = target_date.isoformat()

    dates = _load_schedule_dates()
    if key in dates:
        entries = dates[key]
    elif LIVE_SCHEDULE_FALLBACK:
        entries = _fetch_live(target_date)
    else:
        covered = f"du {min(dates)} au {max(dates)}" if dates else "aucune date (schedule.json absent)"
        raise ScheduleNotCoveredError(
            f"Calendrier non synchronisé pour le {key} (dates couvertes : {covered})."
        )

    return [_scheduled_game(e) for e in entries]


def _scheduled_game(entry: dict) -> ScheduledGame:
    return ScheduledGame(
        game_id=entry["game_id"],
        game_time_utc=entry["game_time_utc"],
        status=entry["status"],
        season_type=_season_type(entry["game_id"]),
        home_team=team_ref(entry["home_team_id"]),
        away_team=team_ref(entry["away_team_id"]),
    )


def find_game(game_id: str, target_date: date | None = None) -> tuple[date, ScheduledGame] | None:
    """
    Retrouve un match par son game_id. Avec une date : cherche parmi les
    matchs de ce jour (repli en direct possible, cf. get_todays_games).
    Sans date : parcourt schedule.json uniquement. None si introuvable.
    """
    if target_date is not None:
        for game in get_todays_games(target_date):
            if game.game_id == game_id:
                return target_date, game
        return None

    for key, entries in _load_schedule_dates().items():
        for entry in entries:
            if entry["game_id"] == game_id:
                return date.fromisoformat(key), _scheduled_game(entry)
    return None


def covered_dates() -> dict[str, int]:
    """Dates présentes dans schedule.json -> nombre de matchs (0 = synchronisée, aucun match)."""
    return {key: len(entries) for key, entries in sorted(_load_schedule_dates().items())}
