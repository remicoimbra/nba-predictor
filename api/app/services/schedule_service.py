"""
schedule_service.py

Récupère les matchs du jour (calendrier), PAS l'historique. Différent de
data/collectors/fetch_games.py, qui ne récupère que des matchs déjà joués.

Utilise ScoreboardV3 (stats.nba.com), PAS l'endpoint "live" de cdn.nba.com
ni ScoreboardV2 :
- cdn.nba.com (endpoint "live") s'est révélé géobloqué depuis certains
  réseaux (constaté : accès refusé par le CDN Akamai de nba.com,
  indépendamment de tout pare-feu local). stats.nba.com, en revanche, est
  le domaine déjà utilisé avec succès par fetch_games.py (7 saisons, 8279
  matchs collectés) sur ce même réseau.
- ScoreboardV2 est explicitement déprécié par nba_api pour la saison
  2025-26 (bug connu sur les données de score en début de saison) ; le
  package recommande ScoreboardV3 en remplacement direct.

Les team_id renvoyés par cet endpoint sont les MÊMES ID nba_api que ceux
déjà utilisés dans tout le pipeline (games_clean.csv, team_state.json,
etc.) — pas de mapping à faire.
"""

from dataclasses import dataclass
from datetime import date

from nba_api.stats.endpoints import scoreboardv3
from nba_api.stats.static import teams as static_teams

# Chargé une fois : mapping team_id -> nom complet / tricode, pour l'affichage
# côté front. Ne nécessite AUCUN appel réseau (dataset statique embarqué
# dans nba_api).
_TEAM_INFO_BY_ID = {t["id"]: t for t in static_teams.get_teams()}


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
    home_team: TeamRef
    away_team: TeamRef


def _team_ref(team_id: int) -> TeamRef:
    info = _TEAM_INFO_BY_ID.get(team_id)
    return TeamRef(
        id=team_id,
        name=info["full_name"] if info else f"Équipe {team_id}",
        tricode=info["abbreviation"] if info else "???",
    )


def get_todays_games(target_date: date | None = None) -> list[ScheduledGame]:
    """
    Renvoie les matchs programmés pour `target_date` (aujourd'hui si non
    précisé) — peut être une liste vide en dehors de la saison régulière/
    playoffs, ou pendant l'intersaison. Ce n'est pas une erreur.

    Peut lever une exception réseau si stats.nba.com est injoignable ou
    bloque la requête (rate limit) : à laisser remonter au routeur, qui
    décide de la réponse HTTP appropriée plutôt que de masquer le
    problème ici.

    Lecture volontaire du JSON brut (get_dict()) plutôt que du DataFrame
    `game_header` de ScoreboardV3 : ce dernier ne contient PAS les team_id
    home/away (juste le statut du match). Les team_id sont dans le dataset
    `LineScore`, mais sans indicateur explicite home/away — seulement
    l'ordre d'insertion (home ajouté avant away). Le JSON brut, lui,
    labellise explicitement `homeTeam`/`awayTeam` : plus robuste qu'une
    déduction par position.
    """
    target_date = target_date or date.today()

    board = scoreboardv3.ScoreboardV3(game_date=target_date.strftime("%Y-%m-%d"), league_id="00")
    raw = board.get_dict()
    games_raw = raw.get("scoreboard", {}).get("games", [])

    games = []
    for g in games_raw:
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        games.append(
            ScheduledGame(
                game_id=str(g.get("gameId")),
                game_time_utc=g.get("gameTimeUTC") or "",
                status=str(g.get("gameStatusText", "")),
                home_team=_team_ref(int(home["teamId"])),
                away_team=_team_ref(int(away["teamId"])),
            )
        )
    return games