"""
routers/games.py

Endpoints liés aux matchs :
- /games/today      : matchs d'un jour + prédiction et résumé léger par équipe
                      (ce qu'affichent les cartes de la page d'accueil)
- /games/calendar   : dates synchronisées -> nombre de matchs (pastilles du
                      calendrier du front)
- /games/{game_id}  : fiche complète d'un match (facteurs de la prédiction,
                      stats comparées, forme récente, courbe Elo)

/games (liste) est dans la liste cible de CLAUDE.md mais pas encore développé.
"""

import logging
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query

from app.core.config import LIVE_SCHEDULE_FALLBACK
from app.services.predictor_service import (
    PredictorNotReadyError,
    PredictorService,
    UnknownTeamError,
    get_predictor_service,
)
from app.services.schedule_service import (
    ScheduledGame,
    ScheduleNotCoveredError,
    TeamRef,
    covered_dates,
    find_game,
    get_todays_games,
    team_ref,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/games", tags=["games"])

LAST_N_FORM = 5  # résultats affichés sur les cartes (V/D)


def _parse_date(on_date: str | None) -> date | None:
    """400 si la date est mal formée (faute de frappe côté appelant, pas un problème de service)."""
    if on_date is None:
        return None
    try:
        return datetime.strptime(on_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Date invalide : '{on_date}' (attendu YYYY-MM-DD)")


def _require_predictor() -> PredictorService:
    """
    Vérifie que le service de prédiction est chargeable AVANT de traiter les
    matchs un par un, pour renvoyer une erreur claire unique plutôt que N
    logs identiques (un par match). PredictorNotReadyError (modèle/team_state
    absent) -> 503 : le service n'est pas mal utilisé, il n'est juste pas
    prêt (déploiement incomplet), le front peut réessayer plus tard.
    """
    try:
        return get_predictor_service()
    except PredictorNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _call_schedule(fn, *args):
    """
    Erreurs de calendrier -> HTTP :
    - date absente de schedule.json, repli en direct désactivé (VPS) -> 404 :
      la requête est valide, mais cette date n'a jamais été synchronisée ;
    - erreur réseau nba_api (repli en direct) -> 502, la source externe est
      en cause, pas notre API.
    """
    try:
        return fn(*args)
    except ScheduleNotCoveredError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Échec de récupération du calendrier NBA (%s)", args)
        raise HTTPException(status_code=502, detail=f"Calendrier NBA indisponible : {e}")


def _team_base(team: TeamRef) -> dict:
    return {"id": team.id, "name": team.name, "tricode": team.tricode}


def _snapshots(service: PredictorService, game: ScheduledGame, as_of_date) -> tuple[dict | None, dict | None]:
    """Stats des deux équipes, None pour une équipe absente de team_state.json."""
    result = []
    for team, is_home in ((game.home_team, True), (game.away_team, False)):
        try:
            result.append(service.team_snapshot(team.id, is_home=is_home, as_of_date=as_of_date))
        except UnknownTeamError:
            result.append(None)
    return result[0], result[1]


def _card_team(team: TeamRef, snapshot: dict | None) -> dict:
    """Équipe + résumé léger pour une carte : Elo, bilan, derniers résultats."""
    data = _team_base(team)
    if snapshot is not None:
        data.update({
            "elo": round(snapshot["stats"]["elo"], 1),
            "record": snapshot["record"],
            "last_results": [g["win"] for g in snapshot["recent_games"][-LAST_N_FORM:]],
        })
    return data


def _detail_team(team: TeamRef, snapshot: dict | None) -> dict:
    """Équipe + toutes les stats de la fiche match (adversaires nommés)."""
    data = _card_team(team, snapshot)
    if snapshot is not None:
        recent = []
        for g in snapshot["recent_games"]:
            opponent = team_ref(g["opponent_id"]) if g["opponent_id"] is not None else None
            recent.append({**g, "opponent": _team_base(opponent) if opponent else None})
        data.update({
            "stats": snapshot["stats"],
            "new_season": snapshot["new_season"],
            "home_record": snapshot["home_record"],
            "away_record": snapshot["away_record"],
            "recent_games": recent,
            "elo_history": snapshot["elo_history"],
        })
    return data


def _prediction(service: PredictorService, game: ScheduledGame, as_of_date, detailed: bool) -> dict | None:
    """
    `None` (pas une erreur HTTP) si UNE équipe du match n'est pas dans
    team_state.json : le front affiche quand même la carte du match, avec
    juste la prédiction manquante, plutôt que de faire échouer tout
    l'endpoint pour un seul match (ex : équipe fraîchement ajoutée à la
    ligue, adversaire hors NBA en présaison).
    """
    try:
        pred = service.predict(home_team_id=game.home_team.id, away_team_id=game.away_team.id, as_of_date=as_of_date)
    except UnknownTeamError as e:
        logger.warning("Prédiction impossible pour %s vs %s : %s", game.home_team.tricode, game.away_team.tricode, e)
        return None

    data = {
        "home_win_probability": pred.home_win_probability,
        "away_win_probability": pred.away_win_probability,
        "predicted_winner": pred.predicted_winner,
        "predicted_home_score": pred.predicted_home_score,
        "predicted_away_score": pred.predicted_away_score,
        "elo_home_probability": pred.elo_home_probability,
    }
    if detailed:
        data["factors"] = pred.factors
        data["base_value"] = pred.base_value
    return data


def _game_base(game: ScheduledGame) -> dict:
    return {
        "game_id": game.game_id,
        "game_time_utc": game.game_time_utc,
        "status": game.status,
        "season_type": game.season_type,
    }


def _serialize_game(service: PredictorService, game: ScheduledGame, as_of_date) -> dict:
    home_snap, away_snap = _snapshots(service, game, as_of_date)
    return {
        **_game_base(game),
        "home_team": _card_team(game.home_team, home_snap),
        "away_team": _card_team(game.away_team, away_snap),
        "prediction": _prediction(service, game, as_of_date, detailed=False),
    }


@router.get("/today")
def get_today_games(
    on_date: str | None = Query(
        default=None,
        alias="date",
        description=(
            "Date à interroger, format YYYY-MM-DD. Par défaut : aujourd'hui. "
            "Utile en intersaison (aucun match programmé aujourd'hui) pour tester "
            "avec une date de la saison 2025-26 déjà présente dans l'historique, "
            "ex: /games/today?date=2026-01-15"
        ),
    ),
):
    """
    Matchs du jour (ou d'une date donnée via ?date=YYYY-MM-DD) + prédiction
    et résumé de chaque équipe (Elo, bilan, 5 derniers résultats).
    Erreurs : 400 date mal formée, 503 modèle non prêt, 404 date non
    synchronisée (repli désactivé), 502 calendrier NBA indisponible.
    """
    target_date = _parse_date(on_date) or date.today()
    service = _require_predictor()
    games = _call_schedule(get_todays_games, target_date)
    return {
        "date": target_date.isoformat(),
        "count": len(games),
        "games": [_serialize_game(service, g, target_date) for g in games],
    }


@router.get("/calendar")
def get_calendar():
    """
    Dates présentes dans schedule.json -> nombre de matchs. Une date absente
    n'est pas synchronisée : si `live_fallback` est vrai, l'API ira quand
    même chercher le calendrier en direct pour cette date.
    (Déclaré avant /{game_id}, sinon "calendar" serait pris pour un game_id.)
    """
    return {
        "today": date.today().isoformat(),
        "live_fallback": LIVE_SCHEDULE_FALLBACK,
        "dates": covered_dates(),
    }


@router.get("/{game_id}")
def get_game(
    game_id: str,
    on_date: str | None = Query(
        default=None,
        alias="date",
        description="Date du match (YYYY-MM-DD). Sans date, le match est cherché dans schedule.json.",
    ),
):
    """
    Fiche complète d'un match : prédiction + facteurs (contributions XGBoost
    regroupées par famille), proba Elo seule, stats des deux équipes, 10
    derniers matchs, évolution Elo. 404 si le match est introuvable.
    """
    target_date = _parse_date(on_date)
    service = _require_predictor()
    found = _call_schedule(find_game, game_id, target_date)
    if found is None:
        where = f"le {target_date.isoformat()}" if target_date else "dans le calendrier synchronisé"
        raise HTTPException(status_code=404, detail=f"Match {game_id} introuvable {where}.")

    game_date, game = found
    home_snap, away_snap = _snapshots(service, game, game_date)
    return {
        "date": game_date.isoformat(),
        **_game_base(game),
        "home_team": _detail_team(game.home_team, home_snap),
        "away_team": _detail_team(game.away_team, away_snap),
        "prediction": _prediction(service, game, game_date, detailed=True),
    }
