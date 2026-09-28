"""
routers/games.py

Endpoints liés aux matchs. Pour l'instant : /games/today uniquement (les
autres — /games, /games/{game_id} — sont dans la liste cible de CLAUDE.md
mais pas encore développés ; même structure de router à réutiliser plus
tard, pas besoin d'un nouveau fichier).
"""

import logging
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query

from app.services.predictor_service import (
    PredictorNotReadyError,
    UnknownTeamError,
    get_predictor_service,
)
from app.services.schedule_service import ScheduledGame, ScheduleNotCoveredError, get_todays_games

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/games", tags=["games"])


def _serialize_game(game: ScheduledGame, as_of_date) -> dict:
    """
    Construit la réponse pour un match : infos de calendrier + prédiction
    si elle a pu être calculée. `prediction: null` (pas une erreur HTTP)
    si UNE équipe du match n'est pas dans team_state.json — ça permet au
    front d'afficher quand même la carte du match (cf. GameCard.jsx) avec
    juste les prédictions manquantes, plutôt que de faire échouer tout
    l'endpoint pour un seul match problématique (ex: équipe fraîchement
    ajoutée à la ligue, team_state.json pas encore resynchronisé).
    """
    base = {
        "game_id": game.game_id,
        "game_time_utc": game.game_time_utc,
        "status": game.status,
        "season_type": game.season_type,
        "home_team": {"id": game.home_team.id, "name": game.home_team.name, "tricode": game.home_team.tricode},
        "away_team": {"id": game.away_team.id, "name": game.away_team.name, "tricode": game.away_team.tricode},
        "prediction": None,
    }

    try:
        service = get_predictor_service()
        pred = service.predict(home_team_id=game.home_team.id, away_team_id=game.away_team.id, as_of_date=as_of_date)
        base["prediction"] = {
            "home_win_probability": pred.home_win_probability,
            "away_win_probability": pred.away_win_probability,
            "predicted_winner": pred.predicted_winner,
            "predicted_home_score": pred.predicted_home_score,
            "predicted_away_score": pred.predicted_away_score,
        }
    except UnknownTeamError as e:
        # Cas attendu et gérable : on logue pour investigation mais on ne
        # fait pas échouer la requête entière pour un match.
        logger.warning("Prédiction impossible pour %s vs %s : %s", game.home_team.tricode, game.away_team.tricode, e)

    return base


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
    pour chacun.

    Erreurs volontairement distinguées :
    - 400 : date mal formée (faute de frappe côté appelant, pas un
      problème de service).
    - PredictorNotReadyError (modèle/team_state absent) -> 503, le service
      n'est pas mal utilisé, il n'est juste pas prêt (déploiement
      incomplet) : le front peut réessayer plus tard sans changer sa requête.
    - Date absente de schedule.json, repli en direct désactivé (VPS) -> 404 :
      la requête est valide, mais cette date n'a jamais été synchronisée.
    - Erreur réseau nba_api (scoreboard indisponible) -> 502, la source de
      données externe est en cause, pas notre API elle-même.
    """
    target_date = date.today()
    if on_date is not None:
        try:
            target_date = datetime.strptime(on_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Date invalide : '{on_date}' (attendu YYYY-MM-DD)")

    try:
        # Vérifie que le service de prédiction est chargeable AVANT de
        # traiter les matchs un par un, pour renvoyer une erreur claire
        # unique plutôt que N logs identiques (un par match).
        get_predictor_service()
    except PredictorNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        games = get_todays_games(target_date)
    except ScheduleNotCoveredError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Échec de récupération du calendrier NBA pour %s", target_date)
        raise HTTPException(status_code=502, detail=f"Calendrier NBA indisponible : {e}")

    return {"date": target_date.isoformat(), "count": len(games), "games": [_serialize_game(g, target_date) for g in games]}