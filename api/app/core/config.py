"""
config.py

Réglages de l'API lus depuis les variables d'environnement, avec des
valeurs par défaut adaptées au développement local.
"""

import os


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Si une date demandée n'est pas dans data/processed/schedule.json, aller
# chercher le calendrier en direct sur stats.nba.com.
# - En local (IP résidentielle) : True, le sélecteur de date du front marche
#   pour n'importe quelle date sans relancer fetch_schedule.py.
# - Sur le VPS : À METTRE À 0. stats.nba.com y est bloqué (la requête pend
#   jusqu'au timeout) : chaque date non synchronisée ferait attendre le
#   visiteur avant une erreur, au lieu d'un 404 immédiat.
LIVE_SCHEDULE_FALLBACK = _env_flag("NBA_LIVE_SCHEDULE_FALLBACK", default=True)
