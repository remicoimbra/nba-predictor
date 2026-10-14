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

# Origines autorisées à appeler l'API depuis un navigateur (CORS), séparées
# par des virgules.
# - En local : le front (localhost:3000) et l'API (:8000) sont deux origines.
# - Sur le VPS : vide. Le front appelle /api sur son propre domaine (Caddy
#   relaie vers l'API), même origine donc pas de CORS.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("NBA_CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# Documentation interactive (/docs, /redoc, /openapi.json). Coupée sur le VPS :
# inutile aux visiteurs, et Swagger UI charge un script depuis un CDN que la
# Content-Security-Policy du site bloque de toute façon.
API_DOCS = _env_flag("NBA_API_DOCS", default=True)
