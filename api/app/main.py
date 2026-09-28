"""
main.py

Point d'entrée FastAPI. Garde volontairement peu de logique ici : la
logique métier vit dans app/services/, les endpoints dans app/routers/
(cf. CLAUDE.md, refactor prévu : "sortir la logique de main.py vers les
routers dédiés"). Comme il n'y avait pas de main.py existant à reprendre,
celui-ci part directement sur cette structure cible plutôt que de repasser
par une étape intermédiaire "tout dans main.py".

Lancement (depuis api/) :
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import games, model

app = FastAPI(title="NBA Predictor API")

# Le front (Next.js, localhost:3000 en dev) tourne sur une origine
# différente de l'API (localhost:8000) : sans CORS explicite, le
# navigateur bloque les requêtes fetch() de api.js malgré un serveur qui
# répond correctement (à ne pas confondre avec une vraie erreur 4xx/5xx
# lors du débogage : dans les DevTools, une erreur CORS apparaît comme un
# échec réseau, pas comme une réponse HTTP visible).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(model.router)


@app.get("/")
def root():
    return {"status": "ok", "service": "nba-predictor-api"}