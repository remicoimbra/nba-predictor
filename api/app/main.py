from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NBA Predictor API")

# Autorise le front (localhost:3000) à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "API en ligne"}

@app.get("/games/today")
def get_today_games():
    # Données fake en attendant nba_api
    return [
        {
            "id": 1,
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "predicted_winner": "Los Angeles Lakers",
            "win_probability": 0.62,
            "predicted_home_score": 108,
            "predicted_away_score": 102,
        },
        {
            "id": 2,
            "home_team": "Golden State Warriors",
            "away_team": "Denver Nuggets",
            "predicted_winner": "Denver Nuggets",
            "win_probability": 0.55,
            "predicted_home_score": 99,
            "predicted_away_score": 104,
        },
    ]