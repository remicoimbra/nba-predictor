"""
routers/model.py

GET /model/metrics : métriques du modèle (accuracy vs baseline, log loss,
importance des features, courbe de calibration) pour la page « Le modèle »
du front. Lit data/models/saved_models/model_metrics.json, écrit par
data/models/train_model.py à chaque entraînement.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
METRICS_PATH = DATA_DIR / "models" / "saved_models" / "model_metrics.json"

router = APIRouter(prefix="/model", tags=["model"])

# Cache invalidé par date de modification (même principe que schedule.json
# et team_state.json) : un réentraînement est pris en compte sans redémarrage.
_cache: dict = {"mtime": None, "data": None}


@router.get("/metrics")
def get_metrics():
    try:
        mtime = METRICS_PATH.stat().st_mtime
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="model_metrics.json introuvable : relancer data/models/train_model.py.",
        )
    if _cache["mtime"] != mtime:
        with open(METRICS_PATH, encoding="utf-8") as f:
            _cache["data"] = json.load(f)
        _cache["mtime"] = mtime
    return _cache["data"]
