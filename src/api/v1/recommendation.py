from fastapi import APIRouter, Depends
from functools import lru_cache
from src.schemas import (
    TrackRequest, TrackResponse,
    TrackBatchRequest, TrackBatchResponse,
)
from src.models.perceptron import Perceptron

router = APIRouter()


# Injeção de Dependência — carrega o modelo uma vez só
@lru_cache
def get_perceptron() -> Perceptron:
    return Perceptron(weights={'energy': 0.8, 'loudness': 0.2}, bias=0.1)


# ---- Endpoint Individual (Semana 02) ----

@router.post("/recommend/predict", response_model=TrackResponse)
def predict_track(request: TrackRequest, model: Perceptron = Depends(get_perceptron)):
    """Predição para UMA música (Perceptron Manual)."""
    result = model.predict(request.features.energy, request.features.loudness)
    mood = "Festa/Agitada" if result["prediction"] == 1 else "Relax/Calma"

    return {
        "track": request.track_name,
        "artist": request.artist_name,
        "recommendation": mood,
        "debug_info": result,
    }