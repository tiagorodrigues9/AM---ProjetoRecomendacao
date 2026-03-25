from fastapi import APIRouter, HTTPException, Depends
from typing import List
import pandas as pd
from src.services.feature_engineer import FeatureEngineer
from pydantic import BaseModel

router = APIRouter()

class RawTrack(BaseModel):
    track_name: str
    track_genre: str
    tempo: float
    popularity: float

class PreprocessRequest(BaseModel):
    tracks: List[RawTrack]

def get_feature_engineer():
    fe = FeatureEngineer()
    if not fe.transformer_path.exists():
        dummy = pd.DataFrame([
            {"tempo": 120, "popularity": 50, "track_genre": "Pop"},
            {"tempo": 80, "popularity": 10, "track_genre": "Rock"},
        ])
        fe.fit(dummy, ["tempo", "popularity"], ["track_genre"])
    else:
        fe.load()
    return fe

@router.post("/preprocess")
def preprocess_data(request: PreprocessRequest, fe: FeatureEngineer = Depends(get_feature_engineer)):
    try:
        df = pd.DataFrame([t.model_dump() for t in request.tracks])
        df_transformed = fe.transform(df)
        return {
            "transformed_shape": list(df_transformed.shape),
            "features": df_transformed.columns.tolist(),
            "data": df_transformed.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))