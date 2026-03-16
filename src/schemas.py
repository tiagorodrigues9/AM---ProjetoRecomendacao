from pydantic import BaseModel
from typing import Optional


class MusicFeatures(BaseModel):
    energy: float
    loudness: float


class TrackRequest(BaseModel):
    track_id: Optional[str] = "unknown"
    track_name: str
    artist_name: str
    features: MusicFeatures


class TrackResponse(BaseModel):
    track: str
    artist: str
    recommendation: str
    debug_info: dict

class BatchTrackItem(BaseModel):
    """Uma música dentro de um lote (batch)."""
    track_name: str
    artist_name: str
    features: MusicFeatures


class TrackBatchRequest(BaseModel):
    """Requisição com várias músicas de uma vez."""
    tracks: list[BatchTrackItem]


class TrackBatchResponse(BaseModel):
    """Resposta com o resultado de todas as músicas."""
    results: list[TrackResponse]
    total: int
    summary: dict

class LibraryUploadResponse(BaseModel):
    """Resposta do upload de biblioteca de músicas."""
    total_received: int
    total_valid: int
    total_invalid: int
    invalid_rows: list[dict]
    sample: list[dict]

class ColumnReport(BaseModel):
    """Relatório de saúde de uma coluna."""
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list

class OutlierReport(BaseModel):
    """Relatório de outliers de uma coluna numérica."""
    column: str
    total_outliers: int
    outlier_pct: float
    lower_bound: float
    upper_bound: float

class DataAuditResponse(BaseModel):
    """Relatório completo de auditoria/saúde de um dataset."""
    total_rows: int
    total_columns: int
    duplicate_rows: int
    columns: list[ColumnReport]
    outliers: list[OutlierReport]
    numeric_summary: dict
    correlations: dict
    health_score: float