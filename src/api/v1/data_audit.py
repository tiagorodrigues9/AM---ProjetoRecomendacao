from fastapi import APIRouter, UploadFile, File, HTTPException
from src.schemas import DataAuditResponse
from src.services.data_cleaner import DataCleaner

router = APIRouter()

@router.post(
    "/data/audit",
    response_model=DataAuditResponse,
    summary="Auditoria de saúde de um dataset",
)
async def audit_data(file: UploadFile = File(..., description="Arquivo CSV para auditoria")):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Apenas arquivos CSV são aceitos.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        cleaner = DataCleaner.from_bytes(content, file_type="csv")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro ao processar o CSV: {str(e)}")

    return cleaner.diagnose()