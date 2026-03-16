from fastapi import APIRouter, UploadFile, File, HTTPException
from src.schemas import LibraryUploadResponse
from src.services.data_cleaner import DataCleaner

router = APIRouter()

@router.post(
    "/library/upload",
    response_model=LibraryUploadResponse,
    summary="Upload de biblioteca de músicas",
)
async def upload_library(file: UploadFile = File(..., description="Arquivo CSV com músicas")):
    # 1. Valida tipo de arquivo
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Apenas arquivos CSV são aceitos.")

    # 2. Lê e processa
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        cleaner = DataCleaner.from_bytes(content, file_type="csv")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro ao processar o CSV: {str(e)}")

    # 3. Valida colunas obrigatórias e retorna resumo
    return cleaner.validate_library_upload()