from fastapi import FastAPI
from src.api.v1.router import api_router

app = FastAPI(
    title="API de Recomendação (Perceptron Manual)",
    description="API simples para testar o Perceptron Manual de Recomendação de Músicas.",
    version="1.0.0"
)

app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "API de Recomendação Online. Use api/v1/recommend/predict para classificar músicas."}