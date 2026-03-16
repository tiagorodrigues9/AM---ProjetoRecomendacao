from fastapi import APIRouter
from src.api.v1 import recommendation, library, data_audit

api_router = APIRouter(prefix="/api/v1")

# ... router de recomendação ...
api_router.include_router(recommendation.router, tags=["recommendation"])

# Novos routers
api_router.include_router(library.router, tags=["library"])
api_router.include_router(data_audit.router, tags=["data-audit"])