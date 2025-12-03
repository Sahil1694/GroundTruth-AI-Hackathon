from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..config import get_settings
from ..models import RecommendationRequest, RecommendationResponse
from ..services import RecommendationService

app = FastAPI(
    title="GroundTruth Intelligent Customer Experience Agent",
    version="0.1.0",
    description="RAG-powered backend for hyper-personalized customer recommendations.",
)

settings = get_settings()
service = RecommendationService(settings=settings)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    try:
        return service.recommend(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail="Recommendation failed.") from exc

