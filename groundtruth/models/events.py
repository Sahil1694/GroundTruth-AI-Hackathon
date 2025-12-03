from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class LiveEvent(BaseModel):
    """Incoming live signal describing the customer's real-time context."""

    customer_id: int = Field(..., description="Unique identifier of the customer.")
    message: str = Field(..., description="Raw customer utterance or event text.")
    latitude: float = Field(..., description="Current latitude of the customer.")
    longitude: float = Field(..., description="Current longitude of the customer.")
    detected_store_id: Optional[int] = Field(
        default=None, description="Identifier of the nearest detected store."
    )
    weather: Optional[str] = Field(default=None, description="Weather context if available.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp in UTC.",
    )

    @validator("message")
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message may not be empty")
        return cleaned


class RecommendationRequest(LiveEvent):
    """Request model used by the FastAPI endpoint (inherits LiveEvent)."""


class RecommendationResponse(BaseModel):
    """Structured JSON response returned to the client."""

    message: str
    reason: str
    sources: list[str]
    latency_ms: int = Field(..., description="End-to-end latency in milliseconds.")

