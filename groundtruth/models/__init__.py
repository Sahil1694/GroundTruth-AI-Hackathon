"""Shared Pydantic models and dataclasses used across the backend."""

from .events import (
    LiveEvent,
    RecommendationRequest,
    RecommendationResponse,
)
from .evidence import ChunkRecord, RetrievedChunk, EvidencePayload, EvidenceSelection
from .summary import CustomerSummary

__all__ = [
    "LiveEvent",
    "RecommendationRequest",
    "RecommendationResponse",
    "ChunkRecord",
    "RetrievedChunk",
    "EvidencePayload",
    "EvidenceSelection",
    "CustomerSummary",
]

