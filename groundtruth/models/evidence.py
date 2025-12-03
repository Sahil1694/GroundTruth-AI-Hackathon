from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ChunkRecord:
    """Single chunk with metadata loaded from the JSONL store."""

    chunk_id: str
    text: str
    meta: Dict[str, Any]


@dataclass(slots=True)
class RetrievedChunk:
    """Chunk plus retrieval metadata from FAISS."""

    chunk: ChunkRecord
    score: float
    rank: int


@dataclass(slots=True)
class EvidencePayload:
    """Chunk content prepared for prompting."""

    chunk_id: str
    text: str
    meta: Dict[str, Any]
    score: float


@dataclass(slots=True)
class EvidenceSelection:
    """Final selection of evidence to feed into the prompt."""

    items: List[EvidencePayload]
    truncated: bool = False
    notes: Optional[str] = None

