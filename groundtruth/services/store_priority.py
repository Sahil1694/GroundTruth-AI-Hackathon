from __future__ import annotations

from typing import Iterable, List, Optional

from ..models import RetrievedChunk


class StorePriorityBooster:
    """Applies heuristic boosts to retrieval scores based on store context."""

    def __init__(
        self,
        store_match_boost: float = 0.08,
        store_keyword_boost: float = 0.05,
    ):
        self._store_match_boost = store_match_boost
        self._store_keyword_boost = store_keyword_boost

    def boost(
        self,
        candidates: Iterable[RetrievedChunk],
        detected_store_id: Optional[int],
    ) -> List[RetrievedChunk]:
        boosted: List[RetrievedChunk] = []
        for candidate in candidates:
            score = candidate.score
            meta = candidate.chunk.meta
            source = str(meta.get("source", "")).lower()
            store_id = meta.get("store_id")

            if detected_store_id and store_id == detected_store_id:
                score += self._store_match_boost
            elif detected_store_id and store_id and store_id != detected_store_id:
                score -= self._store_match_boost / 2

            if "store" in source:
                score += self._store_keyword_boost

            boosted.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    score=score,
                    rank=candidate.rank,
                )
            )

        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted

