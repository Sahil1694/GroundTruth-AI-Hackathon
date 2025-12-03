from __future__ import annotations

from typing import List

from ..models import EvidencePayload, EvidenceSelection, RetrievedChunk


class EvidenceSelector:
    """Selects the final set of evidence chunks for prompting."""

    def __init__(self, top_k: int, max_chars: int = 4000):
        self._top_k = top_k
        self._max_chars = max_chars

    def select(self, candidates: List[RetrievedChunk]) -> EvidenceSelection:
        selected: List[EvidencePayload] = []
        char_budget = 0
        truncated = False

        for candidate in candidates:
            if len(selected) >= self._top_k:
                break
            text = candidate.chunk.text.strip()
            potential_budget = char_budget + len(text)
            if potential_budget > self._max_chars:
                truncated = True
                break

            selected.append(
                EvidencePayload(
                    chunk_id=candidate.chunk.chunk_id,
                    text=text,
                    meta=candidate.chunk.meta,
                    score=candidate.score,
                )
            )
            char_budget = potential_budget

        notes = None
        if truncated:
            notes = "Evidence truncated due to prompt budget."

        return EvidenceSelection(items=selected, truncated=truncated, notes=notes)

