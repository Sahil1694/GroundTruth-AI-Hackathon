from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from ..models import RetrievedChunk


class CrossEncoderReranker:
    """Cross-encoder reranker that refines FAISS candidates."""

    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not candidates:
            return []

        slice_k = min(top_k, len(candidates))
        pairs = [(query, candidate.chunk.text) for candidate in candidates[:slice_k]]
        scores = self._model.predict(pairs)

        reranked: List[RetrievedChunk] = []
        for candidate, score in zip(candidates[:slice_k], scores):
            reranked.append(
                RetrievedChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    rank=candidate.rank,
                )
            )

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked

