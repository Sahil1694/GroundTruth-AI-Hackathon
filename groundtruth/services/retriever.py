from __future__ import annotations

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models import RetrievedChunk
from .chunk_store import ChunkStore


class FaissRetriever:
    """Lightweight FAISS retriever that returns scored chunk candidates."""

    def __init__(
        self,
        index_path,
        chunk_store: ChunkStore,
        model_name: str,
    ):
        self._chunk_store = chunk_store
        self._index = faiss.read_index(str(index_path))
        self._encoder = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        embedding = self._encoder.encode([query], convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        distances, indices = self._index.search(embedding.astype("float32"), top_k)
        results: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._chunk_store):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunk_store[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

