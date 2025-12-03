from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central configuration for the GroundTruth backend."""

    project_root: Path = Field(
        default=Path(__file__).resolve().parent.parent,
        description="Absolute path to the repository root.",
    )
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "Dataset",
        description="Directory containing prepared dataset artifacts.",
    )
    chunks_meta_path: Path = Field(default=None, description="Path to chunks metadata JSONL.")
    faiss_index_path: Path = Field(default=None, description="Path to FAISS IndexFlatIP file.")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model used for query embeddings.",
    )
    cross_encoder_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used for reranking evidence.",
    )
    retrieval_k: int = Field(default=50, description="Number of initial FAISS hits.")
    rerank_k: int = Field(default=12, description="Number of hits to rerank.")
    evidence_top_k: int = Field(default=4, description="Final pieces of evidence to keep.")
    max_prompt_tokens: int = Field(default=1800, description="Max prompt budget for evidence text.")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier for generation.",
    )
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API key sourced from environment.",
    )
    pii_mask_token: str = Field(default="[REDACTED]", description="Token used to mask PII.")

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: dict[str, object]) -> None:
        if self.chunks_meta_path is None:
            self.chunks_meta_path = self.data_dir / "chunks_meta.jsonl"
        if self.faiss_index_path is None:
            self.faiss_index_path = self.data_dir / "faiss_index.index"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()

