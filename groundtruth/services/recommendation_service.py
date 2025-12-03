from __future__ import annotations

import time
from typing import Optional

from ..config import Settings, get_settings
from ..models import LiveEvent, RecommendationResponse
from .chunk_store import ChunkStore
from .customer_summary import CustomerSummaryService
from .evidence_selector import EvidenceSelector
from .llm_client import GeminiClient
from .prompt_builder import PromptBuilder
from .query_builder import QueryBuilder
from .response_validator import ResponseValidator
from .reranker import CrossEncoderReranker
from .retriever import FaissRetriever
from .store_priority import StorePriorityBooster


class RecommendationService:
    """End-to-end orchestration of the GroundTruth recommendation pipeline."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._chunk_store = ChunkStore(self._settings.chunks_meta_path)
        self._summary_service = CustomerSummaryService(self._settings.data_dir)
        self._retriever = FaissRetriever(
            index_path=self._settings.faiss_index_path,
            chunk_store=self._chunk_store,
            model_name=self._settings.embedding_model_name,
        )
        self._booster = StorePriorityBooster()
        self._reranker = CrossEncoderReranker(self._settings.cross_encoder_model_name)
        self._selector = EvidenceSelector(
            top_k=self._settings.evidence_top_k,
            max_chars=self._settings.max_prompt_tokens * 4,  # rough char/token ratio
        )
        self._query_builder = QueryBuilder()
        self._prompt_builder = PromptBuilder(self._settings.pii_mask_token)
        self._llm = GeminiClient(
            api_key=self._settings.gemini_api_key,
            model_name=self._settings.gemini_model,
        )
        self._validator = ResponseValidator()

    def recommend(self, event: LiveEvent) -> RecommendationResponse:
        start = time.perf_counter()
        summary = self._summary_service.summarize(event.customer_id)
        query = self._query_builder.build(event, summary)

        retrieved = self._retriever.search(query, self._settings.retrieval_k)
        boosted = self._booster.boost(retrieved, event.detected_store_id)
        reranked = self._reranker.rerank(query, boosted, self._settings.rerank_k)
        evidence = self._selector.select(reranked)

        prompt = self._prompt_builder.build(event, summary, evidence)
        llm_output = self._llm.generate(prompt)
        parsed = self._validator.parse(llm_output)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return RecommendationResponse(latency_ms=latency_ms, **parsed)

