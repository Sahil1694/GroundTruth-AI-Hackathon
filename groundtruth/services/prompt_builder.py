from __future__ import annotations

import re
from typing import List

from ..models import EvidenceSelection, LiveEvent
from ..models.summary import CustomerSummary


class PromptBuilder:
    """Constructs the final prompt consumed by the LLM."""

    _PII_REGEX = re.compile(r"\b\d{8,}\b")

    def __init__(self, pii_mask_token: str = "[REDACTED]"):
        self._pii_mask_token = pii_mask_token

    def build(
        self,
        event: LiveEvent,
        summary: CustomerSummary,
        evidence: EvidenceSelection,
    ) -> str:
        evidence_blocks = self._format_evidence(evidence)
        mask_message = self._mask(event.message)

        template = f"""
You are GroundTruth's Intelligent Customer Experience Agent.
Use the evidence provided below and the customer context to make the **safest, best-effort recommendation you can**.
If there is no evidence, you may still answer using the customer summary, live message, and weather/location,
but you must not invent specific offers or coupons.
Return STRICT JSON with keys message, reason, sources (list of chunk_ids).

Customer Context:
- customer_id: {event.customer_id}
- loyalty: {summary.loyalty_level or 'unknown'}
- reward_points: {summary.reward_points or 'unknown'}
- detected_store_id: {event.detected_store_id or 'unknown'}
- weather: {event.weather or 'unknown'}
- geo: lat {event.latitude:.4f}, lon {event.longitude:.4f}
- user_message: "{mask_message}"
- summary: {summary.overview}

Evidence:
{evidence_blocks}

JSON Output Requirements:
- Output ONLY a single JSON object, no prose before or after.
- Use double quotes for all keys and string values.
- Ensure all fields are comma-separated (valid JSON, no trailing commas).
- message: friendly recommendation grounded in evidence **or** customer summary.
- reason: short explanation referencing evidence and/or summary.
- sources: array of chunk_ids you relied on (may be empty if no evidence).
- never fabricate chunk_ids.
- respect allergies and preferences.
- encourage nearest store visit if relevant.

Example (structure only, adapt content to the evidence):
{{
  "message": "string",
  "reason": "string",
  "sources": ["chunk_id_1", "chunk_id_2"]
}}
"""
        return template.strip()

    def _format_evidence(self, evidence: EvidenceSelection) -> str:
        if not evidence.items:
            return "- None"
        lines: List[str] = []
        for idx, item in enumerate(evidence.items, start=1):
            text = self._mask(item.text)
            lines.append(f"[{idx}] chunk_id={item.chunk_id} | score={item.score:.4f}\n{text}")
        if evidence.notes:
            lines.append(f"Note: {evidence.notes}")
        return "\n\n".join(lines)

    def _mask(self, value: str) -> str:
        return self._PII_REGEX.sub(self._pii_mask_token, value)

