from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class ResponseValidator:
    """Validates that the LLM output adheres to the required JSON schema."""

    _JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    def parse(self, raw_text: str) -> Dict[str, Any]:
        payload = self._extract_json(raw_text)
        self._validate_structure(payload)
        return payload

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            # remove leading and trailing ```
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        match = self._JSON_PATTERN.search(raw_text)
        if not match:
            raise ValueError("LLM response does not contain JSON.")

        snippet = match.group()
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    @staticmethod
    def _validate_structure(payload: Dict[str, Any]) -> None:
        for key in ("message", "reason", "sources"):
            if key not in payload:
                raise ValueError(f"Missing '{key}' in LLM response.")

        if not isinstance(payload["sources"], list):
            raise ValueError("'sources' must be a list.")
        payload["sources"] = [str(item) for item in payload["sources"]]

