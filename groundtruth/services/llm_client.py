from __future__ import annotations

from typing import Optional

import google.generativeai as genai


class GeminiClient:
    """Thin client around the Gemini SDK."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        if not getattr(response, "text", None):
            raise RuntimeError("Gemini response missing text payload.")
        return response.text.strip()

