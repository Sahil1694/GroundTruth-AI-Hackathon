"""Streamlit frontend to exercise the GroundTruth recommendation pipeline."""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from groundtruth.config import get_settings
from groundtruth.models import LiveEvent
from groundtruth.services import RecommendationService

@st.cache_resource(show_spinner=True)
def load_service() -> RecommendationService:
    """Instantiate the heavy backend service just once."""

    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Configure it before launching Streamlit."
        )
    return RecommendationService(settings=settings)


st.set_page_config(
    page_title="GroundTruth CX Agent",
    page_icon="🛰️",
    layout="centered",
)

st.title("GroundTruth Customer Experience Agent")
st.caption("Test the full RAG pipeline end-to-end with live customer events.")

with st.expander("Environment Check", expanded=False):
    st.write("`GEMINI_API_KEY` present:", bool(os.getenv("GEMINI_API_KEY")))
    st.write("Dataset path:", str(get_settings().data_dir))

service: Optional[RecommendationService] = None
error_placeholder = st.empty()

try:
    service = load_service()
except Exception as exc:  # pragma: no cover - UI feedback
    error_placeholder.error(f"Failed to initialize backend: {exc}")

st.subheader("Live Event Input")
with st.form("recommendation_form"):
    customer_id = st.number_input("Customer ID", min_value=1, value=1001)
    store_id_input = st.text_input("Detected Store ID (optional)", value="2060")
    message = st.text_area(
        "Customer Message",
        value="I'm outside the cafe and need something warm.",
    )
    latitude = st.number_input("Latitude", value=18.4455, format="%.6f")
    longitude = st.number_input("Longitude", value=73.7917, format="%.6f")
    weather = st.text_input("Weather context", value="light rain")
    submitted = st.form_submit_button("Generate Recommendation")

if submitted:
    if not service:
        st.error("Backend service unavailable.")
    else:
        with st.spinner("Running RAG pipeline..."):
            try:
                detected_store_id = (
                    int(store_id_input) if store_id_input.strip() else None
                )
            except ValueError:
                st.error("Store ID must be an integer.")
                detected_store_id = None

            event = LiveEvent(
                customer_id=int(customer_id),
                message=message,
                latitude=float(latitude),
                longitude=float(longitude),
                detected_store_id=detected_store_id,
                weather=weather or None,
            )

            try:
                result = service.recommend(event)
            except Exception as exc:  # pragma: no cover - interactive error
                st.error(f"Failed to generate recommendation: {exc}")
            else:
                st.success(f"Completed in {result.latency_ms} ms")
                st.json(result.dict())

