from __future__ import annotations

from ..models import CustomerSummary, LiveEvent


class QueryBuilder:
    """Builds semantic search queries that blend live context with history."""

    def build(self, event: LiveEvent, summary: CustomerSummary) -> str:
        parts: list[str] = [
            f"Customer ID {event.customer_id}",
            f"Live message: {event.message.strip()}",
            f"Summary: {summary.overview}",
        ]

        if summary.loyalty_level:
            parts.append(f"Loyalty tier: {summary.loyalty_level}")

        if event.detected_store_id:
            parts.append(f"Store context: store {event.detected_store_id}")

        if event.weather:
            parts.append(f"Weather: {event.weather}")

        parts.append(f"Location: lat {event.latitude:.4f}, lon {event.longitude:.4f}")

        return " | ".join(parts)

