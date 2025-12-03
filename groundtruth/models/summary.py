from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class CustomerSummary:
    """Condensed snapshot of the customer's profile and habits."""

    customer_id: int
    overview: str
    loyalty_level: Optional[str] = None
    preferred_items: Optional[str] = None
    last_store_id: Optional[int] = None
    reward_points: Optional[int] = None

