from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..models import CustomerSummary


@dataclass(slots=True)
class _CustomerRow:
    customer_id: int
    name: str
    preferred_drinks: List[str]
    preferred_size: Optional[str]
    allergies: Optional[str]
    usual_order_time: Optional[str]
    last_store_id: Optional[int]
    reward_points: Optional[int]


class CustomerSummaryService:
    """Loads customer data and produces lightweight textual summaries."""

    def __init__(self, data_dir: Path):
        self._customers: Dict[int, _CustomerRow] = {}
        self._history: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        self._customers_path = data_dir / "customers.csv"
        self._history_path = data_dir / "customer_history.csv"
        self._load_customers()
        self._load_history()

    def _load_customers(self) -> None:
        with self._customers_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                customer_id = int(row["customer_id"])
                preferred_drinks = [
                    item.strip()
                    for item in row.get("preferred_drinks", "").split("|")
                    if item.strip()
                ]
                last_store_id = (
                    int(row["last_visited_store_id"])
                    if row.get("last_visited_store_id")
                    else None
                )
                reward_points = (
                    int(row["reward_points"]) if row.get("reward_points") else None
                )
                self._customers[customer_id] = _CustomerRow(
                    customer_id=customer_id,
                    name=row.get("name", ""),
                    preferred_drinks=preferred_drinks,
                    preferred_size=row.get("preferred_size"),
                    allergies=row.get("allergies") or None,
                    usual_order_time=row.get("usual_order_time"),
                    last_store_id=last_store_id,
                    reward_points=reward_points,
                )

    def _load_history(self) -> None:
        with self._history_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                customer_id = int(row["customer_id"])
                self._history[customer_id].append(row)

        for rows in self._history.values():
            rows.sort(
                key=lambda r: datetime.strptime(
                    r["timestamp"], "%d-%m-%Y %H:%M"
                ),
                reverse=True,
            )

    def summarize(self, customer_id: int) -> CustomerSummary:
        customer = self._customers.get(customer_id)
        if not customer:
            return CustomerSummary(
                customer_id=customer_id,
                overview="No profile found; rely on live context only.",
            )

        history_rows = self._history.get(customer_id, [])[:5]
        item_counter = Counter(row["item"] for row in history_rows if row.get("item"))
        top_items = ", ".join(item for item, _ in item_counter.most_common(3))

        preferred_items = (
            ", ".join(customer.preferred_drinks)
            if customer.preferred_drinks
            else top_items
        )

        overview_bits = [
            f"{customer.name} prefers {preferred_items or 'seasonal beverages'}"
        ]

        if customer.preferred_size:
            overview_bits.append(f"usually orders {customer.preferred_size} size")

        if customer.usual_order_time:
            overview_bits.append(f"typically visits during {customer.usual_order_time.lower()}")

        if customer.allergies and customer.allergies.lower() not in ("none", ".", ""):
            overview_bits.append(f"allergic to {customer.allergies}")

        if top_items:
            overview_bits.append(f"recent orders include {top_items}")

        loyalty = self._loyalty_tier(customer.reward_points)

        return CustomerSummary(
            customer_id=customer.customer_id,
            overview=". ".join(overview_bits),
            loyalty_level=loyalty,
            preferred_items=preferred_items or None,
            last_store_id=customer.last_store_id,
            reward_points=customer.reward_points,
        )

    @staticmethod
    def _loyalty_tier(points: Optional[int]) -> Optional[str]:
        if points is None:
            return None
        if points >= 1500:
            return "Platinum"
        if points >= 900:
            return "Gold"
        if points >= 400:
            return "Silver"
        return "Bronze"

