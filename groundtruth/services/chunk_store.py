from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List
import logging

from ..models import ChunkRecord

logger = logging.getLogger(__name__)


class ChunkStore:
    """In-memory store for chunk metadata backed by the prepared JSONL file."""

    def __init__(self, path: Path):
        self._path = path
        self._chunks: List[ChunkRecord] = []
        self._load()

    def _load(self) -> None:
        with self._path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    # Log and skip malformed lines instead of crashing the app
                    logger.warning(
                        "Skipping malformed JSON in %s at line %d: %s",
                        self._path,
                        line_no,
                        exc,
                    )
                    continue

                try:
                    self._chunks.append(
                        ChunkRecord(
                            chunk_id=payload["chunk_id"],
                            text=payload["text"],
                            meta=payload.get("meta", {}),
                        )
                    )
                except KeyError as exc:
                    logger.warning(
                        "Skipping chunk with missing keys in %s at line %d: %s",
                        self._path,
                        line_no,
                        exc,
                    )
                    continue

    @property
    def chunks(self) -> List[ChunkRecord]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> ChunkRecord:
        return self._chunks[index]

    def iter(self) -> Iterable[ChunkRecord]:
        return iter(self._chunks)

