"""
Pinecone vector store wrapper.

Index layout
------------
Each vector has:
  - id        : unique string  (e.g. "text::doc.txt::chunk_3")
  - values    : float list of length EMBEDDING_DIMENSION
  - metadata  : dict with at least {source_type, source_path, text_snippet}

source_type values: "text" | "image" | "video_frame"
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._index_name = settings.PINECONE_INDEX_NAME
        self._index = self._get_or_create_index()

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _get_or_create_index(self):
        existing = {idx.name for idx in self._pc.list_indexes()}
        if self._index_name not in existing:
            logger.info(
                "Creating Pinecone index '%s' (dim=%d) …",
                self._index_name,
                settings.EMBEDDING_DIMENSION,
            )
            self._pc.create_index(
                name=self._index_name,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
            # Wait until the index is ready
            while not self._pc.describe_index(self._index_name).status.get("ready"):
                logger.info("Waiting for index to become ready …")
                time.sleep(2)
        else:
            logger.info("Using existing Pinecone index '%s'.", self._index_name)

        return self._pc.Index(self._index_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
        batch_size: int = 100,
    ) -> None:
        """
        Upsert vectors into the index.

        Parameters
        ----------
        vectors : list of (id, embedding, metadata)
        batch_size : int
            Number of vectors per Pinecone upsert call.
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            records = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in batch
            ]
            self._index.upsert(vectors=records)
            logger.debug("Upserted %d vectors (batch %d).", len(batch), i // batch_size)

    def upsert_one(
        self, vid: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        self.upsert([(vid, embedding, metadata)])

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k nearest neighbours.

        Each result dict contains: id, score, metadata.
        """
        k = top_k or settings.TOP_K
        kwargs: dict[str, Any] = {
            "vector": embedding,
            "top_k": k,
            "include_metadata": True,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        return [
            {
                "id": m.id,
                "score": m.score,
                "metadata": m.metadata or {},
            }
            for m in response.matches
        ]

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_by_source(self, source_path: str) -> None:
        """Delete all vectors whose metadata.source_path matches."""
        self._index.delete(filter={"source_path": {"$eq": source_path}})
        logger.info("Deleted vectors for source '%s'.", source_path)

    def delete_all(self) -> None:
        """Wipe the entire index namespace."""
        self._index.delete(delete_all=True)
        logger.warning("Deleted ALL vectors from index '%s'.", self._index_name)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return self._index.describe_index_stats().to_dict()
