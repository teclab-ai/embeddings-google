"""
Retriever — converts a query (text or image) to an embedding and searches Pinecone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import settings
from embedder import GeminiEmbedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self._embedder = embedder or GeminiEmbedder()
        self._store = store or VectorStore()

    def retrieve_by_text(
        self,
        query: str,
        top_k: int | None = None,
        source_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed a text query and return the top-k most relevant items.

        Parameters
        ----------
        query              : natural-language question or keyword string
        top_k              : number of results (default: settings.TOP_K)
        source_type_filter : restrict to "text", "image", or "video_frame"
        """
        embedding = self._embedder.embed_text(query)
        filter_dict = None
        if source_type_filter:
            filter_dict = {"source_type": {"$eq": source_type_filter}}
        results = self._store.query(embedding, top_k=top_k, filter=filter_dict)
        logger.debug("Text query returned %d results.", len(results))
        return results

    def retrieve_by_image(
        self,
        image_path: str | Path,
        caption: Optional[str] = None,
        top_k: int | None = None,
        source_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed an image (with optional text caption) and return the top-k results.

        Useful for reverse image search or "find images similar to this one".
        """
        embedding = self._embedder.embed_image_file(image_path, caption=caption)
        filter_dict = None
        if source_type_filter:
            filter_dict = {"source_type": {"$eq": source_type_filter}}
        results = self._store.query(embedding, top_k=top_k, filter=filter_dict)
        logger.debug("Image query returned %d results.", len(results))
        return results

    def retrieve_by_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str,
        caption: Optional[str] = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Embed raw image bytes and return top-k results."""
        embedding = self._embedder.embed_image_bytes(
            image_bytes, mime_type, caption=caption
        )
        return self._store.query(embedding, top_k=top_k)

    def format_context(self, results: list[dict]) -> str:
        """
        Convert retrieval results into a compact string for LLM context.

        Each entry includes source type, source name, similarity score,
        and the stored text snippet.
        """
        if not results:
            return "No relevant context found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            score = r.get("score", 0.0)
            src_type = meta.get("source_type", "unknown")
            src_name = meta.get("source_name", meta.get("source_path", "?"))
            snippet = meta.get("text_snippet", "")

            header = f"[{i}] [{src_type}] {src_name}  (score={score:.3f})"
            if src_type == "video_frame":
                ts = meta.get("timestamp_s", "?")
                header += f"  @ {ts}s"
            lines.append(header)
            if snippet:
                lines.append(f"    {snippet[:400]}")
            lines.append("")

        return "\n".join(lines).rstrip()
