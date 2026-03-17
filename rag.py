"""
End-to-end multimodal RAG pipeline.

        ┌─────────────┐     embed     ┌──────────────┐    nearest   ┌───────┐
Query ──►  Embedder   ├──────────────►  Pinecone     ├─────────────► LLM   ├──► Answer
(text/   └─────────────┘              └──────────────┘   context     └───────┘
 image)

Ingestion path:
        ┌──────────┐   chunk/frame   ┌─────────────┐     embed     ┌──────────────┐
Files ──►  Ingester ├────────────────► Embedder     ├──────────────► Pinecone      │
(text/  └──────────┘                 └─────────────┘               └──────────────┘
 image/
 video)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config import settings
from embedder import GeminiEmbedder
from ingestion import Ingester
from llm import BaseLLM, build_llm
from retriever import Retriever
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class MultimodalRAG:
    """
    Convenience façade that wires together ingestion, retrieval, and generation.

    Example
    -------
    >>> rag = MultimodalRAG()
    >>> rag.ingest("docs/report.txt")
    >>> rag.ingest("photos/product.jpg", caption="A red running shoe")
    >>> rag.ingest("videos/demo.mp4")
    >>> answer = rag.query("What does the product look like?")
    >>> print(answer)
    """

    def __init__(self, llm_provider: Optional[str] = None) -> None:
        embedder = GeminiEmbedder()
        store = VectorStore()

        self._ingester = Ingester(embedder=embedder, store=store)
        self._retriever = Retriever(embedder=embedder, store=store)
        self._llm: BaseLLM = build_llm(llm_provider)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, path: str | Path, **kwargs) -> int:
        """
        Ingest any supported file (text / image / video).

        kwargs are forwarded to the underlying ingestion method:
          - caption (str)          → for images
          - frame_interval (int)   → for videos
          - max_frames (int)       → for videos
          - encoding (str)         → for text files
        """
        return self._ingester.ingest_file(path, **kwargs)

    def ingest_text(self, text: str, doc_id: str) -> int:
        """Ingest a raw text string."""
        return self._ingester.ingest_text_string(text, doc_id)

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict:
        """Ingest all supported files under a directory."""
        return self._ingester.ingest_directory(directory, recursive=recursive)

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        source_type_filter: Optional[str] = None,
        return_sources: bool = False,
    ) -> str | tuple[str, list[dict]]:
        """
        Answer a natural-language question using the indexed documents.

        Parameters
        ----------
        question           : user question
        top_k              : number of context items to retrieve
        source_type_filter : restrict retrieval to "text", "image", "video_frame"
        return_sources     : if True, returns (answer, results) tuple

        Returns
        -------
        answer (str) or (answer, sources) if return_sources=True
        """
        k = top_k or settings.TOP_K
        results = self._retriever.retrieve_by_text(
            question, top_k=k, source_type_filter=source_type_filter
        )
        context = self._retriever.format_context(results)
        logger.debug("Context for LLM:\n%s", context)

        answer = self._llm.answer(question, context)

        if return_sources:
            return answer, results
        return answer

    def query_by_image(
        self,
        image_path: str | Path,
        question: str = "What is in this image and what related information do you have?",
        top_k: int | None = None,
        return_sources: bool = False,
    ) -> str | tuple[str, list[dict]]:
        """
        Use an image as the retrieval query, then answer `question` in context.
        """
        k = top_k or settings.TOP_K
        results = self._retriever.retrieve_by_image(
            image_path, caption=question, top_k=k
        )
        context = self._retriever.format_context(results)
        answer = self._llm.answer(question, context)

        if return_sources:
            return answer, results
        return answer

    # ── Utilities ─────────────────────────────────────────────────────────────

    def index_stats(self) -> dict:
        """Return Pinecone index statistics."""
        return self._ingester._store.stats()

    def clear_index(self) -> None:
        """Delete all vectors (use with caution)."""
        self._ingester._store.delete_all()
        logger.warning("Index cleared.")
