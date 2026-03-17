"""
Multimodal embedder using Google Gemini Embedding 2.

Supports:
  - Plain text / text chunks
  - Images (JPEG, PNG, WEBP, GIF, BMP) — passed as inline base64 bytes
  - Video frames — caller extracts frames first (see ingestion.py)

The model `gemini-embedding-2-preview` accepts a sequence of Parts so you can
embed an image together with an optional text caption in a single forward pass.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from config import settings

logger = logging.getLogger(__name__)

# Supported image MIME types
_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


class GeminiEmbedder:
    """Wraps the Gemini Embedding 2 model for text, image, and video-frame input."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model = settings.EMBEDDING_MODEL

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _embed_parts(self, parts: list[types.Part]) -> list[float]:
        """Call the embeddings API with arbitrary Part list."""
        content = types.Content(parts=parts)
        result = self._client.models.embed_content(
            model=self._model,
            contents=content,
        )
        return list(result.embeddings[0].values)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a text string."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> list[float]:
        """
        Embed raw image bytes, optionally paired with a text caption.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data (e.g. JPEG/PNG bytes).
        mime_type : str
            MIME type, e.g. "image/jpeg".
        caption : str, optional
            Short description embedded alongside the image.
        """
        parts: list[types.Part] = [
            types.Part(
                inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
            )
        ]
        if caption:
            parts.append(types.Part(text=caption))
        return self._embed_parts(parts)

    def embed_image_file(
        self, path: str | Path, caption: Optional[str] = None
    ) -> list[float]:
        """Embed an image file by path."""
        p = Path(path)
        mime_type = _IMAGE_MIME.get(p.suffix.lower())
        if mime_type is None:
            raise ValueError(
                f"Unsupported image extension '{p.suffix}'. "
                f"Supported: {list(_IMAGE_MIME)}"
            )
        image_bytes = p.read_bytes()
        return self.embed_image_bytes(image_bytes, mime_type, caption=caption)

    def embed_image_base64(
        self,
        b64_data: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> list[float]:
        """Embed an image supplied as a base64-encoded string."""
        image_bytes = base64.b64decode(b64_data)
        return self.embed_image_bytes(image_bytes, mime_type, caption=caption)

    def embed_text_and_image(
        self,
        text: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> list[float]:
        """
        Embed text and image together in a single call.
        Useful for query expansion: "what is shown here? <image>".
        """
        parts = [
            types.Part(text=text),
            types.Part(
                inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
            ),
        ]
        return self._embed_parts(parts)
