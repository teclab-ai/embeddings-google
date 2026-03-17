"""
Ingestion pipeline for text, image, video, and audio files.

Text   → chunk → embed each chunk → upsert
Image  → embed (+ optional caption) → upsert
Video  → extract keyframes → embed each frame as image → upsert
Audio  → upload to Gemini Files API → transcribe via LLM → chunk → embed text → upsert
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from google import genai
from tqdm import tqdm

from config import settings
from embedder import GeminiEmbedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)

# File extension → media type
_TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".py"}
_PDF_EXTS  = {".pdf"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus"}
_IMAGE_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
_AUDIO_MIME: dict[str, str] = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
    ".flac": "audio/flac",
    ".ogg":  "audio/ogg",
    ".aac":  "audio/aac",
    ".opus": "audio/opus",
}


def _short_id(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of approximately `size` characters."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    """Encode an OpenCV BGR frame as JPEG bytes."""
    success, buf = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return buf.tobytes()


def _transcribe_audio(client, path: Path, mime_type: str, model: str) -> str:
    """
    Upload an audio file to the Gemini Files API, request a transcript,
    then delete the uploaded file resource.

    Returns the transcript as a single string.
    """
    logger.info("Uploading audio '%s' to Gemini Files API …", path.name)
    uploaded = client.files.upload(file=path, config={"mime_type": mime_type})
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                uploaded,
                "Transcribe this audio accurately and completely. "
                "Return only the spoken text, preserving natural sentence boundaries. "
                "If the audio is non-speech (music, sound effects), describe it briefly.",
            ],
        )
        return response.text.strip()
    finally:
        client.files.delete(name=uploaded.name)
        logger.debug("Deleted uploaded file resource '%s'.", uploaded.name)


class Ingester:
    def __init__(
        self,
        embedder: Optional[GeminiEmbedder] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self._embedder = embedder or GeminiEmbedder()
        self._store = store or VectorStore()
        self._genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    # ── Text ──────────────────────────────────────────────────────────────────

    def ingest_text(self, path: str | Path, encoding: str = "utf-8") -> int:
        """
        Read a text file, chunk it, embed each chunk, and upsert.

        Returns the number of chunks ingested.
        """
        p = Path(path)
        text = p.read_text(encoding=encoding, errors="replace")
        chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        logger.info("Ingesting text '%s' → %d chunks.", p.name, len(chunks))

        vectors = []
        for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {p.name}")):
            vid = f"text::{p.name}::chunk_{i}::{_short_id(chunk)}"
            embedding = self._embedder.embed_text(chunk)
            metadata = {
                "source_type": "text",
                "source_path": str(p),
                "source_name": p.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text_snippet": chunk[:500],
            }
            vectors.append((vid, embedding, metadata))

        self._store.upsert(vectors)
        logger.info("Ingested %d text chunks from '%s'.", len(chunks), p.name)
        return len(chunks)

    def ingest_text_string(self, text: str, doc_id: str) -> int:
        """Ingest a raw text string with a caller-supplied identifier."""
        chunks = _chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        vectors = []
        for i, chunk in enumerate(chunks):
            vid = f"text::{doc_id}::chunk_{i}::{_short_id(chunk)}"
            embedding = self._embedder.embed_text(chunk)
            metadata = {
                "source_type": "text",
                "source_path": doc_id,
                "source_name": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text_snippet": chunk[:500],
            }
            vectors.append((vid, embedding, metadata))
        self._store.upsert(vectors)
        return len(chunks)

    # ── Image ─────────────────────────────────────────────────────────────────

    def ingest_image(
        self, path: str | Path, caption: Optional[str] = None
    ) -> None:
        """
        Embed an image file and upsert it.

        Parameters
        ----------
        path    : path to image file
        caption : optional human-readable description embedded alongside the image
        """
        p = Path(path)
        suffix = p.suffix.lower()
        mime_type = _IMAGE_MIME.get(suffix)
        if mime_type is None:
            raise ValueError(f"Unsupported image type: {suffix}")

        logger.info("Ingesting image '%s'.", p.name)
        image_bytes = p.read_bytes()
        embedding = self._embedder.embed_image_bytes(
            image_bytes, mime_type, caption=caption
        )

        vid = f"image::{p.name}::{_short_id(str(p))}"
        metadata = {
            "source_type": "image",
            "source_path": str(p),
            "source_name": p.name,
            "mime_type": mime_type,
            "caption": caption or "",
            "text_snippet": caption or p.name,
        }
        self._store.upsert_one(vid, embedding, metadata)
        logger.info("Ingested image '%s'.", p.name)

    # ── Video ─────────────────────────────────────────────────────────────────

    def ingest_video(
        self,
        path: str | Path,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> int:
        """
        Extract keyframes from a video and embed each as an image.

        Parameters
        ----------
        path           : path to video file
        frame_interval : sample every N frames (default: settings.VIDEO_FRAME_INTERVAL)
        max_frames     : cap on frames extracted (default: settings.MAX_FRAMES_PER_VIDEO)

        Returns the number of frames ingested.
        """
        p = Path(path)
        interval = frame_interval or settings.VIDEO_FRAME_INTERVAL
        cap_frames = max_frames or settings.MAX_FRAMES_PER_VIDEO

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {p}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps
        logger.info(
            "Video '%s': %.1f s, %d total frames, FPS=%.1f",
            p.name, duration_s, total_frames, fps,
        )

        vectors = []
        frame_idx = 0
        sampled = 0

        with tqdm(total=min(cap_frames, max(1, total_frames // interval)),
                  desc=f"Frames {p.name}") as pbar:
            while cap.isOpened() and sampled < cap_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    timestamp_s = frame_idx / fps
                    jpeg_bytes = _frame_to_jpeg_bytes(frame)
                    caption = f"Frame at {timestamp_s:.1f}s from video {p.name}"
                    embedding = self._embedder.embed_image_bytes(
                        jpeg_bytes, "image/jpeg", caption=caption
                    )
                    vid = f"video::{p.name}::frame_{frame_idx}::{_short_id(str(p))}"
                    metadata = {
                        "source_type": "video_frame",
                        "source_path": str(p),
                        "source_name": p.name,
                        "frame_index": frame_idx,
                        "timestamp_s": round(timestamp_s, 2),
                        "total_frames": total_frames,
                        "fps": round(fps, 2),
                        "caption": caption,
                        "text_snippet": caption,
                    }
                    vectors.append((vid, embedding, metadata))
                    sampled += 1
                    pbar.update(1)
                frame_idx += 1

        cap.release()

        if vectors:
            self._store.upsert(vectors)
            logger.info(
                "Ingested %d frames from '%s'.", len(vectors), p.name
            )
        return len(vectors)

    # ── PDF ───────────────────────────────────────────────────────────────────

    def ingest_pdf(self, path: str | Path) -> int:
        """
        Extract text and page images from a PDF.

        - Text: chunked → embedded as text → upserted
        - Pages: rendered as JPEG → embedded as images → upserted (saved to
          data/cache/<stem>/ for later display in the UI)

        Returns total items ingested (text chunks + page images).
        """
        import fitz  # PyMuPDF
        from pypdf import PdfReader

        p = Path(path)

        # ── Text chunks ───────────────────────────────────────────────────────
        reader = PdfReader(str(p))
        pages_text = [pg.extract_text() or "" for pg in reader.pages]

        # Build full text and track each page's character range
        page_offsets: list[tuple[int, int, int]] = []  # (page_num, start, end)
        full_text = ""
        for page_num, text in enumerate(pages_text):
            if text.strip():
                start = len(full_text)
                full_text += text + "\n\n"
                page_offsets.append((page_num, start, len(full_text)))

        def _page_for_offset(offset: int) -> int:
            for pn, start, end in page_offsets:
                if start <= offset < end:
                    return pn
            return page_offsets[-1][0] if page_offsets else 0

        text_vectors = []
        if full_text.strip():
            chunks = _chunk_text(full_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
            # Recompute chunk start positions (mirrors _chunk_text logic)
            step = settings.CHUNK_SIZE - settings.CHUNK_OVERLAP
            chunk_starts = [i * step for i in range(len(chunks))]

            logger.info("Ingesting PDF '%s' → %d text chunks.", p.name, len(chunks))
            for i, chunk in enumerate(tqdm(chunks, desc=f"Text {p.name}")):
                vid = f"pdf::{p.name}::chunk_{i}::{_short_id(chunk)}"
                embedding = self._embedder.embed_text(chunk)
                text_vectors.append((vid, embedding, {
                    "source_type": "pdf",
                    "source_path": str(p),
                    "source_name": p.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "page_number": _page_for_offset(chunk_starts[i]),
                    "text_snippet": chunk[:500],
                }))
            self._store.upsert(text_vectors)

        # ── Page images ───────────────────────────────────────────────────────
        cache_dir = Path("data/cache") / p.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(p))
        total_pages = len(doc)
        page_vectors = []

        for page_num in tqdm(range(total_pages), desc=f"Pages {p.name}"):
            page = doc[page_num]
            mat = fitz.Matrix(1.5, 1.5)  # 150 % zoom → reasonable quality
            pix = page.get_pixmap(matrix=mat)
            jpeg_bytes = pix.tobytes("jpeg")

            img_path = cache_dir / f"page_{page_num:04d}.jpg"
            img_path.write_bytes(jpeg_bytes)

            caption = f"Page {page_num + 1} of {p.name}"
            embedding = self._embedder.embed_image_bytes(jpeg_bytes, "image/jpeg", caption=caption)
            vid = f"pdf_page::{p.name}::page_{page_num}::{_short_id(str(p))}"
            page_vectors.append((vid, embedding, {
                "source_type": "pdf_page",
                "source_path": str(p),
                "source_name": p.name,
                "page_number": page_num,
                "image_path": str(img_path),
                "text_snippet": caption,
            }))

        doc.close()
        if page_vectors:
            self._store.upsert(page_vectors)
        logger.info(
            "Ingested PDF '%s': %d text chunks + %d page images.",
            p.name, len(text_vectors), len(page_vectors),
        )
        return len(text_vectors) + len(page_vectors)

    # ── Audio ─────────────────────────────────────────────────────────────────

    def ingest_audio(
        self,
        path: str | Path,
        max_chunks: Optional[int] = None,
    ) -> int:
        """
        Transcribe an audio file via Gemini, chunk the transcript, embed each
        chunk as text, and upsert.

        Parameters
        ----------
        path       : path to audio file
        max_chunks : cap on chunks to embed (default: settings.MAX_AUDIO_CHUNKS)

        Returns the number of chunks ingested.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        mime_type = _AUDIO_MIME.get(suffix)
        if mime_type is None:
            raise ValueError(f"Unsupported audio type: {suffix}")

        cap = max_chunks if max_chunks is not None else settings.MAX_AUDIO_CHUNKS

        transcript = _transcribe_audio(
            client=self._genai_client,
            path=p,
            mime_type=mime_type,
            model=settings.AUDIO_TRANSCRIPTION_MODEL,
        )

        if not transcript:
            logger.warning("Empty transcript for '%s'; skipping.", p.name)
            return 0

        chunks = _chunk_text(transcript, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if len(chunks) > cap:
            logger.warning(
                "Audio '%s' produced %d chunks; capping at %d.",
                p.name, len(chunks), cap,
            )
            chunks = chunks[:cap]

        logger.info("Ingesting audio '%s' → %d transcript chunks.", p.name, len(chunks))

        vectors = []
        for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {p.name}")):
            vid = f"audio::{p.name}::chunk_{i}::{_short_id(chunk)}"
            embedding = self._embedder.embed_text(chunk)
            metadata = {
                "source_type": "audio",
                "source_path": str(p),
                "source_name": p.name,
                "mime_type": mime_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text_snippet": chunk[:500],
            }
            vectors.append((vid, embedding, metadata))

        self._store.upsert(vectors)
        logger.info("Ingested %d audio transcript chunks from '%s'.", len(vectors), p.name)
        return len(vectors)

    # ── Auto-detect ───────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path, **kwargs) -> int:
        """
        Ingest any supported file, auto-detecting the type by extension.

        kwargs are forwarded to the specific ingestion method.
        Returns the number of items (chunks / frames / 1) ingested.
        """
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in _TEXT_EXTS:
            return self.ingest_text(p, **{k: v for k, v in kwargs.items() if k in ("encoding",)})
        elif suffix in _PDF_EXTS:
            return self.ingest_pdf(p)
        elif suffix in _IMAGE_EXTS:
            self.ingest_image(p, **{k: v for k, v in kwargs.items() if k in ("caption",)})
            return 1
        elif suffix in _VIDEO_EXTS:
            return self.ingest_video(p, **{k: v for k, v in kwargs.items() if k in ("frame_interval", "max_frames")})
        elif suffix in _AUDIO_EXTS:
            return self.ingest_audio(p, **{k: v for k, v in kwargs.items() if k in ("max_chunks",)})
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: text={_TEXT_EXTS}, image={_IMAGE_EXTS}, "
                f"video={_VIDEO_EXTS}, audio={_AUDIO_EXTS}"
            )

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> dict:
        """
        Ingest all supported files in a directory.

        Returns a summary dict: {path: items_ingested or error_message}
        """
        d = Path(directory)
        glob = d.rglob("*") if recursive else d.glob("*")
        all_exts = _TEXT_EXTS | _PDF_EXTS | _IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS
        files = [f for f in glob if f.is_file() and f.suffix.lower() in all_exts]

        summary: dict[str, int | str] = {}
        for f in files:
            try:
                n = self.ingest_file(f)
                summary[str(f)] = n
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", f, exc)
                summary[str(f)] = f"ERROR: {exc}"
        return summary
