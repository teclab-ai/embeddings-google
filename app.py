"""
Gradio web app for the multimodal RAG system.

Run:
    python app.py
    # then open http://127.0.0.1:7860

Tabs
----
Chat   — ask questions, see answers + retrieved sources
Ingest — upload files (text / image / video / audio / PDF) to index them
Stats  — Pinecone index statistics
"""

from __future__ import annotations

import base64
import logging
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import gradio as gr
from google import genai
from google.genai import types as genai_types

from config import settings
from ingestion import _IMAGE_EXTS, _PDF_EXTS, _VIDEO_EXTS, _AUDIO_EXTS
from rag import MultimodalRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Initialise RAG (connect to Pinecone + Gemini once at startup) ─────────────
rag = MultimodalRAG(llm_provider="claude")
_genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# Permanent data directories (created if absent)
_DATA_DIRS = {
    "image":  Path("data/images"),
    "pdf":    Path("data/documents"),
    "text":   Path("data/documents"),
    "video":  Path("data/videos"),
    "audio":  Path("data/audios"),
}
for _d in _DATA_DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dest_for(path: Path) -> Path:
    """Return the permanent data subfolder for a given file extension."""
    s = path.suffix.lower()
    if s in _IMAGE_EXTS:
        return _DATA_DIRS["image"]
    if s in _PDF_EXTS:
        return _DATA_DIRS["pdf"]
    if s in _VIDEO_EXTS:
        return _DATA_DIRS["video"]
    if s in _AUDIO_EXTS:
        return _DATA_DIRS["audio"]
    return _DATA_DIRS["text"]


def _copy_to_data(tmp_path: Path) -> Path:
    """Copy an uploaded file to its permanent data folder and return the new path."""
    dest_dir = _dest_for(tmp_path)
    dest = dest_dir / tmp_path.name
    # Avoid overwriting with a different file
    if dest.exists() and dest.stat().st_size != tmp_path.stat().st_size:
        stem, suffix = tmp_path.stem, tmp_path.suffix
        i = 1
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(tmp_path, dest)
    return dest


def _gemini_caption(image_path: Path) -> str:
    """Ask Gemini to describe an image and return the caption."""
    try:
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp",
            ".gif": "image/gif",  ".bmp":  "image/bmp",
        }.get(image_path.suffix.lower(), "image/jpeg")
        img_bytes = image_path.read_bytes()
        response = _genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                genai_types.Part(inline_data=genai_types.Blob(mime_type=mime, data=img_bytes)),
                genai_types.Part(text="Describe this image concisely in one or two sentences, including any text or brand names visible."),
            ],
        )
        return response.text.strip()
    except Exception as exc:
        logging.getLogger(__name__).warning("Caption generation failed: %s", exc)
        return ""


def _crop_page_to_text(
    pdf_path: str, page_num: int, text: str, min_context: int = 300
) -> bytes | None:
    """Render a full-width strip of `page_num` centred on `text`."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        rects = page.search_for(text[:80])
        if rects:
            bbox = rects[0]
            for r in rects[1:]:
                bbox = bbox | r
            cy = (bbox.y0 + bbox.y1) / 2
            half = max(min_context, (bbox.y1 - bbox.y0) * 3)
            clip = fitz.Rect(
                0,
                max(0, cy - half),
                page.rect.width,
                min(page.rect.height, cy + half),
            )
        else:
            clip = page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=clip)
        doc.close()
        return pix.tobytes("jpeg")
    except Exception:
        return None


def _format_sources_md(results: list[dict]) -> str:
    """Render retrieved context as Markdown for display."""
    if not results:
        return "_No sources retrieved._"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score = r.get("score", 0.0)
        src_type = meta.get("source_type", "?")
        name = meta.get("source_name", "?")
        snippet = meta.get("text_snippet", "")[:300]
        extra = (
            f" @ {meta['timestamp_s']}s"
            if src_type == "video_frame" and "timestamp_s" in meta
            else ""
        )
        lines.append(f"**[{i}] `{src_type}` · {name}{extra}** — score: `{score:.3f}`")
        if src_type == "pdf" and "page_number" in meta and Path(meta.get("source_path", "")).exists():
            jpeg = _crop_page_to_text(
                meta["source_path"],
                meta["page_number"],
                meta.get("text_snippet", ""),
            )
            if jpeg:
                b64 = base64.b64encode(jpeg).decode()
                lines.append(f'<img src="data:image/jpeg;base64,{b64}" width="480"/>')
            elif snippet:
                lines.append(f"> {snippet}")
        elif src_type == "image" and Path(meta.get("source_path", "")).exists():
            img_bytes = Path(meta["source_path"]).read_bytes()
            mime = meta.get("mime_type", "image/jpeg")
            b64 = base64.b64encode(img_bytes).decode()
            lines.append(f'<img src="data:{mime};base64,{b64}" width="480"/>')
            if snippet:
                lines.append(f"> {snippet}")
        elif snippet:
            lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat(message: str, history: list[dict]) -> tuple[list, str]:
    if not message.strip():
        return history, ""
    try:
        answer, sources = rag.query(message, return_sources=True)
        sources_md = _format_sources_md(sources)
    except Exception as exc:
        answer = f"⚠️ Error: {exc}"
        sources_md = ""
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, sources_md


def clear_chat() -> tuple[list, str]:
    return [], ""


# ── Ingest ────────────────────────────────────────────────────────────────────

def on_upload(files) -> str:
    """
    Called as soon as files are dropped into the upload widget.
    Copies files to data/ and returns an auto-generated caption for the
    first image found (shown in the caption textbox for the user to review).
    """
    if not files:
        return ""
    for f in files:
        p = Path(f.name)
        if p.suffix.lower() in _IMAGE_EXTS:
            permanent = _copy_to_data(p)
            caption = _gemini_caption(permanent)
            return caption
    return ""


def ingest_files(files, caption: str) -> str:
    """Ingest uploaded files into the vector index."""
    if not files:
        return "No files selected."
    lines: list[str] = []
    for f in files:
        tmp = Path(f.name)
        # Copy to permanent location
        try:
            permanent = _copy_to_data(tmp)
        except Exception as exc:
            lines.append(f"❌  {tmp.name} — copy failed: {exc}")
            continue
        kwargs = {"caption": caption.strip()} if caption.strip() and tmp.suffix.lower() in _IMAGE_EXTS else {}
        try:
            n = rag.ingest(permanent, **kwargs)
            lines.append(f"✅  {permanent.name} — {n} item(s) ingested (saved to {permanent.parent})")
        except Exception as exc:
            lines.append(f"❌  {permanent.name} — {exc}")
    return "\n".join(lines)


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    try:
        return rag.index_stats()
    except Exception as exc:
        return {"error": str(exc)}


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multimodal RAG") as demo:
    gr.Markdown(
        """
# 🔍 Multimodal RAG
**Embedding:** `gemini-embedding-2-preview` · **LLM:** `claude-sonnet-4-6` · **Store:** Pinecone
        """
    )

    with gr.Tabs():

        # ── Chat tab ──────────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="Conversation", height=460)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about your documents, images, or videos…",
                    label="",
                    scale=5,
                    autofocus=True,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            with gr.Accordion("📄 Retrieved Sources", open=False):
                sources_box = gr.Markdown(value="_Sources will appear here after each query._")

            send_btn.click(
                fn=chat,
                inputs=[msg_box, chatbot],
                outputs=[chatbot, sources_box],
            ).then(lambda: "", outputs=msg_box)

            msg_box.submit(
                fn=chat,
                inputs=[msg_box, chatbot],
                outputs=[chatbot, sources_box],
            ).then(lambda: "", outputs=msg_box)

            clear_btn.click(fn=clear_chat, outputs=[chatbot, sources_box])

        # ── Ingest tab ────────────────────────────────────────────────────────
        with gr.Tab("📥 Ingest"):
            gr.Markdown(
                "Upload one or more files to embed and store. "
                "Supports **text** (.txt .md .csv …), **PDFs** (.pdf), **images** (.jpg .png …), "
                "**videos** (.mp4 .mov …), and **audio** (.mp3 .wav .m4a …). "
                "Files are saved permanently to `data/`. "
                "A caption is auto-generated for images — review and edit it before ingesting."
            )
            upload_widget = gr.File(
                label="Files",
                file_count="multiple",
                file_types=[
                    ".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html",
                    ".pdf",
                    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
                    ".mp4", ".avi", ".mov", ".mkv", ".webm",
                    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus",
                ],
            )
            caption_box = gr.Textbox(
                label="Caption (images only — auto-generated, edit if needed)",
                placeholder="Upload an image to auto-generate a caption…",
                lines=2,
            )
            ingest_btn = gr.Button("Ingest Files", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=6)

            # Auto-generate caption on upload
            upload_widget.upload(
                fn=on_upload,
                inputs=[upload_widget],
                outputs=[caption_box],
            )

            ingest_btn.click(
                fn=ingest_files,
                inputs=[upload_widget, caption_box],
                outputs=status_box,
            )

        # ── Stats tab ─────────────────────────────────────────────────────────
        with gr.Tab("📊 Stats"):
            gr.Markdown("Pinecone index statistics.")
            refresh_btn = gr.Button("Refresh", variant="secondary")
            stats_display = gr.JSON(label="Index Stats")

            refresh_btn.click(fn=get_stats, outputs=stats_display)
            demo.load(fn=get_stats, outputs=stats_display)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
