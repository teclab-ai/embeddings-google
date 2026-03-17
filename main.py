"""
Interactive CLI demo for the multimodal RAG system.

Usage examples
--------------
# Ingest files then run the REPL
python main.py ingest path/to/file.txt
python main.py ingest path/to/image.jpg --caption "A red running shoe"
python main.py ingest path/to/video.mp4 --frame-interval 60 --max-frames 10
python main.py ingest path/to/audio.mp3
python main.py ingest path/to/audio.wav --max-audio-chunks 20
python main.py ingest path/to/directory --recursive
python main.py ingest data/documents/   # ingest all text docs
python main.py ingest data/videos/      # ingest all videos
python main.py ingest data/audios/      # ingest all audio files

# Ingest a directory then start the REPL
python main.py ingest ./data --recursive && python main.py repl

# Ask a single question
python main.py ask "What is the main topic of the documents?"

# Interactive REPL
python main.py repl

# Show index statistics
python main.py stats

# Clear the entire index
python main.py clear
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from config import settings
from rag import MultimodalRAG

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


def _print_sources(sources: list[dict]) -> None:
    table = Table(title="Retrieved context", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Source", style="green")
    table.add_column("Score", width=6)
    table.add_column("Snippet", no_wrap=False)

    for i, r in enumerate(sources, 1):
        meta = r.get("metadata", {})
        table.add_row(
            str(i),
            meta.get("source_type", "?"),
            meta.get("source_name", "?"),
            f"{r.get('score', 0):.3f}",
            meta.get("text_snippet", "")[:120],
        )
    console.print(table)


def cmd_ingest(args) -> None:
    rag = MultimodalRAG(llm_provider=args.llm)
    kwargs = {}
    if args.caption:
        kwargs["caption"] = args.caption
    if args.frame_interval:
        kwargs["frame_interval"] = args.frame_interval
    if args.max_frames:
        kwargs["max_frames"] = args.max_frames
    if args.max_audio_chunks:
        kwargs["max_chunks"] = args.max_audio_chunks

    import os
    from pathlib import Path

    path = Path(args.path)
    if path.is_dir():
        summary = rag.ingest_directory(path, recursive=args.recursive)
        table = Table(title="Ingestion summary")
        table.add_column("File")
        table.add_column("Items", justify="right")
        for f, n in summary.items():
            table.add_row(f, str(n))
        console.print(table)
    else:
        n = rag.ingest(path, **kwargs)
        console.print(f"[green]Ingested[/green] '{path}' → {n} item(s) stored.")


def cmd_ask(args) -> None:
    rag = MultimodalRAG(llm_provider=args.llm)
    question = " ".join(args.question)
    console.print(Panel(f"[bold]{question}[/bold]", title="Question"))

    answer, sources = rag.query(question, return_sources=True)

    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
    if args.show_sources:
        _print_sources(sources)


def cmd_repl(args) -> None:
    rag = MultimodalRAG(llm_provider=args.llm)
    console.print(
        Panel(
            f"[bold]Multimodal RAG — Interactive Mode[/bold]\n"
            f"Embedding : {settings.EMBEDDING_MODEL}\n"
            f"LLM       : {settings.LLM_PROVIDER} / "
            f"{settings.GEMINI_LLM_MODEL if settings.LLM_PROVIDER == 'gemini' else settings.OPENAI_MODEL}\n"
            f"Pinecone  : {settings.PINECONE_INDEX_NAME}\n\n"
            f"Commands  : :stats  :clear  :sources on/off  :quit",
            title="RAG System",
        )
    )

    show_sources = True
    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not question:
            continue
        if question.lower() in (":quit", ":exit", "exit", "quit"):
            break
        if question == ":stats":
            console.print(rag.index_stats())
            continue
        if question == ":clear":
            if console.input("Are you sure? (yes/no): ").strip().lower() == "yes":
                rag.clear_index()
                console.print("[yellow]Index cleared.[/yellow]")
            continue
        if question.startswith(":sources"):
            show_sources = "on" in question
            console.print(f"Source display: {'on' if show_sources else 'off'}")
            continue

        # Image query: prefix with "image:<path> <question>"
        if question.startswith("image:"):
            parts = question[6:].strip().split(" ", 1)
            img_path = parts[0]
            img_q = parts[1] if len(parts) > 1 else "What is in this image?"
            answer, sources = rag.query_by_image(
                img_path, question=img_q, return_sources=True
            )
        else:
            answer, sources = rag.query(question, return_sources=True)

        console.print(Panel(Markdown(answer), title="[green]Answer[/green]", border_style="green"))
        if show_sources:
            _print_sources(sources)


def cmd_stats(args) -> None:
    rag = MultimodalRAG(llm_provider=args.llm)
    import json
    console.print_json(json.dumps(rag.index_stats(), indent=2))


def cmd_clear(args) -> None:
    rag = MultimodalRAG(llm_provider=args.llm)
    answer = console.input("[red]Delete ALL vectors from the index? (yes/no):[/red] ")
    if answer.strip().lower() == "yes":
        rag.clear_index()
        console.print("[yellow]Index cleared.[/yellow]")
    else:
        console.print("Aborted.")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Multimodal RAG with Gemini Embedding 2 + Pinecone",
    )
    parser.add_argument(
        "--llm",
        default=None,
        choices=["gemini", "openai"],
        help="Override LLM provider (default: from .env)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest files into the index")
    p_ingest.add_argument("path", help="File or directory path")
    p_ingest.add_argument("--caption", help="Caption for images")
    p_ingest.add_argument(
        "--frame-interval", type=int, help="Video: sample every N frames"
    )
    p_ingest.add_argument(
        "--max-frames", type=int, help="Video: max frames to extract"
    )
    p_ingest.add_argument(
        "--max-audio-chunks", type=int,
        help="Audio: max transcript chunks to embed per file",
    )
    p_ingest.add_argument(
        "--recursive", action="store_true", default=True,
        help="Recursively scan directories (default: True)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", nargs="+", help="Question text")
    p_ask.add_argument(
        "--show-sources", action="store_true", default=True,
        help="Print retrieved sources (default: True)",
    )
    p_ask.set_defaults(func=cmd_ask)

    # repl
    p_repl = sub.add_parser("repl", help="Start the interactive REPL")
    p_repl.set_defaults(func=cmd_repl)

    # stats
    p_stats = sub.add_parser("stats", help="Show Pinecone index statistics")
    p_stats.set_defaults(func=cmd_stats)

    # clear
    p_clear = sub.add_parser("clear", help="Delete all vectors from the index")
    p_clear.set_defaults(func=cmd_clear)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
