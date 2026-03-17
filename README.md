# embeddings-google

A multimodal Retrieval-Augmented Generation (RAG) system powered by **Google Gemini embeddings**, **Pinecone** vector storage, and a choice of LLM backend (Claude, Gemini, or OpenAI).

## Architecture

```
                    ┌─────────────┐     embed     ┌──────────────┐   nearest   ┌───────┐
Query (text/image) ─►  Embedder   ├───────────────►  Pinecone     ├────────────► LLM   ├──► Answer
                    └─────────────┘               └──────────────┘   context   └───────┘

                    ┌──────────┐   chunk/frame   ┌─────────────┐     embed     ┌──────────────┐
Files ─────────────►  Ingester ├─────────────────►  Embedder   ├───────────────► Pinecone     │
(text/image/video/ └──────────┘                  └─────────────┘               └──────────────┘
 audio/PDF)
```

## Features

- **Multimodal ingestion** — text, images, PDFs (text + page images), videos (keyframes), audio (transcription)
- **Google Gemini embeddings** — `gemini-embedding-2-preview` (3072-dimensional)
- **Pinecone** serverless vector store with cosine similarity search
- **Flexible LLM backend** — Claude (default), Gemini, or OpenAI, switchable via config
- **Gradio web UI** — chat, file upload/ingest, and index stats tabs
- **Rich CLI** — single queries, interactive REPL, batch ingest, index management

## Requirements

- Python 3.12+
- API keys:
  - **Google AI** (Gemini embeddings + optional LLM/audio transcription)
  - **Pinecone** (vector store)
  - **Anthropic** (if using Claude — default)
  - **OpenAI** (if using GPT-4o)

## Setup

```bash
# 1. Clone
git clone https://github.com/teclab-ai/embeddings-google.git
cd embeddings-google

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and fill in your API keys
```

## Usage

### Web UI

```bash
.venv/bin/python app.py
# Open http://127.0.0.1:7860
```

Three tabs:
- **Chat** — ask questions about indexed content; retrieved sources shown below answers
- **Ingest** — upload files (drag & drop); optional caption for images
- **Stats** — live Pinecone index statistics

### CLI

```bash
# Ingest a single file
.venv/bin/python main.py ingest data/documents/report.pdf

# Ingest an entire folder recursively
.venv/bin/python main.py ingest data/ --recursive

# Ask a single question
.venv/bin/python main.py ask "What does the product manual say about temperature settings?"

# Interactive REPL
.venv/bin/python main.py repl
# REPL special commands:
#   :stats          — show index statistics
#   :clear          — delete all vectors
#   :sources on/off — toggle source display
#   image:<path> <question> — image-based query

# Index statistics
.venv/bin/python main.py stats

# Clear the index
.venv/bin/python main.py clear
```

## Supported File Types

| Type   | Extensions |
|--------|-----------|
| Text   | `.txt` `.md` `.rst` `.csv` `.json` `.xml` `.html` `.py` |
| Image  | `.jpg` `.jpeg` `.png` `.webp` `.gif` `.bmp` |
| PDF    | `.pdf` (text chunks + page images) |
| Video  | `.mp4` `.avi` `.mov` `.mkv` `.webm` `.flv` |
| Audio  | `.mp3` `.wav` `.m4a` `.flac` `.ogg` `.aac` `.opus` |

## Data Folders

```
data/
├── documents/   # Text and PDF files
├── images/      # Standalone image files
├── videos/      # Video files
├── audios/      # Audio files
└── cache/       # Auto-generated PDF page renders (do not edit)
```

## Project Structure

```
embeddings-google/
├── app.py           # Gradio web UI
├── main.py          # CLI interface
├── rag.py           # End-to-end RAG façade
├── ingestion.py     # File ingestion pipeline (text/image/video/audio/PDF)
├── embedder.py      # Google Gemini embedding wrapper
├── vector_store.py  # Pinecone vector store
├── retriever.py     # Similarity search and context formatting
├── llm.py           # LLM abstraction (Claude / Gemini / OpenAI)
├── config.py        # Pydantic settings (reads from .env)
├── requirements.txt # Python dependencies
└── .env.example     # Environment variable template
```

## LLM Provider

Set `LLM_PROVIDER` in `.env` to switch backends:

| Value     | Model (default)       |
|-----------|-----------------------|
| `claude`  | `claude-sonnet-4-6`   |
| `gemini`  | `gemini-2.0-flash`    |
| `openai`  | `gpt-4o`              |
