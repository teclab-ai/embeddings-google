from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Google / Gemini ──────────────────────────────────────────────────────
    GOOGLE_API_KEY: str
    EMBEDDING_MODEL: str = "gemini-embedding-2-preview"
    EMBEDDING_DIMENSION: int = 3072

    # ── Pinecone ─────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "multimodal-rag"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    # ── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: Literal["gemini", "openai", "claude"] = "claude"
    GEMINI_LLM_MODEL: str = "gemini-2.0-flash"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # ── Anthropic / Claude ────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    # ── Video processing ─────────────────────────────────────────────────────
    VIDEO_FRAME_INTERVAL: int = 30   # sample every N frames
    MAX_FRAMES_PER_VIDEO: int = 20   # cap on frames per video

    # ── Audio processing ─────────────────────────────────────────────────────
    MAX_AUDIO_CHUNKS: int = 50                           # cap on transcript chunks per audio file
    AUDIO_TRANSCRIPTION_MODEL: str = "gemini-2.0-flash"  # model used to transcribe audio

    # ── Text chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K: int = 5


settings = Settings()
