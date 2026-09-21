"""Central runtime configuration, loaded from environment variables / .env.

Every other module reads configuration through `get_settings()` rather than
touching `os.environ` directly, so config sourcing can change later (e.g. a
FastAPI settings dependency in Phase 2) without touching business logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


# Models offered in the UI model pickers (override with a comma-separated AVAILABLE_MODELS).
DEFAULT_AVAILABLE_MODELS = "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    judge_model: str
    available_models: tuple[str, ...]
    chroma_persist_dir: str
    chroma_collection_name: str
    chunk_size: int
    chunk_overlap: int
    default_top_k: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        available_models=tuple(
            m.strip() for m in os.getenv("AVAILABLE_MODELS", DEFAULT_AVAILABLE_MODELS).split(",") if m.strip()
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "support_tickets"),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
    )


def model_choices(current: str | None = None) -> list[str]:
    """Models to offer in a picker, with `current` (e.g. the .env default) always included."""
    choices = list(get_settings().available_models)
    if current and current not in choices:
        choices.insert(0, current)
    return choices
