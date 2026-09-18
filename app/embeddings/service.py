"""Embedding provider abstraction.

Phase 1 implements OpenAI embeddings. Retrieval/vectorstore code depends only
on this `EmbeddingService` interface (`embed_documents` / `embed_query`), not
on `langchain_openai` directly, so swapping providers later doesn't ripple
through the rest of the app.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.config.settings import get_settings


class EmbeddingService:
    def __init__(self, model: str | None = None, api_key: str | None = None, batch_size: int = 300):
        settings = get_settings()
        self.model_name = model or settings.embedding_model
        self._client = OpenAIEmbeddings(
            model=self.model_name,
            api_key=api_key or settings.openai_api_key,
            # LangChain's default batch of 1000 texts per request can exceed
            # OpenAI's 300k-tokens-per-request cap once chunks are long;
            # 300 keeps typical chunk sizes safely under that limit.
            chunk_size=batch_size,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)
