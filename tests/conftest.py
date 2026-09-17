"""Shared test fixtures.

`FakeEmbeddingService` produces deterministic vectors from a text hash --
tests exercise real Chroma read/write/filter behavior without ever calling
the OpenAI API.
"""

from __future__ import annotations

import random

import pytest

from app.vectorstore.chroma_store import VectorStoreService


class FakeEmbeddingService:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        random.seed(abs(hash(text)) % (2**32))
        return [random.random() for _ in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def vector_store(tmp_path) -> VectorStoreService:
    return VectorStoreService(persist_dir=str(tmp_path / "chroma"), collection_name="test_collection")


@pytest.fixture
def sample_csv_path() -> str:
    return "data/sample/sample_tickets.csv"
