"""Lightweight embedding providers for memory vector retrieval.

Phase 4 intentionally avoids mandatory external dependencies.  The default
HashEmbeddingProvider produces deterministic local embeddings that are good
enough for tests, offline use, and graceful degradation when no API key is
configured.  OpenAICompatibleEmbeddingProvider is optional and uses the stdlib
HTTP stack so the project does not gain a hard dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional


class EmbeddingProvider(ABC):
    """Abstract embedding provider used by VectorMemoryIndex."""

    provider_name: str = "base"
    model: str = ""
    dimensions: int = 0

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for text."""


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider with no external dependencies.

    Tokens are projected into a fixed-size vector through a signed hashing
    trick and then L2-normalized.  It is not a replacement for semantic model
    embeddings, but it provides stable cosine similarity, dimension metadata,
    and fully offline tests.
    """

    provider_name = "hash"

    def __init__(self, dimensions: int = 128, model: str = "hash-bm25-token-v1"):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.model = model

    def embed(self, text: str) -> List[float]:
        from .memory_index import BM25MemoryIndex

        vector = [0.0] * self.dimensions
        tokens = BM25MemoryIndex.tokenize(text or "")
        if not tokens:
            tokens = [str(text or "").lower() or "empty"]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            # Slightly damp repeated very common tokens while preserving term signal.
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Minimal OpenAI-compatible embedding provider.

    The provider is optional.  It can target OpenAI or compatible local servers
    by configuring base_url/api_key/model.  Tests should continue to rely on
    HashEmbeddingProvider so no network is required.
    """

    provider_name = "openai-compatible"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.dimensions = int(dimensions or os.getenv("EMBEDDING_DIMENSIONS", "0") or 0)
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("OpenAICompatibleEmbeddingProvider requires an API key")

        payload: Dict[str, Any] = {"model": self.model, "input": text or ""}
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # nosec - user-configured endpoint
            data = json.loads(response.read().decode("utf-8"))
        embedding = data["data"][0]["embedding"]
        if not isinstance(embedding, list):
            raise RuntimeError("Invalid embedding response")
        vector = [float(value) for value in embedding]
        if self.dimensions and len(vector) != self.dimensions:
            raise RuntimeError(f"Embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}")
        self.dimensions = len(vector)
        return vector


__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
]
