"""Memory-specific search indexes.

Phase 1 introduces BM25MemoryIndex as the retrieval backbone for long-term
MemoryItem / SessionSummary documents.  It keeps the implementation local and
small, but uses the standard BM25 formula with document frequency, document
length normalization, weighted fields, and JSON persistence hooks.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class BM25MemoryDocument:
    """Document accepted by BM25MemoryIndex."""

    doc_id: str
    title: str = ""
    content: str = ""
    concepts: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    kind: str = ""
    error: str = ""
    project: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BM25SearchHit:
    """Ranked BM25 hit."""

    doc_id: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25MemoryIndex:
    """Standard BM25 index for memory documents.

    The index expands field weights by repeating tokens.  This keeps the stored
    schema simple (`term_freqs`, `doc_lengths`, `inverted_index`) while allowing
    memory-specific boosts for title/concepts/files/errors.
    """

    VERSION = 1

    FIELD_WEIGHTS: Dict[str, float] = {
        "title": 2.0,
        "content": 1.0,
        "concepts": 1.8,
        "files": 2.2,
        "kind": 0.8,
        "error": 2.0,
        "project": 0.5,
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.inverted_index: Dict[str, Set[str]] = {}
        self.avg_doc_length: float = 0.0

    def add_or_update(self, document: BM25MemoryDocument) -> None:
        """Add or replace a document in the index."""
        if not document.doc_id:
            raise ValueError("BM25MemoryDocument.doc_id is required")

        self.remove(document.doc_id)
        weighted_terms = self._weighted_document_terms(document)
        if not weighted_terms:
            weighted_terms = [document.doc_id.lower()]

        term_freq: Dict[str, int] = {}
        for term in weighted_terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        self.doc_store[document.doc_id] = {
            "doc_id": document.doc_id,
            "title": document.title,
            "kind": document.kind,
            "project": document.project,
            "files": list(document.files),
            "concepts": list(document.concepts),
            "metadata": dict(document.metadata or {}),
        }
        self.term_freqs[document.doc_id] = term_freq
        self.doc_lengths[document.doc_id] = len(weighted_terms)
        for term in term_freq:
            self.inverted_index.setdefault(term, set()).add(document.doc_id)
        self._update_avg_doc_length()

    def add_documents(self, documents: Iterable[BM25MemoryDocument]) -> None:
        for document in documents:
            self.add_or_update(document)

    def remove(self, doc_id: str) -> None:
        """Remove a document if present."""
        old_terms = self.term_freqs.pop(doc_id, None)
        if old_terms:
            for term in old_terms:
                doc_ids = self.inverted_index.get(term)
                if not doc_ids:
                    continue
                doc_ids.discard(doc_id)
                if not doc_ids:
                    self.inverted_index.pop(term, None)
        self.doc_store.pop(doc_id, None)
        self.doc_lengths.pop(doc_id, None)
        self._update_avg_doc_length()

    def search(self, query: str, top_k: int = 10) -> List[BM25SearchHit]:
        """Search documents using standard BM25."""
        query_terms = self.tokenize(query)
        if not query_terms or not self.doc_store:
            return []

        scores: Dict[str, float] = {}
        matched_terms: Dict[str, Set[str]] = {}
        total_docs = len(self.doc_store)
        avgdl = self.avg_doc_length or 1.0

        for term in query_terms:
            doc_ids = self.inverted_index.get(term)
            if not doc_ids:
                continue
            df = len(doc_ids)
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            for doc_id in doc_ids:
                tf = self.term_freqs[doc_id].get(term, 0)
                dl = self.doc_lengths.get(doc_id, 0) or 1
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                if denominator <= 0:
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denominator
                matched_terms.setdefault(doc_id, set()).add(term)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            BM25SearchHit(
                doc_id=doc_id,
                score=score,
                matched_terms=sorted(matched_terms.get(doc_id, set())),
                metadata=dict(self.doc_store.get(doc_id, {})),
            )
            for doc_id, score in ranked[:top_k]
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "k1": self.k1,
            "b": self.b,
            "doc_store": self.doc_store,
            "term_freqs": self.term_freqs,
            "doc_lengths": self.doc_lengths,
            "inverted_index": {term: sorted(doc_ids) for term, doc_ids in self.inverted_index.items()},
            "avg_doc_length": self.avg_doc_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25MemoryIndex":
        index = cls(k1=float(data.get("k1", 1.5)), b=float(data.get("b", 0.75)))
        index.doc_store = dict(data.get("doc_store", {}))
        index.term_freqs = {
            str(doc_id): {str(term): int(count) for term, count in terms.items()}
            for doc_id, terms in data.get("term_freqs", {}).items()
        }
        index.doc_lengths = {str(doc_id): int(length) for doc_id, length in data.get("doc_lengths", {}).items()}
        index.inverted_index = {
            str(term): {str(doc_id) for doc_id in doc_ids}
            for term, doc_ids in data.get("inverted_index", {}).items()
        }
        index.avg_doc_length = float(data.get("avg_doc_length", 0.0))
        if not index.avg_doc_length:
            index._update_avg_doc_length()
        return index

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25MemoryIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def clear(self) -> None:
        self.doc_store.clear()
        self.term_freqs.clear()
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.avg_doc_length = 0.0

    def __len__(self) -> int:
        return len(self.doc_store)

    def _weighted_document_terms(self, document: BM25MemoryDocument) -> List[str]:
        terms: List[str] = []
        field_values = {
            "title": document.title,
            "content": document.content,
            "concepts": " ".join(document.concepts),
            "files": " ".join(document.files),
            "kind": document.kind,
            "error": document.error,
            "project": document.project,
        }
        for field_name, value in field_values.items():
            field_terms = self.tokenize(value)
            repeat = max(1, round(self.FIELD_WEIGHTS.get(field_name, 1.0)))
            for _ in range(repeat):
                terms.extend(field_terms)
        return terms

    def _update_avg_doc_length(self) -> None:
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize English/code/path terms and Chinese text.

        Code paths and symbols are preserved (`core/memory_layers.py`,
        `memory_layers`, `WinError`, `ModuleNotFoundError`) while also splitting
        useful parts.  jieba is used when installed; otherwise Chinese falls
        back to overlapping 2/3-character fragments.
        """
        if not text:
            return []

        raw = str(text)
        lowered = raw.lower()
        tokens: List[str] = []

        # Preserve path/code-like tokens containing / . _ -.
        code_tokens = re.findall(r"[a-zA-Z0-9_./\\-]*[a-zA-Z_][a-zA-Z0-9_./\\-]*", lowered)
        for token in code_tokens:
            cleaned = token.strip("./\\-_")
            if len(cleaned) < 2:
                continue
            tokens.append(cleaned.replace("\\", "/"))
            for part in re.split(r"[/\\._\-]+", cleaned):
                if len(part) >= 2:
                    tokens.append(part)

        # Split CamelCase and keep error/code symbols useful for traceback search.
        for symbol in re.findall(r"[A-Z][A-Za-z0-9_]*(?:Error|Exception)|[A-Za-z]+[A-Z][A-Za-z0-9_]+", raw):
            tokens.append(symbol.lower())
            for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", symbol):
                if len(part) >= 2:
                    tokens.append(part.lower())

        # Keep standalone numbers such as WinError 5.
        tokens.extend(re.findall(r"\b\d+\b", lowered))

        chinese_text = " ".join(re.findall(r"[\u4e00-\u9fff]+", raw))
        if chinese_text:
            try:
                import jieba  # type: ignore

                tokens.extend(token.lower() for token in jieba.cut(chinese_text) if len(token.strip()) >= 2)
            except Exception:
                chunks = re.findall(r"[\u4e00-\u9fff]+", raw)
                for chunk in chunks:
                    for size in (2, 3):
                        if len(chunk) >= size:
                            tokens.extend(chunk[i:i + size] for i in range(0, len(chunk) - size + 1))
                    if 2 <= len(chunk) <= 8:
                        tokens.append(chunk)

        return tokens
