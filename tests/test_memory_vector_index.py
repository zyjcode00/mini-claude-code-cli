"""Phase 4：轻量 VectorIndex 与 Hybrid Recall 接入测试。"""

from __future__ import annotations

import json

from core.memory_embedding import HashEmbeddingProvider
from core.memory_index import VectorMemoryDocument, VectorMemoryIndex, VectorIndexPersistence
from core.memory_items import MemoryItem, MemoryKind, MemoryStatus
from core.memory_manager import MemoryManager


def test_hash_embedding_provider_is_deterministic_and_dimensioned():
    provider = HashEmbeddingProvider(dimensions=32)

    first = provider.embed("MemoryManager hybrid recall")
    second = provider.embed("MemoryManager hybrid recall")

    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6


def test_vector_memory_index_cosine_search_returns_related_doc():
    provider = HashEmbeddingProvider(dimensions=64)
    index = VectorMemoryIndex(provider=provider)
    index.add_documents([
        VectorMemoryDocument(doc_id="doc_python", text="pytest AssertionError traceback debugging workflow"),
        VectorMemoryDocument(doc_id="doc_docs", text="write product roadmap and meeting notes"),
    ])

    hits = index.search("pytest traceback AssertionError", top_k=2)

    assert hits
    assert hits[0].doc_id == "doc_python"
    assert hits[0].score > 0


def test_vector_index_persistence_disables_dimension_mismatch(tmp_path):
    path = tmp_path / "indexes" / "vector.json"
    provider32 = HashEmbeddingProvider(dimensions=32)
    index = VectorMemoryIndex(provider=provider32)
    index.add_or_update(VectorMemoryDocument(doc_id="doc", text="dimension guard"))
    persistence = VectorIndexPersistence(path, provider=provider32)
    persistence.save(index, force=True)

    provider64 = HashEmbeddingProvider(dimensions=64)
    restarted = VectorIndexPersistence(path, provider=provider64)
    loaded = restarted.load()

    assert loaded is None
    assert restarted.disabled is True
    assert "dimension" in restarted.disabled_reason.lower()


def test_memory_item_save_updates_persisted_vector_index(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_vector_incremental",
        kind=MemoryKind.WORKFLOW.value,
        title="VectorIndex 增量更新",
        content="HashEmbeddingProvider writes embeddings to vector.json for cosine similarity search.",
        concepts=["VectorIndex", "HashEmbeddingProvider"],
    ))

    vector_path = tmp_path / "long_term" / "indexes" / "vector.json"
    assert vector_path.exists()
    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    assert payload["indexes"]["vector"]["vectors"]["mem_vector_incremental"]["dimensions"] == 128

    restarted = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    hits = restarted.long_term_memory.index_manager.vector_search("HashEmbeddingProvider vector cosine", top_k=1)
    assert hits[0].doc_id == "mem_vector_incremental"


def test_vector_hit_enters_hybrid_recall_reason(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_vector_recall",
        kind=MemoryKind.ARCHITECTURE.value,
        title="语义召回架构",
        content="VectorIndex should participate in Hybrid Recall with cosine similarity.",
        concepts=["VectorIndex", "Hybrid Recall"],
        importance=0.1,
    ))

    results = manager.hybrid_recall("cosine similarity VectorIndex", top_k=1)

    assert results
    assert results[0].item.id == "mem_vector_recall"
    assert "Vector相关" in results[0].reason


def test_archived_item_removed_from_vector_index(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    item = MemoryItem(
        id="mem_vector_archived",
        kind=MemoryKind.BUG.value,
        title="Vector archived",
        content="Archived vector item should be removed from vector index.",
        concepts=["ArchivedVector"],
    )
    manager.save_memory_item(item)
    assert "mem_vector_archived" in manager.long_term_memory.index_manager.vector.vectors

    item.status = MemoryStatus.ARCHIVED.value
    manager.save_memory_item(item)

    assert "mem_vector_archived" not in manager.long_term_memory.index_manager.vector.vectors
