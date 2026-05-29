"""Phase 2：IndexPersistence 与增量 BM25 索引测试。"""

from __future__ import annotations

import json

from core.memory_index import BM25MemoryIndex, IndexPersistence
from core.memory_items import MemoryItem, MemoryKind, MemoryStatus
from core.memory_manager import MemoryManager


def test_memory_item_save_incrementally_updates_persisted_bm25_index(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    item = MemoryItem(
        id="mem_phase2_incremental",
        kind=MemoryKind.BUG.value,
        title="Phase2 WinError 5 持久化索引",
        content="PermissionError WinError 5 os.replace bm25.json on Windows",
        concepts=["WinError 5", "IndexPersistence"],
        files=["core/memory_index.py"],
        importance=0.8,
    )

    manager.save_memory_item(item)

    index_path = tmp_path / "long_term" / "indexes" / "bm25.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["embedding_provider"] is None
    assert payload["indexes"]["bm25"]["doc_store"]["mem_phase2_incremental"]["title"].startswith("Phase2")

    results = manager.hybrid_recall("WinError 5 bm25.json", top_k=1)
    assert results[0].item.id == "mem_phase2_incremental"


def test_restart_uses_persisted_bm25_without_loading_all_items_for_scoring(tmp_path, monkeypatch):
    storage_dir = tmp_path / "long_term"
    manager = MemoryManager(long_term_storage_dir=str(storage_dir))
    manager.save_memory_item(MemoryItem(
        id="mem_phase2_restart",
        kind=MemoryKind.WORKFLOW.value,
        title="重启后使用持久化 BM25",
        content="Restart recall should use persisted indexes/bm25.json for MemoryIndexManager search",
        concepts=["persisted_bm25", "restart"],
        files=["core/memory_retrieval.py"],
    ))

    restarted = MemoryManager(long_term_storage_dir=str(storage_dir))

    def fail_build_bm25(_documents):  # pragma: no cover - should not be called for long-term docs
        raise AssertionError("hybrid_recall should not rebuild BM25 from all docs")

    monkeypatch.setattr(restarted.retriever, "_build_bm25_index", fail_build_bm25)
    results = restarted.hybrid_recall("persisted_bm25 restart", top_k=1)

    assert results[0].item.id == "mem_phase2_restart"
    assert "BM25相关" in results[0].reason


def test_corrupt_bm25_index_rebuilds_and_temp_files_are_cleaned(tmp_path):
    storage_dir = tmp_path / "long_term"
    manager = MemoryManager(long_term_storage_dir=str(storage_dir))
    manager.save_memory_item(MemoryItem(
        id="mem_phase2_rebuild",
        kind=MemoryKind.ARCHITECTURE.value,
        title="损坏索引自动重建",
        content="Corrupt bm25 index should rebuild from memory item index",
        concepts=["corrupt_bm25", "rebuild"],
    ))

    index_path = storage_dir / "indexes" / "bm25.json"
    index_path.write_text("{ broken json", encoding="utf-8")

    rebuilt = MemoryManager(long_term_storage_dir=str(storage_dir))
    results = rebuilt.hybrid_recall("corrupt_bm25 rebuild", top_k=1)

    assert results[0].item.id == "mem_phase2_rebuild"
    assert json.loads(index_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert list(index_path.parent.glob("*.tmp")) == []


def test_archived_or_superseded_memory_item_is_removed_from_bm25_index(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    item = MemoryItem(
        id="mem_phase2_archived",
        kind=MemoryKind.BUG.value,
        title="Archived ModuleNotFoundError",
        content="ModuleNotFoundError archived item should disappear from BM25 index",
        concepts=["ModuleNotFoundError"],
    )
    manager.save_memory_item(item)
    assert manager.hybrid_recall("ModuleNotFoundError archived", top_k=1)[0].item.id == "mem_phase2_archived"

    item.status = MemoryStatus.ARCHIVED.value
    manager.save_memory_item(item)

    assert "mem_phase2_archived" not in manager.long_term_memory.index_manager.bm25.doc_store
    assert manager.hybrid_recall("ModuleNotFoundError archived", top_k=1) == []


def test_index_persistence_debounce_flushes_and_loads(tmp_path):
    path = tmp_path / "indexes" / "bm25.json"
    persistence = IndexPersistence(path, debounce_seconds=60.0)
    index = BM25MemoryIndex()

    persistence.save(index)
    assert path.exists()

    index.add_or_update(
        __import__("core.memory_index", fromlist=["BM25MemoryDocument"]).BM25MemoryDocument(
            doc_id="debounced_doc",
            content="debounced write should wait until explicit flush",
        )
    )
    persistence.save(index)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "debounced_doc" not in payload["indexes"]["bm25"]["doc_store"]

    persistence.flush(index)
    loaded = persistence.load()
    assert loaded is not None
    assert loaded.search("debounced write", top_k=1)[0].doc_id == "debounced_doc"
    assert list(path.parent.glob("*.tmp")) == []
