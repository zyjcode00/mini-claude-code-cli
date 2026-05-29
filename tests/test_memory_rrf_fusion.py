"""Phase 5：RRF 融合测试。"""

from __future__ import annotations

from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager


def test_rrf_fusion_reason_shows_ranked_sources(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_rrf_multi_source",
        kind=MemoryKind.ARCHITECTURE.value,
        title="RRF Fusion Architecture",
        content="BM25 and Vector retrieval results should be fused by reciprocal rank fusion.",
        concepts=["RRF", "Vector"],
        files=["core/memory_retrieval.py"],
        importance=0.2,
    ))

    results = manager.hybrid_recall("RRF Vector retrieval", top_k=1, file_path="core/memory_retrieval.py")

    assert results
    assert results[0].item.id == "mem_rrf_multi_source"
    reason = results[0].reason
    assert "RRF" in reason
    assert "BM25 rank" in reason
    assert "Vector rank" in reason
    assert "file exact" in reason


def test_rrf_fusion_prefers_consensus_over_single_strong_metadata_hit(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_rrf_metadata_only",
        kind=MemoryKind.OTHER.value,
        title="Unrelated file note",
        content="This note only shares the file path and has no alpha beta gamma query terms.",
        files=["core/target.py"],
        importance=0.1,
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_rrf_consensus",
        kind=MemoryKind.ARCHITECTURE.value,
        title="Alpha beta gamma architecture",
        content="alpha beta gamma retrieval design appears in both BM25 and vector search.",
        concepts=["alpha", "beta", "gamma"],
        files=["docs/other.md"],
        importance=0.1,
    ))

    results = manager.hybrid_recall("alpha beta gamma", top_k=2, file_path="core/target.py")

    assert [result.item.id for result in results][:2] == ["mem_rrf_consensus", "mem_rrf_metadata_only"]
    assert "RRF" in results[0].reason


def test_error_history_dynamic_rrf_weight_keeps_error_match_strong(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_rrf_error",
        kind=MemoryKind.BUG.value,
        title="ModuleNotFoundError fix",
        content="Fix ModuleNotFoundError by adjusting PYTHONPATH before running pytest.",
        concepts=["ModuleNotFoundError", "pytest"],
        importance=0.1,
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_rrf_other",
        kind=MemoryKind.WORKFLOW.value,
        title="pytest workflow",
        content="pytest workflow without the requested exception name.",
        concepts=["pytest"],
        importance=0.9,
    ))

    results = manager.hybrid_recall("pytest", top_k=1, error_type="ModuleNotFoundError")

    assert results
    assert results[0].item.id == "mem_rrf_error"
    assert "error exact" in results[0].reason or "error partial" in results[0].reason
