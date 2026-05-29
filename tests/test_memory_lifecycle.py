"""Phase 3：MemoryItem 生命周期治理测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.memory_items import MemoryItem, MemoryKind, MemoryStatus
from core.memory_manager import MemoryManager
from tools.memory_tool import MemoryStatsTool


def test_duplicate_memory_save_merges_without_pollution(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    manager.save_memory_item(MemoryItem(
        id="mem_dup_a",
        kind=MemoryKind.WORKFLOW.value,
        title="修改 core 后运行 pytest",
        content="任何 core/ 逻辑修改都必须运行 pytest tests。",
        concepts=["pytest"],
        files=["core/memory_layers.py"],
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_dup_b",
        kind=MemoryKind.WORKFLOW.value,
        title="修改 core 后运行 pytest",
        content="任何 core/ 逻辑修改都必须运行 pytest tests。",
        concepts=["core"],
        files=["tests/test_memory_lifecycle.py"],
    ))

    items = manager.long_term_memory.get_all_items()
    assert len(items) == 1
    assert items[0].id == "mem_dup_a"
    assert "mem_dup_b" in items[0].related_ids
    assert items[0].metadata["deduplicated_count"] == 1

    recalled = manager.recall("core pytest", top_k=5, include_summaries=False)
    assert len(recalled) == 1
    assert recalled[0].item.id == "mem_dup_a"


def test_new_decision_supersedes_old_decision_and_recall_returns_latest_only(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    old = MemoryItem(
        id="mem_decision_old",
        kind=MemoryKind.DECISION.value,
        title="Memory 索引持久化策略",
        content="Memory 索引暂时每次启动全量重建。",
        concepts=["memory", "index"],
        files=["core/memory_index.py"],
    )
    new = MemoryItem(
        id="mem_decision_new",
        kind=MemoryKind.DECISION.value,
        title="Memory 索引持久化策略",
        content="Memory 索引使用 IndexPersistence 持久化，并通过增量 add/remove 维护。",
        concepts=["memory", "index"],
        files=["core/memory_index.py"],
    )

    manager.save_memory_item(old)
    manager.save_memory_item(new)

    old_saved = manager.long_term_memory.retrieve_item("mem_decision_old")
    new_saved = manager.long_term_memory.retrieve_item("mem_decision_new")
    assert old_saved.status == MemoryStatus.SUPERSEDED.value
    assert old_saved.is_latest is False
    assert old_saved.metadata["superseded_by"] == "mem_decision_new"
    assert new_saved.parent_id == "mem_decision_old"
    assert new_saved.supersedes == ["mem_decision_old"]
    assert new_saved.version == 2
    assert new_saved.is_latest is True

    recalled = manager.recall("Memory 索引持久化", top_k=5, include_summaries=False)
    ids = [result.item.id for result in recalled]
    assert "mem_decision_new" in ids
    assert "mem_decision_old" not in ids


def test_low_confidence_or_expired_memory_is_archived_and_not_recalled(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    expired_at = (datetime.now() - timedelta(days=1)).isoformat()

    manager.save_memory_item(MemoryItem(
        id="mem_expired_state",
        kind=MemoryKind.FACT.value,
        title="过期项目状态",
        content="临时分支仍在使用旧的 memory index 方案。",
        confidence=0.9,
        forget_after=expired_at,
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_low_confidence",
        kind=MemoryKind.FACT.value,
        title="低可信噪声",
        content="可能存在但未验证的工具日志。",
        confidence=0.1,
    ))

    expired = manager.long_term_memory.retrieve_item("mem_expired_state")
    low_confidence = manager.long_term_memory.retrieve_item("mem_low_confidence")
    assert expired.status == MemoryStatus.ARCHIVED.value
    assert low_confidence.status == MemoryStatus.ARCHIVED.value

    recalled = manager.recall("工具日志 memory index", top_k=5, include_summaries=False)
    assert recalled == []


def test_recall_tracks_access_count_and_last_accessed_at(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_access_tracking",
        kind=MemoryKind.BUG.value,
        title="ModuleNotFoundError 访问追踪",
        content="pytest ModuleNotFoundError 时检查 PYTHONPATH。",
        concepts=["ModuleNotFoundError", "pytest"],
    ))

    before = manager.long_term_memory.retrieve_item("mem_access_tracking")
    assert before.access_count == 0
    assert before.last_accessed_at is None

    recalled = manager.recall("ModuleNotFoundError pytest", top_k=1, include_summaries=False)
    assert recalled[0].item.id == "mem_access_tracking"

    after = manager.long_term_memory.retrieve_item("mem_access_tracking")
    assert after.access_count == 1
    assert after.last_accessed_at is not None
    assert after.quality_score >= before.quality_score


def test_memory_stats_reports_lifecycle_counts(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(id="mem_active", kind=MemoryKind.FACT.value, title="活跃", content="活跃记忆"))
    manager.save_memory_item(MemoryItem(
        id="mem_archived",
        kind=MemoryKind.FACT.value,
        title="归档",
        content="归档记忆",
        confidence=0.1,
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_old_decision",
        kind=MemoryKind.DECISION.value,
        title="生命周期策略",
        content="旧策略使用简单覆盖。",
        concepts=["lifecycle"],
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_new_decision",
        kind=MemoryKind.DECISION.value,
        title="生命周期策略",
        content="新策略使用 supersede 标记旧版本。",
        concepts=["lifecycle"],
    ))

    stats = manager.get_statistics()["long_term_memory"]
    assert stats["active_count"] == 2
    assert stats["superseded_count"] == 1
    assert stats["archived_count"] == 1
    assert stats["latest_count"] == 2

    output = MemoryStatsTool(memory_manager=manager).run()
    assert "active 2" in output
    assert "superseded 1" in output
    assert "archived 1" in output
