"""阶段 2 MemoryItem 写入/召回管线集成测试。"""

import json
import tempfile
from pathlib import Path

from core.memory_items import MemoryItem, MemoryKind, ObservationType, RawObservation
from core.memory_manager import MemoryManager
from core.memory_models import FileChange, SessionSummary


def make_summary(session_id: str) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        timestamp="2024-01-01T00:00:00",
        summary_text="阶段1新增 MemoryManager 统一编排入口，并保留旧摘要兼容。",
        task_goal="重构长期记忆系统阶段1",
        task_status="completed",
        files_changed=[FileChange(path="core/memory_manager.py", action="modified", summary="新增统一入口")],
        importance=0.7,
    )


def test_long_term_memory_stores_rebuilds_and_searches_memory_items():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        item = MemoryItem(
            id="mem-architecture",
            kind=MemoryKind.ARCHITECTURE,
            title="MemoryManager 统一入口",
            content="阶段2支持 MemoryItem 保存和 recall 召回。",
            concepts=["MemoryManager", "recall"],
            files=["core/memory_manager.py"],
            importance=0.9,
            confidence=0.9,
        )

        stored_path = manager.save_memory_item(item)
        assert Path(stored_path).exists()
        assert manager.long_term_memory.retrieve_item("mem-architecture").title == "MemoryManager 统一入口"

        results = manager.recall("MemoryManager recall", top_k=3, include_summaries=False)
        assert results[0].item.id == "mem-architecture"
        assert results[0].source == "long_term_items"
        assert "匹配关键词" in results[0].reason

        # 模拟跨进程/跨会话重启后从 index.json 恢复索引
        restored = MemoryManager(long_term_storage_dir=temp_dir)
        restored_results = restored.recall("core memory_manager", top_k=3, include_summaries=False)
        assert restored_results[0].item.id == "mem-architecture"


def test_memory_manager_save_observation_promotes_to_memory_item():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        observation = RawObservation(
            id="obs-pytest",
            session_id="session-phase2",
            project="mini-claude-code-cli",
            event_type=ObservationType.TEST_RESULT,
            tool_name="run_pytest",
            tool_output="tests/test_memory_items.py 5 passed",
            files=["tests/test_memory_items.py"],
            metadata={"exit_code": 0},
        )

        item = manager.save_observation(
            observation,
            promote=True,
            kind=MemoryKind.WORKFLOW.value,
            title="阶段2模型测试通过",
            importance=0.8,
            confidence=1.0,
        )

        assert item is not None
        assert item.source_observation_ids == ["obs-pytest"]
        assert item.source_session_ids == ["session-phase2"]
        assert item.metadata["raw_observation"]["tool_output"] == "tests/test_memory_items.py 5 passed"

        results = manager.recall("pytest memory_items", top_k=1, include_summaries=False)
        assert results[0].item.title == "阶段2模型测试通过"
        assert results[0].item.kind == MemoryKind.WORKFLOW.value


def test_memory_manager_export_import_memory_items_and_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=Path(temp_dir) / "source")
        manager.save_memory_item(MemoryItem(
            id="mem-export",
            kind="decision",
            title="保留 SessionSummary 兼容",
            content="阶段2 recall 会把旧摘要包装为 summary MemoryItem。",
            concepts=["SessionSummary", "兼容"],
        ))

        exported = manager.export_memories(history_summary="历史摘要")
        assert exported["memory_items"][0]["id"] == "mem-export"
        assert exported["history_summary"] == "历史摘要"

        restored = MemoryManager(long_term_storage_dir=Path(temp_dir) / "restored")
        restored.import_memories(exported)
        recall_results = restored.recall("SessionSummary 兼容", top_k=1, include_summaries=False)
        assert recall_results[0].item.id == "mem-export"

        output_files = restored.export_to_files(str(Path(temp_dir) / "export"))
        assert Path(output_files["memory_items"]).exists()
        with open(output_files["memory_items"], "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["id"] == "mem-export"


def test_recall_wraps_legacy_session_summaries_for_compatibility():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(episodic_max_size=1, long_term_storage_dir=temp_dir)
        old_summary = make_summary("phase1")
        new_summary = make_summary("phase2")
        new_summary.summary_text = "阶段2新增 MemoryItem 模型。"

        # old_summary 会被淘汰进 LongTermMemory，验证旧摘要仍能通过 recall 统一返回。
        manager.save_summary(old_summary)
        manager.save_summary(new_summary)

        results = manager.recall("阶段1 MemoryManager", top_k=3)
        summary_results = [result for result in results if result.source == "summary_compat"]

        assert summary_results
        assert summary_results[0].item.kind == MemoryKind.SUMMARY.value
        assert summary_results[0].item.id == "summary_phase1"
        assert "core/memory_manager.py" in summary_results[0].item.files
