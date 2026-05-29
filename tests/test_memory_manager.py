# tests/test_memory_manager.py
"""测试阶段 1 新增的 MemoryManager 统一记忆编排入口。"""

import tempfile
from pathlib import Path

import pytest

from core.compression_engine import CompressionStrategy
from core.memory_manager import MemoryManager
from core.memory_models import SessionSummary, FileChange, ErrorRecord


def make_summary(session_id: str, goal: str = "测试 MemoryManager", importance: float = 0.8) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        timestamp="2024-01-01T00:00:00",
        summary_text=f"{goal} 摘要",
        task_goal=goal,
        task_status="done",
        files_changed=[FileChange(path=f"core/{session_id}.py", action="modified", summary="测试文件")],
        errors_encountered=[ErrorRecord(error_type="ValueError", error_message="测试错误", timestamp="2024-01-01T00:00:00")],
        tools_used=[],
        importance=importance,
    )


def test_memory_manager_initializes_all_components():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(
            working_max_size=3,
            episodic_max_size=2,
            long_term_storage_dir=temp_dir,
        )

        assert manager.enabled is True
        assert manager.working_memory.max_size == 3
        assert manager.episodic_memory.max_size == 2
        assert manager.long_term_memory.storage_dir == Path(temp_dir)
        assert manager.retriever.working_memory is manager.working_memory
        assert manager.retriever.episodic_memory is manager.episodic_memory
        assert manager.retriever.long_term_memory is manager.long_term_memory
        assert manager.compression_engine is not None


def test_memory_manager_add_message_and_reset_working_memory():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(working_max_size=2, long_term_storage_dir=temp_dir)

        assert manager.add_message({"role": "user", "content": "第一条"}) is None
        assert manager.add_message({"role": "assistant", "content": "第二条"}) is None
        evicted = manager.add_message({"role": "user", "content": "第三条"})

        assert evicted["content"] == "第一条"
        assert [m["content"] for m in manager.working_memory.get_all()] == ["第二条", "第三条"]

        manager.reset_working_memory([
            {"role": "user", "content": "重置1"},
            {"role": "assistant", "content": "重置2"},
        ])
        assert [m["content"] for m in manager.working_memory.get_all()] == ["重置1", "重置2"]


def test_memory_manager_save_summary_archives_evicted_to_long_term():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(episodic_max_size=1, long_term_storage_dir=temp_dir)

        old_summary = make_summary("old", importance=0.1)
        new_summary = make_summary("new", importance=0.9)

        assert manager.save_summary(old_summary) is None
        evicted = manager.save_summary(new_summary)

        assert evicted.session_id == "old"
        assert [s.session_id for s in manager.episodic_memory.get_all()] == ["new"]
        assert manager.long_term_memory.retrieve("old").session_id == "old"
        assert manager.search_long_term("old", top_k=1)[0].session_id == "old"


def test_memory_manager_search_and_specialized_retrieval():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        summary = make_summary("memory_manager", goal="实现统一记忆管理器", importance=0.9)
        manager.save_summary(summary)

        results = manager.search("统一记忆", top_k=5)
        assert len(results) == 1
        assert results[0].session_id == "memory_manager"

        file_results = manager.retrieve_by_file_path("core/memory_manager.py", top_k=5)
        assert file_results[0][0].session_id == "memory_manager"
        assert file_results[0][2] == "episodic"

        error_results = manager.retrieve_by_error_type("ValueError", top_k=5)
        assert error_results[0][0].session_id == "memory_manager"


def test_memory_manager_export_import_roundtrip_and_compat_fields():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        manager.add_message({"role": "user", "content": "测试导出"})
        episodic_summary = make_summary("episodic_export", goal="导出情景记忆")
        session_summary = make_summary("session_export", goal="导出会话摘要")
        manager.save_summary(episodic_summary)

        exported = manager.export_memories(
            session_summaries=[session_summary],
            history_summary="历史摘要",
        )

        assert exported["history_summary"] == "历史摘要"
        assert len(exported["working_memory"]) == 1
        assert exported["episodic_memory"][0]["session_id"] == "episodic_export"
        assert exported["session_summaries"][0]["session_id"] == "session_export"
        assert exported["memory_items"] == []

        exported_with_items = manager.export_memories(
            session_summaries=[session_summary],
            history_summary="历史摘要",
            include_memory_items=True,
        )
        assert len(exported_with_items["memory_items"]) == len(manager.long_term_memory.item_index)

        restored = MemoryManager(long_term_storage_dir=temp_dir)
        imported = restored.import_memories(exported)

        assert imported["history_summary"] == "历史摘要"
        assert imported["session_summaries"][0].session_id == "session_export"
        assert len(restored.working_memory) == 1
        assert restored.episodic_memory.get_all()[0].session_id == "episodic_export"
        # 旧状态 done 应在导入时归一化
        assert restored.episodic_memory.get_all()[0].task_status == "completed"


def test_memory_manager_import_skips_embedded_memory_items_by_default():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        exported = manager.export_memories(include_memory_items=True)
        exported["memory_items"] = [
            {
                "id": "legacy-embedded-item",
                "kind": "fact",
                "title": "旧 session 内嵌长期记忆",
                "content": "启动恢复 session 时不应重复写入长期记忆。",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "importance": 0.5,
                "confidence": 0.8,
            }
        ]

        restored = MemoryManager(long_term_storage_dir=temp_dir)
        restored.import_memories(exported)
        assert restored.long_term_memory.retrieve_item("legacy-embedded-item") is None

        restored.import_memories(exported, import_memory_items=True)
        assert restored.long_term_memory.retrieve_item("legacy-embedded-item") is not None


def test_memory_manager_export_to_files_and_clear():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        manager.add_message({"role": "user", "content": "待清理"})
        manager.save_summary(make_summary("to_export"))
        manager.long_term_memory.store(make_summary("long_term_to_clear"))

        output_dir = Path(temp_dir) / "export"
        files = manager.export_to_files(str(output_dir))

        assert Path(files["episodic_memory"]).exists()
        assert Path(files["memory_statistics"]).exists()

        manager.clear()
        assert len(manager.working_memory) == 0
        assert len(manager.episodic_memory) == 0
        assert len(manager.long_term_memory) == 0


@pytest.mark.asyncio
async def test_memory_manager_compress_messages_delegates_to_compression_engine():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = MemoryManager(long_term_storage_dir=temp_dir)
        messages = [{"role": "user", "content": f"消息 {i}"} for i in range(10)]

        result = await manager.compress_messages(
            messages=messages,
            llm_summarizer_func=None,
            strategy=CompressionStrategy.SLIDING_WINDOW,
            min_keep=4,
            existing_summary="",
        )

        assert result.success
        assert result.strategy == CompressionStrategy.SLIDING_WINDOW
        assert len(result.compressed_messages) >= 4
        assert len(result.compressed_messages) < len(messages)
