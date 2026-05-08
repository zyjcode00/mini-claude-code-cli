# tests/test_memory_persistence.py
"""
测试记忆持久化功能

验证三层记忆在 session 文件中正确保存和恢复
"""
import pytest
import json
import tempfile
import os
from pathlib import Path

from core.context import ContextManager
from core.memory_models import SessionSummary, FileChange, ErrorRecord, ToolUsage
from core.memory_layers import WorkingMemory, EpisodicMemory


class TestMemoryPersistence:
    """测试记忆持久化到 session 文件"""

    def test_export_memories_basic(self):
        """测试导出记忆为字典格式"""
        context = ContextManager(max_history=100, min_keep=10)

        # 添加消息到工作记忆
        for i in range(5):
            context.add_message({"role": "user", "content": f"测试消息 {i}"})

        # 添加摘要到情景记忆
        summary = SessionSummary(
            session_id="test_session_001",
            timestamp="2024-01-01T12:00:00",
            summary_text="这是一个测试摘要",
            task_goal="测试记忆持久化",
            task_status="in_progress",
            files_changed=[FileChange(path="test.py", action="modified", summary="测试文件修改")],
            errors_encountered=[ErrorRecord(error_type="TestError", error_message="测试错误", timestamp="2024-01-01T12:00:00")],
            tools_used=[ToolUsage(tool_name="read_file", parameters={}, result_summary="读取成功", timestamp="2024-01-01T12:00:00")],
            importance=0.8
        )
        context.episodic_memory.add(summary)

        # 导出记忆
        exported = context.export_memories()

        # 验证导出结构
        assert "working_memory" in exported
        assert "episodic_memory" in exported
        assert "session_summaries" in exported
        assert "history_summary" in exported

        # 验证工作记忆
        assert len(exported["working_memory"]) == 5

        # 验证情景记忆
        assert len(exported["episodic_memory"]) == 1
        assert exported["episodic_memory"][0]["session_id"] == "test_session_001"

    def test_import_memories_basic(self):
        """测试从字典导入记忆"""
        context = ContextManager(max_history=100, min_keep=10)

        # 准备导入数据
        import_data = {
            "history_summary": "这是历史摘要",
            "working_memory": [
                {"role": "user", "content": "消息1"},
                {"role": "assistant", "content": "回复1"}
            ],
            "episodic_memory": [
                {
                    "session_id": "session_001",
                    "timestamp": "2024-01-01T12:00:00",
                    "summary_text": "测试摘要",
                    "task_goal": "测试目标",
                    "task_status": "done",
                    "files_changed": [],
                    "errors_encountered": [],
                    "tools_used": [],
                    "importance": 0.9
                }
            ],
            "session_summaries": []
        }

        # 导入记忆
        context.import_memories(import_data)

        # 验证恢复结果
        assert context.history_summary == "这是历史摘要"
        assert len(context.working_memory) == 2
        assert len(context.episodic_memory.get_all()) == 1

    def test_export_import_roundtrip(self):
        """测试导出后重新导入的完整性"""
        context1 = ContextManager(max_history=100, min_keep=10)

        # 添加数据
        for i in range(10):
            context1.add_message({"role": "user", "content": f"消息 {i}"})

        summary = SessionSummary(
            session_id="roundtrip_test",
            timestamp="2024-01-01T12:00:00",
            summary_text="往返测试",
            task_goal="测试完整性",
            task_status="in_progress",
            files_changed=[
                FileChange(path="file1.py", action="created", summary="创建文件1"),
                FileChange(path="file2.py", action="modified", summary="修改文件2")
            ],
            errors_encountered=[ErrorRecord(error_type="ValueError", error_message="测试错误", timestamp="2024-01-01T12:00:00")],
            tools_used=[ToolUsage(tool_name="edit_file", parameters={}, result_summary="编辑成功", timestamp="2024-01-01T12:00:00")],
            importance=0.75
        )
        context1.episodic_memory.add(summary)
        context1.history_summary = "历史摘要测试"

        # 导出
        exported = context1.export_memories()

        # 创建新的 ContextManager 并导入
        context2 = ContextManager(max_history=100, min_keep=10)
        context2.import_memories(exported)

        # 验证完整性
        assert context2.history_summary == context1.history_summary
        assert len(context2.working_memory) == len(context1.working_memory)
        assert len(context2.episodic_memory.get_all()) == len(context1.episodic_memory.get_all())

        # 验证摘要内容
        restored_summary = context2.episodic_memory.get_all()[0]
        assert restored_summary.session_id == summary.session_id
        assert restored_summary.task_goal == summary.task_goal
        assert len(restored_summary.files_changed) == 2

    def test_persistence_with_session_file(self):
        """测试完整的 session 文件持久化流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_file = Path(temp_dir) / "test_session.json"

            # === 第一步：创建并保存 session ===
            context = ContextManager(max_history=100, min_keep=10)

            # 添加工作记忆
            for i in range(15):
                context.add_message({"role": "user", "content": f"工作记忆消息 {i}"})

            # 添加情景记忆
            summary1 = SessionSummary(
                session_id="session_001",
                timestamp="2024-01-01T10:00:00",
                summary_text="第一个摘要",
                task_goal="任务一",
                task_status="done",
                files_changed=[],
                errors_encountered=[],
                tools_used=[],
                importance=0.8
            )
            summary2 = SessionSummary(
                session_id="session_002",
                timestamp="2024-01-01T11:00:00",
                summary_text="第二个摘要",
                task_goal="任务二",
                task_status="in_progress",
                files_changed=[FileChange(path="test.py", action="modified", summary="修改测试文件")],
                errors_encountered=[],
                tools_used=[ToolUsage(tool_name="pytest", parameters={}, result_summary="测试通过", timestamp="2024-01-01T11:00:00")],
                importance=0.9
            )
            context.episodic_memory.add(summary1)
            context.episodic_memory.add(summary2)
            context.history_summary = "这是历史摘要，包含多个会话的信息"

            # 保存到文件
            session_data = {
                "history_summary": context.history_summary,
                "messages": context.get_serializable_messages(),
                "memories": context.export_memories()
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            # === 第二步：从文件恢复 session ===
            context_restored = ContextManager(max_history=100, min_keep=10)

            with open(session_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            context_restored.history_summary = loaded_data.get("history_summary", "")
            context_restored.messages = loaded_data.get("messages", [])

            if "memories" in loaded_data:
                context_restored.import_memories(loaded_data["memories"])

            # === 验证恢复结果 ===
            assert context_restored.history_summary == context.history_summary
            assert len(context_restored.working_memory) == len(context.working_memory)
            assert len(context_restored.episodic_memory.get_all()) == 2

            # 验证摘要内容
            restored_summaries = context_restored.episodic_memory.get_all()
            assert restored_summaries[0].session_id == "session_001"
            assert restored_summaries[1].session_id == "session_002"
            assert len(restored_summaries[1].files_changed) == 1

    def test_backward_compatibility_no_memories_field(self):
        """测试向后兼容：旧版 session 文件没有 memories 字段"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_file = Path(temp_dir) / "old_session.json"

            # 创建旧版 session 文件（没有 memories 字段）
            old_data = {
                "history_summary": "旧版历史摘要",
                "messages": [
                    {"role": "user", "content": "旧消息1"},
                    {"role": "assistant", "content": "旧回复1"}
                ]
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(old_data, f, ensure_ascii=False, indent=2)

            # 加载旧版 session
            context = ContextManager(max_history=100, min_keep=10)

            with open(session_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            context.history_summary = loaded_data.get("history_summary", "")
            context.messages = loaded_data.get("messages", [])

            # 应该不会报错
            if "memories" in loaded_data:
                context.import_memories(loaded_data["memories"])

            # 验证基本功能正常
            assert context.history_summary == "旧版历史摘要"
            assert len(context.messages) == 2

    def test_empty_memories_export_import(self):
        """测试空记忆的导出和导入"""
        context = ContextManager(max_history=100, min_keep=10)

        # 不添加任何数据，直接导出
        exported = context.export_memories()

        # 验证空导出
        assert len(exported["working_memory"]) == 0
        assert len(exported["episodic_memory"]) == 0
        assert len(exported["session_summaries"]) == 0

        # 导入空数据
        context2 = ContextManager(max_history=100, min_keep=10)
        context2.import_memories(exported)

        # 验证没有报错
        assert len(context2.working_memory) == 0
        assert len(context2.episodic_memory.get_all()) == 0

    def test_large_memory_persistence(self):
        """测试大量记忆数据的持久化"""
        context = ContextManager(max_history=100, min_keep=10)

        # 添加大量工作记忆
        for i in range(100):
            context.add_message({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"消息内容 {i}" * 10  # 较长的消息
            })

        # 添加多个情景记忆
        for i in range(10):
            summary = SessionSummary(
                session_id=f"session_{i:03d}",
                timestamp=f"2024-01-{i+1:02d}T12:00:00",
                summary_text=f"摘要 {i}: " + "测试内容 " * 20,
                task_goal=f"任务目标 {i}",
                task_status="done" if i % 2 == 0 else "in_progress",
                files_changed=[
                    FileChange(path=f"file_{j}.py", action="modified", summary=f"修改文件{j}")
                    for j in range(5)
                ],
                errors_encountered=[
                    ErrorRecord(error_type="TestError", error_message=f"错误 {j}", timestamp=f"2024-01-{i+1:02d}T12:00:00")
                    for j in range(2)
                ],
                tools_used=[
                    ToolUsage(tool_name="edit_file", parameters={}, result_summary=f"工具调用{j}", timestamp=f"2024-01-{i+1:02d}T12:00:00")
                    for j in range(3)
                ],
                importance=0.5 + (i / 20)
            )
            context.episodic_memory.add(summary)

        # 导出
        exported = context.export_memories()

        # 导入到新的 context
        context2 = ContextManager(max_history=100, min_keep=10)
        context2.import_memories(exported)

        # 验证数据完整性
        assert len(context2.working_memory) == len(context.working_memory)
        assert len(context2.episodic_memory.get_all()) == 10

        # 验证第一个和最后一个摘要
        summaries = context2.episodic_memory.get_all()
        assert summaries[0].session_id == "session_000"
        assert summaries[-1].session_id == "session_009"


class TestPhaseCompatibility:
    """测试 Phase 1/2/3 兼容性"""

    def test_phase1_only_session(self):
        """测试只有 Phase 1 数据（session_summaries 和 history_summary）"""
        context = ContextManager(max_history=100, min_keep=10)

        # 只使用 Phase 1 功能
        context.history_summary = "Phase 1 历史摘要"
        context.session_summaries = [
            SessionSummary(
                session_id="phase1_session",
                timestamp="2024-01-01T10:00:00",
                summary_text="Phase 1 摘要",
                task_goal="Phase 1 任务",
                task_status="done",
                files_changed=[],
                errors_encountered=[],
                tools_used=[],
                importance=0.7
            )
        ]

        # 导出
        exported = context.export_memories()

        # 验证
        assert exported["history_summary"] == "Phase 1 历史摘要"
        assert len(exported["session_summaries"]) == 1

        # 导入
        context2 = ContextManager(max_history=100, min_keep=10)
        context2.import_memories(exported)

        assert context2.history_summary == "Phase 1 历史摘要"
        assert len(context2.session_summaries) == 1

    def test_phase2_enabled_automatically(self):
        """测试 Phase 2 记忆层自动启用"""
        context = ContextManager(max_history=100, min_keep=10)

        # 验证三层记忆已启用
        assert context._enable_memory_layers is True
        assert context.working_memory is not None
        assert context.episodic_memory is not None
        assert context.long_term_memory is not None

        # 添加消息后验证工作记忆自动更新
        context.add_message({"role": "user", "content": "测试消息"})
        assert len(context.working_memory) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])