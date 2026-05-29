"""
测试三层记忆架构

测试 WorkingMemory / EpisodicMemory / LongTermMemory 的基本功能。
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path

from core.memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory
from core.memory_models import SessionSummary, FileChange, ErrorRecord, ToolUsage


class TestWorkingMemory:
    """测试工作记忆"""

    def test_init(self):
        """测试初始化"""
        wm = WorkingMemory(max_size=10)
        assert len(wm) == 0
        assert wm.max_size == 10

    def test_add_message(self):
        """测试添加消息"""
        wm = WorkingMemory(max_size=5)

        msg1 = {"role": "user", "content": "Hello"}
        result = wm.add(msg1)

        assert len(wm) == 1
        assert result is None  # 未满，不淘汰
        assert wm.get_all() == [msg1]

    def test_fifo_eviction(self):
        """测试 FIFO 淘汰"""
        wm = WorkingMemory(max_size=3)

        msg1 = {"role": "user", "content": "Message 1"}
        msg2 = {"role": "user", "content": "Message 2"}
        msg3 = {"role": "user", "content": "Message 3"}
        msg4 = {"role": "user", "content": "Message 4"}

        # 添加 3 条消息
        wm.add(msg1)
        wm.add(msg2)
        wm.add(msg3)

        assert len(wm) == 3

        # 添加第 4 条，应该淘汰第 1 条
        evicted = wm.add(msg4)

        assert len(wm) == 3
        assert evicted == msg1
        assert msg1 not in wm.get_all()
        assert msg4 in wm.get_all()

    def test_get_recent(self):
        """测试获取最近消息"""
        wm = WorkingMemory(max_size=10)

        for i in range(5):
            wm.add({"role": "user", "content": f"Message {i}"})

        recent = wm.get_recent(3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 2"
        assert recent[2]["content"] == "Message 4"

    def test_clear(self):
        """测试清空"""
        wm = WorkingMemory(max_size=10)

        for i in range(5):
            wm.add({"role": "user", "content": f"Message {i}"})

        wm.clear()
        assert len(wm) == 0


class TestEpisodicMemory:
    """测试情景记忆"""

    def test_init(self):
        """测试初始化"""
        em = EpisodicMemory(max_size=10)
        assert len(em) == 0
        assert em.max_size == 10

    def test_add_summary(self):
        """测试添加摘要"""
        em = EpisodicMemory(max_size=5)

        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="Test summary",
            task_goal="Test task",
            task_status="completed"
        )

        result = em.add(summary)

        assert len(em) == 1
        assert result is None
        assert summary in em.get_all()

    def test_importance_eviction(self):
        """测试基于重要性淘汰"""
        em = EpisodicMemory(max_size=3)

        # 创建 3 个摘要，重要性不同
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 1",
            task_goal="Task 1",
            task_status="completed",
            importance=0.9  # 高重要性
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 2",
            task_goal="Task 2",
            task_status="completed",
            importance=0.3  # 低重要性
        )

        summary3 = SessionSummary(
            session_id="test_3",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 3",
            task_goal="Task 3",
            task_status="completed",
            importance=0.7
        )

        summary4 = SessionSummary(
            session_id="test_4",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 4",
            task_goal="Task 4",
            task_status="completed",
            importance=0.8
        )

        em.add(summary1)
        em.add(summary2)
        em.add(summary3)

        assert len(em) == 3

        # 添加第 4 个，应该淘汰 summary2（重要性最低）
        evicted = em.add(summary4)

        assert len(em) == 3
        assert evicted.session_id == "test_2"
        assert summary2 not in em.get_all()
        assert summary4 in em.get_all()

    def test_search(self):
        """测试关键词搜索"""
        em = EpisodicMemory(max_size=10)

        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构代码，优化性能",
            task_goal="代码重构",
            task_status="completed"
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="修复 Bug，添加测试",
            task_goal="Bug 修复",
            task_status="completed"
        )

        em.add(summary1)
        em.add(summary2)

        # 搜索关键词
        results = em.search("重构")
        assert len(results) == 1
        assert results[0].session_id == "test_1"

        results = em.search("Bug")
        assert len(results) == 1
        assert results[0].session_id == "test_2"

    def test_get_recent(self):
        """测试获取最近摘要"""
        em = EpisodicMemory(max_size=10)

        for i in range(5):
            em.add(SessionSummary(
                session_id=f"test_{i}",
                timestamp="2024-01-01T00:00:00",
                summary_text=f"Summary {i}",
                task_goal=f"Task {i}",
                task_status="completed"
            ))

        recent = em.get_recent(3)
        assert len(recent) == 3
        assert recent[0].session_id == "test_2"
        assert recent[2].session_id == "test_4"


class TestLongTermMemory:
    """测试长期记忆"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        yield
        # 测试后清理
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """测试初始化"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        assert ltm.count() == 0
        assert Path(self.temp_dir).exists()

    def test_store_and_retrieve(self):
        """测试存储和检索"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="Test summary",
            task_goal="Test task",
            task_status="completed"
        )

        # 存储
        file_path = ltm.store(summary)
        assert file_path != ""
        assert ltm.count() == 1

        # 检索
        retrieved = ltm.retrieve("test_1")
        assert retrieved is not None
        assert retrieved.session_id == "test_1"
        assert retrieved.summary_text == "Test summary"

    def test_search(self):
        """测试关键词搜索"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构代码，优化性能",
            task_goal="代码重构",
            task_status="completed"
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="修复 Bug，添加测试",
            task_goal="Bug 修复",
            task_status="completed"
        )

        ltm.store(summary1)
        ltm.store(summary2)

        # 搜索关键词
        results = ltm.search("重构")
        assert len(results) >= 1

        results = ltm.search("Bug")
        assert len(results) >= 1

    def test_rebuild_index(self):
        """测试重建索引"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        # 存储 2 个摘要
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 1",
            task_goal="Task 1",
            task_status="completed"
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="Summary 2",
            task_goal="Task 2",
            task_status="completed"
        )

        ltm.store(summary1)
        ltm.store(summary2)

        # 清空索引
        ltm.index.clear()
        ltm.inverted_index.clear()

        # 重建索引
        ltm._rebuild_index()

        # 验证索引
        assert len(ltm.index) == 2
        assert "test_1" in ltm.index
        assert "test_2" in ltm.index

    def test_search_indexes_file_change_path_field(self):
        """测试长期记忆使用 FileChange.path 字段建立文件路径索引"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        summary = SessionSummary(
            session_id="file_path_index_test",
            timestamp="2024-01-01T00:00:00",
            summary_text="修复长期记忆文件路径索引",
            task_goal="修改 memory layer",
            task_status="completed",
            files_changed=[
                FileChange(path="core/memory_layers.py", action="modified", summary="修复 path/file_path 不一致")
            ]
        )

        ltm.store(summary)

        assert ltm.search("memory_layers")
        assert ltm.search("core")

    def test_rebuild_index_supports_legacy_file_path_field(self):
        """测试重建索引时兼容旧 JSON 中的 file_path 字段"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)
        legacy_file = Path(self.temp_dir) / "summary_legacy_2024-01-01T00-00-00.json"
        legacy_file.write_text(
            '''{
                "session_id": "legacy_file_path_test",
                "timestamp": "2024-01-01T00:00:00",
                "summary_text": "旧版 file_path 字段测试",
                "task_goal": "兼容历史长期记忆",
                "task_status": "done",
                "files_changed": [
                    {
                        "file_path": "tools/legacy_memory_tool.py",
                        "action": "modified",
                        "summary": "旧版字段"
                    }
                ],
                "errors_encountered": [],
                "tools_used": []
            }''',
            encoding="utf-8"
        )

        ltm._rebuild_index()

        results = ltm.search("legacy_memory_tool")
        assert len(results) == 1
        assert results[0].session_id == "legacy_file_path_test"
        assert results[0].task_status == "completed"
        assert results[0].files_changed[0].path == "tools/legacy_memory_tool.py"

    def test_corrupt_index_is_backed_up_and_rebuilt(self):
        """损坏 index.json 时应备份现场并重建出可解析的新索引。"""
        summary_file = Path(self.temp_dir) / "summary_corrupt_2024-01-01T00-00-00.json"
        summary_file.write_text(
            json.dumps({
                "session_id": "corrupt_index_rebuild",
                "timestamp": "2024-01-01T00:00:00",
                "summary_text": "索引损坏恢复测试",
                "task_goal": "重建索引",
                "task_status": "completed",
                "files_changed": [],
                "errors_encountered": [],
                "tools_used": [],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        index_file = Path(self.temp_dir) / "index.json"
        index_file.write_text('{"index": {"broken": "missing comma" "x": 1}', encoding="utf-8")

        ltm = LongTermMemory(storage_dir=self.temp_dir)

        backups = list(Path(self.temp_dir).glob("index.json.corrupt_*.bak"))
        assert len(backups) == 1
        assert json.loads(index_file.read_text(encoding="utf-8"))
        assert ltm.retrieve("corrupt_index_rebuild") is not None

    def test_save_index_writes_valid_json_atomically(self):
        """保存索引应通过临时文件原子替换，最终 index.json 始终为合法 JSON。"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)
        summary = SessionSummary(
            session_id="atomic_save",
            timestamp="2024-01-01T00:00:00",
            summary_text="原子保存索引",
            task_goal="避免半写入 JSON",
            task_status="completed",
        )

        ltm.store(summary)

        index_file = Path(self.temp_dir) / "index.json"
        data = json.loads(index_file.read_text(encoding="utf-8"))
        assert "atomic_save" in data["index"]
        assert not list(Path(self.temp_dir).glob("index.*.tmp"))

    def test_save_index_skips_unchanged_content(self, monkeypatch):
        """索引内容未变化时不应重复 os.replace，降低启动期 Windows 文件占用冲突。"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)
        summary = SessionSummary(
            session_id="skip_unchanged",
            timestamp="2024-01-01T00:00:00",
            summary_text="跳过重复保存",
            task_goal="降低启动写入",
            task_status="completed",
        )
        ltm.store(summary)

        calls = []

        def fail_if_called(*args, **kwargs):
            calls.append(args)
            raise AssertionError("unchanged index should not be replaced")

        monkeypatch.setattr("core.memory_layers.os.replace", fail_if_called)
        ltm._save_index()

        assert calls == []

    def test_save_index_permission_error_keeps_pending_snapshot_and_cleans_tmp(self, monkeypatch):
        """Windows 下 index.json 被占用导致 replace 拒绝访问时，应降级且不遗留 index.*.tmp。"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)
        ltm.item_index["locked_item"] = Path(self.temp_dir) / "memory_item_locked.json"
        real_replace = os.replace

        def locked_replace(src, dst):
            if Path(dst).name == "index.json":
                raise PermissionError(5, "拒绝访问", str(dst))
            return real_replace(src, dst)

        monkeypatch.setattr("core.memory_layers.os.replace", locked_replace)
        ltm._save_index()

        pending_file = Path(self.temp_dir) / "index.pending.json"
        assert pending_file.exists()
        pending_data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert "locked_item" in pending_data["item_index"]
        assert not list(Path(self.temp_dir).glob("index.*.tmp"))

    def test_clear(self):
        """测试清空"""
        ltm = LongTermMemory(storage_dir=self.temp_dir)

        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="Test summary",
            task_goal="Test task",
            task_status="completed"
        )

        ltm.store(summary)
        assert ltm.count() == 1

        # 清空
        ltm.clear()
        assert ltm.count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
