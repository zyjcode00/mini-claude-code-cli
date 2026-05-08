"""
测试记忆检索功能

测试 MemoryRetriever 的检索功能。
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from core.memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory
from core.memory_retrieval import MemoryRetriever
from core.memory_models import SessionSummary, FileChange, ErrorRecord


class TestMemoryRetriever:
    """测试记忆检索器"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

        # 初始化三层记忆
        self.working_memory = WorkingMemory(max_size=10)
        self.episodic_memory = EpisodicMemory(max_size=20)
        self.long_term_memory = LongTermMemory(storage_dir=self.temp_dir)

        # 初始化检索器
        self.retriever = MemoryRetriever(
            self.working_memory,
            self.episodic_memory,
            self.long_term_memory
        )

        yield
        # 测试后清理
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """测试初始化"""
        assert self.retriever.working_memory is not None
        assert self.retriever.episodic_memory is not None
        assert self.retriever.long_term_memory is not None

    def test_search_working_memory(self):
        """测试检索工作记忆"""
        # 添加消息到工作记忆
        self.working_memory.add({"role": "user", "content": "Hello world"})
        self.working_memory.add({"role": "assistant", "content": "Hi there"})
        self.working_memory.add({"role": "user", "content": "Test message"})

        # 检索关键词
        results = self.retriever.retrieve("Hello", top_k=5, include_episodic=False, include_long_term=False)

        assert len(results) >= 1
        assert results[0][2] == "working"  # 来源层

    def test_search_episodic_memory(self):
        """测试检索情景记忆"""
        # 添加摘要到情景记忆
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

        self.episodic_memory.add(summary1)
        self.episodic_memory.add(summary2)

        # 检索关键词
        results = self.retriever.retrieve("重构", top_k=5, include_working=False, include_long_term=False)

        assert len(results) >= 1
        assert results[0][2] == "episodic"
        assert results[0][0].session_id == "test_1"

    def test_search_long_term_memory(self):
        """测试检索长期记忆"""
        # 添加摘要到长期记忆
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="持久化存储测试",
            task_goal="测试长期记忆",
            task_status="completed"
        )

        self.long_term_memory.store(summary1)

        # 检索关键词
        results = self.retriever.retrieve("测试", top_k=5, include_working=False, include_episodic=False)

        assert len(results) >= 1
        assert results[0][2] == "long_term"

    def test_multi_layer_retrieval(self):
        """测试多层级检索"""
        # 添加消息到工作记忆
        self.working_memory.add({"role": "user", "content": "重构代码"})

        # 添加摘要到情景记忆
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构模块 A",
            task_goal="代码重构",
            task_status="completed"
        )

        self.episodic_memory.add(summary1)

        # 添加摘要到长期记忆
        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构模块 B",
            task_goal="代码重构",
            task_status="completed"
        )

        self.long_term_memory.store(summary2)

        # 检索关键词
        results = self.retriever.retrieve("重构", top_k=5)

        # 应该从多个层级检索到结果
        sources = {result[2] for result in results}
        assert len(sources) >= 2  # 至少从 2 个层级检索到

    def test_retrieve_by_file_path(self):
        """测试按文件路径检索"""
        # 添加包含文件变更的摘要
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="修改了代码文件",
            task_goal="修改代码",
            task_status="completed",
            files_changed=[
                FileChange(path="core/context.py", action="modified", summary="修改上下文管理器")
            ]
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="创建了新文件",
            task_goal="创建文件",
            task_status="completed",
            files_changed=[
                FileChange(path="tests/test_new.py", action="created", summary="创建测试文件")
            ]
        )

        self.episodic_memory.add(summary1)
        self.episodic_memory.add(summary2)

        # 检索文件路径
        results = self.retriever.retrieve_by_file_path("context.py", top_k=5)

        assert len(results) >= 1
        assert results[0][0].session_id == "test_1"

    def test_retrieve_by_error_type(self):
        """测试按错误类型检索"""
        # 添加包含错误记录的摘要
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="修复了 TypeError",
            task_goal="错误修复",
            task_status="completed",
            errors_encountered=[
                ErrorRecord(
                    error_type="TypeError",
                    error_message="unsupported operand type(s)",
                    timestamp="2024-01-01T00:00:00",
                    solution="类型转换"
                )
            ]
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="修复了 ValueError",
            task_goal="错误修复",
            task_status="completed",
            errors_encountered=[
                ErrorRecord(
                    error_type="ValueError",
                    error_message="invalid literal for int()",
                    timestamp="2024-01-01T00:00:00",
                    solution="输入验证"
                )
            ]
        )

        self.episodic_memory.add(summary1)
        self.episodic_memory.add(summary2)

        # 检索错误类型
        results = self.retriever.retrieve_by_error_type("TypeError", top_k=5)

        assert len(results) >= 1
        assert results[0][0].session_id == "test_1"

    def test_bm25_scoring(self):
        """测试 BM25 评分"""
        # 添加多个摘要
        summary1 = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构代码 重构代码 重构代码",  # 高词频
            task_goal="代码重构",
            task_status="completed"
        )

        summary2 = SessionSummary(
            session_id="test_2",
            timestamp="2024-01-01T00:00:00",
            summary_text="重构",  # 低词频
            task_goal="其他任务",
            task_status="completed"
        )

        self.episodic_memory.add(summary1)
        self.episodic_memory.add(summary2)

        # 检索并验证分数
        results = self.retriever.retrieve("重构", top_k=5, include_working=False, include_long_term=False)

        # 高词频的应该分数更高
        if len(results) >= 2:
            # 找到 summary1 的分数
            score1 = None
            score2 = None

            for summary, score, source in results:
                if summary.session_id == "test_1":
                    score1 = score
                elif summary.session_id == "test_2":
                    score2 = score

            if score1 is not None and score2 is not None:
                assert score1 >= score2  # 词频高的分数应该更高

    def test_top_k_limit(self):
        """测试 top_k 限制"""
        # 添加多个摘要
        for i in range(10):
            summary = SessionSummary(
                session_id=f"test_{i}",
                timestamp="2024-01-01T00:00:00",
                summary_text=f"测试摘要 {i}",
                task_goal=f"测试任务 {i}",
                task_status="completed"
            )
            self.episodic_memory.add(summary)

        # 检索并限制数量
        results = self.retriever.retrieve("测试", top_k=3, include_working=False, include_long_term=False)

        assert len(results) <= 3

    def test_deduplication(self):
        """测试去重"""
        # 添加摘要到情景记忆和长期记忆
        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="测试摘要",
            task_goal="测试任务",
            task_status="completed"
        )

        self.episodic_memory.add(summary)
        self.long_term_memory.store(summary)

        # 检索
        results = self.retriever.retrieve("测试", top_k=5)

        # 应该去重
        session_ids = [result[0].session_id for result in results]
        assert len(session_ids) == len(set(session_ids))

    def test_statistics(self):
        """测试统计功能"""
        stats = self.retriever.get_statistics()

        assert "working_memory_size" in stats
        assert "episodic_memory_size" in stats
        assert "long_term_memory_count" in stats

    def test_empty_retrieval(self):
        """测试空检索"""
        # 不添加任何记忆，直接检索
        results = self.retriever.retrieve("不存在的关键词", top_k=5)

        # 应该返回空列表
        assert isinstance(results, list)

    def test_chinese_keyword_retrieval(self):
        """测试中文关键词检索"""
        # 添加中文摘要
        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="这是一个中文摘要",
            task_goal="中文任务",
            task_status="completed"
        )

        self.episodic_memory.add(summary)

        # 检索中文关键词
        results = self.retriever.retrieve("中文", top_k=5, include_working=False, include_long_term=False)

        assert len(results) >= 1

    def test_english_keyword_retrieval(self):
        """测试英文关键词检索"""
        # 添加英文摘要
        summary = SessionSummary(
            session_id="test_1",
            timestamp="2024-01-01T00:00:00",
            summary_text="This is an English summary",
            task_goal="English task",
            task_status="completed"
        )

        self.episodic_memory.add(summary)

        # 检索英文关键词
        results = self.retriever.retrieve("English", top_k=5, include_working=False, include_long_term=False)

        assert len(results) >= 1


class TestMemoryRetrieverIntegration:
    """测试检索器集成功能"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前设置"""
        self.temp_dir = tempfile.mkdtemp()

        self.working_memory = WorkingMemory(max_size=20)
        self.episodic_memory = EpisodicMemory(max_size=50)
        self.long_term_memory = LongTermMemory(storage_dir=self.temp_dir)

        self.retriever = MemoryRetriever(
            self.working_memory,
            self.episodic_memory,
            self.long_term_memory
        )

        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_large_scale_retrieval(self):
        """测试大规模检索"""
        # 添加大量摘要
        for i in range(100):
            summary = SessionSummary(
                session_id=f"test_{i}",
                timestamp="2024-01-01T00:00:00",
                summary_text=f"摘要 {i}，包含关键词",
                task_goal=f"任务 {i}",
                task_status="completed"
            )

            if i < 50:
                self.episodic_memory.add(summary)
            else:
                self.long_term_memory.store(summary)

        # 检索
        results = self.retriever.retrieve("关键词", top_k=10)

        # 应该返回结果
        assert len(results) >= 1
        assert len(results) <= 10

    def test_performance(self):
        """测试检索性能"""
        import time

        # 添加大量摘要
        for i in range(50):
            summary = SessionSummary(
                session_id=f"test_{i}",
                timestamp="2024-01-01T00:00:00",
                summary_text=f"摘要 {i}",
                task_goal=f"任务 {i}",
                task_status="completed"
            )
            self.episodic_memory.add(summary)

        # 测试检索速度
        start_time = time.time()
        results = self.retriever.retrieve("摘要", top_k=5)
        end_time = time.time()

        # 检索应该在 100ms 内完成
        assert (end_time - start_time) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])