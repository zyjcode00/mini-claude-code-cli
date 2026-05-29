"""测试记忆检索优化：任务目标传递、相关性过滤、任务复杂度评估"""

import pytest
import tempfile
import os
from pathlib import Path


class TestTaskComplexityAssessment:
    """测试任务复杂度评估"""

    def test_simple_interaction_detection(self):
        """测试简单交互识别"""
        from core.engine import AgentEngine
        from core.plan import PlanManager

        # 创建模拟工具列表
        tools = []

        # 创建 engine
        plan_manager = PlanManager()
        engine = AgentEngine(
            tools=tools,
            model="gpt-4",
            plan_manager=plan_manager,
            session_id="test_complexity"
        )

        # 测试简单交互
        assert engine._is_simple_interaction("你好") == True
        assert engine._is_simple_interaction("hello") == True
        assert engine._is_simple_interaction("继续") == True
        assert engine._is_simple_interaction("ok") == True
        assert engine._is_simple_interaction("好的") == True

        # 测试非简单交互
        assert engine._is_simple_interaction("请帮我重构这个函数") == False
        assert engine._is_simple_interaction("测试失败，如何修复") == False

    def test_task_complexity_estimation(self):
        """测试任务复杂度评估"""
        from core.engine import AgentEngine
        from core.plan import PlanManager

        tools = []
        plan_manager = PlanManager()
        engine = AgentEngine(
            tools=tools,
            model="gpt-4",
            plan_manager=plan_manager,
            session_id="test_complexity"
        )

        # 简单交互
        assert engine._estimate_task_complexity("你好") == "simple"
        assert engine._estimate_task_complexity("ok") == "simple"

        # 复杂任务
        assert engine._estimate_task_complexity("请帮我重构这个函数") == "complex"
        assert engine._estimate_task_complexity("测试失败，如何修复") == "complex"
        assert engine._estimate_task_complexity("实现一个新的API接口") == "complex"

        # 中等复杂度
        assert engine._estimate_task_complexity("查看这个文件") == "medium"


class TestMemoryInjectionBudget:
    """测试记忆注入预算控制"""

    def test_simple_interaction_skips_memory_injection(self):
        """测试简单交互跳过记忆注入"""
        from core.engine import AgentEngine
        from core.plan import PlanManager

        tools = []
        plan_manager = PlanManager()
        engine = AgentEngine(
            tools=tools,
            model="gpt-4",
            plan_manager=plan_manager,
            session_id="test_budget"
        )

        # 启用记忆层
        engine.context._enable_memory_layers = True

        # 简单交互应该返回空字符串
        result = engine._build_relevant_memory_context("你好")
        assert result == ""

        result = engine._build_relevant_memory_context("ok")
        assert result == ""

    def test_complex_task_uses_memory_budget(self):
        """测试复杂任务使用记忆预算"""
        from core.engine import AgentEngine
        from core.plan import PlanManager

        tools = []
        plan_manager = PlanManager()
        engine = AgentEngine(
            tools=tools,
            model="gpt-4",
            plan_manager=plan_manager,
            session_id="test_budget_complex"
        )

        # 启用记忆层
        engine.context._enable_memory_layers = True

        # 复杂任务应该注入记忆（如果有的话）
        # 由于没有记忆数据，这里只验证不会崩溃
        try:
            result = engine._build_relevant_memory_context("请帮我重构这个函数")
            # 结果可能是空字符串（没有记忆）或非空字符串（有记忆）
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"复杂任务记忆注入失败: {e}")


class TestTaskGoalRelevance:
    """测试任务目标相关性过滤"""

    def test_hybrid_recall_with_task_goal(self):
        """测试 hybrid_recall 支持任务目标参数"""
        from core.memory_manager import MemoryManager
        from core.memory_items import MemoryItem

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 MemoryManager
            manager = MemoryManager(long_term_storage_dir=tmpdir)

            # 添加测试记忆
            item1 = MemoryItem(
                id="test1",
                kind="bug",
                title="修复Bug",
                content="修复了 core/engine.py 中的内存泄漏问题",
                created_at="2024-01-01",
                updated_at="2024-01-01",
            )
            manager.save_memory_item(item1)

            item2 = MemoryItem(
                id="test2",
                kind="feature",
                title="添加功能",
                content="在 main.py 中添加了新的命令行参数解析",
                created_at="2024-01-02",
                updated_at="2024-01-02",
            )
            manager.save_memory_item(item2)

            # 调用 hybrid_recall，传递任务目标
            results = manager.hybrid_recall(
                query="engine.py",
                top_k=5,
                current_task_goal="修复 core/engine.py 的 Bug"
            )

            # 验证：应该返回结果
            assert len(results) > 0
            # 相关性应该被计算
            assert all(hasattr(r, 'score') for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])