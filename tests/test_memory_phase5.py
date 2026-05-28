"""阶段5：Prompt 注入、Token Budget、自动历史召回与统计工具测试。"""

import asyncio
import tempfile

import pytest

from core.memory_context_builder import MemoryContextBuilder
from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager
from tools import get_default_tools
from tools.memory_tool import MemoryStatsTool
from core.engine import AgentEngine


class _FakePlanManager:
    current_goal = ""

    def __init__(self, plan_id=None):
        self.plan_id = plan_id

    def get_formatted_plan(self):
        return ""

    def get_completed_goals(self):
        return []

    def get_plan_id(self):
        return self.plan_id

    def is_plan_complete(self):
        return False


class _DummyTool:
    name = "dummy"

    def to_anthropic_spec(self):
        return {"name": self.name, "description": "dummy", "input_schema": {"type": "object", "properties": {}}}

    def run(self, **kwargs):
        return "ok"


def _result(item, score=1.0, source="long_term_items", reason="test"):
    from core.memory_items import MemoryRecallResult

    return MemoryRecallResult(item=item, score=score, source=source, reason=reason)


def test_memory_context_builder_respects_token_budget_and_deduplicates():
    builder = MemoryContextBuilder(default_token_budget=80, max_items=5, per_item_char_limit=120)
    results = [
        _result(MemoryItem(kind=MemoryKind.BUG.value, title="A", content="重复内容 " * 30, files=["core/a.py"], confidence=0.9), 3.0),
        _result(MemoryItem(kind=MemoryKind.BUG.value, title="A duplicate", content="重复内容 " * 30, files=["core/a.py"], confidence=0.9), 2.9),
        _result(MemoryItem(kind=MemoryKind.WORKFLOW.value, title="B", content="运行 pytest 后修复失败", files=["tests/test_a.py"], confidence=0.9), 2.0),
    ]

    context = builder.build("修改 core/a.py 并运行测试", results, token_budget=80)

    assert "### 相关长期记忆" in context
    assert context.count("A duplicate") == 0
    assert "core/a.py" in context
    assert len(context) <= 340  # 80 token 粗略预算约 320 字符，允许少量标题开销


def test_memory_context_builder_prioritizes_task_type_kinds():
    builder = MemoryContextBuilder(default_token_budget=500, max_items=5)
    results = [
        _result(MemoryItem(kind=MemoryKind.PREFERENCE.value, title="中文文档", content="文档优先中文。"), 5.0),
        _result(MemoryItem(kind=MemoryKind.BUG.value, title="pytest 失败", content="pytest 失败时先看 traceback。"), 1.0),
        _result(MemoryItem(kind=MemoryKind.ARCHITECTURE.value, title="架构", content="MemoryManager 是统一入口。"), 1.0),
    ]

    test_context = builder.build("pytest 报错 AssertionError", results, task_type="test_failure")
    doc_context = builder.build("写一份架构文档", results, task_type="documentation")

    assert test_context.index("pytest 失败") < test_context.index("中文文档")
    assert doc_context.index("中文文档") < doc_context.index("pytest 失败")


def test_memory_manager_build_prompt_memory_context_uses_budget_and_task_type(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(kind=MemoryKind.BUG.value, title="错误历史", content="pytest AssertionError 修复经验。", concepts=["pytest"]))
    manager.save_memory_item(MemoryItem(kind=MemoryKind.ARCHITECTURE.value, title="架构历史", content="MemoryManager 统一编排记忆。", concepts=["MemoryManager"]))

    context = manager.build_prompt_memory_context("pytest AssertionError", token_budget=120, task_type="test_failure", top_k=5)

    assert "### 相关长期记忆" in context
    assert "错误历史" in context
    assert len(context) < 600


def test_memory_stats_tool_and_default_registration(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(kind=MemoryKind.FACT.value, title="事实", content="一条事实"))

    stats_output = MemoryStatsTool(memory_manager=manager).run()
    names = {tool.name for tool in get_default_tools(plan_manager=None, memory_manager=manager)}

    assert "📊 记忆系统统计" in stats_output
    assert "长期记忆" in stats_output
    assert "memory_stats" in names


def test_agent_engine_builds_file_and_error_history_context(tmp_path):
    engine = AgentEngine(tools=[_DummyTool()], model="fake-model", plan_manager=_FakePlanManager(), base_url="http://example.invalid", api_key="x")
    engine.context.memory_manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    engine.context._enable_memory_layers = True
    engine.context.memory_manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="编辑 engine.py 前注意",
        content="修改 core/engine.py 前需要保留 reasoning_content。",
        files=["core/engine.py"],
        concepts=["reasoning_content"],
    ))
    engine.context.memory_manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="AssertionError 历史",
        content="遇到 AssertionError 时先读取失败断言附近代码。",
        concepts=["AssertionError"],
    ))

    file_context = engine._build_pre_tool_memory_context("edit_file", {"path": "core/engine.py"})
    error_context = engine._build_post_tool_failure_memory_context("run_pytest", "❌ 测试失败 AssertionError: boom")

    assert "自动文件历史召回" in file_context
    assert "编辑 engine.py 前注意" in file_context
    assert "自动错误历史召回" in error_context
    assert "AssertionError 历史" in error_context


def test_agent_engine_initializes_current_plan_branch_and_starts_plan_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = AgentEngine(
        tools=[_DummyTool()],
        model="fake-model",
        plan_manager=_FakePlanManager(plan_id="abc123"),
        base_url="http://example.invalid",
        api_key="x",
        session_id="plan_branch_regression",
    )

    assert hasattr(engine, "current_plan_branch")
    assert engine.current_plan_branch is None

    calls = []

    def fake_start_plan_branch(plan_id):
        calls.append(plan_id)
        return True, "created"

    async def fake_compress_messages():
        return None

    async def fake_call_llm(relevant_history="", user_input=""):
        return [{"type": "text", "text": "done"}], "end_turn"

    monkeypatch.setattr("core.engine.start_plan_branch", fake_start_plan_branch)
    monkeypatch.setattr(engine, "compress_messages", fake_compress_messages)
    monkeypatch.setattr(engine, "_call_llm", fake_call_llm)

    result = asyncio.run(engine.execute_query("制定一个 plan"))

    assert result == "done"
    assert calls == ["abc123"]
    assert engine.current_plan_branch == "abc123"
