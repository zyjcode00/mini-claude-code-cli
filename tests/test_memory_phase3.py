"""阶段 3：AgentEngine 主动记忆与 memory 工具接口测试。"""

import tempfile
from pathlib import Path

import pytest

from core.engine import AgentEngine
from core.memory_items import MemoryKind, ObservationType
from tools import get_default_tools
from tools.memory_tool import MemoryRecallTool, MemorySaveTool


class DummyPlanManager:
    current_goal = "阶段3主动记忆测试"

    def get_formatted_plan(self):
        return "暂无计划"

    def get_completed_goals(self):
        return []

    def get_plan_id(self):
        return None

    def is_plan_complete(self):
        return False

    def has_incomplete_tasks(self):
        return False

    def to_dict(self):
        return {}

    def from_dict(self, data):
        return None

    def validate_state(self):
        return True, []

    def auto_fix(self):
        return None

    def clear_plan(self):
        return None


@pytest.fixture
def isolated_memory_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sessions").mkdir()
    Path("memory/long_term").mkdir(parents=True)
    return tmp_path / "memory" / "long_term"


def test_memory_save_and_recall_tools_share_memory_manager(isolated_memory_dir):
    save_tool = MemorySaveTool(long_term_storage_dir=str(isolated_memory_dir))
    recall_tool = MemoryRecallTool(memory_manager=save_tool.memory_manager)

    save_result = save_tool.run(
        kind="bug",
        title="DeepSeek reasoning_content 必须回传",
        content="OpenAI 兼容接口返回 reasoning_content 时，下一轮 messages 必须原样保留。",
        files=["core/engine.py"],
        concepts=["deepseek", "reasoning_content"],
        importance=0.9,
    )

    assert "✅ 已保存长期记忆" in save_result
    assert "DeepSeek reasoning_content 必须回传" in save_result

    recall_result = recall_tool.run(query="reasoning_content openai", top_k=3)
    assert "🧠 召回 1 条长期记忆" in recall_result
    assert "DeepSeek reasoning_content 必须回传" in recall_result
    assert "core/engine.py" in recall_result


def test_default_tools_include_memory_tools_with_shared_manager(isolated_memory_dir):
    tools = get_default_tools(memory_storage_dir=str(isolated_memory_dir))
    tool_map = {tool.name: tool for tool in tools}

    assert "memory_save" in tool_map
    assert "memory_recall" in tool_map
    assert tool_map["memory_save"].memory_manager is tool_map["memory_recall"].memory_manager

    tool_map["memory_save"].run(
        kind="workflow",
        title="修改 core 后运行 pytest",
        content="任何 core/ 逻辑修改都必须运行 pytest tests。",
        concepts=["pytest", "core"],
    )
    result = tool_map["memory_recall"].run(query="core pytest", top_k=1)
    assert "修改 core 后运行 pytest" in result


def test_agent_engine_recalls_memory_before_llm_call(isolated_memory_dir):
    engine = AgentEngine(
        tools=get_default_tools(memory_storage_dir=str(isolated_memory_dir)),
        model="claude-test",
        plan_manager=DummyPlanManager(),
        session_id="phase3_recall",
    )
    engine.context.memory_manager.save_memory_item(
        MemorySaveTool(memory_manager=engine.context.memory_manager).create_item(
            kind="architecture",
            title="AgentEngine 接入主动召回",
            content="任务开始前应调用 MemoryManager.recall 并注入相关长期记忆。",
            concepts=["AgentEngine", "recall"],
            files=["core/engine.py"],
        )
    )

    relevant_history = engine._build_relevant_memory_context("AgentEngine recall", top_k=3)

    assert "[相关长期记忆]" in relevant_history
    assert "AgentEngine 接入主动召回" in relevant_history
    assert "core/engine.py" in relevant_history


def test_agent_engine_records_prompt_tool_and_completion_observations(isolated_memory_dir):
    engine = AgentEngine(
        tools=get_default_tools(memory_storage_dir=str(isolated_memory_dir)),
        model="claude-test",
        plan_manager=DummyPlanManager(),
        session_id="phase3_observe",
    )

    prompt_item = engine._observe_prompt_submit("帮我修复 pytest 失败")
    tool_item = engine._observe_tool_result(
        tool_name="run_pytest",
        tool_input={"path": "tests/test_memory_phase3.py"},
        result="✅ 测试通过！ 1 passed",
    )
    failure_item = engine._observe_tool_result(
        tool_name="run_pytest",
        tool_input={"path": "tests/test_memory_phase3.py"},
        result="❌ 测试失败 Traceback File tests/test_memory_phase3.py",
    )
    completion_item = engine._remember_task_completion(
        user_input="帮我修复 pytest 失败",
        final_answer="已修复 tests/test_memory_phase3.py 并通过 pytest。",
    )

    assert prompt_item.kind == MemoryKind.TASK.value
    assert prompt_item.metadata["raw_observation"]["event_type"] == ObservationType.PROMPT_SUBMIT.value
    assert tool_item.kind == MemoryKind.WORKFLOW.value
    assert tool_item.metadata["raw_observation"]["event_type"] == ObservationType.TEST_RESULT.value
    assert failure_item.kind == MemoryKind.BUG.value
    assert failure_item.metadata["raw_observation"]["event_type"] == ObservationType.TOOL_FAILURE.value
    assert completion_item.kind == MemoryKind.TASK.value
    assert "pytest" in " ".join(completion_item.concepts)

    recall_results = engine.context.memory_manager.recall("pytest phase3", top_k=5, include_summaries=False)
    titles = [result.item.title for result in recall_results]
    assert any("工具执行成功" in title or "工具执行失败" in title for title in titles)
    assert any("任务完成" in title for title in titles)
