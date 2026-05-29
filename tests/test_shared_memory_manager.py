"""共享 MemoryManager 启动注入回归测试。"""

from pathlib import Path

from core.context import ContextManager
from core.engine import AgentEngine
from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager
from tools import get_default_tools


class DummyPlanManager:
    current_goal = "共享 MemoryManager 测试"

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


class DummyTool:
    name = "dummy"

    def to_anthropic_spec(self):
        return {"name": self.name, "description": "dummy", "input_schema": {"type": "object", "properties": {}}}

    def run(self, **kwargs):
        return "ok"


def _memory_tool_map(tools):
    return {tool.name: tool for tool in tools if tool.name.startswith("memory_")}


def test_context_manager_accepts_injected_memory_manager(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    context = ContextManager(memory_manager=manager, plan_manager=DummyPlanManager())

    assert context.memory_manager is manager
    assert context.working_memory is manager.working_memory
    assert context.episodic_memory is manager.episodic_memory
    assert context.long_term_memory is manager.long_term_memory
    assert context.compression_engine is manager.compression_engine


def test_agent_engine_and_memory_tools_share_injected_memory_manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sessions").mkdir()
    plan_manager = DummyPlanManager()
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "memory" / "long_term"), plan_manager=plan_manager)
    tools = get_default_tools(plan_manager=plan_manager, memory_manager=manager)

    engine = AgentEngine(
        tools=tools,
        model="fake-model",
        plan_manager=plan_manager,
        base_url="http://example.invalid",
        api_key="x",
        session_id="shared_memory",
        memory_manager=manager,
    )

    assert engine.context.memory_manager is manager
    memory_tools = _memory_tool_map(tools)
    assert memory_tools
    assert all(tool.memory_manager is manager for tool in memory_tools.values())

    memory_tools["memory_save"].run(
        kind=MemoryKind.ARCHITECTURE.value,
        title="共享 MemoryManager",
        content="tools 与 AgentEngine 应共用同一个 MemoryManager 实例。",
        concepts=["shared-memory-manager"],
    )
    relevant_history = engine._build_relevant_memory_context("shared-memory-manager", top_k=3)

    assert "共享 MemoryManager" in relevant_history


def test_agent_engine_keeps_existing_default_memory_manager_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sessions").mkdir()

    engine = AgentEngine(
        tools=[DummyTool()],
        model="fake-model",
        plan_manager=DummyPlanManager(),
        base_url="http://example.invalid",
        api_key="x",
        session_id="default_memory_manager",
    )

    assert isinstance(engine.context.memory_manager, MemoryManager)

    engine.context.memory_manager.save_memory_item(MemoryItem(
        kind=MemoryKind.FACT.value,
        title="默认 MemoryManager 仍可用",
        content="未注入共享实例时，ContextManager 会自行创建 MemoryManager。",
        concepts=["default-memory-manager"],
    ))
    relevant_history = engine._build_relevant_memory_context("default-memory-manager", top_k=3)

    assert "默认 MemoryManager 仍可用" in relevant_history
