"""测试阶段 2 新增的统一记忆数据模型。"""

import json

from core.memory_items import (
    MemoryItem,
    MemoryKind,
    MemoryRecallResult,
    MemoryStatus,
    ObservationType,
    RawObservation,
)


def test_raw_observation_roundtrip_and_defaults():
    observation = RawObservation(
        session_id="session-1",
        project="mini-claude-code-cli",
        cwd="D:/LLM/mini-claude-code-cli",
        event_type=ObservationType.POST_TOOL_USE,
        tool_name="run_pytest",
        tool_input={"path": "tests"},
        tool_output="98 passed",
        files=("tests/test_memory_items.py",),
        metadata={"exit_code": 0},
    )

    data = observation.to_dict()
    restored = RawObservation.from_dict(data)

    assert observation.id.startswith("obs_")
    assert data["event_type"] == "post_tool_use"
    assert restored.session_id == "session-1"
    assert restored.tool_name == "run_pytest"
    assert restored.files == ["tests/test_memory_items.py"]
    assert restored.metadata["exit_code"] == 0


def test_raw_observation_json_roundtrip_and_unknown_type_fallback():
    raw_json = json.dumps({
        "id": "obs-fixed",
        "event_type": "unknown_event",
        "user_prompt": "继续阶段2",
    })

    restored = RawObservation.from_json(raw_json)

    assert restored.id == "obs-fixed"
    assert restored.event_type == ObservationType.OTHER.value
    assert restored.user_prompt == "继续阶段2"


def test_memory_item_roundtrip_with_enum_and_type_alias():
    item = MemoryItem(
        kind=MemoryKind.ARCHITECTURE,
        title="MemoryManager 统一记忆入口",
        content="MemoryManager 统一编排 Working/Episodic/LongTerm 记忆。",
        project="mini-claude-code-cli",
        concepts=["MemoryManager", "长期记忆"],
        files=["core/memory_manager.py"],
        source_observation_ids=["obs-1"],
        source_session_ids=["session-1"],
        importance=1.5,
        confidence="0.9",
        metadata={"phase": 2},
    )

    data = item.to_dict()
    restored = MemoryItem.from_dict(data)

    assert item.id.startswith("mem_")
    assert data["kind"] == "architecture"
    assert data["type"] == "architecture"
    assert restored.kind == MemoryKind.ARCHITECTURE.value
    assert restored.status == MemoryStatus.ACTIVE.value
    assert restored.importance == 1.0
    assert restored.confidence == 0.9
    assert restored.files == ["core/memory_manager.py"]
    assert "MemoryManager" in restored.searchable_text()


def test_memory_item_from_legacy_type_alias_and_json_roundtrip():
    item = MemoryItem.from_dict({
        "id": "mem-fixed",
        "type": "bug",
        "title": "长期记忆索引字段兼容",
        "content": "FileChange 使用 path 字段，旧 file_path 需要兼容。",
        "status": "archived",
    })

    restored = MemoryItem.from_json(item.to_json())

    assert restored.id == "mem-fixed"
    assert restored.kind == MemoryKind.BUG.value
    assert restored.status == MemoryStatus.ARCHIVED.value
    assert restored.content.startswith("FileChange")


def test_memory_recall_result_roundtrip_accepts_item_dict():
    item = MemoryItem(kind="workflow", title="测试驱动", content="修改 core 后必须跑 pytest。")
    result = MemoryRecallResult(item=item.to_dict(), score=0.87, source="long_term_items", reason="匹配 pytest")

    data = result.to_dict()
    restored = MemoryRecallResult.from_dict(data)

    assert result.item.kind == MemoryKind.WORKFLOW.value
    assert restored.item.title == "测试驱动"
    assert restored.score == 0.87
    assert restored.source == "long_term_items"
    assert restored.reason == "匹配 pytest"
