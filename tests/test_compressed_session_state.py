"""Phase 3 tests for structured compressed session state."""

import pytest

from core.compression_engine import (
    CompressedSessionState,
    CompressionEngine,
    CompressionStrategy,
)
from core.turn_builder import TurnBuilder


def phase3_messages():
    return [
        {"role": "user", "content": "实现 Phase 3 CompressedSessionState，并更新 tests/test_compressed_session_state.py"},
        {"role": "assistant", "content": "计划：\n✅ 已完成: 阅读架构文档\n⏳ 未完成: 运行 pytest\n决定采用规则提取作为 deterministic fallback"},
        {
            "role": "assistant",
            "content": "调用写文件工具",
            "tool_calls": [
                {"id": "call_write", "type": "function", "function": {"name": "write_full_file", "arguments": '{"path":"core/compression_engine.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_write", "name": "write_full_file", "content": "成功修改 core/compression_engine.py 和 tests/test_compressed_session_state.py"},
        {"role": "assistant", "content": "运行 python -m pytest tests/test_compressed_session_state.py -q"},
        {"role": "tool", "tool_call_id": "orphan", "content": "孤立 tool，不能进入状态"},
        {"role": "assistant", "content": "❌ 测试失败: AssertionError in tests/test_compressed_session_state.py"},
        {"role": "assistant", "content": "修复后 ✅ 2 passed in 0.10s。注意: 不要保留半截 assistant tool call/tool 响应"},
    ]


def test_compressed_session_state_serialization_and_prompt_rendering():
    state = CompressedSessionState(
        task_goal="目标",
        current_status="in_progress",
        completed_steps=["完成 A"],
        pending_steps=["待办 B"],
        files_changed=[{"path": "core/a.py", "action": "modified"}],
        tests_run=[{"command": "pytest", "status": "passed"}],
        errors_encountered=[{"message": "Error: boom"}],
        key_decisions=["采用 deterministic fallback"],
    )

    restored = CompressedSessionState.from_dict(state.to_dict())
    rendered = restored.render_for_prompt()

    assert restored.task_goal == "目标"
    assert "### 压缩后的会话状态" in rendered
    assert "文件变更" in rendered
    assert "关键错误" in rendered


def test_extract_compressed_state_from_turn_metadata():
    engine = CompressionEngine()
    turns = TurnBuilder().build(phase3_messages())

    state = engine._extract_compressed_state(turns)

    assert state.task_goal.startswith("实现 Phase 3")
    assert any("阅读架构文档" in step for step in state.completed_steps)
    assert any("运行 pytest" in step for step in state.pending_steps)
    assert any(item["path"] == "core/compression_engine.py" for item in state.files_changed)
    assert any(item["status"] in {"failed", "resolved"} for item in state.errors_encountered)
    assert any("deterministic fallback" in decision for decision in state.key_decisions)
    assert any(test["status"] == "passed" for test in state.tests_run)
    assert any("半截" in note for note in state.tool_safety_notes)
    assert all("orphan" not in turn_id for turn_id in state.source_turn_ids)


@pytest.mark.asyncio
async def test_keyframe_result_carries_structured_state_and_preserves_tool_pairs():
    engine = CompressionEngine()

    result = await engine.compress(
        phase3_messages(),
        strategy=CompressionStrategy.KEYFRAME,
        target_ratio=0.45,
        min_keep=4,
    )

    assert result.success
    assert result.compressed_state is not None
    assert result.metadata["compressed_state"] == result.compressed_state.to_dict()
    assert "压缩后的会话状态" in result.metadata["compressed_state_prompt"]
    assert "tool_calls" not in result.summary.summary_text
    assert engine._validate_message_ordering(result.compressed_messages)
    assert not any(msg.get("tool_call_id") == "orphan" for msg in result.compressed_messages)


@pytest.mark.asyncio
async def test_llm_summary_result_uses_structured_state_when_llm_returns_plain_text():
    engine = CompressionEngine()

    async def summarizer(prompt: str) -> str:
        return "自然语言摘要，无 JSON。关键决策: LLM 只作为增强，不作为唯一来源。"

    result = await engine.compress(
        phase3_messages(),
        strategy=CompressionStrategy.LLM_SUMMARY,
        llm_summarizer_func=summarizer,
        min_keep=3,
    )

    assert result.success
    assert result.compressed_state is not None
    assert result.summary is not None
    assert result.summary.task_goal.startswith("实现 Phase 3")
    assert "压缩后的会话状态" in result.summary.summary_text
    assert any("LLM 只作为增强" in decision for decision in result.compressed_state.key_decisions)
    assert result.metadata["compressed_state"]["task_goal"].startswith("实现 Phase 3")
