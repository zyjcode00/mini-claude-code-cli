# tests/test_turn_builder.py
"""Tests for provider-safe conversation turn construction."""

import pytest

from core.compression_engine import CompressionEngine, CompressionStrategy
from core.turn_builder import TurnBuilder


def tool_call(call_id: str, name: str = "echo"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def assert_valid_openai_tool_pairs(messages):
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            contiguous_ids = set()
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                contiguous_ids.add(messages[j].get("tool_call_id"))
                j += 1
            assert expected_ids.issubset(contiguous_ids)

        if msg.get("role") == "tool":
            block_start = i
            while block_start > 0 and messages[block_start - 1].get("role") == "tool":
                block_start -= 1
            assert block_start > 0
            prev_msg = messages[block_start - 1]
            assert prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls")
            assert msg.get("tool_call_id") in {tc["id"] for tc in prev_msg["tool_calls"]}


def test_turn_builder_keeps_plain_messages_as_individual_turns():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "next"},
    ]

    builder = TurnBuilder()
    turns = builder.build(messages)

    assert len(turns) == 3
    assert [turn.messages for turn in turns] == [[msg] for msg in messages]
    assert builder.flatten(turns) == messages
    assert all(turn.is_valid_openai_tool_turn for turn in turns)


def test_turn_builder_groups_complete_openai_tool_call_pair_atomically():
    messages = [
        {"role": "user", "content": "run tools"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_1"), tool_call("call_2")],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "echo", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "name": "echo", "content": "two"},
        {"role": "assistant", "content": "done"},
    ]

    turns = TurnBuilder().build(messages)

    assert len(turns) == 3
    tool_turn = turns[1]
    assert tool_turn.start_index == 1
    assert tool_turn.end_index == 3
    assert tool_turn.messages == messages[1:4]
    assert tool_turn.has_tool_calls
    assert tool_turn.is_tool_pair_complete
    assert tool_turn.is_valid_openai_tool_turn
    assert tool_turn.missing_tool_call_ids == []
    assert tool_turn.orphan_tool_call_ids == []


def test_turn_builder_marks_missing_tool_response_incomplete():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_missing")],
        },
        {"role": "assistant", "content": "next message interrupts tool block"},
    ]

    turns = TurnBuilder().build(messages)

    assert len(turns) == 2
    assert turns[0].has_tool_calls
    assert not turns[0].is_tool_pair_complete
    assert not turns[0].is_valid_openai_tool_turn
    assert turns[0].missing_tool_call_ids == ["call_missing"]


def test_turn_builder_marks_orphan_tool_message_incomplete():
    messages = [
        {"role": "tool", "tool_call_id": "call_orphan", "name": "echo", "content": "orphan"},
        {"role": "user", "content": "continue"},
    ]

    turns = TurnBuilder().build(messages)

    assert len(turns) == 2
    assert not turns[0].has_tool_calls
    assert not turns[0].is_tool_pair_complete
    assert not turns[0].is_valid_openai_tool_turn
    assert turns[0].orphan_tool_call_ids == ["call_orphan"]


def test_turn_builder_build_complete_messages_drops_incomplete_tool_turns():
    messages = [
        {"role": "user", "content": "keep"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_missing")],
        },
        {"role": "tool", "tool_call_id": "call_orphan", "name": "echo", "content": "orphan"},
        {"role": "assistant", "content": "also keep"},
    ]

    complete_messages = TurnBuilder().build_complete_messages(messages)

    assert complete_messages == [
        {"role": "user", "content": "keep"},
        {"role": "assistant", "content": "also keep"},
    ]
    assert_valid_openai_tool_pairs(complete_messages)


@pytest.mark.asyncio
async def test_sliding_window_selects_complete_turns_without_splitting_tool_pair():
    engine = CompressionEngine()
    messages = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_1"), tool_call("call_2")],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "echo", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "name": "echo", "content": "two"},
        {"role": "assistant", "content": "final"},
    ]

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.4,
        min_keep=2,
    )

    assert result.success
    assert result.compressed_messages == messages[1:]
    assert_valid_openai_tool_pairs(result.compressed_messages)


def test_sanitizer_uses_turn_builder_to_remove_missing_and_orphan_tool_turns():
    engine = CompressionEngine()
    messages = [
        {"role": "user", "content": "keep"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_missing")],
        },
        {"role": "tool", "tool_call_id": "call_orphan", "name": "echo", "content": "orphan"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_ok")],
        },
        {"role": "tool", "tool_call_id": "call_ok", "name": "echo", "content": "ok"},
    ]

    sanitized = engine._sanitize_openai_tool_pairs(messages)

    assert sanitized == [messages[0], messages[3], messages[4]]
    assert_valid_openai_tool_pairs(sanitized)
