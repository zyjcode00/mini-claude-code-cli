import pytest
from pydantic import BaseModel

from core.engine import AgentEngine
from core.plan import PlanManager
from tools.base import BaseTool


class DummyArgs(BaseModel):
    value: str = "ok"


class EchoTool(BaseTool):
    name = "echo"
    description = "echo tool"
    args_schema = DummyArgs

    def run(self, **kwargs) -> str:
        return f"echo:{kwargs.get('value', 'ok')}"


class DummyPlanManager(PlanManager):
    pass


def make_engine():
    return AgentEngine(
        tools=[EchoTool()],
        model="gpt-test",
        plan_manager=DummyPlanManager(),
        base_url="http://example.invalid/v1",
        api_key="test",
        session_id="test_tool_pairing_unit",
    )


def assert_valid_openai_tool_pairs(messages):
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            contiguous_ids = set()
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                contiguous_ids.add(messages[j].get("tool_call_id"))
                j += 1
            assert expected_ids.issubset(contiguous_ids), (
                f"assistant tool_calls {expected_ids - contiguous_ids} are not followed "
                "by contiguous tool responses"
            )

        if msg.get("role") == "tool":
            block_start = i
            while block_start > 0 and messages[block_start - 1].get("role") == "tool":
                block_start -= 1
            assert block_start > 0
            prev_msg = messages[block_start - 1]
            assert prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls")
            assert msg.get("tool_call_id") in {tc["id"] for tc in prev_msg["tool_calls"]}


@pytest.mark.asyncio
async def test_pre_tool_memory_context_does_not_split_assistant_tool_pair(monkeypatch):
    engine = make_engine()

    async def fake_compress_messages():
        return None

    monkeypatch.setattr(engine, "compress_messages", fake_compress_messages)
    monkeypatch.setattr(engine, "_build_relevant_memory_context", lambda user_input: "")
    monkeypatch.setattr(engine, "_observe_prompt_submit", lambda user_input: None)
    monkeypatch.setattr(engine, "_observe_tool_result", lambda name, inp, res: None)
    monkeypatch.setattr(engine, "_build_post_tool_failure_memory_context", lambda name, res: "")
    monkeypatch.setattr(engine, "save_session", lambda: None)
    monkeypatch.setattr(engine, "_remember_task_completion", lambda user_input, final_ans: None)

    calls = {"count": 0}

    async def fake_call_llm(relevant_history="", user_input=""):
        calls["count"] += 1
        if calls["count"] == 1:
            engine.last_oa_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"value":"ok"}'},
                    }
                ],
            }
            return [
                {"type": "tool_use", "id": "call_1", "name": "echo", "input": {"value": "ok"}}
            ], "tool_use"

        engine.last_oa_msg = {"role": "assistant", "content": "done"}
        return [{"type": "text", "text": "done"}], "end_turn"

    monkeypatch.setattr(engine, "_call_llm", fake_call_llm)
    monkeypatch.setattr(engine, "_build_pre_tool_memory_context", lambda name, inp: "memory hint")

    result = await engine.execute_query("start")

    assert result == "done"
    assert_valid_openai_tool_pairs(engine.context.messages)

    assistant_index = next(i for i, m in enumerate(engine.context.messages) if m.get("tool_calls"))
    tool_index = next(i for i, m in enumerate(engine.context.messages) if m.get("role") == "tool")
    inserted_user_indices = [
        i for i, m in enumerate(engine.context.messages)
        if m.get("role") == "user" and m.get("content") == "memory hint"
    ]

    assert tool_index == assistant_index + 1
    assert not any(assistant_index < i < tool_index for i in inserted_user_indices)


def test_llm_summary_keep_messages_sanitizes_half_tool_pair():
    from core.compression_engine import CompressionEngine

    engine = CompressionEngine()
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "older"},
        {"role": "user", "content": "before"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_missing_tool",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "newer"},
        {"role": "assistant", "content": "latest"},
    ]

    sanitized = engine._sanitize_openai_tool_pairs(messages[-4:])

    assert all(
        not (msg.get("role") == "assistant" and msg.get("tool_calls"))
        for msg in sanitized
    )
    assert_valid_openai_tool_pairs(sanitized)


def test_sanitizer_removes_tool_pair_split_by_inserted_user_message():
    from core.compression_engine import CompressionEngine

    engine = CompressionEngine()
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_split",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "memory hint inserted in wrong place"},
        {"role": "tool", "tool_call_id": "call_split", "name": "echo", "content": "ok"},
    ]

    sanitized = engine._sanitize_openai_tool_pairs(messages)

    assert sanitized == [{"role": "user", "content": "memory hint inserted in wrong place"}]
    assert engine._validate_message_ordering(sanitized)


def test_engine_sanitizes_snapshot_before_openai_call(monkeypatch):
    engine = make_engine()
    captured = {}

    bad_messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
    ]
    engine.context.messages = bad_messages[:]

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]

        class Message:
            content = "ok"
            tool_calls = None

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()

    monkeypatch.setattr(engine.client.chat.completions, "create", fake_create)

    import asyncio

    content_blocks, stop_reason = asyncio.run(engine._call_llm(user_input="hello"))

    assert stop_reason == "end_turn"
    sent_messages = captured["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1:] == [{"role": "user", "content": "hello"}]
    assert_valid_openai_tool_pairs(sent_messages[1:])


@pytest.mark.asyncio
async def test_context_failed_compression_sanitizes_existing_messages():
    from core.compression_engine import CompressionStrategy
    from core.context import ContextManager

    ctx = ContextManager(max_history=1, min_keep=4)
    ctx.messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_failed_compress",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
    ]

    async def timeout_like_summarizer(prompt):
        return None

    success = await ctx.compress(timeout_like_summarizer, strategy=CompressionStrategy.LLM_SUMMARY)

    assert success is False
    assert ctx.messages == [{"role": "user", "content": "hello"}]
    assert_valid_openai_tool_pairs(ctx.messages)
