"""Phase 5 ContextAssembler budget and provider-safety tests."""

from core.context_assembler import ContextAssembler, ContextBudget


def _tool_pair():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "ok"},
    ]


def test_context_assembler_preserves_current_user_request_under_tight_budget():
    assembler = ContextAssembler(ContextBudget(total=220, system=30, memory=20, compressed_state=20, recent_turns=10, current_user=80))
    messages = [
        {"role": "user", "content": "old request " * 80},
        {"role": "assistant", "content": "old answer " * 80},
        {"role": "user", "content": "CURRENT USER REQUEST: implement phase 5"},
    ]

    result = assembler.assemble(
        base_system_prompt="system rules",
        memory_context="memory " * 100,
        compressed_state="summary " * 100,
        messages=messages,
        provider="openai",
    )

    assert result.messages[-1]["role"] == "user"
    assert "CURRENT USER REQUEST" in result.messages[-1]["content"]
    assert result.token_estimate > 0


def test_context_assembler_keeps_tool_pair_atomic_when_recent_turn_budget_small():
    assembler = ContextAssembler(ContextBudget(total=500, system=40, memory=10, compressed_state=10, recent_turns=1, current_user=80))
    messages = [
        {"role": "user", "content": "before"},
        *_tool_pair(),
        {"role": "user", "content": "current"},
    ]

    result = assembler.assemble(base_system_prompt="sys", messages=messages, provider="openai")

    roles = [msg["role"] for msg in result.messages]
    assert roles[-1] == "user"
    assert "assistant" in roles
    assert "tool" in roles
    assistant_index = roles.index("assistant")
    tool_index = roles.index("tool")
    assert tool_index == assistant_index + 1
    assert result.messages[assistant_index]["tool_calls"][0]["id"] == result.messages[tool_index]["tool_call_id"]


def test_context_assembler_drops_incomplete_tool_pairs_for_provider_validation():
    assembler = ContextAssembler()
    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "missing_tool",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "current"},
    ]

    result = assembler.assemble(base_system_prompt="sys", messages=messages, provider="openai")

    assert all("tool_calls" not in msg for msg in result.messages)
    assert [msg["role"] for msg in result.messages] == ["user", "user"]


def test_context_assembler_injects_memory_into_system_not_between_tool_pair():
    assembler = ContextAssembler(ContextBudget(memory=20))
    messages = [*_tool_pair(), {"role": "user", "content": "current"}]

    result = assembler.assemble(
        base_system_prompt="base system",
        memory_context="MEMORY_CONTEXT_SHOULD_BE_SYSTEM_ONLY " * 20,
        messages=messages,
        provider="openai",
    )

    assert "MEMORY_CONTEXT_SHOULD_BE_SYSTEM_ONLY" in result.system_prompt
    assert all("MEMORY_CONTEXT_SHOULD_BE_SYSTEM_ONLY" not in str(msg) for msg in result.messages)
    roles = [msg["role"] for msg in result.messages]
    assert roles[:2] == ["assistant", "tool"]


def test_context_assembler_openai_messages_starts_with_single_system_message():
    assembler = ContextAssembler()
    result = assembler.assemble(
        base_system_prompt="base",
        plan_context="PLAN_CONTEXT",
        memory_context="MEMORY_CONTEXT",
        compressed_state="COMPRESSED_STATE",
        messages=[{"role": "user", "content": "current"}],
        provider="openai",
    )

    openai_messages = result.openai_messages
    assert openai_messages[0] == {"role": "system", "content": result.system_prompt}
    assert [msg["role"] for msg in openai_messages].count("system") == 1
    assert "PLAN_CONTEXT" in result.system_prompt
    assert "MEMORY_CONTEXT" in result.system_prompt
    assert "COMPRESSED_STATE" in result.system_prompt
    assert openai_messages[-1]["content"] == "current"
