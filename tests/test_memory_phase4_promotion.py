"""Phase 4 tests for promoting compressed session state into long-term memory."""

import pytest

from core.compression_engine import CompressionEngine, CompressionStrategy
from core.memory_items import MemoryKind
from core.memory_manager import MemoryManager


def phase4_messages():
    return [
        {"role": "user", "content": "继续 Phase 4：把压缩状态晋升为长期记忆，并跟 memory 模块对齐"},
        {"role": "assistant", "content": "用户偏好: 要测试驱动，要求长期记忆有来源引用\n决定采用 MemoryManager.save_memory_item 作为唯一保存入口"},
        {
            "role": "assistant",
            "content": "调用编辑工具修改 core/compression_engine.py",
            "tool_calls": [
                {"id": "call_edit", "type": "function", "function": {"name": "edit_file", "arguments": '{"path":"core/compression_engine.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_edit", "name": "edit_file", "content": "成功修改 core/compression_engine.py"},
        {"role": "assistant", "content": "运行 python -m pytest tests/test_memory_phase4_promotion.py -q"},
        {"role": "assistant", "content": "❌ 测试失败: AssertionError: source_turn_ids missing in core/compression_engine.py"},
        {"role": "assistant", "content": "修复 metadata 后 ✅ 1 passed。完成: 使用 MemoryManager 检索验证 promotion"},
    ]


@pytest.mark.asyncio
async def test_compression_promotes_bug_decision_preference_and_workflow_to_memory(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    result = await manager.compress_messages(
        phase4_messages(),
        llm_summarizer_func=None,
        strategy=CompressionStrategy.KEYFRAME,
        min_keep=4,
    )

    assert result.success
    promoted = result.metadata["promoted_memory_items"]
    promoted_kinds = {item["kind"] for item in promoted}
    assert MemoryKind.BUG.value in promoted_kinds
    assert MemoryKind.DECISION.value in promoted_kinds
    assert MemoryKind.PREFERENCE.value in promoted_kinds
    assert MemoryKind.WORKFLOW.value in promoted_kinds
    assert all(item["metadata"].get("source") == "compression_promotion" for item in promoted)
    assert all(item["metadata"].get("source_turn_ids") for item in promoted)

    stored_items = manager.long_term_memory.get_all_items()
    assert len(stored_items) == len(promoted)
    assert {item.id for item in stored_items} == set(result.metadata["promoted_memory_item_ids"])


@pytest.mark.asyncio
async def test_promoted_bug_and_file_history_are_retrievable_through_memory_manager(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    await manager.compress_messages(
        phase4_messages(),
        llm_summarizer_func=None,
        strategy=CompressionStrategy.KEYFRAME,
        min_keep=4,
    )

    error_results = manager.retrieve_error_history("AssertionError", top_k=5)
    file_results = manager.retrieve_file_history("core/compression_engine.py", top_k=5)

    assert error_results
    assert error_results[0].item.kind == MemoryKind.BUG.value
    assert "AssertionError" in error_results[0].item.searchable_text()
    assert error_results[0].item.metadata["source_turn_ids"]

    assert file_results
    assert any("core/compression_engine.py" in result.item.files for result in file_results)
    assert any(result.item.metadata.get("source") == "compression_promotion" for result in file_results)


@pytest.mark.asyncio
async def test_compression_promotion_deduplicates_existing_memory_items(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))

    first = await manager.compress_messages(
        phase4_messages(),
        llm_summarizer_func=None,
        strategy=CompressionStrategy.KEYFRAME,
        min_keep=4,
    )
    second = await manager.compress_messages(
        phase4_messages(),
        llm_summarizer_func=None,
        strategy=CompressionStrategy.KEYFRAME,
        min_keep=4,
    )

    assert first.metadata["promoted_memory_items"]
    assert second.metadata["promoted_memory_items"] == []
    assert len(manager.long_term_memory.get_all_items()) == len(first.metadata["promoted_memory_items"])


@pytest.mark.asyncio
async def test_compression_engine_can_accept_optional_memory_manager_directly(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    engine = CompressionEngine(memory_manager=manager)

    result = await engine.compress(
        phase4_messages(),
        strategy=CompressionStrategy.KEYFRAME,
        min_keep=4,
    )

    assert result.metadata["promoted_memory_item_ids"]
    assert manager.retrieve_file_history("core/compression_engine.py", top_k=3)
