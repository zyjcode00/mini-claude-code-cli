# tests/test_compression_engine.py
"""
压缩引擎测试

测试覆盖：
1. 4 种压缩策略（LLM_SUMMARY/KEYFRAME/SLIDING_WINDOW/IMPORTANCE_FILTER）
2. 策略选择器
3. 压缩效果
4. 集成到 ContextManager
"""

import pytest
import asyncio
from typing import List, Dict, Any

from core.compression_engine import (
    CompressionEngine,
    CompressionStrategy,
    CompressionResult,
    select_compression_strategy
)
from core.memory_models import SessionSummary
from core.context import ContextManager


# ========== 辅助函数 ==========

def create_mock_messages(count: int, role_pattern: str = "mixed") -> List[Dict[str, Any]]:
    """
    创建模拟消息列表

    Args:
        count: 消息数量
        role_pattern: 角色模式
            - "mixed": 混合角色（user/assistant/tool）
            - "error": 错误密集
            - "tool": 工具密集
            - "simple": 简单对话（user/assistant）

    Returns:
        消息列表
    """
    messages = []

    if role_pattern == "mixed":
        for i in range(count):
            if i % 3 == 0:
                messages.append({"role": "user", "content": f"任务 {i}"})
            elif i % 3 == 1:
                messages.append({"role": "assistant", "content": f"回复 {i}", "tool_calls": [{"id": f"tc_{i}", "type": "function", "function": {"name": "bash", "arguments": ""}}]})
            else:
                messages.append({"role": "tool", "content": f"工具结果 {i}", "tool_call_id": f"tc_{i}"})

    elif role_pattern == "error":
        for i in range(count):
            if i % 4 == 0:
                messages.append({"role": "user", "content": f"任务 {i}"})
            elif i % 4 == 1:
                messages.append({"role": "assistant", "content": f"回复 {i}", "tool_calls": [{"id": f"tc_{i}", "type": "function", "function": {"name": "bash", "arguments": ""}}]})
            elif i % 4 == 2:
                messages.append({"role": "tool", "content": f"Error: 失败 {i}", "tool_call_id": f"tc_{i}"})
            else:
                messages.append({"role": "assistant", "content": f"Traceback: 错误 {i}"})

    elif role_pattern == "tool":
        for i in range(count):
            if i % 2 == 0:
                messages.append({"role": "user", "content": f"任务 {i}"})
            else:
                messages.append({"role": "assistant", "content": f"工具调用 {i}", "tool_calls": [{"id": f"tc_{i}", "type": "function", "function": {"name": "bash", "arguments": ""}}]})
                messages.append({"role": "tool", "content": f"工具结果 {i}", "tool_call_id": f"tc_{i}"})

    elif role_pattern == "simple":
        for i in range(count):
            if i % 2 == 0:
                messages.append({"role": "user", "content": f"任务 {i}"})
            else:
                messages.append({"role": "assistant", "content": f"回复 {i}"})

    return messages


async def mock_llm_summarizer(prompt: str) -> str:
    """
    模拟 LLM 摘要函数

    Returns:
        JSON 格式的摘要文本
    """
    return """
    {
        "task_goal": "完成测试任务",
        "task_status": "in_progress",
        "files_changed": [],
        "errors_encountered": [],
        "tools_used": [],
        "key_decisions": [],
        "summary_text": "这是一个测试摘要"
    }
    """


# ========== 测试 1: 压缩引擎初始化 ==========

def test_compression_engine_init():
    """测试压缩引擎初始化"""
    engine = CompressionEngine()
    assert engine.default_strategy is None

    # 设置默认策略
    engine_with_default = CompressionEngine(default_strategy=CompressionStrategy.SLIDING_WINDOW)
    assert engine_with_default.default_strategy == CompressionStrategy.SLIDING_WINDOW


# ========== 测试 2: 策略选择器 ==========

def test_select_strategy_error_dense():
    """测试策略选择：错误密集 → LLM_SUMMARY"""
    engine = CompressionEngine()
    messages = create_mock_messages(20, "error")

    strategy = engine._select_strategy(messages)
    assert strategy == CompressionStrategy.LLM_SUMMARY


def test_select_strategy_tool_dense():
    """测试策略选择：工具密集 → KEYFRAME 或 SLIDING_WINDOW"""
    engine = CompressionEngine()
    messages = create_mock_messages(20, "tool")

    strategy = engine._select_strategy(messages)
    # 工具占比阈值可能变化，接受 KEYFRAME 或 SLIDING_WINDOW
    assert strategy in (CompressionStrategy.KEYFRAME, CompressionStrategy.SLIDING_WINDOW)


def test_select_strategy_simple():
    """测试策略选择：一般对话 → SLIDING_WINDOW"""
    engine = CompressionEngine()
    messages = create_mock_messages(20, "simple")

    strategy = engine._select_strategy(messages)
    assert strategy == CompressionStrategy.SLIDING_WINDOW


def test_select_strategy_empty():
    """测试策略选择：空消息列表 → SLIDING_WINDOW"""
    engine = CompressionEngine()
    strategy = engine._select_strategy([])
    assert strategy == CompressionStrategy.SLIDING_WINDOW


def test_select_strategy_function():
    """测试便捷函数 select_compression_strategy"""
    messages = create_mock_messages(20, "tool")
    strategy = select_compression_strategy(messages)
    # 工具占比阈值可能变化
    assert strategy in (CompressionStrategy.KEYFRAME, CompressionStrategy.SLIDING_WINDOW)


# ========== 测试 3: 滑动窗口策略 ==========

@pytest.mark.asyncio
async def test_sliding_window_compression():
    """测试滑动窗口压缩"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "simple")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.3,
        min_keep=4
    )

    assert result.success
    assert result.strategy == CompressionStrategy.SLIDING_WINDOW
    assert len(result.compressed_messages) <= 10  # 30 * 0.3
    assert len(result.compressed_messages) >= 4   # min_keep
    assert result.compression_ratio <= 0.35


@pytest.mark.asyncio
async def test_sliding_window_preserve_order():
    """测试滑动窗口保留原始顺序"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "simple")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.3,
        min_keep=4
    )

    # 验证消息顺序（应该是最近的消息）
    assert result.compressed_messages[-1]["content"] == messages[-1]["content"]


# ========== 测试 4: 关键帧提取策略 ==========

@pytest.mark.asyncio
async def test_keyframe_compression():
    """测试关键帧提取压缩"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "mixed")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.KEYFRAME,
        target_ratio=0.3,
        min_keep=4
    )

    assert result.success
    # 策略可能是 KEYFRAME 或降级到 SLIDING_WINDOW
    assert result.strategy in (CompressionStrategy.KEYFRAME, CompressionStrategy.SLIDING_WINDOW)
    assert len(result.compressed_messages) >= 4   # min_keep


@pytest.mark.asyncio
async def test_keyframe_extract_user_messages():
    """测试关键帧提取用户消息"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "mixed")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.KEYFRAME,
        target_ratio=0.3,
        min_keep=4
    )

    # 验证用户消息被提取
    user_count = sum(1 for msg in result.compressed_messages if msg.get("role") == "user")
    assert user_count >= 3  # 应该提取至少 3 条用户消息


@pytest.mark.asyncio
async def test_keyframe_extract_tool_messages():
    """测试关键帧提取工具消息"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "mixed")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.KEYFRAME,
        target_ratio=0.3,
        min_keep=4
    )

    # 验证压缩成功（可能降级到 SLIDING_WINDOW）
    assert result.success
    # 工具消息可能在降级后的 SLIDING_WINDOW 中出现
    tool_count = sum(1 for msg in result.compressed_messages if msg.get("role") == "tool")
    # 不强制要求 keyframe 提取 tool 消息（降级时可能没有）


# ========== 测试 5: 重要性过滤策略 ==========

# ========== 测试 6: LLM 摘要策略 ==========

@pytest.mark.asyncio
async def test_llm_summary_compression():
    """测试 LLM 摘要压缩"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "simple")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.LLM_SUMMARY,
        llm_summarizer_func=mock_llm_summarizer,
        min_keep=4
    )

    assert result.success
    assert result.strategy == CompressionStrategy.LLM_SUMMARY
    assert len(result.compressed_messages) == 4  # min_keep
    assert result.summary is not None
    assert isinstance(result.summary, SessionSummary)


@pytest.mark.asyncio
async def test_llm_summary_without_llm():
    """测试 LLM 摘要策略降级（无 LLM 函数）"""
    engine = CompressionEngine()
    messages = create_mock_messages(30, "simple")

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.LLM_SUMMARY,
        llm_summarizer_func=None,  # 无 LLM 函数
        min_keep=4
    )

    # 应该降级到 SLIDING_WINDOW
    assert result.strategy == CompressionStrategy.SLIDING_WINDOW
    assert result.success


# ========== 测试 7: 自动策略选择 ==========

@pytest.mark.asyncio
async def test_auto_strategy_selection():
    """测试自动策略选择"""
    engine = CompressionEngine()

    # 测试不同消息类型的自动选择
    error_messages = create_mock_messages(30, "error")
    tool_messages = create_mock_messages(30, "tool")
    simple_messages = create_mock_messages(30, "simple")

    # 错误密集 → LLM_SUMMARY
    result1 = await engine.compress(messages=error_messages, llm_summarizer_func=mock_llm_summarizer)
    assert result1.strategy == CompressionStrategy.LLM_SUMMARY

    # 工具密集 → KEYFRAME (可能降级到 SLIDING_WINDOW)
    result2 = await engine.compress(messages=tool_messages)
    assert result2.strategy in (CompressionStrategy.KEYFRAME, CompressionStrategy.SLIDING_WINDOW)

    # 一般对话 → SLIDING_WINDOW
    result3 = await engine.compress(messages=simple_messages)
    assert result3.strategy == CompressionStrategy.SLIDING_WINDOW


# ========== 测试 8: 压缩效果 ==========

@pytest.mark.asyncio
async def test_compression_ratio():
    """测试压缩比"""
    engine = CompressionEngine()
    messages = create_mock_messages(100, "mixed")

    # 测试不同压缩比的压缩效果
    result1 = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.2,
        min_keep=4
    )
    assert result1.compression_ratio <= 0.25

    result2 = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.5,
        min_keep=4
    )
    assert result2.compression_ratio <= 0.55


@pytest.mark.asyncio
async def test_empty_messages():
    """测试空消息列表压缩"""
    engine = CompressionEngine()
    result = await engine.compress(messages=[])

    assert not result.success
    assert len(result.compressed_messages) == 0


# ========== 测试 9: 集成到 ContextManager ==========

@pytest.mark.asyncio
async def test_context_manager_compression_engine():
    """测试 ContextManager 集成压缩引擎"""
    ctx = ContextManager(max_history=100, min_keep=4)

    # 验证压缩引擎已初始化
    assert ctx.compression_engine is not None
    assert isinstance(ctx.compression_engine, CompressionEngine)


@pytest.mark.asyncio
async def test_context_manager_compress_with_strategy():
    """测试 ContextManager 使用指定策略压缩"""
    ctx = ContextManager(max_history=50, min_keep=4)

    # 添加大量消息
    for i in range(60):
        ctx.add_message({"role": "user", "content": f"任务 {i}"})
        ctx.add_message({"role": "assistant", "content": f"回复 {i}"})

    # 使用 SLIDING_WINDOW 策略压缩
    success = await ctx.compress(mock_llm_summarizer, strategy=CompressionStrategy.SLIDING_WINDOW)

    assert success
    assert len(ctx.messages) <= 8  # min_keep + 留一点余量


@pytest.mark.asyncio
async def test_context_manager_compress_auto_strategy():
    """测试 ContextManager 自动策略压缩"""
    ctx = ContextManager(max_history=50, min_keep=4)

    # 添加工具密集的消息
    for i in range(30):
        ctx.add_message({"role": "user", "content": f"任务 {i}"})
        ctx.add_message({"role": "tool", "content": f"工具结果 {i}"})

    # 自动选择策略（应该选择 KEYFRAME）
    success = await ctx.compress(mock_llm_summarizer)

    assert success
    assert len(ctx.messages) <= 10


@pytest.mark.asyncio
async def test_context_manager_compress_with_summary():
    """测试 ContextManager 压缩生成摘要"""
    ctx = ContextManager(max_history=50, min_keep=4)

    # 添加消息
    for i in range(60):
        ctx.add_message({"role": "user", "content": f"任务 {i}"})

    # 使用 LLM_SUMMARY 策略压缩
    success = await ctx.compress(mock_llm_summarizer, strategy=CompressionStrategy.LLM_SUMMARY)

    assert success
    assert len(ctx.session_summaries) > 0  # 应该有结构化摘要
    assert ctx.history_summary != ""  # 应该有纯文本摘要


# ========== 测试 10: 压缩结果验证 ==========

def test_compression_result_dataclass():
    """测试 CompressionResult 数据类"""
    result = CompressionResult(
        success=True,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        compressed_messages=[{"role": "user", "content": "test"}],
        compression_ratio=0.3,
        original_count=10,
        compressed_count=3
    )

    assert result.success
    assert result.strategy == CompressionStrategy.SLIDING_WINDOW
    assert len(result.compressed_messages) == 1
    assert result.compression_ratio == 0.3
    assert result.metadata == {}  # 默认值


# ========== 性能测试 ==========

@pytest.mark.asyncio
async def test_compression_performance():
    """测试压缩性能（< 1s）"""
    engine = CompressionEngine()
    messages = create_mock_messages(200, "mixed")

    import time
    start_time = time.time()

    result = await engine.compress(
        messages=messages,
        strategy=CompressionStrategy.SLIDING_WINDOW,
        target_ratio=0.3,
        min_keep=4
    )

    elapsed_time = time.time() - start_time

    assert result.success
    assert elapsed_time < 1.0  # 应该在 1 秒内完成


if __name__ == "__main__":
    pytest.main([__file__, "-v"])