"""Phase 0：长期记忆召回质量基线测试。

这些测试不追求当前检索算法已经完美，而是用固定 fixtures 度量
MemoryManager.hybrid_recall 在 architecture / workflow / preference /
file history / error history 等典型场景下的 top-k 命中表现。

后续 Phase 1+ 调整 BM25、RRF、向量召回时，应优先保证这些基线不退化，
再逐步收紧 top1/top3 断言。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager


@dataclass(frozen=True)
class RecallEvalCase:
    name: str
    query: str
    expected_id: str
    top_k: int = 5
    recall: str = "hybrid"
    description: str = ""


def _seed_recall_fixture(manager: MemoryManager) -> None:
    """写入一组覆盖 Phase 0 目标场景的固定评测记忆。"""
    items = [
        MemoryItem(
            id="mem_arch_memory_manager_unified_entry",
            kind=MemoryKind.ARCHITECTURE.value,
            title="MemoryManager 是长期记忆统一入口",
            content=(
                "ContextManager、AgentEngine 和 memory 工具必须通过 "
                "MemoryManager.hybrid_recall 访问长期记忆，不再维护独立 BM25 索引。"
            ),
            concepts=["MemoryManager", "hybrid_recall", "architecture"],
            files=["core/memory_manager.py", "core/memory_retrieval.py"],
            importance=0.95,
            confidence=0.95,
        ),
        MemoryItem(
            id="mem_workflow_core_pytest_required",
            kind=MemoryKind.WORKFLOW.value,
            title="修改 core 后必须运行 pytest",
            content=(
                "任何对 core/ 或 tools/ 的逻辑修改都必须新增或更新 tests/test_*.py，"
                "并运行 python -m pytest 对应测试，失败时先读 traceback 再修复。"
            ),
            concepts=["pytest", "TDD", "core", "workflow"],
            files=["CLAUDE.md", "tests/test_memory_phase5.py"],
            importance=0.9,
            confidence=0.9,
        ),
        MemoryItem(
            id="mem_preference_chinese_concise_report",
            kind=MemoryKind.PREFERENCE.value,
            title="用户偏好中文且报告要简洁",
            content="用户希望最终答复使用中文，说明已改文件和测试结果，避免冗长无关展开。",
            concepts=["用户偏好", "中文", "简洁报告"],
            files=["CLAUDE.md"],
            importance=0.8,
            confidence=0.9,
        ),
        MemoryItem(
            id="mem_bug_engine_openai_tool_pairing",
            kind=MemoryKind.BUG.value,
            title="OpenAI tool_calls 必须紧邻 tool response",
            content=(
                "修复 AgentEngine 中 OpenAI strict tool message 顺序问题：assistant tool_calls "
                "后必须紧跟对应 tool response，不能在中间插入 memory hint。"
            ),
            concepts=["OpenAI", "tool_calls", "tool response", "message ordering"],
            files=["core/engine.py", "tests/test_openai_tool_pairing.py"],
            importance=0.92,
            confidence=0.9,
            metadata={"error_type": "BadRequestError"},
        ),
        MemoryItem(
            id="mem_bug_winerror5_index_replace",
            kind=MemoryKind.BUG.value,
            title="WinError 5 写入 index.json 时需要保留 pending snapshot",
            content=(
                "Windows 下 os.replace 写 index.json 可能触发 PermissionError: [WinError 5] 拒绝访问。"
                "LongTermMemory._save_index 应清理临时文件并保留 _pending_index_snapshot。"
            ),
            concepts=["PermissionError", "WinError 5", "index.json", "os.replace"],
            files=["core/memory_layers.py", "memory/long_term/index.json"],
            importance=0.95,
            confidence=0.9,
            metadata={"error_type": "PermissionError"},
        ),
        MemoryItem(
            id="mem_bug_module_not_found_core_pytest",
            kind=MemoryKind.BUG.value,
            title="pytest ModuleNotFoundError 需要检查 PYTHONPATH",
            content=(
                "pytest 失败：ModuleNotFoundError: No module named core。"
                "通常是测试启动目录或 PYTHONPATH 不正确，应从项目根目录运行 python -m pytest。"
            ),
            concepts=["ModuleNotFoundError", "pytest", "PYTHONPATH"],
            files=["tests/test_memory_phase4.py", "core/memory_retrieval.py"],
            importance=0.88,
            confidence=0.9,
            metadata={"error_type": "ModuleNotFoundError"},
        ),
        MemoryItem(
            id="mem_fact_context_assembler_budget",
            kind=MemoryKind.FACT.value,
            title="ContextAssembler 有 memory token budget",
            content=(
                "ContextAssembler 将 plan、memory、compressed_state 注入 system prompt，"
                "并用 ContextBudget 控制 memory_context 长度。"
            ),
            concepts=["ContextAssembler", "ContextBudget", "memory_context"],
            files=["core/context_assembler.py", "tests/test_context_assembly_budget.py"],
            importance=0.75,
            confidence=0.85,
        ),
    ]
    for item in items:
        manager.save_memory_item(item)


def _ids(results: Iterable) -> list[str]:
    return [result.item.id for result in results]


def _assert_expected_hit(results: Iterable, expected_id: str) -> None:
    result_ids = _ids(results)
    assert expected_id in result_ids, f"expected {expected_id!r} in ranked ids {result_ids!r}"


RECALL_EVAL_CASES = [
    RecallEvalCase(
        name="architecture_query_hits_memory_manager",
        query="当前长期记忆 architecture 统一入口 MemoryManager hybrid_recall",
        expected_id="mem_arch_memory_manager_unified_entry",
        top_k=3,
    ),
    RecallEvalCase(
        name="workflow_query_hits_pytest_rule",
        query="修改 core 逻辑后应该跑什么测试 workflow pytest",
        expected_id="mem_workflow_core_pytest_required",
        top_k=3,
    ),
    RecallEvalCase(
        name="chinese_preference_query_hits_user_preference",
        query="用户偏好 最终答复 中文 简洁",
        expected_id="mem_preference_chinese_concise_report",
        top_k=5,
    ),
    RecallEvalCase(
        name="file_path_query_hits_context_budget_fact",
        query="tests/test_context_assembly_budget.py memory_context budget",
        expected_id="mem_fact_context_assembler_budget",
        top_k=5,
    ),
    RecallEvalCase(
        name="traceback_query_hits_module_not_found_bug",
        query="Traceback ModuleNotFoundError No module named core pytest",
        expected_id="mem_bug_module_not_found_core_pytest",
        top_k=3,
    ),
]


def test_phase0_recall_quality_topk_baseline(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    _seed_recall_fixture(manager)

    for case in RECALL_EVAL_CASES:
        results = manager.hybrid_recall(case.query, top_k=case.top_k)
        _assert_expected_hit(results, case.expected_id)


def test_phase0_recall_quality_reports_top1_top3_top5_metrics(tmp_path):
    """记录当前基线命中率，后续优化可在这里收紧阈值。"""
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    _seed_recall_fixture(manager)

    metrics = {"top1": 0, "top3": 0, "top5": 0}
    for case in RECALL_EVAL_CASES:
        results = manager.hybrid_recall(case.query, top_k=5)
        result_ids = _ids(results)
        if result_ids[:1] and case.expected_id in result_ids[:1]:
            metrics["top1"] += 1
        if case.expected_id in result_ids[:3]:
            metrics["top3"] += 1
        if case.expected_id in result_ids[:5]:
            metrics["top5"] += 1

    total = len(RECALL_EVAL_CASES)
    # Phase 0 先建立可运行质量闸门：top5 必须全命中，top3 至少覆盖主要场景。
    assert metrics["top5"] == total, metrics
    assert metrics["top3"] >= total - 1, metrics


def test_phase0_file_history_eval_uses_memory_manager_hybrid_recall(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    _seed_recall_fixture(manager)

    results = manager.retrieve_file_history("core/engine.py", top_k=3)

    assert results
    assert results[0].item.id == "mem_bug_engine_openai_tool_pairing"
    assert all("core/engine.py" in result.item.files for result in results)
    assert "文件匹配" in results[0].reason


def test_phase0_error_history_eval_handles_traceback_query(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    _seed_recall_fixture(manager)

    traceback = (
        "Traceback (most recent call last):\n"
        "  File tests/test_memory_phase4.py, line 1, in <module>\n"
        "ModuleNotFoundError: No module named core"
    )
    results = manager.retrieve_error_history(traceback, top_k=5)

    _assert_expected_hit(results, "mem_bug_module_not_found_core_pytest")
    assert results[0].item.kind == MemoryKind.BUG.value


def test_phase0_recall_quality_main_path_only_uses_memory_manager(tmp_path, monkeypatch):
    """Phase 0 清理约束：质量评测主线只允许走 MemoryManager.hybrid_recall。"""
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    _seed_recall_fixture(manager)
    calls = []
    original: Callable = manager.hybrid_recall

    def spy_hybrid_recall(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "hybrid_recall", spy_hybrid_recall)

    results = manager.recall("MemoryManager hybrid_recall architecture", top_k=3)

    assert calls
    assert results[0].item.id == "mem_arch_memory_manager_unified_entry"
