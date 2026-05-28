"""阶段4：统一检索层与 Hybrid Recall 测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager
from core.memory_models import ErrorRecord, FileChange, SessionSummary
from core.memory_retrieval import MemoryDocument
from tools import get_default_tools
from tools.memory_tool import MemoryErrorHistoryTool, MemoryFileHistoryTool


def _summary(session_id: str, goal: str, text: str, file_path: str, error_type: str = "") -> SessionSummary:
    errors = []
    if error_type:
        errors.append(ErrorRecord(
            error_type=error_type,
            error_message=f"{error_type}: import failed in {file_path}",
            timestamp=datetime.now().isoformat(),
            file_path=file_path,
            solution="补充缺失依赖并重新运行 pytest",
            resolved=True,
        ))
    return SessionSummary(
        session_id=session_id,
        timestamp=(datetime.now() - timedelta(days=1)).isoformat(),
        summary_text=text,
        task_goal=goal,
        task_status="completed",
        files_changed=[FileChange(path=file_path, action="modified", summary=text, importance=0.8)],
        errors_encountered=errors,
        key_decisions=["统一使用 MemoryRetriever.hybrid_recall"],
        importance=0.7,
    )


def test_hybrid_recall_unifies_memory_items_and_session_summaries(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    summary = _summary(
        "s1",
        "修复检索工具",
        "SearchMemoryTool 需要接入统一 Hybrid Recall 检索层",
        "tools/retrieval_tool.py",
    )
    manager.save_summary(summary)
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.ARCHITECTURE.value,
        title="Hybrid Recall 架构",
        content="MemoryRetriever 将 MemoryItem 与 SessionSummary 统一为 MemoryDocument 后加权排序。",
        files=["core/memory_retrieval.py"],
        concepts=["hybrid", "recall", "MemoryDocument"],
        importance=0.95,
    ))

    results = manager.hybrid_recall("Hybrid Recall MemoryRetriever", top_k=5)

    assert len(results) >= 2
    assert all(isinstance(result.item, MemoryItem) for result in results)
    assert results[0].item.title == "Hybrid Recall 架构"
    assert results[0].source in {"memory_item", "long_term_memory_item", "long_term_items"}
    assert any(result.item.kind == MemoryKind.SUMMARY.value for result in results)
    assert all(result.reason for result in results)


def test_memory_document_from_summary_preserves_file_error_metadata():
    summary = _summary(
        "s_error",
        "修复 pytest 导入错误",
        "pytest 运行失败，原因是 ModuleNotFoundError，需要调整依赖。",
        "tests/test_memory_phase4.py",
        "ModuleNotFoundError",
    )

    document = MemoryDocument.from_session_summary(summary, source_type="episodic")

    assert document.source_type == "episodic"
    assert document.files == ["tests/test_memory_phase4.py"]
    assert "ModuleNotFoundError" in document.error_types
    assert "pytest" in document.content
    assert document.to_memory_item().kind == MemoryKind.SUMMARY.value


def test_file_history_recall_weights_exact_file_match(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="engine.py 工具调用顺序修复",
        content="修复 AgentEngine 中工具调用消息顺序，避免 OpenAI strict tool message 报错。",
        files=["core/engine.py"],
        concepts=["tool_calls", "openai"],
        importance=0.9,
    ))
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="无关文件修复",
        content="修复 README 中的说明。",
        files=["README.md"],
        concepts=["docs"],
        importance=0.9,
    ))

    results = manager.retrieve_file_history("core/engine.py", top_k=3)

    assert results
    assert results[0].item.title == "engine.py 工具调用顺序修复"
    assert "文件匹配" in results[0].reason
    assert all("core/engine.py" in result.item.files for result in results)


def test_error_history_recall_filters_error_type_and_bug_kind(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="ModuleNotFoundError 修复",
        content="pytest 失败：ModuleNotFoundError: No module named core，解决方案是补充 PYTHONPATH。",
        files=["tests/test_memory_phase4.py"],
        concepts=["ModuleNotFoundError", "pytest"],
        importance=0.85,
    ))
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.WORKFLOW.value,
        title="普通测试流程",
        content="运行 pytest tests 验证全部测试。",
        concepts=["pytest"],
        importance=0.6,
    ))

    results = manager.retrieve_error_history("ModuleNotFoundError", top_k=3)

    assert results
    assert results[0].item.kind == MemoryKind.BUG.value
    assert "ModuleNotFoundError" in results[0].item.searchable_text()
    assert all("ModuleNotFoundError" in result.item.searchable_text() or result.item.kind == MemoryKind.BUG.value for result in results)


def test_file_and_error_history_tools_registered_and_format_output(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        kind=MemoryKind.BUG.value,
        title="phase4 文件历史",
        content="修改 tests/test_memory_phase4.py 前要注意 ModuleNotFoundError 历史。",
        files=["tests/test_memory_phase4.py"],
        concepts=["ModuleNotFoundError"],
        importance=0.9,
    ))

    tools = get_default_tools(memory_manager=manager)
    names = {tool.name for tool in tools}
    assert {"memory_file_history", "memory_error_history"}.issubset(names)

    file_tool = next(tool for tool in tools if tool.name == "memory_file_history")
    error_tool = next(tool for tool in tools if tool.name == "memory_error_history")
    assert isinstance(file_tool, MemoryFileHistoryTool)
    assert isinstance(error_tool, MemoryErrorHistoryTool)

    file_output = file_tool.run(path="tests/test_memory_phase4.py", top_k=2)
    error_output = error_tool.run(error="ModuleNotFoundError", top_k=2)

    assert "📁 文件历史" in file_output
    assert "phase4 文件历史" in file_output
    assert "🐞 错误历史" in error_output
    assert "ModuleNotFoundError" in error_output
