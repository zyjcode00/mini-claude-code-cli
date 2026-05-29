"""Phase 1：BM25MemoryIndex 标准 BM25 与集成测试。"""

from __future__ import annotations

from core.memory_index import BM25MemoryDocument, BM25MemoryIndex
from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager


def test_bm25_index_prioritizes_rare_error_terms():
    index = BM25MemoryIndex()
    index.add_documents([
        BM25MemoryDocument(
            doc_id="common_pytest",
            title="pytest 通用失败处理",
            content="pytest failed test error error error should inspect traceback",
            kind="bug",
        ),
        BM25MemoryDocument(
            doc_id="module_not_found",
            title="ModuleNotFoundError 修复",
            content="Traceback ModuleNotFoundError No module named core; run python -m pytest from project root",
            concepts=["ModuleNotFoundError", "PYTHONPATH"],
            kind="bug",
            error="ModuleNotFoundError",
        ),
        BM25MemoryDocument(
            doc_id="winerror5",
            title="WinError 5 index replace",
            content="PermissionError WinError 5 os.replace index.json denied on Windows",
            concepts=["PermissionError", "WinError 5"],
            files=["memory/long_term/index.json"],
            kind="bug",
            error="PermissionError WinError 5",
        ),
    ])

    assert index.search("ModuleNotFoundError No module named core", top_k=1)[0].doc_id == "module_not_found"
    assert index.search("WinError 5 index.json", top_k=1)[0].doc_id == "winerror5"


def test_bm25_index_preserves_file_paths_symbols_and_persists(tmp_path):
    index = BM25MemoryIndex()
    index.add_or_update(BM25MemoryDocument(
        doc_id="context_budget",
        title="ContextAssembler memory budget",
        content="memory_context is controlled by ContextBudget",
        files=["tests/test_context_assembly_budget.py", "core/context_assembler.py"],
        concepts=["ContextAssembler", "ContextBudget", "memory_context"],
        kind="fact",
    ))
    index.add_or_update(BM25MemoryDocument(
        doc_id="unrelated",
        title="其他测试",
        content="pytest workflow for unrelated files",
        files=["tests/test_other.py"],
        kind="workflow",
    ))

    results = index.search("tests/test_context_assembly_budget.py memory_context", top_k=2)
    assert results[0].doc_id == "context_budget"
    assert "tests/test_context_assembly_budget.py" in index.inverted_index
    assert "memory_context" in index.inverted_index

    path = tmp_path / "indexes" / "bm25.json"
    index.save(path)
    loaded = BM25MemoryIndex.load(path)

    assert loaded.avg_doc_length == index.avg_doc_length
    assert loaded.search("core/context_assembler.py ContextBudget", top_k=1)[0].doc_id == "context_budget"


def test_bm25_index_chinese_query_uses_meaningful_fragments_not_single_chars():
    index = BM25MemoryIndex()
    tokens = index.tokenize("用户偏好中文简洁报告")

    assert "用户" in tokens
    assert "偏好" in tokens
    assert "中文" in tokens
    assert "简洁" in tokens
    assert "用" not in tokens

    index.add_documents([
        BM25MemoryDocument(
            doc_id="preference",
            title="用户偏好中文且报告要简洁",
            content="用户希望最终答复使用中文，说明已改文件和测试结果，避免冗长无关展开。",
            concepts=["用户偏好", "中文", "简洁报告"],
            kind="preference",
        ),
        BM25MemoryDocument(
            doc_id="noise",
            title="中文分词噪音样例",
            content="这个文档只有一些常见中文句子，没有报告偏好要求。",
            kind="fact",
        ),
    ])

    assert index.search("用户偏好 最终答复 中文 简洁", top_k=1)[0].doc_id == "preference"


def test_memory_manager_hybrid_recall_uses_bm25_index_reason_and_ranking(tmp_path):
    manager = MemoryManager(long_term_storage_dir=str(tmp_path / "long_term"))
    manager.save_memory_item(MemoryItem(
        id="mem_general_pytest_noise",
        kind=MemoryKind.WORKFLOW.value,
        title="pytest 通用工作流",
        content="pytest failed error traceback test should be inspected carefully",
        concepts=["pytest", "traceback"],
        importance=0.5,
        confidence=0.8,
    ))
    manager.save_memory_item(MemoryItem(
        id="mem_bug_module_not_found_core_pytest",
        kind=MemoryKind.BUG.value,
        title="pytest ModuleNotFoundError 需要检查 PYTHONPATH",
        content="pytest 失败：ModuleNotFoundError: No module named core。应从项目根目录运行 python -m pytest。",
        concepts=["ModuleNotFoundError", "pytest", "PYTHONPATH"],
        files=["tests/test_memory_phase4.py", "core/memory_retrieval.py"],
        importance=0.7,
        confidence=0.9,
        metadata={"error_type": "ModuleNotFoundError"},
    ))

    results = manager.hybrid_recall("Traceback ModuleNotFoundError No module named core pytest", top_k=2)

    assert results[0].item.id == "mem_bug_module_not_found_core_pytest"
    assert "BM25相关" in results[0].reason
    assert "modulenotfounderror" in results[0].reason.lower()
