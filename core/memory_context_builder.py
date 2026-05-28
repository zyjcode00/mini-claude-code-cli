"""Prompt 记忆上下文构建器。

阶段5引入：把 Hybrid Recall 结果按任务类型、置信度、去重和 token budget
格式化为可控的 prompt 注入片段，避免直接拼接全部记忆造成上下文污染。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set

from core.memory_items import MemoryKind, MemoryRecallResult


class MemoryContextBuilder:
    """根据 token budget 构造可注入系统提示词的长期记忆上下文。"""

    TASK_KIND_PRIORITY = {
        "code_edit": [MemoryKind.BUG.value, MemoryKind.WORKFLOW.value, MemoryKind.DECISION.value, MemoryKind.ARCHITECTURE.value],
        "architecture": [MemoryKind.ARCHITECTURE.value, MemoryKind.DECISION.value, MemoryKind.FACT.value, MemoryKind.WORKFLOW.value],
        "test_failure": [MemoryKind.BUG.value, MemoryKind.PROCEDURAL.value, MemoryKind.WORKFLOW.value, MemoryKind.FACT.value],
        "documentation": [MemoryKind.PREFERENCE.value, MemoryKind.ARCHITECTURE.value, MemoryKind.DECISION.value, MemoryKind.TASK.value],
        "git": [MemoryKind.WORKFLOW.value, MemoryKind.PREFERENCE.value, MemoryKind.BUG.value, MemoryKind.PROCEDURAL.value],
        "general": [MemoryKind.BUG.value, MemoryKind.ARCHITECTURE.value, MemoryKind.WORKFLOW.value, MemoryKind.DECISION.value, MemoryKind.PREFERENCE.value],
    }

    def __init__(
        self,
        default_token_budget: int = 1500,
        max_items: int = 8,
        per_item_char_limit: int = 200,
        min_confidence: float = 0.2,
    ):
        self.default_token_budget = default_token_budget
        self.max_items = max_items
        self.per_item_char_limit = per_item_char_limit
        self.min_confidence = min_confidence

    def build(
        self,
        query: str,
        memories: Sequence[MemoryRecallResult],
        token_budget: Optional[int] = None,
        task_type: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> str:
        """构造受预算约束的记忆上下文。"""
        if not memories:
            return ""

        budget = max(0, token_budget or self.default_token_budget)
        if budget <= 0:
            return ""

        inferred_task_type = task_type or self.infer_task_type(query)
        char_budget = budget * 4  # 粗略估算：1 token ~= 4 chars
        selected = self._select_memories(memories, inferred_task_type, max_items or self.max_items)

        lines: List[str] = ["\n[相关长期记忆]", "### 相关长期记忆", f"任务类型: {inferred_task_type}"]
        used_chars = sum(len(line) + 1 for line in lines)

        for index, result in enumerate(selected, 1):
            item = result.item
            files = ", ".join(item.files[:4]) if item.files else "无"
            sources = ", ".join(item.source_session_ids[:3]) if item.source_session_ids else result.source
            content = self._truncate_clean(item.content, self.per_item_char_limit)
            block_lines = [
                f"{index}. [{item.kind}] {item.title}",
                f"   来源: {files}; {sources}",
                f"   相关度: {result.score:.2f}; 原因: {result.reason}",
                f"   内容: {content}",
            ]
            block = "\n".join(block_lines)
            block_cost = len(block) + 1
            if used_chars + block_cost > char_budget:
                remaining = char_budget - used_chars - 80
                if remaining > 60:
                    short_content = self._truncate_clean(item.content, min(remaining, self.per_item_char_limit))
                    block = "\n".join(block_lines[:-1] + [f"   内容: {short_content}"])
                    lines.append(block)
                break
            lines.append(block)
            used_chars += block_cost

        if len(lines) <= 2:
            return ""
        return "\n".join(lines) + "\n"

    def _select_memories(
        self,
        memories: Sequence[MemoryRecallResult],
        task_type: str,
        max_items: int,
    ) -> List[MemoryRecallResult]:
        priority = self.TASK_KIND_PRIORITY.get(task_type, self.TASK_KIND_PRIORITY["general"])
        priority_map = {kind: idx for idx, kind in enumerate(priority)}
        seen: Set[str] = set()
        unique: List[MemoryRecallResult] = []

        for result in memories:
            item = result.item
            if item.confidence < self.min_confidence:
                continue
            signature = self._signature(item.title, item.content)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(result)

        unique.sort(
            key=lambda result: (
                priority_map.get(result.item.kind, len(priority_map) + 1),
                -float(result.score or 0),
                -float(result.item.importance or 0),
            )
        )
        return unique[:max(1, max_items)]

    @staticmethod
    def infer_task_type(query: str) -> str:
        text = (query or "").lower()
        if any(word in text for word in ["pytest", "traceback", "assertionerror", "测试失败", "报错", "exception", "error"]):
            return "test_failure"
        if any(word in text for word in ["架构", "architecture", "设计", "方案", "roadmap"]):
            return "architecture"
        if any(word in text for word in ["文档", "document", "docs", "说明", "readme"]):
            return "documentation"
        if any(word in text for word in ["git", "commit", "branch", "rollback", "提交", "回滚"]):
            return "git"
        if any(word in text for word in ["修改", "实现", "修复", "edit", "write", ".py", "代码"]):
            return "code_edit"
        return "general"

    @staticmethod
    def _truncate_clean(text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @classmethod
    def _signature(cls, title: str, content: str) -> str:
        normalized = cls._truncate_clean(f"{title} {content}", 160).lower()
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)[:120]
