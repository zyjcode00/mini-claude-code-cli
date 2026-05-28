"""长期记忆工具：memory_save / memory_recall。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.memory_items import MemoryItem, MemoryKind
from core.memory_manager import MemoryManager
from tools.base import BaseTool


class MemorySaveArgs(BaseModel):
    kind: str = Field(default=MemoryKind.FACT.value, description="记忆类型，如 fact/architecture/preference/bug/workflow/decision/task")
    title: str = Field(..., description="记忆标题，简短描述这条长期记忆")
    content: str = Field(..., description="记忆正文，记录可复用的事实、经验、决策或工作流")
    files: List[str] = Field(default_factory=list, description="相关文件路径列表")
    concepts: List[str] = Field(default_factory=list, description="相关概念/关键词列表")
    importance: float = Field(default=0.6, description="重要性 0~1")
    confidence: float = Field(default=0.8, description="置信度 0~1")
    project: str = Field(default="", description="项目名称，可选")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据，可选")


class MemoryRecallArgs(BaseModel):
    query: str = Field(..., description="召回查询文本")
    top_k: int = Field(default=5, description="最多返回几条记忆")
    include_summaries: bool = Field(default=True, description="是否包含旧 SessionSummary 兼容召回")


class MemorySaveTool(BaseTool):
    name = "memory_save"
    description = "保存一条可长期复用的 MemoryItem，例如架构决策、Bug 修复经验、用户偏好或工作流。"
    args_schema = MemorySaveArgs

    def __init__(self, memory_manager: Optional[MemoryManager] = None, long_term_storage_dir: str = "memory/long_term"):
        self.memory_manager = memory_manager or MemoryManager(long_term_storage_dir=long_term_storage_dir)

    def create_item(
        self,
        kind: str = MemoryKind.FACT.value,
        title: str = "",
        content: str = "",
        files: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        importance: float = 0.6,
        confidence: float = 0.8,
        project: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        return MemoryItem(
            kind=kind,
            title=title,
            content=content,
            project=project,
            files=files or [],
            concepts=concepts or [],
            importance=importance,
            confidence=confidence,
            metadata=metadata or {},
        )

    def run(
        self,
        kind: str = MemoryKind.FACT.value,
        title: str = "",
        content: str = "",
        files: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        importance: float = 0.6,
        confidence: float = 0.8,
        project: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        try:
            if not title or not content:
                return "❌ 保存失败: title 和 content 不能为空"

            item = self.create_item(
                kind=kind,
                title=title,
                content=content,
                files=files,
                concepts=concepts,
                importance=importance,
                confidence=confidence,
                project=project,
                metadata=metadata,
            )
            path = self.memory_manager.save_memory_item(item)
            return (
                "✅ 已保存长期记忆\n"
                f"ID: {item.id}\n"
                f"类型: {item.kind}\n"
                f"标题: {item.title}\n"
                f"文件: {', '.join(item.files) if item.files else '无'}\n"
                f"概念: {', '.join(item.concepts) if item.concepts else '无'}\n"
                f"路径: {path}"
            )
        except Exception as e:
            return f"❌ 保存长期记忆失败: {e}"


class MemoryRecallTool(BaseTool):
    name = "memory_recall"
    description = "从长期记忆中召回与查询相关的 MemoryItem，返回标题、内容、文件、概念和来源。"
    args_schema = MemoryRecallArgs

    def __init__(self, memory_manager: Optional[MemoryManager] = None, long_term_storage_dir: str = "memory/long_term"):
        self.memory_manager = memory_manager or MemoryManager(long_term_storage_dir=long_term_storage_dir)

    def run(self, query: str, top_k: int = 5, include_summaries: bool = True, **kwargs) -> str:
        try:
            results = self.memory_manager.recall(query=query, top_k=top_k, include_summaries=include_summaries)
            if not results:
                return f"未找到与 '{query}' 相关的长期记忆"

            lines = [f"🧠 召回 {len(results)} 条长期记忆："]
            for index, result in enumerate(results, 1):
                item = result.item
                files = ", ".join(item.files) if item.files else "无"
                concepts = ", ".join(item.concepts) if item.concepts else "无"
                content = item.content[:500] + "..." if len(item.content) > 500 else item.content
                lines.extend([
                    f"\n{index}. [{item.kind}] {item.title}",
                    f"   ID: {item.id}",
                    f"   相关度: {result.score:.3f} | 来源: {result.source} | 原因: {result.reason}",
                    f"   内容: {content}",
                    f"   文件: {files}",
                    f"   概念: {concepts}",
                ])
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 召回长期记忆失败: {e}"


__all__ = ["MemorySaveTool", "MemoryRecallTool", "MemorySaveArgs", "MemoryRecallArgs"]
