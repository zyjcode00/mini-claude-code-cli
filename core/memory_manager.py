"""
统一记忆管理器

MemoryManager 是阶段 1 重构新增的记忆编排入口，负责统一持有并协调：
- WorkingMemory：当前工作记忆
- EpisodicMemory：会话内结构化摘要
- LongTermMemory：跨会话持久化摘要
- MemoryRetriever：多层检索器
- CompressionEngine：上下文压缩引擎

底层 memory_layers 仍保持轻量数据结构职责；ContextManager/AgentEngine 应逐步只通过
MemoryManager 访问记忆能力，减少分散编排逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.compression_engine import CompressionEngine, CompressionStrategy, CompressionResult
from core.memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory
from core.memory_items import MemoryItem, MemoryKind, MemoryRecallResult, RawObservation, MemoryStatus
from core.memory_models import SessionSummary
from core.memory_retrieval import MemoryRetriever
from core.memory_context_builder import MemoryContextBuilder


class MemoryManager:
    """
    统一记忆编排入口。

    该类不替代底层三层记忆的数据结构，而是提供稳定的高层 API：
    - add_message / save_summary
    - compress_messages
    - search / retrieve_by_file_path / retrieve_by_error_type
    - export_memories / import_memories
    - get_statistics / clear
    """

    def __init__(
        self,
        working_max_size: int = 20,
        episodic_max_size: int = 50,
        long_term_storage_dir: str = "memory/long_term",
        plan_manager: Any = None,
        enabled: bool = True,
        working_memory: Optional[WorkingMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        compression_engine: Optional[CompressionEngine] = None,
    ):
        self.enabled = enabled
        self.working_memory = working_memory or WorkingMemory(max_size=working_max_size)
        self.episodic_memory = episodic_memory or EpisodicMemory(max_size=episodic_max_size)
        self.long_term_memory = long_term_memory or LongTermMemory(storage_dir=long_term_storage_dir)
        self.compression_engine = compression_engine or CompressionEngine(
            plan_manager=plan_manager,
            memory_manager=self,
        )
        if compression_engine is not None and getattr(compression_engine, "memory_manager", None) is None:
            compression_engine.memory_manager = self
        self.retriever = MemoryRetriever(
            self.working_memory,
            self.episodic_memory,
            self.long_term_memory,
        )
        self.context_builder = MemoryContextBuilder()

    def add_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """添加消息到工作记忆，返回被 FIFO 淘汰的消息。"""
        if not self.enabled:
            return None
        return self.working_memory.add(message)

    def reset_working_memory(self, messages: List[Dict[str, Any]]) -> None:
        """用给定消息重建工作记忆。"""
        if not self.enabled:
            return
        self.working_memory.clear()
        for msg in messages:
            self.working_memory.add(msg)

    def save_summary(self, summary: SessionSummary) -> Optional[SessionSummary]:
        """
        保存摘要到情景记忆。

        如果情景记忆因容量限制淘汰旧摘要，则自动归档淘汰摘要到长期记忆。
        返回被归档的摘要；没有淘汰则返回 None。
        """
        if not self.enabled:
            return None

        evicted_summary = self.episodic_memory.add(summary)
        if evicted_summary:
            self.long_term_memory.store(evicted_summary)
        return evicted_summary

    def save_memory_item(self, item: MemoryItem) -> str:
        """保存标准 MemoryItem 到长期记忆。"""
        if not self.enabled:
            return ""
        return self.long_term_memory.store_item(item)

    def save_observation(
        self,
        observation: RawObservation,
        promote: bool = False,
        kind: str = MemoryKind.FACT.value,
        title: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 0.8,
    ) -> Optional[MemoryItem]:
        """
        记录 RawObservation，并可选择立即提升为 MemoryItem。

        第一版暂不单独落盘 Observation Store，而是把 observation 作为来源元数据写入
        MemoryItem，保证写入管线和来源追溯先打通。
        """
        if not self.enabled or not promote:
            return None

        if observation.event_type == "session_end" and observation.assistant_message:
            content = f"用户任务: {observation.user_prompt or ''}\n完成结果: {observation.assistant_message}"
        else:
            content = (
                observation.user_prompt
                or observation.assistant_message
                or observation.tool_output
                or observation.error
                or ""
            )
        item = MemoryItem(
            kind=kind,
            title=title or f"{observation.event_type}: {observation.tool_name or observation.session_id or observation.id}",
            content=content,
            project=observation.project,
            concepts=[observation.event_type, observation.tool_name] if observation.tool_name else [observation.event_type],
            files=observation.files,
            source_observation_ids=[observation.id],
            source_session_ids=[observation.session_id] if observation.session_id else [],
            importance=importance,
            confidence=confidence,
            metadata={"raw_observation": observation.to_dict()},
        )
        self.save_memory_item(item)
        return item

    def recall(self, query: str, top_k: int = 5, include_summaries: bool = True) -> List[MemoryRecallResult]:
        """
        统一召回长期记忆。

        阶段4 起默认委托 MemoryRetriever.hybrid_recall，统一整合 MemoryItem 与
        SessionSummary，并使用 BM25/关键词/元数据加权排序。
        """
        if not self.enabled:
            return []

        return self.hybrid_recall(query=query, top_k=top_k, include_summaries=include_summaries)

    def hybrid_recall(
        self,
        query: str = "",
        top_k: int = 5,
        file_path: Optional[str] = None,
        error_type: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        include_summaries: bool = True,
        include_items: bool = True,
    ) -> List[MemoryRecallResult]:
        """统一 Hybrid Recall 入口，返回标准 MemoryRecallResult。"""
        if not self.enabled:
            return []
        return self.retriever.hybrid_recall(
            query=query,
            top_k=top_k,
            file_path=file_path,
            error_type=error_type,
            kinds=kinds,
            include_summaries=include_summaries,
            include_items=include_items,
        )

    def retrieve_file_history(self, path: str, top_k: int = 5) -> List[MemoryRecallResult]:
        """按文件路径召回 MemoryItem 与 SessionSummary 历史。"""
        if not self.enabled:
            return []
        return self.hybrid_recall(query=path, top_k=top_k, file_path=path)

    def retrieve_error_history(self, error: str, top_k: int = 5) -> List[MemoryRecallResult]:
        """按错误类型/错误文本召回历史 Bug、测试失败与相关摘要。"""
        if not self.enabled:
            return []
        return self.hybrid_recall(query=error, top_k=top_k, error_type=error)

    def build_prompt_memory_context(
        self,
        query: str,
        token_budget: int = 1500,
        task_type: Optional[str] = None,
        top_k: int = 8,
        file_path: Optional[str] = None,
        error_type: Optional[str] = None,
        kinds: Optional[List[str]] = None,
    ) -> str:
        """召回并构建受 token budget 限制的 prompt 记忆上下文。"""
        if not self.enabled or not query:
            return ""
        results = self.hybrid_recall(
            query=query,
            top_k=max(top_k * 2, top_k),
            file_path=file_path,
            error_type=error_type,
            kinds=kinds,
            include_summaries=True,
            include_items=True,
        )
        return self.context_builder.build(
            query=query,
            memories=results,
            token_budget=token_budget,
            task_type=task_type,
            max_items=top_k,
        )

    def _summary_to_memory_item(self, summary: SessionSummary) -> MemoryItem:
        """把旧 SessionSummary 包装为 MemoryItem 召回结果。"""
        return MemoryItem(
            id=f"summary_{summary.session_id}",
            kind=MemoryKind.SUMMARY.value,
            title=summary.task_goal,
            content=summary.summary_text,
            created_at=summary.timestamp,
            updated_at=summary.timestamp,
            concepts=summary.get_keywords(),
            files=summary.get_file_paths(),
            source_session_ids=[summary.session_id],
            importance=summary.importance,
            confidence=0.7,
            metadata={"session_summary": summary.to_dict()},
        )

    async def compress_messages(
        self,
        messages: List[Dict[str, Any]],
        llm_summarizer_func,
        strategy: CompressionStrategy = None,
        min_keep: int = 4,
        existing_summary: str = "",
    ) -> CompressionResult:
        """使用压缩引擎压缩消息。"""
        target_ratio = min_keep / len(messages) if messages else 0.3
        return await self.compression_engine.compress(
            messages=messages,
            strategy=strategy,
            llm_summarizer_func=llm_summarizer_func,
            target_ratio=target_ratio,
            min_keep=min_keep,
            existing_summary=existing_summary,
        )

    def search(self, query: str, top_k: int = 5) -> List[SessionSummary]:
        """检索情景记忆和长期记忆，返回去重后的摘要列表。"""
        if not self.enabled:
            return []

        ranked_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            include_working=False,
            include_episodic=True,
            include_long_term=True,
        )
        return [summary for summary, _score, _source in ranked_results]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_working: bool = True,
        include_episodic: bool = True,
        include_long_term: bool = True,
    ) -> List[Tuple[SessionSummary, float, str]]:
        """返回带分数和来源层的多层检索结果。"""
        if not self.enabled:
            return []
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            include_working=include_working,
            include_episodic=include_episodic,
            include_long_term=include_long_term,
        )

    def search_long_term(self, query: str, top_k: int = 5) -> List[SessionSummary]:
        """仅检索长期记忆。"""
        if not self.enabled:
            return []
        return self.long_term_memory.search(query=query, top_k=top_k)

    def retrieve_by_file_path(self, file_path: str, top_k: int = 5) -> List[Tuple[SessionSummary, float, str]]:
        """按文件路径检索相关摘要。"""
        if not self.enabled:
            return []
        return self.retriever.retrieve_by_file_path(file_path=file_path, top_k=top_k)

    def retrieve_by_error_type(self, error_type: str, top_k: int = 5) -> List[Tuple[SessionSummary, float, str]]:
        """按错误类型检索相关摘要。"""
        if not self.enabled:
            return []
        return self.retriever.retrieve_by_error_type(error_type=error_type, top_k=top_k)

    def get_recent_summaries(self, n: int = 5) -> List[SessionSummary]:
        """获取最近情景摘要。"""
        if not self.enabled:
            return []
        return self.episodic_memory.get_recent(n)

    def get_statistics(self) -> Dict[str, Any]:
        """获取三层记忆统计信息。"""
        if not self.enabled:
            return {
                "enabled": False,
                "working_memory": {"size": 0, "max_size": self.working_memory.max_size, "usage_rate": 0},
                "episodic_memory": {"size": 0, "max_size": self.episodic_memory.max_size, "usage_rate": 0},
                "long_term_memory": {
                    "count": 0,
                    "summary_count": 0,
                    "item_count": 0,
                    "storage_dir": str(self.long_term_memory.storage_dir),
                },
            }

        items = self.long_term_memory.get_all_items()
        item_status_counts = {
            MemoryStatus.ACTIVE.value: 0,
            MemoryStatus.SUPERSEDED.value: 0,
            MemoryStatus.ARCHIVED.value: 0,
        }
        latest_count = 0
        for item in items:
            item_status_counts[item.status] = item_status_counts.get(item.status, 0) + 1
            if item.is_latest:
                latest_count += 1

        working_max = max(self.working_memory.max_size, 1)
        episodic_max = max(self.episodic_memory.max_size, 1)
        return {
            "enabled": True,
            "working_memory": {
                "size": len(self.working_memory),
                "max_size": self.working_memory.max_size,
                "usage_rate": len(self.working_memory) / working_max,
            },
            "episodic_memory": {
                "size": len(self.episodic_memory),
                "max_size": self.episodic_memory.max_size,
                "usage_rate": len(self.episodic_memory) / episodic_max,
            },
            "long_term_memory": {
                "count": len(self.long_term_memory),
                "summary_count": len(self.long_term_memory.index),
                "item_count": len(self.long_term_memory.item_index),
                "item_status_counts": item_status_counts,
                "active_count": item_status_counts.get(MemoryStatus.ACTIVE.value, 0),
                "superseded_count": item_status_counts.get(MemoryStatus.SUPERSEDED.value, 0),
                "archived_count": item_status_counts.get(MemoryStatus.ARCHIVED.value, 0),
                "latest_count": latest_count,
                "storage_dir": str(self.long_term_memory.storage_dir),
            },
        }

    def export_memories(
        self,
        session_summaries: Optional[List[SessionSummary]] = None,
        history_summary: str = "",
        include_memory_items: bool = False,
    ) -> Dict[str, Any]:
        """
        导出可写入 session 文件的会话局部记忆数据。

        注意：长期 MemoryItem 已持久化在 memory/long_term 目录，并由 index.json 管理。
        默认不再把所有长期 MemoryItem 嵌入每个 session JSON，否则会导致：
        1. session 文件随长期记忆规模线性膨胀；
        2. 启动恢复 session 时重复 store_item 写回几百/几千条长期记忆；
        3. Windows 下反复写 index.json，显著拖慢启动。
        """
        session_summaries = session_summaries or []
        memory_items = []
        if include_memory_items and self.enabled:
            memory_items = [item.to_dict() for item in self.long_term_memory.get_all_items()]

        if not self.enabled:
            return {
                "working_memory": [],
                "episodic_memory": [],
                "memory_items": memory_items,
                "session_summaries": [s.to_dict() for s in session_summaries],
                "history_summary": history_summary,
            }

        return {
            "working_memory": self.working_memory.get_all(),
            "episodic_memory": [s.to_dict() for s in self.episodic_memory.get_all()],
            "memory_items": memory_items,
            "session_summaries": [s.to_dict() for s in session_summaries],
            "history_summary": history_summary,
        }

    def import_memories(self, data: Dict[str, Any], import_memory_items: bool = False) -> Dict[str, Any]:
        """
        导入 session 文件中的会话局部记忆数据。

        返回 Phase 1 兼容字段：history_summary 与 session_summaries。
        默认跳过旧 session 内嵌的 memory_items，因为这些长期记忆已经在
        memory/long_term/index.json 中加载；再次导入会造成启动期大量重复写盘。
        """
        session_summaries: List[SessionSummary] = []
        for s_dict in data.get("session_summaries", []):
            try:
                session_summaries.append(SessionSummary.from_dict(s_dict))
            except Exception as e:
                print(f"⚠️ 恢复会话摘要失败: {e}")

        if self.enabled:
            self.working_memory.clear()
            for msg in data.get("working_memory", []):
                self.working_memory.add(msg)

            self.episodic_memory.clear()
            for s_dict in data.get("episodic_memory", []):
                try:
                    self.episodic_memory.add(SessionSummary.from_dict(s_dict))
                except Exception as e:
                    print(f"⚠️ 恢复情景记忆失败: {e}")

            if import_memory_items:
                for item_dict in data.get("memory_items", []):
                    try:
                        self.long_term_memory.store_item(MemoryItem.from_dict(item_dict))
                    except Exception as e:
                        print(f"⚠️ 恢复 MemoryItem 失败: {e}")

        return {
            "history_summary": data.get("history_summary", ""),
            "session_summaries": session_summaries,
        }

    def export_to_files(self, output_dir: str = "memory/export") -> Dict[str, str]:
        """导出情景记忆和统计信息到独立文件，返回文件路径。"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        episodic_file = output_path / "episodic_memory.json"
        memory_items_file = output_path / "memory_items.json"
        stats_file = output_path / "memory_statistics.json"

        with open(episodic_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.episodic_memory.get_all()], f, ensure_ascii=False, indent=2)

        with open(memory_items_file, "w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in self.long_term_memory.get_all_items()], f, ensure_ascii=False, indent=2)

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.get_statistics(), f, ensure_ascii=False, indent=2)

        return {
            "episodic_memory": str(episodic_file),
            "memory_items": str(memory_items_file),
            "memory_statistics": str(stats_file),
        }

    def clear(self) -> None:
        """清空三层记忆。"""
        if self.enabled:
            self.working_memory.clear()
            self.episodic_memory.clear()
            self.long_term_memory.clear()

    def __repr__(self) -> str:
        return (
            f"MemoryManager(enabled={self.enabled}, "
            f"working={len(self.working_memory)}, "
            f"episodic={len(self.episodic_memory)}, "
            f"long_term={len(self.long_term_memory)})"
        )
