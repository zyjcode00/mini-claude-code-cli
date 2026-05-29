"""
记忆检索模块

实现多层级检索功能，支持从三层记忆中快速定位相关信息。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from .memory_models import SessionSummary
from .memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory
from .memory_items import MemoryItem, MemoryKind, MemoryRecallResult, MemoryStatus
from .memory_index import BM25MemoryDocument, BM25MemoryIndex


@dataclass
class MemoryDocument:
    """统一检索文档：把 MemoryItem / SessionSummary 映射到同一排序空间。"""

    id: str
    source_type: str
    content: str
    project: str = ""
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    timestamp: str = ""
    importance: float = 0.5
    kind: str = MemoryKind.OTHER.value
    item: Optional[MemoryItem] = None
    summary: Optional[SessionSummary] = None
    error_types: List[str] = field(default_factory=list)

    @classmethod
    def from_memory_item(cls, item: MemoryItem, source_type: str = "long_term_items") -> "MemoryDocument":
        raw_observation = item.metadata.get("raw_observation", {}) if item.metadata else {}
        error_types = []
        if item.kind == MemoryKind.BUG.value:
            error_types.extend(item.concepts)
        if isinstance(raw_observation, dict):
            error = raw_observation.get("error") or raw_observation.get("tool_output") or ""
            error_types.extend(_extract_error_types(error))

        return cls(
            id=item.id,
            source_type=source_type,
            content=item.searchable_text(),
            project=item.project,
            session_id=item.source_session_ids[0] if item.source_session_ids else None,
            files=list(item.files),
            concepts=list(item.concepts),
            timestamp=item.updated_at or item.created_at,
            importance=item.importance,
            kind=item.kind,
            item=item,
            error_types=list(dict.fromkeys(error_types)),
        )

    @classmethod
    def from_session_summary(cls, summary: SessionSummary, source_type: str = "session_summary") -> "MemoryDocument":
        error_types = [error.error_type for error in summary.errors_encountered]
        content_parts = [
            summary.task_goal,
            summary.summary_text,
            " ".join(summary.key_decisions),
            " ".join(fc.summary for fc in summary.files_changed),
            " ".join(f"{er.error_type} {er.error_message} {er.solution or ''}" for er in summary.errors_encountered),
            " ".join(tu.tool_name for tu in summary.tools_used),
        ]
        return cls(
            id=f"summary_{summary.session_id}",
            source_type=source_type,
            content="\n".join(part for part in content_parts if part),
            session_id=summary.session_id,
            files=summary.get_file_paths(),
            concepts=summary.get_keywords(),
            timestamp=summary.timestamp,
            importance=summary.importance,
            kind=MemoryKind.SUMMARY.value,
            summary=summary,
            error_types=error_types,
        )

    def to_memory_item(self) -> MemoryItem:
        if self.item:
            return self.item
        if self.summary:
            return MemoryItem(
                id=self.id,
                kind=MemoryKind.SUMMARY.value,
                title=self.summary.task_goal,
                content=self.summary.summary_text,
                created_at=self.summary.timestamp,
                updated_at=self.summary.timestamp,
                concepts=self.summary.get_keywords(),
                files=self.summary.get_file_paths(),
                source_session_ids=[self.summary.session_id],
                importance=self.summary.importance,
                confidence=0.7,
                metadata={"session_summary": self.summary.to_dict()},
            )
        return MemoryItem(
            id=self.id,
            kind=self.kind,
            title=self.id,
            content=self.content,
            project=self.project,
            concepts=self.concepts,
            files=self.files,
            source_session_ids=[self.session_id] if self.session_id else [],
            importance=self.importance,
            confidence=0.7,
        )


def _extract_error_types(text: str) -> List[str]:
    import re
    return list(dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b", text or "")))


class MemoryRetriever:
    """
    记忆检索器

    支持从三层记忆中检索相关信息：
    1. 工作记忆：关键词匹配（当前对话上下文）
    2. 情景记忆：BM25 检索（历史会话摘要）
    3. 长期记忆：倒排索引（持久化摘要）
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        episodic_memory: EpisodicMemory,
        long_term_memory: LongTermMemory
    ):
        """
        初始化记忆检索器

        Args:
            working_memory: 工作记忆实例
            episodic_memory: 情景记忆实例
            long_term_memory: 长期记忆实例
        """
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.long_term_memory = long_term_memory

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_working: bool = True,
        include_episodic: bool = True,
        include_long_term: bool = True
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        从三层记忆中检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量
            include_working: 是否检索工作记忆
            include_episodic: 是否检索情景记忆
            include_long_term: 是否检索长期记忆

        Returns:
            元组列表：(摘要, 相关性分数, 来源层)
        """
        results = []

        # 1. 检索工作记忆（关键词匹配）
        if include_working:
            working_results = self._search_working_memory(query, top_k)
            results.extend(working_results)

        # 2. 检索情景记忆（关键词匹配）
        if include_episodic and len(results) < top_k:
            episodic_results = self._search_episodic_memory(query, top_k - len(results))
            results.extend(episodic_results)

        # 3. 检索长期记忆（倒排索引）
        if include_long_term and len(results) < top_k:
            long_term_results = self._search_long_term_memory(query, top_k - len(results))
            results.extend(long_term_results)

        # 4. 去重（按 session_id）
        seen_ids = set()
        unique_results = []
        for summary, score, source in results:
            if summary.session_id not in seen_ids:
                seen_ids.add(summary.session_id)
                unique_results.append((summary, score, source))

        # 5. 按相关性分数排序
        unique_results.sort(key=lambda x: x[1], reverse=True)

        return unique_results[:top_k]

    def _search_working_memory(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        从工作记忆中检索消息

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            元组列表：(摘要, 相关性分数, "working")
        """
        results = []
        query_lower = query.lower()

        # 工作记忆存储的是消息，不是摘要
        # 这里我们返回匹配的消息（转换为简单的摘要格式）
        messages = self.working_memory.get_all()

        for i, msg in enumerate(messages):
            content = str(msg.get("content", ""))
            if query_lower in content.lower():
                # 计算相关性分数（简单的词频）
                score = content.lower().count(query_lower) / max(len(content), 1)

                # 创建临时摘要
                temp_summary = SessionSummary(
                    session_id=f"working_msg_{i}",
                    timestamp="",
                    summary_text=content[:200],  # 截取前 200 字符
                    task_goal="当前对话",
                    task_status="in_progress",
                    files_changed=[],
                    errors_encountered=[],
                    tools_used=[],
                    key_decisions=[],
                    importance=0.5 + score,
                    message_count=1,
                    token_count=len(content)
                )

                results.append((temp_summary, score, "working"))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _search_episodic_memory(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        从情景记忆中检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            元组列表：(摘要, 相关性分数, "episodic")
        """
        results = []

        # 使用情景记忆的搜索功能
        summaries = self.episodic_memory.search(query, top_k * 2)  # 获取更多结果用于排序

        for summary in summaries:
            # 计算 BM25 分数
            score = self._calculate_bm25_score(query, summary)
            results.append((summary, score, "episodic"))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _search_long_term_memory(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        从长期记忆中检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            元组列表：(摘要, 相关性分数, "long_term")
        """
        results = []

        # 使用长期记忆的搜索功能
        summaries = self.long_term_memory.search(query, top_k * 2)

        for summary in summaries:
            # 计算 BM25 分数
            score = self._calculate_bm25_score(query, summary)
            results.append((summary, score, "long_term"))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _calculate_bm25_score(
        self,
        query: str,
        summary: SessionSummary,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """
        计算 BM25 相关性分数

        Args:
            query: 查询字符串
            summary: 会话摘要
            k1: BM25 参数
            b: BM25 参数

        Returns:
            BM25 分数
        """
        import math

        # 分词
        query_terms = self._tokenize(query)

        # 文档内容
        doc_text = f"{summary.summary_text} {summary.task_goal}"
        doc_terms = self._tokenize(doc_text)

        # 文档长度
        doc_len = len(doc_terms)

        # 平均文档长度（简化处理）
        avg_doc_len = 100  # 假设平均长度为 100

        # 计算词频
        term_freq = {}
        for term in doc_terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        # 计算 BM25 分数
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]

                # BM25 公式
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))

                score += numerator / denominator

        return score

    def _tokenize(self, text: str) -> List[str]:
        """统一使用 BM25MemoryIndex 的 Phase 1 分词器。"""
        return BM25MemoryIndex.tokenize(text)

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
        """统一 Hybrid Recall：整合 MemoryItem、Episodic SessionSummary 与长期 SessionSummary。"""
        documents = self._build_documents(include_summaries=include_summaries, include_items=include_items)
        bm25_query = " ".join(part for part in [query, file_path or "", error_type or ""] if part)
        query_terms = self._tokenize(bm25_query)
        bm25_hits = self._search_bm25(documents, bm25_query, top_k=max(len(documents), top_k))
        ranked: List[Tuple[MemoryDocument, float, List[str]]] = []

        normalized_file = self._normalize_path(file_path) if file_path else ""
        normalized_error = (error_type or "").lower()
        allowed_kinds = {kind.lower() for kind in kinds} if kinds else None

        for doc in documents:
            if allowed_kinds and doc.kind.lower() not in allowed_kinds:
                continue

            file_match = self._file_match_score(doc, normalized_file) if normalized_file else 0.0
            error_match = self._error_match_score(doc, normalized_error) if normalized_error else 0.0
            if normalized_file and file_match <= 0:
                continue
            if normalized_error and error_match <= 0 and doc.kind != MemoryKind.BUG.value:
                continue

            bm25_hit = bm25_hits.get(doc.id)
            bm25_score = bm25_hit.score if bm25_hit else 0.0
            importance_weight = max(0.0, min(doc.importance, 1.0)) * 0.35
            recency_weight = self._recency_weight(doc.timestamp)
            type_weight = self._type_weight(doc.kind, query)
            score = bm25_score + importance_weight + recency_weight + file_match + error_match + type_weight

            reasons = []
            if bm25_score:
                matched = ", ".join((bm25_hit.matched_terms if bm25_hit else [])[:6])
                suffix = f" ({matched})" if matched else ""
                reasons.append(f"BM25相关 {bm25_score:.2f}{suffix}")
            if importance_weight:
                reasons.append(f"重要性 {importance_weight:.2f}")
            if recency_weight:
                reasons.append(f"时间 {recency_weight:.2f}")
            if file_match:
                reasons.append(f"文件匹配 {file_match:.2f}")
            if error_match:
                reasons.append(f"错误匹配 {error_match:.2f}")
            if type_weight:
                reasons.append(f"类型匹配 {type_weight:.2f}")

            if score > 0 or not query_terms:
                ranked.append((doc, score, reasons or ["默认召回"]))

        ranked.sort(key=lambda item: item[1], reverse=True)
        results: List[MemoryRecallResult] = []
        seen = set()
        for doc, score, reasons in ranked:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            recalled_item = doc.to_memory_item()
            if doc.item and doc.source_type == "long_term_items":
                try:
                    self.long_term_memory._write_item(self.long_term_memory.maintenance.record_access(doc.item))
                    recalled_item = doc.item
                except Exception:
                    pass
            results.append(MemoryRecallResult(
                item=recalled_item,
                score=score,
                source=doc.source_type,
                reason="; ".join(reasons),
            ))
            if len(results) >= top_k:
                break
        return results

    def _build_documents(self, include_summaries: bool = True, include_items: bool = True) -> List[MemoryDocument]:
        documents: List[MemoryDocument] = []
        if include_items:
            documents.extend(
                MemoryDocument.from_memory_item(item)
                for item in self.long_term_memory.get_all_items()
                if item.status == MemoryStatus.ACTIVE.value and item.is_latest and not item.is_expired()
            )
        if include_summaries:
            documents.extend(MemoryDocument.from_session_summary(summary, source_type="summary_compat") for summary in self.episodic_memory.get_all())
            for session_id in self.long_term_memory.get_all_session_ids():
                summary = self.long_term_memory.retrieve(session_id)
                if summary:
                    documents.append(MemoryDocument.from_session_summary(summary, source_type="summary_compat"))
        return documents

    def _search_bm25(self, documents: List[MemoryDocument], query: str, top_k: int) -> Dict[str, Any]:
        """Search BM25, preferring Phase 2 persisted index for long-term docs."""
        if not query:
            return {}

        persisted_hits: Dict[str, Any] = {}
        if hasattr(self.long_term_memory, "index_manager"):
            try:
                valid_long_term_ids = {
                    doc.id
                    for doc in documents
                    if doc.source_type in {"long_term_items", "summary_compat"}
                }
                raw_hits = self.long_term_memory.index_manager.search(query, top_k=max(top_k, len(valid_long_term_ids)))
                persisted_hits = {hit.doc_id: hit for hit in raw_hits if hit.doc_id in valid_long_term_ids}
            except Exception:
                try:
                    self.long_term_memory.rebuild_search_index()
                    raw_hits = self.long_term_memory.index_manager.search(query, top_k=top_k)
                    valid_ids = {doc.id for doc in documents}
                    persisted_hits = {hit.doc_id: hit for hit in raw_hits if hit.doc_id in valid_ids}
                except Exception:
                    persisted_hits = {}

        transient_docs = [
            doc for doc in documents
            if doc.source_type not in {"long_term_items", "summary_compat"}
        ]
        if not transient_docs:
            return persisted_hits

        transient_index = self._build_bm25_index(transient_docs)
        for hit in transient_index.search(query, top_k=max(top_k, len(transient_docs))):
            persisted_hits[hit.doc_id] = hit
        return persisted_hits

    def _build_bm25_index(self, documents: List[MemoryDocument]) -> BM25MemoryIndex:
        index = BM25MemoryIndex()
        for doc in documents:
            index.add_or_update(BM25MemoryDocument(
                doc_id=doc.id,
                title=self._document_title(doc),
                content=doc.content,
                concepts=doc.concepts,
                files=doc.files,
                kind=doc.kind,
                error=" ".join(doc.error_types),
                project=doc.project,
                metadata={"source_type": doc.source_type, "session_id": doc.session_id},
            ))
        return index

    def _document_title(self, doc: MemoryDocument) -> str:
        if doc.item:
            return doc.item.title
        if doc.summary:
            return doc.summary.task_goal
        return doc.id

    def _calculate_text_bm25_score(self, query_terms: List[str], text: str) -> float:
        doc_terms = self._tokenize(text)
        if not query_terms or not doc_terms:
            return 0.0
        term_freq: Dict[str, int] = {}
        for term in doc_terms:
            term_freq[term] = term_freq.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if term in term_freq:
                tf = term_freq[term]
                score += (tf * 2.5) / (tf + 1.5)
        return score

    def _file_match_score(self, doc: MemoryDocument, normalized_file: str) -> float:
        for candidate in doc.files:
            normalized_candidate = self._normalize_path(candidate)
            if normalized_candidate == normalized_file:
                return 1.2
            if normalized_file in normalized_candidate or normalized_candidate in normalized_file:
                return 0.75
            if normalized_candidate.split("/")[-1] == normalized_file.split("/")[-1]:
                return 0.45
        return 0.0

    def _error_match_score(self, doc: MemoryDocument, normalized_error: str) -> float:
        haystack = " ".join(doc.error_types + doc.concepts + [doc.content]).lower()
        if not normalized_error:
            return 0.0
        if normalized_error in haystack:
            return 1.1
        tokens = self._tokenize(normalized_error)
        if tokens and any(token in haystack for token in tokens):
            return 0.55
        return 0.0

    def _recency_weight(self, timestamp: str) -> float:
        if not timestamp:
            return 0.0
        try:
            ts = datetime.fromisoformat(timestamp)
        except Exception:
            return 0.0
        age_days = max((datetime.now() - ts).days, 0)
        return max(0.0, 0.2 - min(age_days, 30) * 0.005)

    def _type_weight(self, kind: str, query: str) -> float:
        q = (query or "").lower()
        if kind == MemoryKind.BUG.value and any(word in q for word in ["error", "exception", "失败", "错误", "bug"]):
            return 0.35
        if kind == MemoryKind.ARCHITECTURE.value and any(word in q for word in ["架构", "architecture", "设计"]):
            return 0.3
        if kind == MemoryKind.WORKFLOW.value and any(word in q for word in ["流程", "workflow", "pytest", "测试"]):
            return 0.25
        return 0.0

    @staticmethod
    def _normalize_path(path: Optional[str]) -> str:
        return (path or "").replace("\\", "/").lower().strip()

    def retrieve_by_file_path(
        self,
        file_path: str,
        top_k: int = 5
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        根据文件路径检索相关摘要

        Args:
            file_path: 文件路径
            top_k: 返回结果数量

        Returns:
            元组列表：(摘要, 相关性分数, 来源层)
        """
        results = []

        # 1. 检索情景记忆
        for summary in self.episodic_memory.get_all():
            for fc in summary.files_changed:
                if file_path in fc.path:
                    score = 1.0 if fc.path == file_path else 0.5
                    results.append((summary, score, "episodic"))
                    break

        # 2. 检索长期记忆
        if len(results) < top_k:
            # 使用文件名作为关键词搜索
            filename = file_path.split("/")[-1].split("\\")[-1]
            long_term_results = self._search_long_term_memory(filename, top_k - len(results))
            results.extend(long_term_results)

        # 去重并排序
        seen_ids = set()
        unique_results = []
        for summary, score, source in results:
            if summary.session_id not in seen_ids:
                seen_ids.add(summary.session_id)
                unique_results.append((summary, score, source))

        unique_results.sort(key=lambda x: x[1], reverse=True)

        return unique_results[:top_k]

    def retrieve_by_error_type(
        self,
        error_type: str,
        top_k: int = 5
    ) -> List[Tuple[SessionSummary, float, str]]:
        """
        根据错误类型检索相关摘要

        Args:
            error_type: 错误类型
            top_k: 返回结果数量

        Returns:
            元组列表：(摘要, 相关性分数, 来源层)
        """
        results = []

        # 检索情景记忆
        for summary in self.episodic_memory.get_all():
            for error in summary.errors_encountered:
                if error_type.lower() in error.error_type.lower():
                    score = 1.0 if error.error_type == error_type else 0.7
                    results.append((summary, score, "episodic"))
                    break

        # 检索长期记忆
        if len(results) < top_k:
            long_term_results = self._search_long_term_memory(error_type, top_k - len(results))
            results.extend(long_term_results)

        # 去重并排序
        seen_ids = set()
        unique_results = []
        for summary, score, source in results:
            if summary.session_id not in seen_ids:
                seen_ids.add(summary.session_id)
                unique_results.append((summary, score, source))

        unique_results.sort(key=lambda x: x[1], reverse=True)

        return unique_results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索器统计信息

        Returns:
            统计信息字典
        """
        return {
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "long_term_memory_count": len(self.long_term_memory),
            "long_term_memory_dir": str(self.long_term_memory.storage_dir)
        }

    def __repr__(self) -> str:
        return (f"MemoryRetriever(working={len(self.working_memory)}, "
                f"episodic={len(self.episodic_memory)}, "
                f"long_term={len(self.long_term_memory)})")