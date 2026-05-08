"""
记忆检索模块

实现多层级检索功能，支持从三层记忆中快速定位相关信息。
"""

from typing import List, Dict, Any, Optional, Tuple
from .memory_models import SessionSummary
from .memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory


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
        """
        简单分词（支持中英文）

        Args:
            text: 输入文本

        Returns:
            词项列表
        """
        import re

        # 英文单词
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

        # 中文分词（简单实现：提取 2-3 字的片段）
        # 注意：这是简化版本，实际应用中应使用 jieba 等分词工具
        chinese_2char = re.findall(r'[\u4e00-\u9fa5]{2}', text)
        chinese_3char = re.findall(r'[\u4e00-\u9fa5]{3}', text)

        # 合并并去重
        chinese = list(set(chinese_2char + chinese_3char))

        return words + chinese

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