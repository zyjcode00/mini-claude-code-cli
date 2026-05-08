"""
关键词索引器

实现倒排索引（Inverted Index），支持快速关键词检索。
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import re
import time


@dataclass
class IndexEntry:
    """索引条目"""
    doc_id: str  # 文档 ID
    positions: List[int] = field(default_factory=list)  # 词项在文档中的位置
    tf: int = 0  # 词频（Term Frequency）

    def add_position(self, position: int):
        """添加位置信息"""
        self.positions.append(position)
        self.tf += 1


class KeywordIndexer:
    """
    关键词索引器

    实现基于倒排索引的关键词检索，支持：
    1. 快速索引构建（< 50ms）
    2. 快速检索（< 10ms）
    3. 中英文分词
    4. 词频统计
    """

    def __init__(self):
        """初始化索引器"""
        # 倒排索引：term -> {doc_id: IndexEntry}
        self.inverted_index: Dict[str, Dict[str, IndexEntry]] = {}

        # 文档存储：doc_id -> document_text
        self.documents: Dict[str, str] = {}

        # 文档长度统计
        self.doc_lengths: Dict[str, int] = {}

        # 统计信息
        self.stats = {
            "total_docs": 0,
            "total_terms": 0,
            "avg_doc_length": 0.0,
            "index_time_ms": 0.0,
            "search_time_ms": 0.0
        }

    def index_document(self, doc_id: str, document: str) -> float:
        """
        索引单个文档

        Args:
            doc_id: 文档 ID
            document: 文档内容

        Returns:
            索引耗时（毫秒）
        """
        start_time = time.time()

        # 如果文档已存在，先删除旧索引
        if doc_id in self.documents:
            self._remove_document(doc_id)

        # 存储文档
        self.documents[doc_id] = document

        # 分词
        terms = self._tokenize(document)
        self.doc_lengths[doc_id] = len(terms)

        # 构建倒排索引
        for position, term in enumerate(terms):
            if term not in self.inverted_index:
                self.inverted_index[term] = {}

            if doc_id not in self.inverted_index[term]:
                self.inverted_index[term][doc_id] = IndexEntry(doc_id=doc_id)

            self.inverted_index[term][doc_id].add_position(position)

        # 更新统计信息
        self.stats["total_docs"] = len(self.documents)
        self.stats["total_terms"] = len(self.inverted_index)
        self.stats["avg_doc_length"] = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["index_time_ms"] = elapsed_ms

        return elapsed_ms

    def index_documents(self, documents: Dict[str, str]) -> float:
        """
        批量索引文档

        Args:
            documents: 文档字典 {doc_id: document}

        Returns:
            总索引耗时（毫秒）
        """
        start_time = time.time()

        for doc_id, document in documents.items():
            self.index_document(doc_id, document)

        elapsed_ms = (time.time() - start_time) * 1000
        return elapsed_ms

    def search(
        self,
        query: str,
        top_k: int = 10,
        require_all_terms: bool = False
    ) -> List[Tuple[str, float]]:
        """
        检索文档

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            require_all_terms: 是否要求包含所有查询词

        Returns:
            结果列表 [(doc_id, score)]
        """
        start_time = time.time()

        # 分词
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # 检索文档
        candidate_docs: Dict[str, float] = {}

        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, entry in self.inverted_index[term].items():
                    if doc_id not in candidate_docs:
                        candidate_docs[doc_id] = 0.0

                    # 简单的 TF 计分
                    candidate_docs[doc_id] += entry.tf

        # 如果要求包含所有查询词，过滤结果
        if require_all_terms:
            filtered_docs = {}
            for doc_id, score in candidate_docs.items():
                doc_text = self.documents[doc_id]
                contains_all = all(
                    term.lower() in doc_text.lower()
                    for term in query_terms
                )
                if contains_all:
                    filtered_docs[doc_id] = score
            candidate_docs = filtered_docs

        # 按分数排序
        results = sorted(
            candidate_docs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["search_time_ms"] = elapsed_ms

        return results[:top_k]

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        获取文档内容

        Args:
            doc_id: 文档 ID

        Returns:
            文档内容（如果存在）
        """
        return self.documents.get(doc_id)

    def get_term_info(self, term: str) -> Dict[str, IndexEntry]:
        """
        获取词项的索引信息

        Args:
            term: 词项

        Returns:
            词项索引信息 {doc_id: IndexEntry}
        """
        return self.inverted_index.get(term.lower(), {})

    def get_statistics(self) -> Dict[str, any]:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        return {
            **self.stats,
            "documents_count": len(self.documents),
            "terms_count": len(self.inverted_index),
            "avg_doc_length": self.stats["avg_doc_length"]
        }

    def _remove_document(self, doc_id: str):
        """
        删除文档的索引

        Args:
            doc_id: 文档 ID
        """
        if doc_id not in self.documents:
            return

        # 从倒排索引中删除
        for term in list(self.inverted_index.keys()):
            if doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]

            # 如果词项不再对应任何文档，删除该词项
            if not self.inverted_index[term]:
                del self.inverted_index[term]

        # 删除文档
        del self.documents[doc_id]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]

    def _tokenize(self, text: str) -> List[str]:
        """
        分词（支持中英文）

        Args:
            text: 输入文本

        Returns:
            词项列表
        """
        if not text:
            return []

        terms = []

        # 1. 提取英文单词（长度 >= 2）
        english_words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        terms.extend(english_words)

        # 2. 提取中文词汇（改进：使用滑动窗口提取所有可能的词片段）
        # 提取所有中文字符
        chinese_text = re.findall(r'[\u4e00-\u9fa5]+', text)

        for segment in chinese_text:
            # 使用滑动窗口提取 2-4 字的词
            for length in [2, 3, 4]:
                for i in range(len(segment) - length + 1):
                    word = segment[i:i+length]
                    terms.append(word)

        # 3. 提取数字
        numbers = re.findall(r'\b\d+\b', text)
        terms.extend(numbers)

        return terms

    def clear(self):
        """清空索引"""
        self.inverted_index.clear()
        self.documents.clear()
        self.doc_lengths.clear()
        self.stats = {
            "total_docs": 0,
            "total_terms": 0,
            "avg_doc_length": 0.0,
            "index_time_ms": 0.0,
            "search_time_ms": 0.0
        }

    def __len__(self) -> int:
        """返回文档数量"""
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"KeywordIndexer(docs={len(self.documents)}, "
            f"terms={len(self.inverted_index)}, "
            f"avg_len={self.stats['avg_doc_length']:.1f})"
        )


# 性能测试函数
def test_performance():
    """测试索引器性能"""
    indexer = KeywordIndexer()

    # 测试数据
    test_docs = {
        f"doc_{i}": f"这是第 {i} 个测试文档，包含一些关键词和内容。"
        for i in range(100)
    }

    # 测试索引构建速度
    index_time = indexer.index_documents(test_docs)
    print(f"索引构建时间: {index_time:.2f}ms")
    print(f"平均每文档: {index_time / len(test_docs):.2f}ms")

    # 测试检索速度
    results = indexer.search("测试 文档")
    print(f"检索时间: {indexer.stats['search_time_ms']:.2f}ms")
    print(f"检索结果: {len(results)} 个")

    # 打印统计信息
    stats = indexer.get_statistics()
    print(f"统计信息: {stats}")

    return indexer


if __name__ == "__main__":
    test_performance()
