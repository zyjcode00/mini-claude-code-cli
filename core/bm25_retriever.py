"""
BM25 检索器

实现 BM25（Best Matching 25）算法，一种经典的信息检索算法。
BM25 在搜索引擎和文档检索中广泛应用，比简单的关键词匹配更智能。
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math
import re
import time


@dataclass
class BM25Document:
    """BM25 文档"""
    doc_id: str  # 文档 ID
    content: str  # 文档内容
    terms: List[str] = field(default_factory=list)  # 分词后的词项
    term_freq: Dict[str, int] = field(default_factory=dict)  # 词频统计
    doc_length: int = 0  # 文档长度（词项数量）


class BM25Retriever:
    """
    BM25 检索器

    实现 BM25 算法的核心功能：
    1. 文档索引和词频统计
    2. IDF（逆文档频率）计算
    3. BM25 相关性评分
    4. 高效检索

    参数说明：
    - k1: 控制词频饱和度的参数（通常 1.2-2.0，默认 1.5）
    - b: 控制文档长度归一化的参数（通常 0.75，默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25 检索器

        Args:
            k1: 词频饱和度参数
            b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b

        # 文档存储：doc_id -> BM25Document
        self.documents: Dict[str, BM25Document] = {}

        # 倒排索引：term -> set(doc_id)
        self.inverted_index: Dict[str, set] = {}

        # IDF 缓存：term -> idf_score
        self.idf_cache: Dict[str, float] = {}

        # 统计信息
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0

        # 性能统计
        self.stats = {
            "index_time_ms": 0.0,
            "search_time_ms": 0.0,
            "total_indexed": 0
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

        # 如果文档已存在，先删除
        if doc_id in self.documents:
            self._remove_document(doc_id)

        # 分词
        terms = self._tokenize(document)
        term_freq = {}
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        # 创建文档对象
        doc = BM25Document(
            doc_id=doc_id,
            content=document,
            terms=terms,
            term_freq=term_freq,
            doc_length=len(terms)
        )
        self.documents[doc_id] = doc

        # 更新倒排索引
        for term in term_freq.keys():
            if term not in self.inverted_index:
                self.inverted_index[term] = set()
            self.inverted_index[term].add(doc_id)

        # 清空 IDF 缓存（因为文档集合变了）
        self.idf_cache.clear()

        # 更新统计信息
        self._update_statistics()

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["index_time_ms"] = elapsed_ms
        self.stats["total_indexed"] += 1

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
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        检索文档

        Args:
            query: 查询字符串
            top_k: 返回结果数量
            min_score: 最低分数阈值

        Returns:
            结果列表 [(doc_id, score)]
        """
        start_time = time.time()

        # 分词
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # 计算每个文档的 BM25 分数
        scores: Dict[str, float] = {}

        for term in query_terms:
            # 获取包含该词项的文档
            if term not in self.inverted_index:
                continue

            doc_ids = self.inverted_index[term]

            # 计算 IDF
            idf = self._calculate_idf(term)

            for doc_id in doc_ids:
                doc = self.documents[doc_id]

                # 计算 BM25 分数
                tf = doc.term_freq.get(term, 0)
                doc_length = doc.doc_length

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                term_score = idf * (numerator / denominator)

                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += term_score

        # 过滤低分结果并排序
        results = [
            (doc_id, score)
            for doc_id, score in scores.items()
            if score >= min_score
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["search_time_ms"] = elapsed_ms

        return results[:top_k]

    def search_with_details(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        检索文档（带详细信息）

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            结果列表 [{doc_id, score, content, highlights}]
        """
        results = self.search(query, top_k)

        detailed_results = []
        query_terms = set(self._tokenize(query))

        for doc_id, score in results:
            doc = self.documents[doc_id]

            # 提取高亮片段
            highlights = self._extract_highlights(doc.content, query_terms)

            detailed_results.append({
                "doc_id": doc_id,
                "score": score,
                "content": doc.content,
                "highlights": highlights,
                "doc_length": doc.doc_length
            })

        return detailed_results

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        获取文档内容

        Args:
            doc_id: 文档 ID

        Returns:
            文档内容（如果存在）
        """
        if doc_id in self.documents:
            return self.documents[doc_id].content
        return None

    def get_statistics(self) -> Dict[str, any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_docs": self.total_docs,
            "total_terms": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b,
            **self.stats
        }

    def _calculate_idf(self, term: str) -> float:
        """
        计算 IDF（逆文档频率）

        Args:
            term: 词项

        Returns:
            IDF 分数
        """
        # 检查缓存
        if term in self.idf_cache:
            return self.idf_cache[term]

        # 计算包含该词项的文档数
        n_qi = len(self.inverted_index.get(term, set()))

        # IDF 公式（使用平滑版本，避免除零）
        idf = math.log(
            (self.total_docs - n_qi + 0.5) / (n_qi + 0.5) + 1
        )

        # 缓存结果
        self.idf_cache[term] = idf

        return idf

    def _update_statistics(self):
        """更新统计信息"""
        self.total_docs = len(self.documents)

        if self.total_docs > 0:
            total_length = sum(
                doc.doc_length for doc in self.documents.values()
            )
            self.avg_doc_length = total_length / self.total_docs
        else:
            self.avg_doc_length = 0.0

    def _remove_document(self, doc_id: str):
        """
        删除文档

        Args:
            doc_id: 文档 ID
        """
        if doc_id not in self.documents:
            return

        doc = self.documents[doc_id]

        # 从倒排索引中删除
        for term in doc.term_freq.keys():
            if term in self.inverted_index:
                self.inverted_index[term].discard(doc_id)
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        # 删除文档
        del self.documents[doc_id]

        # 清空 IDF 缓存
        self.idf_cache.clear()

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

    def _extract_highlights(
        self,
        content: str,
        query_terms: set,
        context_length: int = 50
    ) -> List[str]:
        """
        提取高亮片段

        Args:
            content: 文档内容
            query_terms: 查询词项集合
            context_length: 上下文长度

        Returns:
            高亮片段列表
        """
        highlights = []

        # 分句
        sentences = re.split(r'[。！？\n]', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查句子是否包含查询词
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # 截取适当长度
                if len(sentence) > context_length * 2:
                    # 找到第一个匹配词的位置
                    for term in query_terms:
                        pos = sentence_lower.find(term)
                        if pos >= 0:
                            start = max(0, pos - context_length)
                            end = min(len(sentence), pos + len(term) + context_length)
                            highlights.append(sentence[start:end])
                            break
                else:
                    highlights.append(sentence)

        return highlights[:3]  # 最多返回 3 个片段

    def clear(self):
        """清空索引"""
        self.documents.clear()
        self.inverted_index.clear()
        self.idf_cache.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.stats = {
            "index_time_ms": 0.0,
            "search_time_ms": 0.0,
            "total_indexed": 0
        }

    def __len__(self) -> int:
        """返回文档数量"""
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"BM25Retriever(docs={self.total_docs}, "
            f"terms={len(self.inverted_index)}, "
            f"avg_len={self.avg_doc_length:.1f}, "
            f"k1={self.k1}, b={self.b})"
        )


# 性能测试函数
def test_bm25_performance():
    """测试 BM25 检索器性能"""
    retriever = BM25Retriever()

    # 测试数据
    test_docs = {
        "doc_1": "Python 是一门流行的编程语言，广泛用于数据科学和人工智能。",
        "doc_2": "JavaScript 是网页开发的核心语言，支持前后端开发。",
        "doc_3": "机器学习是人工智能的重要分支，使用 Python 进行开发。",
        "doc_4": "深度学习是机器学习的子领域，基于神经网络技术。",
        "doc_5": "自然语言处理是人工智能的应用领域，涉及文本分析。"
    }

    # 索引构建
    index_time = retriever.index_documents(test_docs)
    print(f"索引构建时间: {index_time:.2f}ms")

    # 检索测试
    queries = ["Python", "人工智能", "机器学习", "编程"]

    for query in queries:
        results = retriever.search(query, top_k=3)
        print(f"\n查询: {query}")
        print(f"检索时间: {retriever.stats['search_time_ms']:.2f}ms")
        print("结果:")
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.3f}")

    # 打印统计信息
    stats = retriever.get_statistics()
    print(f"\n统计信息: {stats}")

    return retriever


if __name__ == "__main__":
    test_bm25_performance()