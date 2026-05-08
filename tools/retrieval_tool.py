"""
检索工具

封装检索功能为 Agent 可调用的工具，支持：
1. 关键词检索
2. BM25 检索
3. 历史记忆检索
"""

from typing import Type, Dict, Any, List, Optional
from pydantic import BaseModel, Field
from tools.base import BaseTool

# 导入检索器
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.keyword_indexer import KeywordIndexer
from core.bm25_retriever import BM25Retriever


class SearchInput(BaseModel):
    """检索工具输入参数"""
    query: str = Field(..., description="检索查询字符串")
    top_k: int = Field(default=5, description="返回结果数量，默认 5")
    search_type: str = Field(
        default="bm25",
        description="检索类型：keyword（关键词）或 bm25（默认）"
    )


class SearchResult(BaseModel):
    """单个检索结果"""
    doc_id: str = Field(..., description="文档 ID")
    score: float = Field(..., description="相关性分数")
    content: str = Field(..., description="文档内容")
    highlight: Optional[str] = Field(None, description="高亮片段")


class SearchOutput(BaseModel):
    """检索工具输出"""
    success: bool = Field(..., description="是否成功")
    results: List[SearchResult] = Field(default_factory=list, description="检索结果")
    message: str = Field("", description="结果消息")
    query: str = Field(..., description="查询字符串")
    total: int = Field(0, description="总结果数")


class SearchKeywordTool(BaseTool):
    """关键词检索工具"""
    
    name: str = "search_keyword"
    description: str = "使用倒排索引进行关键词检索，快速定位文档"
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(self, indexer: Optional[KeywordIndexer] = None):
        """
        初始化关键词检索工具
        
        Args:
            indexer: 关键词索引器实例（可选）
        """
        super().__init__()
        self.indexer = indexer or KeywordIndexer()
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> str:
        """
        执行关键词检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
        
        Returns:
            格式化的检索结果
        """
        try:
            # 检索
            results = self.indexer.search(query, top_k=top_k)
            
            if not results:
                return f"未找到与 '{query}' 相关的结果"
            
            # 格式化输出
            output_lines = [f"找到 {len(results)} 个与 '{query}' 相关的结果：\n"]
            
            for i, (doc_id, score) in enumerate(results, 1):
                content = self.indexer.get_document(doc_id)
                output_lines.append(f"{i}. [{doc_id}] (分数: {score:.2f})")
                if content:
                    # 截取前 100 字符
                    preview = content[:100] + "..." if len(content) > 100 else content
                    output_lines.append(f"   {preview}\n")
            
            return "\n".join(output_lines)
        
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    def index_documents(self, documents: Dict[str, str]) -> str:
        """
        索引文档
        
        Args:
            documents: 文档字典 {doc_id: content}
        
        Returns:
            索引结果消息
        """
        try:
            index_time = self.indexer.index_documents(documents)
            return f"成功索引 {len(documents)} 个文档，耗时 {index_time:.2f}ms"
        except Exception as e:
            return f"索引失败: {str(e)}"


class SearchBM25Tool(BaseTool):
    """BM25 检索工具"""
    
    name: str = "search_bm25"
    description: str = "使用 BM25 算法进行智能检索，比关键词匹配更准确"
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(self, retriever: Optional[BM25Retriever] = None):
        """
        初始化 BM25 检索工具
        
        Args:
            retriever: BM25 检索器实例（可选）
        """
        super().__init__()
        self.retriever = retriever or BM25Retriever()
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> str:
        """
        执行 BM25 检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
        
        Returns:
            格式化的检索结果
        """
        try:
            # 检索（带详细信息）
            detailed_results = self.retriever.search_with_details(query, top_k=top_k)
            
            if not detailed_results:
                return f"未找到与 '{query}' 相关的结果"
            
            # 格式化输出
            output_lines = [f"找到 {len(detailed_results)} 个与 '{query}' 相关的结果：\n"]
            
            for i, result in enumerate(detailed_results, 1):
                output_lines.append(
                    f"{i}. [{result['doc_id']}] (分数: {result['score']:.3f}, "
                    f"长度: {result['doc_length']})"
                )
                
                # 添加高亮片段
                if result['highlights']:
                    output_lines.append("   相关片段：")
                    for highlight in result['highlights']:
                        output_lines.append(f"   - {highlight}")
                
                output_lines.append("")
            
            return "\n".join(output_lines)
        
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    def index_documents(self, documents: Dict[str, str]) -> str:
        """
        索引文档
        
        Args:
            documents: 文档字典 {doc_id: content}
        
        Returns:
            索引结果消息
        """
        try:
            index_time = self.retriever.index_documents(documents)
            return f"成功索引 {len(documents)} 个文档，耗时 {index_time:.2f}ms"
        except Exception as e:
            return f"索引失败: {str(e)}"


class SearchMemoryTool(BaseTool):
    """
    检索历史记忆工具
    
    支持从三层记忆中检索相关信息：
    1. 工作记忆（当前对话）
    2. 情景记忆（历史会话）
    3. 长期记忆（持久化摘要）
    """
    
    name: str = "search_memory"
    description: str = "从历史记忆中检索相关信息，支持关键词和 BM25 检索"
    args_schema: Type[BaseModel] = SearchInput
    
    def __init__(
        self,
        keyword_indexer: Optional[KeywordIndexer] = None,
        bm25_retriever: Optional[BM25Retriever] = None
    ):
        """
        初始化记忆检索工具
        
        Args:
            keyword_indexer: 关键词索引器（可选）
            bm25_retriever: BM25 检索器（可选）
        """
        super().__init__()
        self.keyword_indexer = keyword_indexer or KeywordIndexer()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
    
    def run(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "bm25",
        **kwargs
    ) -> str:
        """
        执行记忆检索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            search_type: 检索类型（keyword 或 bm25）
        
        Returns:
            格式化的检索结果
        """
        try:
            # 根据检索类型选择检索器
            if search_type.lower() == "keyword":
                results = self.keyword_indexer.search(query, top_k=top_k)
                indexer = self.keyword_indexer
                get_content = indexer.get_document
            else:
                results = self.bm25_retriever.search(query, top_k=top_k)
                indexer = self.bm25_retriever
                get_content = indexer.get_document
            
            if not results:
                return f"未在历史记忆中找到与 '{query}' 相关的信息"
            
            # 格式化输出
            output_lines = [
                f"从历史记忆中找到 {len(results)} 个相关信息：\n"
            ]
            
            for i, (doc_id, score) in enumerate(results, 1):
                content = get_content(doc_id)
                output_lines.append(
                    f"{i}. [{doc_id}] (相关度: {score:.3f})"
                )
                
                if content:
                    # 截取前 150 字符作为预览
                    preview = content[:150] + "..." if len(content) > 150 else content
                    output_lines.append(f"   {preview}\n")
            
            return "\n".join(output_lines)
        
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    def index_memory(self, doc_id: str, content: str, search_type: str = "both") -> str:
        """
        索引记忆内容
        
        Args:
            doc_id: 文档 ID
            content: 记忆内容
            search_type: 索引类型（keyword, bm25, both）
        
        Returns:
            索引结果消息
        """
        try:
            messages = []
            
            if search_type in ["keyword", "both"]:
                index_time = self.keyword_indexer.index_document(doc_id, content)
                messages.append(f"关键词索引完成，耗时 {index_time:.2f}ms")
            
            if search_type in ["bm25", "both"]:
                index_time = self.bm25_retriever.index_document(doc_id, content)
                messages.append(f"BM25 索引完成，耗时 {index_time:.2f}ms")
            
            return " | ".join(messages)
        
        except Exception as e:
            return f"索引失败: {str(e)}"


# 工具导出
__all__ = [
    "SearchKeywordTool",
    "SearchBM25Tool",
    "SearchMemoryTool",
    "SearchInput",
    "SearchOutput",
    "SearchResult"
]


# 测试代码
if __name__ == "__main__":
    # 测试 BM25 工具
    tool = SearchBM25Tool()
    
    # 索引测试文档
    test_docs = {
        "session_1": "完成了用户登录功能的开发，使用了 JWT 认证机制。",
        "session_2": "修复了数据库连接池的 bug，提高了并发性能。",
        "session_3": "实现了文件上传功能，支持图片和文档格式。"
    }
    
    print("=== 索引文档 ===")
    print(tool.index_documents(test_docs))
    
    print("\n=== 检索测试 ===")
    queries = ["登录", "bug", "文件"]
    for query in queries:
        print(f"\n查询: {query}")
        print(tool.run(query))
        print("-" * 50)