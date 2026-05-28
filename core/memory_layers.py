"""
三层记忆架构实现

实现 WorkingMemory / EpisodicMemory / LongTermMemory 三层记忆系统。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .memory_models import SessionSummary


class WorkingMemory:
    """
    工作记忆层：存储最近 10-20 条消息

    特点：
    - FIFO 淘汰策略
    - 快速访问
    - 用于当前对话上下文
    """

    def __init__(self, max_size: int = 20):
        """
        初始化工作记忆

        Args:
            max_size: 最大消息数量（默认 20）
        """
        self.max_size = max_size
        self.data: List[Dict[str, Any]] = []

    def add(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        添加消息到工作记忆

        Args:
            message: 消息字典

        Returns:
            如果满了，返回被淘汰的消息；否则返回 None
        """
        self.data.append(message)

        # 如果超过最大大小，淘汰最旧的消息（FIFO）
        if len(self.data) > self.max_size:
            evicted = self.data.pop(0)
            return evicted

        return None

    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有消息"""
        return self.data.copy()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近 n 条消息"""
        return self.data[-n:] if n < len(self.data) else self.data.copy()

    def clear(self):
        """清空工作记忆"""
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"WorkingMemory(size={len(self.data)}/{self.max_size})"


class EpisodicMemory:
    """
    情景记忆层：存储结构化摘要

    特点：
    - 基于重要性淘汰（低重要性优先淘汰）
    - 支持关键词检索
    - 用于存储历史会话摘要
    """

    def __init__(self, max_size: int = 50):
        """
        初始化情景记忆

        Args:
            max_size: 最大摘要数量（默认 50）
        """
        self.max_size = max_size
        self.data: List[SessionSummary] = []

    def add(self, summary: SessionSummary) -> Optional[SessionSummary]:
        """
        添加摘要到情景记忆

        Args:
            summary: 会话摘要

        Returns:
            如果满了，返回被淘汰的摘要；否则返回 None
        """
        self.data.append(summary)

        # 如果超过最大大小，淘汰重要性最低的摘要
        if len(self.data) > self.max_size:
            # 找到重要性最低的摘要（如果重要性相同，淘汰最旧的）
            min_importance = min(s.importance for s in self.data)
            for i, s in enumerate(self.data):
                if s.importance == min_importance:
                    evicted = self.data.pop(i)
                    return evicted

        return None

    def search(self, query: str, top_k: int = 5) -> List[SessionSummary]:
        """
        关键词检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            匹配的摘要列表
        """
        results = []
        query_lower = query.lower()

        for summary in self.data:
            # 在摘要文本、任务目标、关键词中搜索
            if (query_lower in summary.summary_text.lower() or
                query_lower in summary.task_goal.lower() or
                any(query_lower in kw.lower() for kw in summary.get_keywords())):
                results.append(summary)

        # 按重要性排序
        results.sort(key=lambda s: s.importance, reverse=True)

        return results[:top_k]

    def get_all(self) -> List[SessionSummary]:
        """获取所有摘要"""
        return self.data.copy()

    def get_recent(self, n: int = 10) -> List[SessionSummary]:
        """获取最近 n 条摘要"""
        return self.data[-n:] if n < len(self.data) else self.data.copy()

    def clear(self):
        """清空情景记忆"""
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"EpisodicMemory(size={len(self.data)}/{self.max_size})"


class LongTermMemory:
    """
    长期记忆层：基于文件系统的持久化存储

    特点：
    - 存储在磁盘上（JSON 格式）
    - 支持关键词检索
    - 用于存储历史所有会话摘要
    """

    def __init__(self, storage_dir: str = "memory/long_term"):
        """
        初始化长期记忆

        Args:
            storage_dir: 存储目录路径
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 内存索引：{session_id: file_path}
        self.index: Dict[str, Path] = {}

        # 倒排索引：{keyword: [session_ids]}
        self.inverted_index: Dict[str, List[str]] = {}

        # 初始化时加载索引
        self._load_index()

    def _load_index(self):
        """从磁盘加载索引"""
        index_file = self.storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index = {k: Path(v) for k, v in data.get("index", {}).items()}
                    self.inverted_index = data.get("inverted_index", {})
            except Exception as e:
                # 如果加载失败，重建索引
                print(f"⚠️ 加载索引失败，将重建索引: {e}")
                self._rebuild_index()

    def _save_index(self):
        """保存索引到磁盘"""
        index_file = self.storage_dir / "index.json"
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "index": {k: str(v) for k, v in self.index.items()},
                    "inverted_index": self.inverted_index
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存索引失败: {e}")

    def _rebuild_index(self):
        """重建索引"""
        self.index.clear()
        self.inverted_index.clear()

        # 遍历所有摘要文件
        for summary_file in self.storage_dir.glob("summary_*.json"):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_id = data.get("session_id")
                    if session_id:
                        self.index[session_id] = summary_file

                        # 更新倒排索引
                        keywords = self._extract_keywords(data)
                        for keyword in keywords:
                            if keyword not in self.inverted_index:
                                self.inverted_index[keyword] = []
                            if session_id not in self.inverted_index[keyword]:
                                self.inverted_index[keyword].append(session_id)
            except Exception as e:
                print(f"⚠️ 重建索引时跳过文件 {summary_file}: {e}")

        # 保存索引
        self._save_index()

    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        """从摘要数据中提取关键词"""
        keywords = []

        # 从摘要文本中提取
        summary_text = data.get("summary_text", "")
        keywords.extend(self._tokenize(summary_text))

        # 从任务目标中提取
        task_goal = data.get("task_goal", "")
        keywords.extend(self._tokenize(task_goal))

        # 从文件路径中提取
        files_changed = data.get("files_changed", [])
        for fc in files_changed:
            if not isinstance(fc, dict):
                continue
            # FileChange 当前规范字段是 path；兼容旧数据中可能存在的 file_path
            file_path = fc.get("path") or fc.get("file_path", "")
            keywords.extend(self._tokenize(file_path))

        return list(set(keywords))

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（支持中英文）"""
        import re

        # 英文/路径片段：保留下划线文件名（如 memory_layers），同时拆分为普通英文词
        raw_words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{1,}\b', text.lower())
        words = []
        for word in raw_words:
            words.append(word)
            words.extend(part for part in word.split('_') if len(part) >= 2)

        # 中文分词（简单实现：提取 2-3 字的片段）
        # 注意：这是简化版本，实际应用中应使用 jieba 等分词工具
        chinese_2char = re.findall(r'[\u4e00-\u9fa5]{2}', text)
        chinese_3char = re.findall(r'[\u4e00-\u9fa5]{3}', text)

        # 合并并去重
        chinese = list(set(chinese_2char + chinese_3char))

        return words + chinese

    def store(self, summary: SessionSummary) -> str:
        """
        存储摘要到长期记忆

        Args:
            summary: 会话摘要

        Returns:
            存储的文件路径
        """
        # 生成文件名
        timestamp = summary.timestamp.replace(":", "-").replace(" ", "_")
        filename = f"summary_{summary.session_id}_{timestamp}.json"
        file_path = self.storage_dir / filename

        # 保存到磁盘
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

            # 更新索引
            self.index[summary.session_id] = file_path

            # 更新倒排索引
            keywords = self._extract_keywords(summary.to_dict())
            for keyword in keywords:
                if keyword not in self.inverted_index:
                    self.inverted_index[keyword] = []
                if summary.session_id not in self.inverted_index[keyword]:
                    self.inverted_index[keyword].append(summary.session_id)

            # 保存索引
            self._save_index()

            return str(file_path)
        except Exception as e:
            print(f"⚠️ 存储摘要失败: {e}")
            return ""

    def retrieve(self, session_id: str) -> Optional[SessionSummary]:
        """
        根据 session_id 检索摘要

        Args:
            session_id: 会话 ID

        Returns:
            会话摘要，如果不存在返回 None
        """
        file_path = self.index.get(session_id)
        if not file_path or not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SessionSummary.from_dict(data)
        except Exception as e:
            print(f"⚠️ 读取摘要失败: {e}")
            return None

    def search(self, query: str, top_k: int = 10) -> List[SessionSummary]:
        """
        关键词检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            匹配的摘要列表
        """
        # 分词
        keywords = self._tokenize(query)

        # 查找匹配的 session_id
        matched_session_ids = set()
        for keyword in keywords:
            if keyword in self.inverted_index:
                matched_session_ids.update(self.inverted_index[keyword])

        # 加载匹配的摘要
        results = []
        for session_id in matched_session_ids:
            summary = self.retrieve(session_id)
            if summary:
                results.append(summary)

        # 按重要性排序
        results.sort(key=lambda s: s.importance, reverse=True)

        return results[:top_k]

    def get_all_session_ids(self) -> List[str]:
        """获取所有会话 ID"""
        return list(self.index.keys())

    def count(self) -> int:
        """获取摘要总数"""
        return len(self.index)

    def clear(self):
        """清空长期记忆（删除所有文件）"""
        import shutil

        # 删除所有摘要文件
        for file_path in self.index.values():
            if file_path.exists():
                file_path.unlink()

        # 清空索引
        self.index.clear()
        self.inverted_index.clear()

        # 保存空索引
        self._save_index()

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"LongTermMemory(count={len(self.index)}, dir={self.storage_dir})"