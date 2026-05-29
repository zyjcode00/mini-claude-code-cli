"""
三层记忆架构实现

实现 WorkingMemory / EpisodicMemory / LongTermMemory 三层记忆系统。
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .memory_models import SessionSummary
from .memory_items import MemoryItem, MemoryKind, MemoryRecallResult, MemoryStatus
from .memory_index import BM25MemoryDocument, MemoryIndexManager


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

        # MemoryItem 索引：{memory_item_id: file_path}
        self.item_index: Dict[str, Path] = {}

        # 倒排索引：{keyword: [session_ids]}
        self.inverted_index: Dict[str, List[str]] = {}

        # MemoryItem 倒排索引：{keyword: [memory_item_ids]}
        self.item_inverted_index: Dict[str, List[str]] = {}

        # Phase 2: 持久化 BM25 索引，随 store/store_item 增量更新。
        self.index_manager = MemoryIndexManager(self.storage_dir)

        # 初始化时加载索引
        self._load_index()

        # 如果 bm25.json 缺失/损坏/版本不一致，则从现有轻量 index.json 重建一次。
        if self.index_manager.needs_rebuild or len(self.index_manager) == 0:
            self.rebuild_search_index()

    def _load_index(self):
        """从磁盘加载索引"""
        index_file = self.storage_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index = {k: Path(v) for k, v in data.get("index", {}).items()}
                    self.item_index = {k: Path(v) for k, v in data.get("item_index", {}).items()}
                    self.inverted_index = data.get("inverted_index", {})
                    self.item_inverted_index = data.get("item_inverted_index", {})
            except Exception as e:
                # 如果加载失败，重建索引
                backup_path = self._backup_corrupt_index(index_file)
                backup_hint = f"，已备份损坏索引到: {backup_path}" if backup_path else ""
                print(f"⚠️ 加载索引失败，将重建索引: {e}{backup_hint}")
                self._rebuild_index()

    def _backup_corrupt_index(self, index_file: Path) -> str:
        """备份无法解析的索引文件，避免重建时直接覆盖现场。"""
        if not index_file.exists():
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = index_file.with_name(f"{index_file.name}.corrupt_{timestamp}.bak")
        try:
            index_file.replace(backup_path)
            return str(backup_path)
        except Exception as backup_error:
            print(f"⚠️ 备份损坏索引失败: {backup_error}")
            return ""

    def _serialize_index(self) -> Dict[str, Any]:
        """把内存索引转换为可持久化 JSON 的普通字典。"""
        return {
            "index": {k: str(v) for k, v in self.index.items()},
            "item_index": {k: str(v) for k, v in self.item_index.items()},
            "inverted_index": self.inverted_index,
            "item_inverted_index": self.item_inverted_index,
        }

    def _save_index(self):
        """保存索引到磁盘。

        Windows 下 os.replace 在目标文件被杀毒软件、编辑器、同步盘或另一个
        Python 进程短暂占用时可能抛 PermissionError(WinError 5)。索引只是缓存，
        单次保存失败不应影响启动/工具调用，因此这里采用：
        1. 内容未变化时跳过写入，降低启动期无意义 replace；
        2. PermissionError 短暂重试；
        3. 仍失败时保留一个 fallback 快照并清理临时文件，避免刷屏和 tmp 堆积。
        """
        index_file = self.storage_dir / "index.json"
        payload = self._serialize_index()

        try:
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            if index_file.exists():
                try:
                    if index_file.read_text(encoding='utf-8') == serialized:
                        return
                except OSError:
                    # 读旧索引失败时继续尝试写入；写入失败会走下面的降级逻辑。
                    pass

            fd, tmp_name = tempfile.mkstemp(
                prefix="index.",
                suffix=".tmp",
                dir=str(self.storage_dir),
                text=True,
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(serialized)
                    f.flush()
                    os.fsync(f.fileno())

                last_error = None
                for attempt in range(5):
                    try:
                        os.replace(tmp_path, index_file)
                        return
                    except PermissionError as replace_error:
                        last_error = replace_error
                        time.sleep(0.05 * (attempt + 1))

                # 降级：目标 index.json 暂时不可替换时，保留最新快照，下一次重建/保存仍可继续。
                fallback_file = self.storage_dir / "index.pending.json"
                try:
                    os.replace(tmp_path, fallback_file)
                    print(f"⚠️ 保存索引失败，已保留待恢复快照 {fallback_file}: {last_error}")
                except Exception:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                    print(f"⚠️ 保存索引失败: {last_error}")
            except Exception:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                raise
        except Exception as e:
            print(f"⚠️ 保存索引失败: {e}")

    def _rebuild_index(self):
        """重建索引"""
        self.index.clear()
        self.item_index.clear()
        self.inverted_index.clear()
        self.item_inverted_index.clear()

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

        # 遍历所有 MemoryItem 文件
        for item_file in self.storage_dir.glob("memory_item_*.json"):
            try:
                with open(item_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    item_id = data.get("id")
                    if item_id:
                        self.item_index[item_id] = item_file

                        keywords = self._extract_item_keywords(data)
                        for keyword in keywords:
                            if keyword not in self.item_inverted_index:
                                self.item_inverted_index[keyword] = []
                            if item_id not in self.item_inverted_index[keyword]:
                                self.item_inverted_index[keyword].append(item_id)
            except Exception as e:
                print(f"⚠️ 重建 MemoryItem 索引时跳过文件 {item_file}: {e}")

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

    def _extract_item_keywords(self, data: Dict[str, Any]) -> List[str]:
        """从 MemoryItem 数据中提取关键词。"""
        keywords = []

        for field_name in ("kind", "type", "title", "content", "project", "status"):
            keywords.extend(self._tokenize(str(data.get(field_name, ""))))

        for concept in data.get("concepts", []):
            keywords.extend(self._tokenize(str(concept)))

        for file_path in data.get("files", []):
            keywords.extend(self._tokenize(str(file_path)))

        for session_id in data.get("source_session_ids", []):
            keywords.extend(self._tokenize(str(session_id)))

        return list(set(keywords))

    def _summary_to_bm25_document(self, summary: SessionSummary) -> BM25MemoryDocument:
        """把 SessionSummary 映射为可持久化 BM25 文档。"""
        error_text = " ".join(
            f"{error.error_type} {error.error_message} {error.solution or ''}"
            for error in summary.errors_encountered
        )
        content_parts = [
            summary.summary_text,
            " ".join(summary.key_decisions),
            " ".join(fc.summary for fc in summary.files_changed),
            error_text,
            " ".join(tool.tool_name for tool in summary.tools_used),
        ]
        return BM25MemoryDocument(
            doc_id=f"summary_{summary.session_id}",
            title=summary.task_goal,
            content="\n".join(part for part in content_parts if part),
            concepts=summary.get_keywords(),
            files=summary.get_file_paths(),
            kind=MemoryKind.SUMMARY.value,
            error=error_text,
            metadata={"source_type": "summary_compat", "session_id": summary.session_id},
        )

    def _item_to_bm25_document(self, item: MemoryItem) -> BM25MemoryDocument:
        """把 MemoryItem 映射为可持久化 BM25 文档。"""
        raw_observation = item.metadata.get("raw_observation", {}) if item.metadata else {}
        error_parts = []
        if item.kind == MemoryKind.BUG.value:
            error_parts.extend(item.concepts)
        if isinstance(raw_observation, dict):
            error_parts.append(str(raw_observation.get("error") or raw_observation.get("tool_output") or ""))
        if item.metadata:
            error_parts.append(str(item.metadata.get("error_type", "")))
        return BM25MemoryDocument(
            doc_id=item.id,
            title=item.title,
            content=item.searchable_text(),
            concepts=list(item.concepts),
            files=list(item.files),
            kind=item.kind,
            error=" ".join(part for part in error_parts if part),
            project=item.project,
            metadata={"source_type": "long_term_items"},
        )

    def rebuild_search_index(self) -> None:
        """从轻量文件索引重建持久化 BM25 索引。"""
        documents: List[BM25MemoryDocument] = []
        for item_id in list(self.item_index.keys()):
            item = self.retrieve_item(item_id)
            if item and item.status == MemoryStatus.ACTIVE.value:
                documents.append(self._item_to_bm25_document(item))
        for session_id in list(self.index.keys()):
            summary = self.retrieve(session_id)
            if summary:
                documents.append(self._summary_to_bm25_document(summary))
        self.index_manager.rebuild(documents, force=True)

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
            self.index_manager.add_or_update(self._summary_to_bm25_document(summary), force=True)

            return str(file_path)
        except Exception as e:
            print(f"⚠️ 存储摘要失败: {e}")
            return ""

    def store_item(self, item: MemoryItem) -> str:
        """
        存储 MemoryItem 到长期记忆。

        Args:
            item: 长期知识条目

        Returns:
            存储的文件路径
        """
        timestamp = item.updated_at.replace(":", "-").replace(" ", "_")
        filename = f"memory_item_{item.id}_{timestamp}.json"
        file_path = self.storage_dir / filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(item.to_dict(), f, ensure_ascii=False, indent=2)

            old_file = self.item_index.get(item.id)
            if old_file and old_file != file_path and old_file.exists():
                old_file.unlink()

            self.item_index[item.id] = file_path
            self._remove_id_from_inverted_index(self.item_inverted_index, item.id)

            keywords = self._extract_item_keywords(item.to_dict())
            for keyword in keywords:
                if keyword not in self.item_inverted_index:
                    self.item_inverted_index[keyword] = []
                if item.id not in self.item_inverted_index[keyword]:
                    self.item_inverted_index[keyword].append(item.id)

            self._save_index()
            if item.status == MemoryStatus.ACTIVE.value:
                self.index_manager.add_or_update(self._item_to_bm25_document(item), force=True)
            else:
                self.index_manager.remove(item.id, force=True)
            return str(file_path)
        except Exception as e:
            print(f"⚠️ 存储 MemoryItem 失败: {e}")
            return ""

    def retrieve_item(self, item_id: str) -> Optional[MemoryItem]:
        """根据 item_id 检索 MemoryItem。"""
        file_path = self.item_index.get(item_id)
        if not file_path or not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return MemoryItem.from_dict(data)
        except Exception as e:
            print(f"⚠️ 读取 MemoryItem 失败: {e}")
            return None

    def search_items(self, query: str, top_k: int = 10, include_archived: bool = False) -> List[MemoryRecallResult]:
        """关键词检索 MemoryItem，返回统一召回结果。"""
        query_keywords = self._tokenize(query)
        if not query_keywords:
            return []

        scores: Dict[str, float] = {}
        matched_terms: Dict[str, List[str]] = {}
        for keyword in query_keywords:
            for item_id in self.item_inverted_index.get(keyword, []):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0
                matched_terms.setdefault(item_id, []).append(keyword)

        results: List[MemoryRecallResult] = []
        for item_id, keyword_score in scores.items():
            item = self.retrieve_item(item_id)
            if not item:
                continue
            if not include_archived and item.status != MemoryStatus.ACTIVE.value:
                continue

            normalized_score = keyword_score + item.importance + item.confidence * 0.5
            reason = "匹配关键词: " + ", ".join(sorted(set(matched_terms.get(item_id, []))))
            results.append(MemoryRecallResult(
                item=item,
                score=normalized_score,
                source="long_term_items",
                reason=reason,
            ))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def get_all_items(self) -> List[MemoryItem]:
        """获取所有 MemoryItem。"""
        items = []
        for item_id in list(self.item_index.keys()):
            item = self.retrieve_item(item_id)
            if item:
                items.append(item)
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items

    def _remove_id_from_inverted_index(self, inverted_index: Dict[str, List[str]], record_id: str) -> None:
        """从倒排索引中移除某个记录 ID，用于更新同 ID 文件。"""
        empty_keywords = []
        for keyword, record_ids in inverted_index.items():
            inverted_index[keyword] = [existing_id for existing_id in record_ids if existing_id != record_id]
            if not inverted_index[keyword]:
                empty_keywords.append(keyword)
        for keyword in empty_keywords:
            inverted_index.pop(keyword, None)

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

        # 删除所有 MemoryItem 文件
        for file_path in self.item_index.values():
            if file_path.exists():
                file_path.unlink()

        # 清空索引
        self.index.clear()
        self.item_index.clear()
        self.inverted_index.clear()
        self.item_inverted_index.clear()

        # 保存空索引
        self._save_index()
        self.index_manager.rebuild([], force=True)

    def __len__(self) -> int:
        return len(self.index) + len(self.item_index)

    def __repr__(self) -> str:
        return f"LongTermMemory(summaries={len(self.index)}, items={len(self.item_index)}, dir={self.storage_dir})"