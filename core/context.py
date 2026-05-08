# core/context.py
"""
上下文管理器 V2

Phase 1 增强功能：
- 结构化摘要（SessionSummary）
- 向后兼容旧版摘要

Phase 2 增强功能：
- 三层记忆架构（Working/Episodic/Long-term）
- 自动流转和淘汰

Phase 3 增强功能：
- 并发安全机制（asyncio.Lock 保护 messages 访问）
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid

# Phase 1 新增导入
from core.memory_models import SessionSummary, FileChange, ErrorRecord, ToolUsage
from core.prompts import SUMMARY_PROMPT_TEMPLATE_V2

# Phase 2 新增导入
from core.memory_layers import WorkingMemory, EpisodicMemory, LongTermMemory

# Phase 3 新增导入
from core.compression_engine import CompressionEngine, CompressionStrategy


class ContextManager:
    """
    上下文管理器

    Phase 1 增强功能：
    - 结构化摘要存储（session_summaries）
    - 兼容旧版纯文本摘要（history_summary）

    Phase 2 增强功能：
    - 三层记忆架构（working_memory / episodic_memory / long_term_memory）
    - 自动流转和淘汰

    Phase 3 增强功能：
    - 多策略压缩引擎（CompressionEngine）
    - 根据消息特征自动选择压缩策略
    - 支持 3 种压缩策略：LLM_SUMMARY/KEYFRAME/SLIDING_WINDOW
    """

    def __init__(self, max_history=100, min_keep=4, plan_manager=None):
        self.max_history = max_history
        self.min_keep = min_keep
        self.plan_manager = plan_manager  # 🔥 新增：保存 Plan 状态引用

        # 旧版摘要（纯文本，向后兼容）
        self.history_summary = ""

        # Phase 1: 结构化摘要列表（将被 Phase 2 替代，但保持向后兼容）
        self.session_summaries: List[SessionSummary] = []

        # Phase 2: 三层记忆架构
        self.working_memory = WorkingMemory(max_size=20)
        self.episodic_memory = EpisodicMemory(max_size=50)
        self.long_term_memory = LongTermMemory(storage_dir="memory/long_term")

        # 消息列表（保留，用于兼容）
        self.messages = []
        
        # 🔥🔥🔥 Phase 4 新增：并发锁，保护 messages 访问
        self.messages_lock = asyncio.Lock()

        # Phase 3: 压缩引擎（传入 plan_manager）
        self.compression_engine = CompressionEngine(plan_manager=plan_manager)

        # 会话 ID（用于生成摘要 ID）
        self._session_id_counter = 0

        # Phase 2: 是否启用三层记忆（默认启用）
        self._enable_memory_layers = True

    def add_message(self, message):
        """
        添加消息到上下文（Phase 2 增强版）

        流程：
        1. 添加到消息列表（兼容）
        2. 添加到工作记忆
        3. 如果工作记忆满了，自动触发压缩和流转
        
        ⚠️ 注意：这个方法是同步的，不进行锁保护。
           用于 Agent loop 中添加消息。
           对于 LLM 调用，请使用 get_messages_snapshot() 获取原子快照。
        """
        # 1. 添加到消息列表（保持兼容）
        self.messages.append(message)

        # 2. 添加到工作记忆
        if self._enable_memory_layers:
            evicted_message = self.working_memory.add(message)

            # 3. 如果工作记忆满了，触发异步压缩（需要在外部调用 compress）
            # 注意：这里不直接调用 compress，因为 compress 是异步的
            # 压缩会在 messages 超过 max_history 时自动触发

    async def get_messages_snapshot(self) -> List[Dict]:
        """
        🔥🔥🔥 获取当前消息的原子快照（并发安全）
        
        用于 LLM 调用时获取一致的消息列表。
        确保在快照期间，messages 不会被压缩修改。
        
        Returns:
            messages 的深拷贝
        """
        async with self.messages_lock:
            return [dict(msg) if isinstance(msg, dict) else msg for msg in self.messages]

    async def add_message_safe(self, message):
        """
        🔥🔥🔥 异步安全的消息添加（带锁保护）
        
        可选方法，用于需要严格并发控制的场景。
        当前 Agent loop 不需要使用此方法。
        
        Args:
            message: 消息字典
        """
        async with self.messages_lock:
            self.add_message(message)  # 调用同步方法

    async def _evict_from_working_memory(self, llm_summarizer_func):
        """
        从工作记忆淘汰消息，压缩为摘要

        Args:
            llm_summarizer_func: 异步回调函数，用于生成摘要

        Returns:
            bool: 是否成功流转
        """
        if not self._enable_memory_layers:
            return False

        # 获取工作记忆中被淘汰的消息
        evicted_messages = []
        while len(self.working_memory) > self.working_memory.max_size:
            # FIFO 淘汰
            if self.working_memory.data:
                evicted_messages.append(self.working_memory.data.pop(0))

        if not evicted_messages:
            return False

        print(f"\n[🧠 Phase 2] 从工作记忆淘汰 {len(evicted_messages)} 条消息")

        # 压缩为摘要
        summary = await self._compress_messages_to_summary(evicted_messages, llm_summarizer_func)

        if summary:
            # 添加到情景记忆
            evicted_summary = self.episodic_memory.add(summary)

            if evicted_summary:
                print(f"[🧠 Phase 2] 情景记忆已满，归档到长期记忆")
                # 归档到长期记忆
                self.long_term_memory.store(evicted_summary)

            # 兼容：添加到 session_summaries
            self.session_summaries.append(summary)

            return True

        return False

    async def _compress_messages_to_summary(
        self,
        messages: List[Dict[str, Any]],
        llm_summarizer_func
    ) -> Optional[SessionSummary]:
        """
        将消息压缩为摘要

        Args:
            messages: 消息列表
            llm_summarizer_func: 异步回调函数

        Returns:
            SessionSummary 实例，失败则返回 None
        """
        # 生成提示词
        prompt = SUMMARY_PROMPT_TEMPLATE_V2.format(
            existing_summary=self.history_summary or "无",
            messages_to_summarize=str(messages)
        )

        try:
            # 调用 LLM 生成摘要
            summary_text = await llm_summarizer_func(prompt)

            if not summary_text:
                return None

            # 解析结构化摘要
            summary = self._parse_structured_summary(summary_text, messages, 0.5)

            return summary

        except Exception as e:
            print(f"   [错误] 压缩消息失败: {str(e)}")
            return None

    def get_serializable_messages(self):
        """获取可序列化的消息列表"""
        serializable = []
        for msg in self.messages:
            if hasattr(msg, "model_dump"):
                serializable.append(msg.model_dump())
            elif isinstance(msg, dict):
                serializable.append(msg)
            else:
                serializable.append(dict(msg))
        return serializable

    async def compress(self, llm_summarizer_func, strategy: CompressionStrategy = None):
        """
        执行异步压缩逻辑（Phase 3 增强版）

        流程：
        1. 检查是否需要压缩（基于 messages 或 working_memory）
        2. 使用 CompressionEngine 选择压缩策略
        3. 执行压缩
        4. 存储摘要到三层记忆

        Args:
            llm_summarizer_func: 异步回调函数，接收 prompt，返回摘要文本
            strategy: 指定压缩策略（None 表示自动选择）

        Returns:
            bool: 是否执行了压缩
        """
        # 检查限制（支持 Phase 1, Phase 2, Phase 3）
        current_tokens = sum(len(str(msg)) for msg in self.messages)
        need_compress = current_tokens > self.max_history

        # Phase 2: 检查工作记忆是否需要压缩
        if self._enable_memory_layers and len(self.working_memory) >= self.working_memory.max_size:
            need_compress = True

        if not need_compress:
            return False

        print(f"\n[🧠 s06] 正在异步压缩上下文 (当前: {len(self.messages)}条)...")
        if self._enable_memory_layers:
            print(f"   [Phase 2] 工作记忆: {len(self.working_memory)}/{self.working_memory.max_size}")

        # Phase 3: 使用压缩引擎
        result = await self.compression_engine.compress(
            messages=self.messages,
            strategy=strategy,
            llm_summarizer_func=llm_summarizer_func,
            target_ratio=self.min_keep / len(self.messages) if self.messages else 0.3,
            min_keep=self.min_keep,
            existing_summary=self.history_summary
        )

        if not result.success:
            print("   [Phase 3] 压缩失败")
            return False

        # 更新消息列表
        self.messages = result.compressed_messages

        # Phase 2: 同步更新工作记忆
        if self._enable_memory_layers:
            # 清空工作记忆，只保留最新的消息
            self.working_memory.clear()
            for msg in result.compressed_messages:
                self.working_memory.add(msg)

        # 如果有结构化摘要，存储到三层记忆
        if result.summary:
            # Phase 2: 添加到情景记忆
            if self._enable_memory_layers:
                evicted_summary = self.episodic_memory.add(result.summary)

                if evicted_summary:
                    print(f"   [Phase 2] 情景记忆已满，归档到长期记忆: {evicted_summary.session_id}")
                    self.long_term_memory.store(evicted_summary)

                print(f"   [Phase 2] 摘要已添加到情景记忆: {result.summary.session_id}")

            # 兼容 Phase 1: 存储到 session_summaries
            self.session_summaries.append(result.summary)

            # 兼容旧版：也更新纯文本摘要
            self.history_summary = result.summary.summary_text
            print(f"   [Phase 3] 结构化摘要已保存: {result.summary.session_id}")
            print(f"   [Phase 3] 压缩比: {result.compression_ratio:.2%}")
            print(f"   [Phase 3] 任务: {result.summary.task_goal}")
            print(f"   [Phase 3] 状态: {result.summary.task_status}")

        return True

    def _parse_structured_summary(
        self,
        summary_text: str,
        messages: List[Any],
        avg_importance: float
    ) -> Optional[SessionSummary]:
        """
        解析 LLM 返回的结构化摘要

        Args:
            summary_text: LLM 返回的文本（可能包含 JSON）
            messages: 被压缩的消息列表
            avg_importance: 平均重要性分数

        Returns:
            SessionSummary 实例，如果解析失败则返回 None
        """
        try:
            # 尝试提取 JSON（可能包含 markdown 代码块）
            json_str = self._extract_json(summary_text)

            if not json_str:
                return None

            # 解析 JSON
            data = json.loads(json_str)

            # 验证必需字段
            required_fields = ["task_goal", "task_status", "summary_text"]
            for field in required_fields:
                if field not in data:
                    print(f"   [警告] 缺少必需字段: {field}")
                    return None

            # 生成会话 ID
            self._session_id_counter += 1
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._session_id_counter}"

            # 解析文件变更
            files_changed = []
            for fc_data in data.get("files_changed", []):
                try:
                    files_changed.append(FileChange.from_dict(fc_data))
                except Exception as e:
                    print(f"   [警告] 解析文件变更失败: {str(e)}")

            # 解析错误记录
            errors_encountered = []
            for er_data in data.get("errors_encountered", []):
                try:
                    errors_encountered.append(ErrorRecord.from_dict(er_data))
                except Exception as e:
                    print(f"   [警告] 解析错误记录失败: {str(e)}")

            # 解析工具使用
            tools_used = []
            for tu_data in data.get("tools_used", []):
                try:
                    tools_used.append(ToolUsage.from_dict(tu_data))
                except Exception as e:
                    print(f"   [警告] 解析工具使用失败: {str(e)}")

            # 创建 SessionSummary
            # 🔥🔥🔥 关键修复：优先使用真实 Plan 状态覆盖 LLM 推断
            task_status_from_llm = data["task_status"]
            real_task_status = task_status_from_llm  # 默认使用 LLM 推断

            if self.plan_manager and hasattr(self.plan_manager, 'is_plan_complete'):
                if self.plan_manager.is_plan_complete():
                    # Plan 已全部完成，强制覆盖 LLM 推断
                    real_task_status = "completed"
                    print(f"   [Phase 3] ✅ 覆盖 LLM 状态推断: {task_status_from_llm} → completed (Plan 已完成)")
                elif self.plan_manager.has_incomplete_tasks():
                    # Plan 还有未完成步骤，强制使用 "in_progress"
                    real_task_status = "in_progress"
                    print(f"   [Phase 3] ⏳ 覆盖 LLM 状态推断: {task_status_from_llm} → in_progress (Plan 未完成)")

            summary = SessionSummary(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                summary_text=data["summary_text"],
                task_goal=data["task_goal"],
                task_status=real_task_status,  # 🔥 使用真实状态（覆盖 LLM 推断）
                files_changed=files_changed,
                errors_encountered=errors_encountered,
                tools_used=tools_used,
                key_decisions=data.get("key_decisions", []),
                importance=data.get("importance", avg_importance),
                message_count=len(messages),
                token_count=sum(len(str(msg)) for msg in messages)
            )

            return summary

        except json.JSONDecodeError as e:
            print(f"   [警告] JSON 解析错误: {str(e)}")
            return None
        except Exception as e:
            print(f"   [错误] 解析结构化摘要失败: {str(e)}")
            return None

    def _extract_json(self, text: str) -> Optional[str]:
        """
        从文本中提取 JSON 字符串

        支持两种格式：
        1. 纯 JSON 字符串
        2. 包含在 markdown 代码块中的 JSON
        """
        text = text.strip()

        # 尝试直接解析（纯 JSON）
        if text.startswith("{"):
            return text

        # 尝试提取 markdown 代码块
        import re
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 尝试提取普通代码块
        pattern = r'```\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def check_tool_identity(self, msg):
        """检查消息是否为工具调用"""
        if isinstance(msg, dict):
            return msg.get("role") == "tool"
        return getattr(msg, "role", None) == "tool"

    # ==================== Phase 1 新增方法 ====================

    def get_structured_summaries(self) -> List[SessionSummary]:
        """获取所有结构化摘要"""
        return self.session_summaries

    def get_recent_files(self, limit: int = 10) -> List[str]:
        """
        获取最近操作的文件路径

        Args:
            limit: 返回的最大文件数

        Returns:
            文件路径列表（去重）
        """
        all_files = []
        for summary in reversed(self.session_summaries):
            all_files.extend(summary.get_file_paths())
            if len(all_files) >= limit:
                break

        # 去重并保持顺序
        seen = set()
        unique_files = []
        for file_path in all_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)

        return unique_files[:limit]

    def get_recent_errors(self, limit: int = 5) -> List[ErrorRecord]:
        """
        获取最近的错误记录

        Args:
            limit: 返回的最大错误数

        Returns:
            错误记录列表
        """
        all_errors = []
        for summary in reversed(self.session_summaries):
            all_errors.extend(summary.errors_encountered)
            if len(all_errors) >= limit:
                break

        return all_errors[:limit]

    def search_summaries(self, keyword: str) -> List[SessionSummary]:
        """
        在摘要中搜索关键词

        Args:
            keyword: 搜索关键词

        Returns:
            包含关键词的摘要列表
        """
        results = []
        keyword_lower = keyword.lower()

        for summary in self.session_summaries:
            # 搜索摘要文本
            if keyword_lower in summary.summary_text.lower():
                results.append(summary)
                continue

            # 搜索任务目标
            if keyword_lower in summary.task_goal.lower():
                results.append(summary)
                continue

            # 搜索文件路径
            for file_change in summary.files_changed:
                if keyword_lower in file_change.path.lower():
                    results.append(summary)
                    break

            # 搜索错误类型
            for error in summary.errors_encountered:
                if keyword_lower in error.error_type.lower():
                    results.append(summary)
                    break

        return results

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取摘要统计信息

        Returns:
            统计信息字典
        """
        if not self.session_summaries:
            return {
                "total_summaries": 0,
                "total_messages": 0,
                "total_tokens": 0,
                "avg_importance": 0.0
            }

        total_messages = sum(s.message_count for s in self.session_summaries)
        total_tokens = sum(s.token_count for s in self.session_summaries)
        avg_importance = sum(s.importance for s in self.session_summaries) / len(self.session_summaries)

        return {
            "total_summaries": len(self.session_summaries),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "avg_importance": round(avg_importance, 2),
            "total_files_changed": sum(len(s.files_changed) for s in self.session_summaries),
            "total_errors": sum(len(s.errors_encountered) for s in self.session_summaries)
        }

    # ==================== Phase 2 新增方法 ====================

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        获取三层记忆统计信息

        Returns:
            统计信息字典
        """
        if not self._enable_memory_layers:
            return self.get_summary_statistics()

        return {
            "working_memory": {
                "size": len(self.working_memory),
                "max_size": self.working_memory.max_size,
                "usage_rate": len(self.working_memory) / self.working_memory.max_size
            },
            "episodic_memory": {
                "size": len(self.episodic_memory),
                "max_size": self.episodic_memory.max_size,
                "usage_rate": len(self.episodic_memory) / self.episodic_memory.max_size
            },
            "long_term_memory": {
                "count": len(self.long_term_memory),
                "storage_dir": str(self.long_term_memory.storage_dir)
            }
        }

    def search_all_memories(self, query: str, top_k: int = 5) -> List[SessionSummary]:
        """
        在三层记忆中检索摘要

        Args:
            query: 查询关键词
            top_k: 返回结果数量

        Returns:
            匹配的摘要列表
        """
        results = []

        if not self._enable_memory_layers:
            return self.search_summaries(query)[:top_k]

        # 1. 检索情景记忆
        episodic_results = self.episodic_memory.search(query, top_k)
        results.extend(episodic_results)

        # 2. 检索长期记忆
        if len(results) < top_k:
            long_term_results = self.long_term_memory.search(query, top_k - len(results))
            results.extend(long_term_results)

        # 3. 去重（基于 session_id）
        seen_ids = set()
        unique_results = []
        for summary in results:
            if summary.session_id not in seen_ids:
                seen_ids.add(summary.session_id)
                unique_results.append(summary)

        return unique_results[:top_k]

    def get_recent_summaries_from_episodic(self, n: int = 5) -> List[SessionSummary]:
        """
        从情景记忆获取最近的摘要

        Args:
            n: 返回数量

        Returns:
            摘要列表
        """
        if not self._enable_memory_layers:
            return self.session_summaries[-n:] if n < len(self.session_summaries) else self.session_summaries

        return self.episodic_memory.get_recent(n)

    def clear_all_memories(self):
        """清空所有三层记忆"""
        if self._enable_memory_layers:
            self.working_memory.clear()
            self.episodic_memory.clear()
            self.long_term_memory.clear()

        # 兼容 Phase 1
        self.session_summaries.clear()
        self.history_summary = ""
        self.messages.clear()

        print("[Phase 2] 所有记忆已清空")

    def export_memories(self) -> Dict[str, Any]:
        """
        导出所有记忆为可序列化的字典（用于持久化到 session 文件）

        Returns:
            包含三层记忆数据的字典
        """
        if not self._enable_memory_layers:
            return {
                "working_memory": [],
                "episodic_memory": [],
                "session_summaries": [s.to_dict() for s in self.session_summaries],
                "history_summary": self.history_summary
            }

        # 导出工作记忆
        working_memory_data = self.working_memory.get_all()

        # 导出情景记忆
        episodic_memory_data = [s.to_dict() for s in self.episodic_memory.get_all()]

        # 导出会话摘要（Phase 1 兼容）
        session_summaries_data = [s.to_dict() for s in self.session_summaries]

        return {
            "working_memory": working_memory_data,
            "episodic_memory": episodic_memory_data,
            "session_summaries": session_summaries_data,
            "history_summary": self.history_summary
        }

    def import_memories(self, data: Dict[str, Any]):
        """
        从字典导入记忆数据（用于从 session 文件恢复）

        Args:
            data: 包含记忆数据的字典
        """
        from .memory_models import SessionSummary

        # 恢复历史摘要
        self.history_summary = data.get("history_summary", "")

        # 恢复会话摘要（Phase 1 兼容）
        summaries_data = data.get("session_summaries", [])
        self.session_summaries = []
        for s_dict in summaries_data:
            try:
                summary = SessionSummary.from_dict(s_dict)
                self.session_summaries.append(summary)
            except Exception as e:
                print(f"⚠️ 恢复会话摘要失败: {e}")

        # 恢复三层记忆（Phase 2）
        if not self._enable_memory_layers:
            print("[Phase 1] 已恢复历史摘要和会话摘要")
            return

        # 恢复工作记忆
        working_memory_data = data.get("working_memory", [])
        self.working_memory.clear()
        for msg in working_memory_data:
            self.working_memory.add(msg)

        # 恢复情景记忆
        episodic_memory_data = data.get("episodic_memory", [])
        self.episodic_memory.clear()
        for s_dict in episodic_memory_data:
            try:
                summary = SessionSummary.from_dict(s_dict)
                self.episodic_memory.add(summary)
            except Exception as e:
                print(f"⚠️ 恢复情景记忆失败: {e}")

        print(f"[Phase 2] 已恢复三层记忆:")
        print(f"   - 工作记忆: {len(self.working_memory)} 条")
        print(f"   - 情景记忆: {len(self.episodic_memory.get_all())} 条")
        print(f"   - 会话摘要: {len(self.session_summaries)} 条")

    def export_memories_to_files(self, output_dir: str = "memory/export"):
        """
        导出所有记忆到独立文件（用于备份/迁移）

        Args:
            output_dir: 输出目录
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 导出情景记忆
        episodic_file = output_path / "episodic_memory.json"
        episodic_data = [s.to_dict() for s in self.episodic_memory.get_all()]

        with open(episodic_file, 'w', encoding='utf-8') as f:
            json.dump(episodic_data, f, ensure_ascii=False, indent=2)

        print(f"[Phase 2] 情景记忆已导出: {episodic_file}")
        print(f"   - 摘要数量: {len(episodic_data)}")

        # 导出统计信息
        stats = self.get_memory_statistics()
        stats_file = output_path / "memory_statistics.json"

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"[Phase 2] 统计信息已导出: {stats_file}")