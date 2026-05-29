# core/compression_engine.py
"""
多策略压缩引擎

Phase 3 增强功能：
- 支持多种压缩策略（LLM_SUMMARY/KEYFRAME/SLIDING_WINDOW/IMPORTANCE_FILTER）
- 根据消息特征智能选择策略
- 可配置压缩比
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json
import hashlib
import time
from datetime import datetime

from core.memory_models import SessionSummary, FileChange, ErrorRecord, ToolUsage
from core.prompts import SUMMARY_PROMPT_TEMPLATE_V2
from core.turn_builder import ConversationTurn, TurnBuilder


class CompressionStrategy(Enum):
    """压缩策略枚举"""
    LLM_SUMMARY = "llm_summary"           # 调用 LLM 生成摘要
    KEYFRAME = "keyframe"                # 提取关键帧（工具调用、错误消息）
    SLIDING_WINDOW = "sliding_window"    # 滑动窗口保留最近消息


@dataclass
class CompressionResult:
    """压缩结果"""
    success: bool
    strategy: CompressionStrategy
    compressed_messages: List[Dict[str, Any]]
    summary: Optional[SessionSummary] = None
    compression_ratio: float = 0.0
    original_count: int = 0
    compressed_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CompressionEngine:
    """
    多策略压缩引擎

    支持 3 种压缩策略：
    1. LLM_SUMMARY: 调用 LLM 生成结构化摘要
    2. KEYFRAME: 提取关键帧（工具调用、错误消息）
    3. SLIDING_WINDOW: 滑动窗口保留最近消息（默认、最可靠）
    """

    def __init__(self, default_strategy: CompressionStrategy = None, plan_manager=None):
        """
        初始化压缩引擎

        Args:
            default_strategy: 默认压缩策略（None 表示自动选择）
            plan_manager: Plan 状态管理器（用于查询真实任务状态）
        """
        self.default_strategy = default_strategy
        self.plan_manager = plan_manager  # Plan 状态管理器

        # 策略选择阈值
        self.error_dense_threshold = 0.3      # 错误消息占比 > 30% → LLM_SUMMARY
        self.tool_dense_threshold = 0.4       # 工具调用占比 > 40% → KEYFRAME

        # 🔥🔥🔥 Phase 2: LLM_SUMMARY 优化配置
        self.summary_cache = {}                # 缓存已有的摘要
        self.llm_call_times = []               # 记录 LLM 调用时间（用于频率限制）
        self.llm_call_limit = 10               # 每小时最多 10 次 LLM 调用
        self.llm_call_window = 3600            # 时间窗口：1 小时（秒）
        self.cache_max_size = 100              # 缓存最大条目数

    async def compress(
        self,
        messages: List[Dict[str, Any]],
        strategy: CompressionStrategy = None,
        llm_summarizer_func: Optional[Callable] = None,
        target_ratio: float = 0.3,
        min_keep: int = 4,
        existing_summary: str = ""
    ) -> CompressionResult:
        """
        执行压缩

        Args:
            messages: 待压缩的消息列表
            strategy: 压缩策略（None 表示自动选择）
            llm_summarizer_func: LLM 摘要函数（LLM_SUMMARY 策略需要）
            target_ratio: 目标压缩比（保留消息的比例）
            min_keep: 最少保留的消息数量
            existing_summary: 已有的摘要文本

        Returns:
            CompressionResult: 压缩结果
        """
        if not messages:
            return CompressionResult(
                success=False,
                strategy=strategy or CompressionStrategy.SLIDING_WINDOW,
                compressed_messages=[],
                original_count=0,
                compressed_count=0
            )

        # 如果没有指定策略，自动选择
        if strategy is None:
            if self.default_strategy:
                strategy = self.default_strategy
            else:
                strategy = self._select_strategy(messages)

        print(f"\n[🧠 压缩引擎] 选择策略: {strategy.value}")
        print(f"   目标压缩比: {target_ratio:.2%}")
        print(f"   原始消息数: {len(messages)}")

        # 执行对应策略
        if strategy == CompressionStrategy.LLM_SUMMARY:
            return await self._compress_with_llm(
                messages, llm_summarizer_func, existing_summary, min_keep
            )
        elif strategy == CompressionStrategy.KEYFRAME:
            return self._compress_with_keyframe(messages, target_ratio, min_keep)
        elif strategy == CompressionStrategy.SLIDING_WINDOW:
            return self._compress_with_sliding_window(messages, target_ratio, min_keep)
        else:
            raise ValueError(f"未知的压缩策略: {strategy}")

    def _select_strategy(self, messages: List[Dict[str, Any]]) -> CompressionStrategy:
        """
        根据消息特征自动选择压缩策略

        决策规则：
        1. 错误密集（错误占比 > 30%）→ LLM_SUMMARY
        2. 工具密集（工具调用占比 > 40%）→ KEYFRAME
        3. 其他情况 → SLIDING_WINDOW（最可靠）

        Args:
            messages: 消息列表

        Returns:
            CompressionStrategy: 选择的策略
        """
        if not messages:
            return CompressionStrategy.SLIDING_WINDOW

        # 统计消息特征
        error_count = 0
        tool_count = 0

        for msg in messages:
            # 统计错误消息
            content = str(msg.get("content", ""))
            if any(keyword in content for keyword in ["Error", "错误", "Exception", "失败", "Traceback"]):
                error_count += 1

            # 统计工具调用
            if msg.get("role") == "tool":
                tool_count += 1

        # 计算占比
        total = len(messages)
        error_ratio = error_count / total
        tool_ratio = tool_count / total

        print(f"   [策略选择] 错误占比: {error_ratio:.2%}, 工具占比: {tool_ratio:.2%}")

        # 决策逻辑
        if error_ratio > self.error_dense_threshold:
            print(f"   [策略选择] 错误密集 → LLM_SUMMARY")
            return CompressionStrategy.LLM_SUMMARY

        if tool_ratio > self.tool_dense_threshold:
            print(f"   [策略选择] 工具密集 → KEYFRAME")
            return CompressionStrategy.KEYFRAME

        print(f"   [策略选择] 一般对话 → SLIDING_WINDOW")
        return CompressionStrategy.SLIDING_WINDOW

    # ========== 策略 1: LLM 摘要 ==========

    async def _compress_with_llm(
        self,
        messages: List[Dict[str, Any]],
        llm_summarizer_func: Optional[Callable],
        existing_summary: str,
        min_keep: int
    ) -> CompressionResult:
        """
        使用 LLM 生成摘要（当前策略）

        🔥 Phase 2 优化：
        - 添加缓存机制（避免重复调用）
        - 添加频率限制（每小时最多 10 次）
        - 超限时降级到 SLIDING_WINDOW

        Args:
            messages: 消息列表
            llm_summarizer_func: LLM 摘要函数
            existing_summary: 已有的摘要
            min_keep: 最少保留的消息数量

        Returns:
            CompressionResult
        """
        if not llm_summarizer_func:
            print("   [警告] LLM_SUMMARY 策略需要 llm_summarizer_func，降级到 SLIDING_WINDOW")
            return self._compress_with_sliding_window(messages, 0.3, min_keep)

        # 划分消息：最近完整 turn 原样保留，较旧完整 turn 作为摘要候选。
        keep_messages = self._select_recent_complete_turn_messages(messages, min_keep)
        keep_start = len(messages)
        if keep_messages:
            keep_start = min(
                messages.index(msg)
                for msg in keep_messages
                if msg in messages
            )
        to_summarize = self._select_summary_candidate_messages(messages[:keep_start])

        # 🔥 Phase 2: 检查缓存
        messages_hash = self._get_messages_hash(to_summarize)
        if messages_hash in self.summary_cache:
            print(f"   ✅ 命中缓存：直接使用已有摘要")
            cached_summary = self.summary_cache[messages_hash]
            return CompressionResult(
                success=True,
                strategy=CompressionStrategy.LLM_SUMMARY,
                compressed_messages=keep_messages,
                summary=cached_summary,
                compression_ratio=len(keep_messages) / len(messages) if messages else 0,
                original_count=len(messages),
                compressed_count=len(keep_messages),
                metadata={"from_cache": True, "cache_hit": messages_hash}
            )

        # 🔥 Phase 2: 检查频率限制
        if not self._should_call_llm():
            print(f"   ⚠️  LLM 调用超限，降级到 SLIDING_WINDOW")
            return self._compress_with_sliding_window(messages, 0.3, min_keep)

        # 生成摘要
        prompt = SUMMARY_PROMPT_TEMPLATE_V2.format(
            existing_summary=existing_summary or "无",
            messages_to_summarize=str(to_summarize)
        )

        try:
            summary_text = await llm_summarizer_func(prompt)

            # 🔥 Phase 2: 记录调用
            self._record_llm_call()

            if not summary_text:
                return CompressionResult(
                    success=False,
                    strategy=CompressionStrategy.LLM_SUMMARY,
                    compressed_messages=keep_messages,
                    original_count=len(messages),
                    compressed_count=len(keep_messages)
                )

            # 解析结构化摘要
            summary = self._parse_structured_summary(
                summary_text, to_summarize, 0.5
            )

            # 🔥 Phase 2: 缓存结果
            if len(self.summary_cache) >= self.cache_max_size:
                # 如果缓存满了，删除最早的条目
                first_key = next(iter(self.summary_cache))
                del self.summary_cache[first_key]
                print(f"   🗑️  缓存已满，删除最早条目")

            self.summary_cache[messages_hash] = summary
            print(f"   💾 摘要已缓存，当前缓存大小: {len(self.summary_cache)}/{self.cache_max_size}")

            return CompressionResult(
                success=True,
                strategy=CompressionStrategy.LLM_SUMMARY,
                compressed_messages=keep_messages,
                summary=summary,
                compression_ratio=len(keep_messages) / len(messages) if messages else 0,
                original_count=len(messages),
                compressed_count=len(keep_messages),
                metadata={"summary_text": summary_text, "from_cache": False}
            )

        except Exception as e:
            print(f"   [错误] LLM 摘要失败: {str(e)}")
            return CompressionResult(
                success=False,
                strategy=CompressionStrategy.LLM_SUMMARY,
                compressed_messages=keep_messages,
                original_count=len(messages),
                compressed_count=len(keep_messages),
                metadata={"error": str(e)}
            )

    # ========== 策略 2: 关键帧提取 ==========

    def _compress_with_keyframe(
        self,
        messages: List[Dict[str, Any]],
        target_ratio: float,
        min_keep: int
    ) -> CompressionResult:
        """
        提取关键帧（工具调用、错误消息、用户消息）

        策略：
        1. 提取所有用户消息
        2. 提取所有错误消息
        3. 提取工具调用及其结果
        4. 如果还有空间，按重要性补充

        Args:
            messages: 消息列表
            target_ratio: 目标压缩比
            min_keep: 最少保留的消息数量

        Returns:
            CompressionResult
        """
        target_count = max(min_keep, int(len(messages) * target_ratio))
        builder = TurnBuilder()
        turns = builder.complete_turns_only(builder.build(messages))
        selected_turns = self._select_keyframe_turns(turns, target_count, min_keep)
        compressed_messages = builder.flatten(selected_turns)

        # 确保至少保留 min_keep 条消息。补齐时仍按完整 turn 选择，避免切断 tool pair。
        if len(compressed_messages) < min_keep:
            if len(messages) <= min_keep:
                compressed_messages = self._sanitize_openai_tool_pairs(messages[:])
            else:
                recent_messages = self._select_recent_complete_turn_messages(messages, min_keep)
                merged = compressed_messages + [msg for msg in recent_messages if msg not in compressed_messages]
                compressed_messages = self._sanitize_openai_tool_pairs(merged)

        print(f"   [关键帧提取] 提取了 {len(compressed_messages)} 条关键消息（基于 turn 元数据）")

        # 🔥 关键验证：检查最终的消息序列是否有效
        is_valid = self._validate_message_ordering(compressed_messages)
        if not is_valid:
            print(f"   ⚠️  警告: 关键帧提取后消息序列无效，降级到 SLIDING_WINDOW")
            # 降级到 SLIDING_WINDOW 策略
            return self._compress_with_sliding_window(messages, 0.3, min_keep)

        selected_ids = {turn.id for turn in selected_turns}
        selected_categories = sorted({category for turn in selected_turns for category in turn.categories})
        avg_importance = (
            sum(turn.importance for turn in selected_turns) / len(selected_turns)
            if selected_turns else 0.5
        )

        # 生成简化版摘要
        summary = self._generate_summary_from_messages(
            compressed_messages, CompressionStrategy.KEYFRAME, avg_importance
        )

        return CompressionResult(
            success=True,
            strategy=CompressionStrategy.KEYFRAME,
            compressed_messages=compressed_messages,
            summary=summary,
            compression_ratio=len(compressed_messages) / len(messages) if messages else 0,
            original_count=len(messages),
            compressed_count=len(compressed_messages),
            metadata={
                "keyframe_count": len(compressed_messages),
                "selected_turn_ids": list(selected_ids),
                "selected_categories": selected_categories,
                "avg_turn_importance": avg_importance,
                "user_messages": sum(1 for msg in compressed_messages if msg.get("role") == "user"),
                "error_messages": sum(1 for turn in selected_turns if "error" in turn.categories),
                "tool_messages": sum(1 for msg in compressed_messages if msg.get("role") == "tool"),
                "files_touched": sorted({path for turn in selected_turns for path in turn.files_touched})[:20],
            }
        )

    # ========== 策略 3: 滑动窗口 ==========

    def _compress_with_sliding_window(
        self,
        messages: List[Dict[str, Any]],
        target_ratio: float,
        min_keep: int
    ) -> CompressionResult:
        """
        滑动窗口保留最近消息

        策略：
        1. 保留最近 target_ratio 比例的消息
        2. 确保不截断工具调用（重要修复）
        3. 至少保留 min_keep 条消息

        Args:
            messages: 消息列表
            target_ratio: 目标压缩比
            min_keep: 最少保留的消息数量

        Returns:
            CompressionResult
        """
        target_count = max(min_keep, int(len(messages) * target_ratio))
        compressed_messages = self._select_recent_complete_turn_messages(messages, target_count)

        if len(compressed_messages) < min_keep and len(messages) <= min_keep:
            compressed_messages = self._sanitize_openai_tool_pairs(messages[:])

        print(f"   [滑动窗口] 保留最近 {len(compressed_messages)} 条消息（基于完整 turn）")

        # 🔥 关键验证：检查最终的消息序列是否有效
        is_valid = self._validate_message_ordering(compressed_messages)
        if not is_valid:
            print(f"   ⚠️  警告: 压缩后的消息序列无效，执行安全清理")
            compressed_messages = self._sanitize_openai_tool_pairs(compressed_messages)

        # 生成简化版摘要
        summary = self._generate_summary_from_messages(
            compressed_messages, CompressionStrategy.SLIDING_WINDOW, 0.5
        )

        return CompressionResult(
            success=True,
            strategy=CompressionStrategy.SLIDING_WINDOW,
            compressed_messages=compressed_messages,
            summary=summary,
            compression_ratio=len(compressed_messages) / len(messages) if messages else 0,
            original_count=len(messages),
            compressed_count=len(compressed_messages),
            metadata={"window_size": len(compressed_messages)}
        )

    # ========== 策略 4: 重要性过滤 ==========

    # ========== LLM_SUMMARY 相关方法 ==========

    def _get_messages_hash(self, messages: List[Dict[str, Any]]) -> str:
        """
        计算消息列表的哈希值（用于缓存）

        Args:
            messages: 消息列表

        Returns:
            哈希值字符串
        """
        # 简化方案：只对消息内容进行哈希
        content = json.dumps([
            {
                "role": msg.get("role"),
                "content": str(msg.get("content", ""))[:500]  # 只取前500字符
            }
            for msg in messages
        ], ensure_ascii=False, sort_keys=True)

        return hashlib.md5(content.encode()).hexdigest()

    def _should_call_llm(self) -> bool:
        """
        检查是否应该调用 LLM（频率限制）

        🔥 Phase 2 优化：每小时最多 10 次 LLM 调用

        Returns:
            True 如果可以调用，False 如果超过限制
        """
        current_time = time.time()

        # 清除时间窗口外的记录
        self.llm_call_times = [
            t for t in self.llm_call_times
            if current_time - t < self.llm_call_window
        ]

        # 检查是否超过限制
        if len(self.llm_call_times) >= self.llm_call_limit:
            print(f"   ⚠️  LLM 调用频率限制：已在最近 {self.llm_call_window}s 内调用 {len(self.llm_call_times)} 次，上限 {self.llm_call_limit}")
            return False

        return True

    def _record_llm_call(self):
        """
        记录一次 LLM 调用（用于频率限制）
        """
        self.llm_call_times.append(time.time())
        print(f"   📝 LLM 调用已记录：本小时已调用 {len(self.llm_call_times)}/{self.llm_call_limit} 次")

    # ========== 辅助方法 ==========

    def _generate_summary_from_messages(
        self,
        messages: List[Dict[str, Any]],
        strategy: CompressionStrategy,
        avg_importance: float
    ) -> SessionSummary:
        """
        从消息列表生成简化版的 SessionSummary（用于非 LLM 策略）

        Args:
            messages: 被压缩的消息列表
            strategy: 使用的压缩策略
            avg_importance: 平均重要性分数

        Returns:
            SessionSummary 实例
        """
        # 提取文件变更
        files_changed = []
        for msg in messages:
            content = str(msg.get("content", ""))
            # 检测文件操作
            if any(keyword in content for keyword in ["创建", "修改", "删除", "created", "modified", "deleted"]):
                # 简单提取文件路径
                import re
                file_patterns = [
                    r'([a-zA-Z0-9_\-/\\]+\.(py|js|ts|java|cpp|c|h|json|md|txt))',
                    r'文件[:\s]+([a-zA-Z0-9_\-/\\]+)',
                    r'file[:\s]+([a-zA-Z0-9_\-/\\]+)'
                ]
                for pattern in file_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        path = match[0] if isinstance(match, tuple) else match
                        files_changed.append(FileChange(
                            path=path,
                            action="modified",
                            summary=f"文件操作（{strategy.value}策略）",
                            importance=avg_importance
                        ))

        # 提取错误记录
        errors_encountered = []
        for i, msg in enumerate(messages):
            content = str(msg.get("content", ""))
            if any(keyword in content for keyword in ["Error", "错误", "Exception", "失败", "Traceback"]):
                errors_encountered.append(ErrorRecord(
                    error_type="unknown",
                    error_message=content[:200],  # 截取前200字符
                    timestamp=datetime.now().isoformat()
                ))

        # 提取工具使用
        tools_used = []
        tool_counts = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tool_name = msg.get("name", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        for tool_name, count in tool_counts.items():
            tools_used.append(ToolUsage(
                tool_name=tool_name,
                parameters={},
                result_summary=f"调用了 {count} 次",
                timestamp=datetime.now().isoformat(),
                success=True,
                importance=0.5
            ))

        # 生成摘要文本（改进版：提取关键帧的实际内容）
        summary_parts = []
        task_goal = None
        key_decisions = []

        # 1. 提取用户消息（任务目标）
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if user_messages:
            first_user_msg = user_messages[0].get("content", "")
            if isinstance(first_user_msg, str):
                task_goal = first_user_msg[:100]  # 提取前100字符作为任务目标
                summary_parts.append(f"任务: {task_goal}")

        # 2. 提取关键消息的实际内容
        key_content_count = 0
        for msg in messages:
            if key_content_count >= 5:  # 只提取前5条关键内容
                break

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            role = msg.get("role", "")

            # 提取错误消息
            if any(keyword in content for keyword in ["Error", "错误", "Exception", "失败", "Traceback"]):
                error_preview = content[:80].replace("\n", " ").strip()
                summary_parts.append(f"错误: {error_preview}...")
                key_content_count += 1

            # 提取关键决策（assistant 消息中的关键内容）
            elif role == "assistant" and len(content) > 20:
                # 检测关键决策关键词
                decision_keywords = ["已完成", "已创建", "已修改", "成功", "完成", "建议", "决策", "选择"]
                if any(keyword in content for keyword in decision_keywords):
                    decision_preview = content[:80].replace("\n", " ").strip()
                    summary_parts.append(f"进展: {decision_preview}...")
                    key_decisions.append(decision_preview)
                    key_content_count += 1

        # 3. 统计数据作为补充
        stats_parts = []
        stats_parts.append(f"压缩了 {len(messages)} 条消息")
        if files_changed:
            stats_parts.append(f"{len(files_changed)} 个文件变更")
        if errors_encountered:
            stats_parts.append(f"{len(errors_encountered)} 个错误")
        if tools_used:
            stats_parts.append(f"使用 {len(tools_used)} 种工具")

        summary_text = "\n".join(summary_parts + ["统计: " + "，".join(stats_parts)])

        # 如果没有提取到任何关键内容，使用简化版摘要
        if not summary_parts:
            summary_text = f"使用{strategy.value}策略压缩了 {len(messages)} 条消息（无关键内容提取）"

        # 🔥 新增：检测任务状态（优先使用真实 Plan 状态）
        task_status = "in_progress"  # 默认值

        # 🔥🔥🔥 关键修复：优先检查 Plan 状态（真实数据源）
        if self.plan_manager and hasattr(self.plan_manager, 'is_plan_complete'):
            if self.plan_manager.is_plan_complete():
                # Plan 已全部完成，强制使用 "completed" 状态
                task_status = "completed"
                print(f"   [Phase 3] ✅ 检测到 Plan 已完成，状态同步为 completed")
            elif self.plan_manager.has_incomplete_tasks():
                # Plan 还有未完成的步骤，强制使用 "in_progress"
                task_status = "in_progress"
                print(f"   [Phase 3] ⏳ Plan 尚未完成，状态保持 in_progress")
        else:
            # 没有 plan_manager，使用推断逻辑（备用）
            if errors_encountered:
                unresolved_errors = [
                    e for e in errors_encountered
                    if not (e.get("resolved", False) if isinstance(e, dict) else e.resolved)
                ]
                if unresolved_errors:
                    task_status = "in_progress"
                else:
                    task_status = "completed"
            else:
                task_status = "completed"

        return SessionSummary(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(messages)}",
            timestamp=datetime.now().isoformat(),
            summary_text=summary_text,
            task_goal=task_goal or f"压缩策略: {strategy.value}",
            task_status=task_status,  # 🔥 使用检测到的状态
            files_changed=files_changed[:10],  # 限制数量
            errors_encountered=errors_encountered[:5],
            tools_used=tools_used[:10],
            key_decisions=key_decisions if key_decisions else [f"使用{strategy.value}策略进行压缩"],
            importance=avg_importance,
            message_count=len(messages),
            token_count=sum(len(str(msg)) for msg in messages)
        )

    def _is_tool_call_start(self, message: Dict[str, Any]) -> bool:
        """
        判断是否是工具调用的开始（即 assistant 消息包含 tool_calls）

        Args:
            message: 消息对象

        Returns:
            bool: 是否是工具调用开始
        """
        # 如果是 tool 消息，需要前一条 assistant 消息
        if message.get("role") == "tool":
            return True

        # 如果是 assistant 消息包含 tool_calls
        if message.get("role") == "assistant" and "tool_calls" in message:
            return True

        return False

    def _validate_message_ordering(self, messages: List[Dict[str, Any]]) -> bool:
        """
        🔥 新增验证：检查消息序列的工具调用完整性

        确保：
        1. 每个 'tool' 消息都有对应的前置 'assistant' 消息包含 tool_calls
        2. 每个包含 tool_calls 的 'assistant' 消息都有对应的后续 'tool' 消息

        🔥🔥🔥 修复：GPT-5.5 严格要求双向验证，GLM-5 可能宽松

        Args:
            messages: 消息列表

        Returns:
            bool: 消息序列是否有效
        """
        for i, msg in enumerate(messages):
            # 检查 1: tool 消息是否有对应的相邻 assistant(tool_calls)
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if not tool_call_id:
                    print(f"   ❌ 错误: Tool 消息 [{i}] 缺少 tool_call_id")
                    return False

                # OpenAI strict tools 要求 tool 消息必须位于 assistant(tool_calls) 后的连续 tool 响应块中。
                block_start = i
                while block_start > 0 and messages[block_start - 1].get("role") == "tool":
                    block_start -= 1
                prev_msg = messages[block_start - 1] if block_start > 0 else None
                if not (prev_msg and prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls")):
                    print(f"   ❌ 错误: Tool 消息 [{i}] 不在 assistant(tool_calls) 后的连续响应块中")
                    return False

                expected_ids = {tc.get("id") for tc in prev_msg.get("tool_calls", []) if tc.get("id")}
                if tool_call_id not in expected_ids:
                    print(f"   ❌ 错误: Tool 消息 [{i}] (tool_call_id={tool_call_id}) 没有对应的相邻 assistant with tool_calls")
                    return False

            # 🔥🔥🔥 检查 2: assistant with tool_calls 是否有对应的 tool 消息（向后查找）
            # 这是 GPT-5.5 严格要求的关键检查！
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])

                for tc in tool_calls:
                    tool_call_id = tc.get("id")
                    if not tool_call_id:
                        continue

                    # 向后检查紧邻的连续 tool 响应块，不能跨过 user/assistant/system 等其它消息查找。
                    expected_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                    contiguous_tool_ids = set()
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        contiguous_tool_ids.add(messages[j].get("tool_call_id"))
                        j += 1
                    found_tool = tool_call_id in contiguous_tool_ids

                    # 🔥🔥🔥 GPT-5.5 严格要求：每个 tool_call_id 都必须有对应的 tool 消息
                    if not found_tool:
                        print(f"   ❌ 错误: Assistant 消息 [{i}] 的 tool_call_id={tool_call_id} 没有对应的 tool 消息")
                        print(f"   💡 这是 GPT-5.5 严格要求：每个 tool_calls 都必须有对应的 tool 消息响应")
                        return False

        return True

    def _sanitize_openai_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        删除 OpenAI strict tools 不接受的半截工具调用消息。

        OpenAI 兼容接口要求 assistant(tool_calls) 与后续 tool 响应必须成对出现。
        压缩、LLM 摘要超时回退、滑动窗口补齐等路径都可能只保留 pair 的一半，
        因此发送前/压缩后统一做一次安全清理：
        - tool 消息若找不到前置 assistant(tool_calls)，删除该 tool；
        - assistant(tool_calls) 若任一 tool_call_id 找不到后续 tool，删除该 assistant。
        """
        sanitized = [dict(msg) if isinstance(msg, dict) else msg for msg in messages]
        turns = TurnBuilder().build(sanitized)
        complete_messages = TurnBuilder().build_complete_messages(sanitized)
        if len(complete_messages) != len(sanitized):
            removed = len(sanitized) - len(complete_messages)
            incomplete_turns = [turn for turn in turns if not turn.is_valid_openai_tool_turn]
            print(f"   [工具调用清理] 删除 {removed} 条半截 tool pair 消息，涉及 {len(incomplete_turns)} 个不完整 turn")
        return complete_messages

    def _select_recent_complete_turn_messages(self, messages: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Select recent complete turns without splitting OpenAI tool pairs."""
        builder = TurnBuilder()
        complete_turns = builder.complete_turns_only(builder.build(messages))
        selected = []
        selected_count = 0

        for turn in reversed(complete_turns):
            turn_size = len(turn.messages)
            if selected and selected_count >= target_count:
                break
            selected.append(turn)
            selected_count += turn_size

        selected.reverse()
        return builder.flatten(selected)

    def _select_keyframe_turns(
        self,
        turns: List[ConversationTurn],
        target_count: int,
        min_keep: int,
    ) -> List[ConversationTurn]:
        """Select keyframe turns using Turn metadata while preserving order."""
        if not turns:
            return []

        selected_by_id: Dict[str, ConversationTurn] = {}

        # Always keep the most recent complete turns for task continuity.
        recent_budget = max(min_keep, int(target_count * 0.45))
        recent_count = 0
        for turn in reversed(turns):
            if selected_by_id and recent_count >= recent_budget:
                break
            selected_by_id[turn.id] = turn
            recent_count += len(turn.messages)

        critical_categories = {"error", "test", "code_edit", "planning"}
        for turn in turns:
            if critical_categories & set(turn.categories):
                selected_by_id[turn.id] = turn

        # Fill remaining budget by importance.  Complete tool-call turns are
        # added whole, so the resulting message list never contains half pairs.
        ranked = sorted(turns, key=lambda t: (t.importance, t.end_index), reverse=True)
        selected_count = sum(len(turn.messages) for turn in selected_by_id.values())
        for turn in ranked:
            if selected_count >= target_count and len(selected_by_id) >= 1:
                break
            if turn.id in selected_by_id:
                continue
            if "bulk_output" in turn.categories and turn.importance < 0.5:
                continue
            selected_by_id[turn.id] = turn
            selected_count += len(turn.messages)

        return sorted(selected_by_id.values(), key=lambda turn: turn.start_index)

    def _select_summary_candidate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select older complete turns worth sending to the LLM summarizer.

        This removes incomplete tool pairs and deprioritizes low-value bulk
        output while keeping errors/tests/file edits/planning turns as summary
        candidates.
        """
        builder = TurnBuilder()
        complete_turns = builder.complete_turns_only(builder.build(messages))
        if not complete_turns:
            return []

        critical = {"error", "test", "code_edit", "planning"}
        selected_turns = [
            turn for turn in complete_turns
            if (critical & set(turn.categories)) or turn.importance >= 0.35
        ]
        if not selected_turns:
            selected_turns = complete_turns[-min(4, len(complete_turns)):]

        return builder.flatten(selected_turns)

    def _parse_structured_summary(
        self,
        summary_text: str,
        messages: List[Dict[str, Any]],
        avg_importance: float
    ) -> Optional[SessionSummary]:
        """
        解析结构化摘要

        Args:
            summary_text: LLM 返回的摘要文本（JSON 格式）
            messages: 原始消息列表
            avg_importance: 平均重要性

        Returns:
            SessionSummary 或 None
        """
        try:
            # 尝试提取 JSON
            json_start = summary_text.find("{")
            json_end = summary_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                print("   [警告] 未找到 JSON 格式摘要")
                return None

            json_str = summary_text[json_start:json_end]
            data = json.loads(json_str)

            # 辅助函数：安全地转换错误记录
            def convert_error(er):
                if isinstance(er, dict):
                    return ErrorRecord(**{**er, "timestamp": er.get("timestamp", datetime.now().isoformat())})
                return er

            # 辅助函数：安全地转换工具使用记录
            def convert_tool(tu):
                if isinstance(tu, dict):
                    return ToolUsage(**{
                        **tu,
                        "timestamp": tu.get("timestamp", datetime.now().isoformat()),
                        "parameters": tu.get("parameters", {})
                    })
                return tu

            # 构建 SessionSummary
            summary = SessionSummary(
                session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(messages)}",
                timestamp=datetime.now().isoformat(),
                summary_text=data.get("summary_text", ""),
                task_goal=data.get("task_goal", ""),
                task_status=data.get("task_status", "in_progress"),
                files_changed=[
                    FileChange(**fc) if isinstance(fc, dict) else fc
                    for fc in data.get("files_changed", [])
                ],
                errors_encountered=[
                    convert_error(er)
                    for er in data.get("errors_encountered", [])
                ],
                tools_used=[
                    convert_tool(tu)
                    for tu in data.get("tools_used", [])
                ],
                key_decisions=data.get("key_decisions", []),
                importance=avg_importance,
                message_count=len(messages),
                token_count=sum(len(str(msg)) for msg in messages)
            )

            return summary

        except json.JSONDecodeError as e:
            print(f"   [警告] JSON 解析失败: {str(e)}")
            return None
        except Exception as e:
            print(f"   [警告] 解析结构化摘要失败: {str(e)}")
            return None


def select_compression_strategy(
    messages: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> CompressionStrategy:
    """
    根据消息特征选择压缩策略（便捷函数）

    Args:
        messages: 消息列表
        context: 上下文信息（可选）

    Returns:
        CompressionStrategy: 推荐的压缩策略
    """
    engine = CompressionEngine()
    return engine._select_strategy(messages)