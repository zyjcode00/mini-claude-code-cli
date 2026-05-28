import json
import os
from typing import List, Dict, Any
import anthropic
import openai
import asyncio
from tools.base import BaseTool
from core.prompts import get_system_prompt
from core.context import ContextManager  # <--- 导入新管家

# Git 自动化保险导入
from tools.git_tool import create_snapshot, rollback_to, has_uncommitted_changes, start_task_branch, finalize_task, start_plan_branch, finalize_plan

# 检索增强导入
from core.keyword_indexer import KeywordIndexer
from core.bm25_retriever import BM25Retriever

class AgentEngine:
    def __init__(self, tools: List[BaseTool], model: str, plan_manager, # <--- 传入管家
                 base_url: str = None, api_key: str = None,
                 max_history: int = 100, min_keep: int = 4, session_id="default"):
        self.tools = tools
        self.model = model
        self.plan_manager = plan_manager  # <--- 保存管家引用
        self.tool_map = {t.name: t for t in tools}
        self.tool_specs = [t.to_anthropic_spec() for t in tools]

        # --- 核心修改：使用 ContextManager 替代原有的 self.messages ---
        # 🔥 新增：传入 plan_manager，让压缩引擎能查询真实任务状态
        self.context = ContextManager(max_history=max_history, min_keep=min_keep, plan_manager=plan_manager)
        self.last_oa_msg = None
        self.session_id = session_id
        self.session_path = f"sessions/{session_id}.json"
        # ----------------------------------------------------------

        # ========== 检索增强：初始化检索器 ==========
        self.keyword_indexer = KeywordIndexer()
        self.bm25_retriever = BM25Retriever()
        self.retrieval_enabled = True  # 检索增强开关
        # ===========================================

        # ========== Git 自动化保险状态追踪 ==========
        self.edit_failures = {}  # {"file_path": failure_count}
        self.last_snapshot_plan_step = None  # 记录上次快照时的计划步骤
        self.current_plan_branch = None  # 当前影子分支对应的 Plan ID
        # ===========================================

        self.is_openai_compat = base_url is not None or "claude" not in model.lower()

        # --- 异步适配：使用 Async 客户端 ---
        if self.is_openai_compat:
            self.client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        os.makedirs("sessions", exist_ok=True)
        self.load_session()



    async def _call_llm(self, relevant_history: str = "", user_input: str = ""):
        """核心推理：改为异步调用"""
        current_plan = self.plan_manager.get_formatted_plan()

        # Phase 2/3: 获取三层记忆数据
        episodic_memories = self.context.episodic_memory.get_all() if self.context._enable_memory_layers else []
        session_summaries = self.context.session_summaries

        # 🔥🔥🔥 跨会话长期记忆检索：从磁盘检索所有会话的长期记忆
        long_term_memories = []
        if self.context._enable_memory_layers and user_input:
            try:
                # 从长期记忆中检索与当前任务相关的摘要（跨所有会话）
                long_term_memories = self.context.memory_manager.search_long_term(
                    query=user_input,
                    top_k=5  # 检索最相关的 5 条长期记忆
                )
                if long_term_memories:
                    print(f" [🧠] 检索到 {len(long_term_memories)} 条跨会话长期记忆")
            except Exception as e:
                print(f" [⚠️] 长期记忆检索失败: {e}")

        # 🔥 新增：获取已完成任务列表
        completed_goals = self.plan_manager.get_completed_goals()

        system_ptr = get_system_prompt(
            summary=self.context.history_summary,
            plan=current_plan,
            episodic_memories=episodic_memories,
            session_summaries=session_summaries,
            long_term_memories=long_term_memories,  # 🔥🔥🔥 新增参数：跨会话长期记忆
            completed_goals=completed_goals,  # 🔥 新增参数
            user_input=user_input  # 🔥 新增：用于 CLAUDE.md 按需注入
        )

        # ========== 检索增强：注入相关历史 ==========
        if relevant_history:
            system_ptr += relevant_history
        # ===========================================

        # 🔥🔥🔥 Phase 4: 获取消息快照（并发安全）
        messages_snapshot = await self.context.get_messages_snapshot()

        if self.is_openai_compat:
            oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in self.tool_specs]

            # 使用 await 调用异步客户端
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_ptr}] + messages_snapshot,
                tools=oa_tools if oa_tools else None
            )

            msg = resp.choices[0].message
            # 🔥 修复：转换为字典格式，避免后续 .get() 报错
            # 🔥🔥🔥 DeepSeek 修复：保存 reasoning_content（思考模式必须回传）
            self.last_oa_msg = {
                "role": "assistant",
                "content": msg.content or "",
            }
            # DeepSeek reasoning_content 回传：必须原样保存在消息历史中
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                self.last_oa_msg["reasoning_content"] = msg.reasoning_content
            # 如果有工具调用，添加 tool_calls 字段
            if msg.tool_calls:
                self.last_oa_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]

            content_blocks = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use", "id": tc.id, "name": tc.function.name,
                        "input": json.loads(tc.function.arguments)
                    })
            return content_blocks, "tool_use" if msg.tool_calls else "end_turn"

        else:
            # Anthropic 异步调用
            # 🔥🔥🔥 Phase 4: 使用快照确保并发安全
            resp = await self.client.messages.create(
                model=self.model, system=system_ptr, tools=self.tool_specs,
                messages=messages_snapshot, max_tokens=4096
            )
            return resp.content, resp.stop_reason

    async def execute_query(self, user_input: str):
        self.context.add_message({"role": "user", "content": user_input})

        # ========== 检索增强：自动检索相关历史 ==========
        relevant_history = ""
        if self.retrieval_enabled and len(self.bm25_retriever) > 0:
            try:
                # 使用 BM25 检索相关历史
                results = self.bm25_retriever.search(user_input, top_k=3)

                if results:
                    # 格式化检索结果
                    history_lines = ["\n[相关历史记忆]"]
                    for i, (doc_id, score) in enumerate(results[:3], 1):
                        content = self.bm25_retriever.get_document(doc_id)
                        if content:
                            # 截取前 200 字符
                            preview = content[:200] + "..." if len(content) > 200 else content
                            history_lines.append(f"{i}. (相关度: {score:.2f}) {preview}")

                    relevant_history = "\n".join(history_lines) + "\n"
                    print(f" [🔍] 检索到 {len(results)} 条相关历史")
            except Exception as e:
                print(f" [⚠️] 检索失败: {e}")
        # ============================================

        await self.compress_messages()

        step = 0
        max_steps = 80

        while step < max_steps:
            step += 1
            print(f" [DEBUG] 正在执行第 {step} 步并发思考...")

            # ========== 影子分支逻辑：Plan 开始时创建分支 ==========
            # 检查是否有 Plan 且当前不在影子分支上
            plan_id = self.plan_manager.get_plan_id()
            if plan_id and not self.current_plan_branch:
                print(f" [🌿] 检测到 Plan，创建影子分支 agent/plan-{plan_id}...")
                success, msg = start_plan_branch(plan_id)
                if success:
                    print(f" [✅] {msg}")
                    self.current_plan_branch = plan_id
                else:
                    print(f" [⚠️] 创建影子分支失败: {msg}")
            # =======================================================

            # 这里必须 await！
            # 🔥 修改：传递 user_input 用于 CLAUDE.md 按需注入
            content_blocks, stop_reason = await self._call_llm(relevant_history, user_input=user_input)

            if self.is_openai_compat and self.last_oa_msg:
                self.context.add_message(self.last_oa_msg)
            else:
                self.context.add_message({"role": "assistant", "content": content_blocks})

            if stop_reason != "tool_use":
                final_ans = "".join([b["text"] for b in content_blocks if b["type"] == "text"])
                self.save_session()
                return final_ans

            # --- 核心并发逻辑 ---
            tasks = []
            tool_calls_info = []

            for block in content_blocks:
                if block["type"] == "tool_use":
                    t_id, t_name, t_input = block["id"], block["name"], block["input"]
                    tool_obj = self.tool_map[t_name]
                    # 包装同步工具到线程中运行，避免阻塞
                    tasks.append(asyncio.to_thread(tool_obj.run, **t_input))
                    tool_calls_info.append((t_id, t_name, t_input))  # 保存输入参数

            if tasks:
                print(f" ⚡ 正在并行执行 {len(tasks)} 个工具: {[n for _, n, _ in tool_calls_info]}...")

                # 🔥🔥🔥 关键修复：添加异常处理，确保工具调用完整性
                # 如果工具执行失败，删除已添加的 assistant 消息，避免孤立
                try:
                    results = await asyncio.gather(*tasks)
                except Exception as tool_exec_error:
                    print(f" [❌] 工具执行异常: {tool_exec_error}")

                    # 🔥🔥🔥 关键：删除刚才添加的 assistant 消息（避免孤立）
                    if self.is_openai_compat and self.last_oa_msg:
                        # 检查最后一条消息是否是刚才添加的 assistant
                        if self.context.messages and self.context.messages[-1] == self.last_oa_msg:
                            self.context.messages.pop()
                            print(f" [🧹] 已删除孤立的 assistant 消息（工具执行失败）")

                    # 添加错误提示消息
                    self.context.add_message({
                        "role": "user",
                        "content": f"⚠️ 工具执行失败: {str(tool_exec_error)}。请重新尝试。"
                    })

                    self.save_session()
                    continue  # 继续下一轮循环，让 LLM 重新处理

                tool_results_content = []
                should_rollback = False
                rollback_reason = ""

                for (t_id, t_name, t_input), res in zip(tool_calls_info, results):
                    # ========== Git 自动化保险逻辑 ==========
                    # 1. 检测 edit_file 失败
                    if t_name == "edit_file":
                        file_path = t_input.get("path", "unknown")
                        if "错误" in str(res) or "失败" in str(res):
                            self.edit_failures[file_path] = self.edit_failures.get(file_path, 0) + 1
                            print(f" [⚠️] edit_file 失败计数: {file_path} -> {self.edit_failures[file_path]}")
                            if self.edit_failures[file_path] >= 2:
                                should_rollback = True
                                rollback_reason = f"edit_file 在 {file_path} 连续失败 2 次"
                        else:
                            # 成功则重置计数
                            self.edit_failures[file_path] = 0

                    # 2. 检测 mark_task_done 成功，检查 Plan 是否完成
                    if t_name == "mark_task_done" and "✅" in str(res):
                        # 检查 Plan 是否全部完成
                        if self.plan_manager.is_plan_complete():
                            print(f" [📸] 检测到 Plan 全部完成，正在归档影子分支...")

                            # 获取 Plan 描述
                            plan_desc = self.plan_manager.current_goal

                            success, msg = finalize_plan(self.current_plan_branch, plan_desc)
                            if success:
                                print(f" [✅] {msg}")
                                self.current_plan_branch = None  # 重置影子分支状态
                                # 🔥 新增：清除已完成的 Plan，避免重复执行
                                self.plan_manager.clear_plan()
                            else:
                                print(f" [⚠️] 归档失败: {msg}")
                                # 归档失败时，仍创建普通快照作为后备
                                success2, msg2 = create_snapshot("🎯 [Fallback] Plan completed")
                                if success2:
                                    print(f" [✅] 已创建后备快照: {msg2}")
                                # 即使归档失败，也要清除 Plan（避免重复执行）
                                self.plan_manager.clear_plan()
                        else:
                            # Plan 未完成，创建普通快照
                            task_id = t_input.get("task_id", 0)
                            print(f" [📸] 任务 {task_id} 完成，创建普通 Git 快照...")
                            success, msg = create_snapshot(f"🎯 [Task-{task_id}] Auto snapshot")
                            if success:
                                print(f" [✅] {msg}")
                            else:
                                print(f" [⚠️] 快照创建失败: {msg}")
                    # =========================================

                    if self.is_openai_compat:
                        self.context.add_message({
                            "role": "tool", "tool_call_id": t_id, "name": t_name, "content": str(res)
                        })
                    else:
                        tool_results_content.append({
                            "type": "tool_result", "tool_use_id": t_id, "content": res
                        })

                if not self.is_openai_compat and tool_results_content:
                    self.context.add_message({"role": "user", "content": tool_results_content})

                # ========== 自动回滚逻辑 ==========
                if should_rollback:
                    print(f"\n [🚨] 触发自动回滚: {rollback_reason}")
                    success, msg = rollback_to("HEAD~1")
                    if success:
                        print(f" [✅] {msg}")
                        self.context.add_message({
                            "role": "user", "content": f"⚠️ 系统已自动回滚代码，原因: {rollback_reason}。请重新思考解决方案。"
                        })
                    else:
                        print(f" [❌] 回滚失败: {msg}")
                # ================================

            self.save_session()

        return "任务达到最大思考步数限制。"



    def save_session(self):
        """利用 context 模块的序列化能力进行保存，同时保存 plan 状态和三层记忆"""
        # 🔥🔥🔥 核心修复：保存前强制同步所有相关摘要的状态
        if self.plan_manager.is_plan_complete() and self.plan_manager.current_goal:
            current_goal = self.plan_manager.current_goal

            # 1. 更新情景记忆中的相关摘要
            if self.context._enable_memory_layers:
                episodic_summaries = self.context.memory_manager.episodic_memory.get_all()
                updated_count = 0
                for summary in episodic_summaries:
                    # 匹配规则：goal 完全一致，或包含关系
                    if summary.task_goal == current_goal or \
                       current_goal in summary.task_goal or \
                       summary.task_goal in current_goal:
                        if summary.task_status != "completed":
                            summary.task_status = "completed"
                            updated_count += 1
                            print(f"   [状态同步] 情景记忆摘要 {summary.session_id} → completed")

                if updated_count > 0:
                    print(f"   [✅] 已更新 {updated_count} 条情景记忆摘要的状态")

            # 2. 更新 session_summaries 中的相关摘要
            updated_count2 = 0
            for summary in self.context.session_summaries:
                if summary.task_goal == current_goal or \
                   current_goal in summary.task_goal or \
                   summary.task_goal in current_goal:
                    if summary.task_status != "completed":
                        summary.task_status = "completed"
                        updated_count2 += 1
                        print(f"   [状态同步] 会话摘要 {summary.session_id} → completed")

            if updated_count2 > 0:
                print(f"   [✅] 已更新 {updated_count2} 条会话摘要的状态")

        data = {
            "history_summary": self.context.history_summary,
            "messages": self.context.get_serializable_messages(),
            "plan": self.plan_manager.to_dict(),  # 🔥 保存计划状态
            # 🔥 新增：保存三层记忆数据（Phase 2/3）
            "memories": self.context.export_memories()
        }
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f" [💾] 会话已存档: {self.session_id}")

    def load_session(self):
        """恢复记忆和计划状态，以及三层记忆数据"""
        if os.path.exists(self.session_path):
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.context.history_summary = data.get("history_summary", "")
                    self.context.messages = data.get("messages", [])
                    # 🔥 新增：恢复计划状态
                    if "plan" in data:
                        self.plan_manager.from_dict(data["plan"])

                        # 🔥🔥🔥 问题 4 修复：验证 Plan 状态一致性
                        is_valid, issues = self.plan_manager.validate_state()
                        if not is_valid:
                            print(f"\n[⚠️] 检测到 Plan 状态不一致:")
                            for issue in issues:
                                print(f"   - {issue}")
                            # 自动修复
                            self.plan_manager.auto_fix()

                        # 🔥🔥🔥 核心修复：自动清理已完成的 Plan
                        if self.plan_manager.is_plan_complete() and self.plan_manager.current_goal:
                            print("[✅] 检测到上一个 Plan 已全部完成，自动清理...")
                            self.plan_manager.clear_plan()
                            # 立即保存清理后的状态
                            self.save_session()

                    # 🔥 新增：恢复三层记忆数据（Phase 2/3）
                    if "memories" in data:
                        self.context.import_memories(data["memories"])
                    else:
                        # 兼容旧版 session 文件（没有 memories 字段）
                        print("[兼容模式] 旧版 session 文件，跳过三层记忆恢复")

                    # 🔥🔥🔥 新增：自动清理孤立消息（GPT-5.5 严格要求）
                    self._clean_orphaned_tool_calls()

                print(f"[📂] 已成功恢复会话: {self.session_id}，摘要长度: {len(self.context.history_summary)}")
            except Exception as e:
                print(f"[❌] 恢复会话失败: {e}")

    def _clean_orphaned_tool_calls(self):
        """
        🔥🔥🔥 新增：自动清理孤立消息（GPT-5.5 严格要求）

        清理规则：
        1. 删除没有对应 assistant 的 tool 消息
        2. 删除没有对应 tool 的 assistant with tool_calls 消息

        这是 GPT-5.5 严格要求：不允许孤立的工具调用
        """
        if not self.context.messages:
            return

        print(f"\n[🧹] 正在清理孤立消息（当前: {len(self.context.messages)} 条）...")

        # 收集所有有效的 tool_call_ids
        valid_tool_call_ids = set()
        for msg in self.context.messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    valid_tool_call_ids.add(tc.get("id"))

        # 收集所有已有的 tool_call_ids（在 tool 消息中）
        existing_tool_call_ids = set()
        for msg in self.context.messages:
            if msg.get("role") == "tool":
                existing_tool_call_ids.add(msg.get("tool_call_id"))

        # 找出缺失的 tool_call_ids（assistant 有，但 tool 没有）
        missing_tool_call_ids = valid_tool_call_ids - existing_tool_call_ids

        if missing_tool_call_ids:
            print(f"   [⚠️] 发现 {len(missing_tool_call_ids)} 个孤立的 tool_calls")

            # 需要删除包含这些 tool_calls 的 assistant 消息
            indices_to_remove = []
            for i, msg in enumerate(self.context.messages):
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    # 检查是否有缺失的 tool_call_id
                    has_missing = any(
                        tc.get("id") in missing_tool_call_ids
                        for tc in msg.get("tool_calls", [])
                    )
                    if has_missing:
                        indices_to_remove.append(i)
                        print(f"   [删除] Assistant 消息 [{i}] 包含孤立的 tool_calls")

            # 删除标记的消息
            for i in indices_to_remove:
                self.context.messages.pop(i)
                # 调整后续索引
                for j in range(len(indices_to_remove)):
                    if indices_to_remove[j] > i:
                        indices_to_remove[j] -= 1

        # 找出孤立的 tool 消息（没有对应的 assistant）
        indices_to_remove_tool = []
        for i, msg in enumerate(self.context.messages):
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                # 检查是否有对应的 assistant（向前查找）
                found = False
                for j in range(i - 1, -1, -1):
                    prev_msg = self.context.messages[j]
                    if prev_msg.get("role") == "assistant" and "tool_calls" in prev_msg:
                        if any(tc.get("id") == tool_call_id for tc in prev_msg.get("tool_calls", [])):
                            found = True
                            break

                if not found:
                    indices_to_remove_tool.append(i)
                    print(f"   [删除] Tool 消息 [{i}] 没有对应的 assistant")

        # 删除孤立的 tool 消息
        for i in indices_to_remove_tool:
            self.context.messages.pop(i)

        # 再次验证
        remaining_orphaned = 0
        for i, msg in enumerate(self.context.messages):
            if msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                found = any(
                    self.context.messages[j].get("role") == "assistant" and
                    "tool_calls" in self.context.messages[j] and
                    any(tc.get("id") == tool_call_id for tc in self.context.messages[j].get("tool_calls", []))
                    for j in range(i)
                )
                if not found:
                    remaining_orphaned += 1

            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    tool_call_id = tc.get("id")
                    found = any(
                        self.context.messages[j].get("role") == "tool" and
                        self.context.messages[j].get("tool_call_id") == tool_call_id
                        for j in range(i + 1, len(self.context.messages))
                    )
                    if not found:
                        remaining_orphaned += 1

        if remaining_orphaned == 0:
            print(f"   [✅] 清理完成，消息序列完整（剩余: {len(self.context.messages)} 条）")
        else:
            print(f"   [⚠️] 仍有 {remaining_orphaned} 个孤立消息，可能需要手动检查")

    async def compress_messages(self): # <--- 改为 async
        """定义异步摘要回调"""
        async def llm_summarizer(prompt_text): # <--- 内部回调也改为 async
            # 加上 await
            blocks, _ = await self._call_llm_internal(user_input=prompt_text)
            if blocks:
                return "".join([b["text"] for b in blocks if b["type"] == "text"])
            return None

        # 注意：如果你的 context.compress 还没改异步，这里需要修改 context 模块
        # 或者在这里直接 await context.compress
        await self.context.compress(llm_summarizer)

        # ========== 检索增强：压缩后自动索引新摘要 ==========
        if self.retrieval_enabled:
            # 索引 history_summary
            if self.context.history_summary:
                self.index_content(
                    f"summary_{self.session_id}",
                    self.context.history_summary
                )
        # ===========================================

    async def _call_llm_internal(self, user_input: str): # <--- 加 async
        """摘要专用：异步调用 (包含超时保护)"""
        temp_messages = [{"role": "user", "content": user_input}]
        try:
            # 🔥🔥🔥 问题 1 修复：添加超时保护（30秒）
            if self.is_openai_compat:
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(model=self.model, messages=temp_messages),
                    timeout=30.0
                )
                content = resp.choices[0].message.content
                return [{"type": "text", "text": content}], "end_turn"
            else:
                resp = await asyncio.wait_for(
                    self.client.messages.create(model=self.model, max_tokens=1024, messages=temp_messages),
                    timeout=30.0
                )
                return resp.content, resp.stop_reason
        except asyncio.TimeoutError:
            print(f"[⚠️] 压缩 LLM 调用超时 (30s)，使用快速回退策略")
            return None, "error"
        except Exception as e:
            print(f"[摘要调用失败]: {e}")
            return None, "error"

    # ========== 检索增强：索引管理方法 ==========
    def index_content(self, doc_id: str, content: str):
        """
        索引内容到检索器

        Args:
            doc_id: 文档 ID（通常是会话 ID 或摘要 ID）
            content: 内容文本
        """
        try:
            # 同时索引到关键词索引器和 BM25 检索器
            self.keyword_indexer.index_document(doc_id, content)
            self.bm25_retriever.index_document(doc_id, content)
        except Exception as e:
            print(f" [⚠️] 索引失败: {e}")

    def index_session_summaries(self):
        """索引所有会话摘要"""
        try:
            summaries = getattr(self.context, 'session_summaries', [])
            for summary in summaries:
                doc_id = getattr(summary, "session_id", None) or str(len(self.bm25_retriever))
                content = self._summary_to_text(summary)
                self.index_content(doc_id, content)

            print(f" [🔍] 已索引 {len(summaries)} 个会话摘要")
        except Exception as e:
            print(f" [⚠️] 索引会话摘要失败: {e}")

    def _summary_to_text(self, summary) -> str:
        """
        将结构化摘要转换为文本

        Args:
            summary: SessionSummary 对象或字典

        Returns:
            文本内容
        """
        if hasattr(summary, 'summary_text'):
            # SessionSummary 对象
            parts = [
                summary.summary_text,
                f"任务目标: {summary.task_goal}",
                f"任务状态: {summary.task_status}"
            ]

            if summary.files_changed:
                parts.append("文件变更: " + ", ".join([fc.path for fc in summary.files_changed]))

            if summary.errors_encountered:
                parts.append("错误: " + ", ".join([er.error_type for er in summary.errors_encountered]))

            if summary.tools_used:
                parts.append("工具: " + ", ".join([tu.tool_name for tu in summary.tools_used]))

            return "\n".join(parts)
        else:
            # 字典格式
            return str(summary)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索器统计信息

        Returns:
            统计信息字典
        """
        return {
            "keyword_indexer": self.keyword_indexer.get_statistics(),
            "bm25_retriever": self.bm25_retriever.get_statistics(),
            "retrieval_enabled": self.retrieval_enabled
        }
    # ===========================================