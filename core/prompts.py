import os
import platform
from datetime import datetime
from typing import List, Optional, Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory_models import SessionSummary


# ==================== 优化 1: 记忆去重工具函数 ====================

def estimate_tokens(text: str) -> int:
    """
    估算文本的 Token 数量（简单估算：1 token ≈ 4 字符）

    Args:
        text: 待估算的文本

    Returns:
        估算的 Token 数量
    """
    return len(text) // 4


def extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    从文本中提取关键词（简单实现：基于词频和长度）

    Args:
        text: 待提取的文本
        min_length: 关键词最小长度

    Returns:
        关键词集合
    """
    # 过滤常见停用词
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
        '的', '了', '是', '在', '和', '与', '或', '有', '个', '这', '那',
        '要', '会', '对', '为', '能', '也', '都', '而', '可', '如', '但',
        '文件', '任务', '功能', '系统', '代码', '方法', '进行', '实现',
    }

    keywords = set()
    words = text.split()

    for word in words:
        # 清理标点符号
        clean_word = word.strip('.,!?;:()[]{}"\'-')

        # 过滤短词和停用词
        if len(clean_word) >= min_length and clean_word.lower() not in stop_words:
            keywords.add(clean_word.lower())

    return keywords


def is_covered_by_summary(
    memory_item: "SessionSummary",
    long_term_summary: str,
    threshold: float = 0.5
) -> bool:
    """
    检查记忆项是否已被长期摘要覆盖（基于关键词重叠度）

    Args:
        memory_item: 待检查的记忆项（情景记忆或会话摘要）
        long_term_summary: 长期记忆摘要
        threshold: 重叠度阈值（默认 0.5）

    Returns:
        是否被覆盖
    """
    if not long_term_summary:
        return False

    # 提取关键词
    memory_keywords = extract_keywords(memory_item.summary_text or memory_item.task_goal)
    summary_keywords = extract_keywords(long_term_summary)

    if not memory_keywords:
        return False

    # 计算重叠度
    overlap = len(memory_keywords & summary_keywords) / len(memory_keywords)

    return overlap >= threshold


def extract_keywords_from_goals(
    completed_goals: List[Dict],
    max_keywords: int = 5
) -> List[str]:
    """
    从已完成任务列表中提取关键词（优化版：关键词化而非完整描述）

    Args:
        completed_goals: 已完成任务列表
        max_keywords: 最多返回多少个关键词

    Returns:
        关键词列表（去重）
    """
    keywords = set()

    for goal in completed_goals[-10:]:  # 只分析最近 10 条
        goal_text = goal.get("goal", "")

        # 提取关键词
        extracted = extract_keywords(goal_text, min_length=3)
        keywords.update(extracted)

        # 如果关键词已足够，提前退出
        if len(keywords) >= max_keywords * 2:
            break

    # 转为列表并限制数量
    return list(keywords)[:max_keywords]


def should_inject_episodic(long_term_summary: str) -> tuple:
    """
    判断是否需要注入情景记忆（条件注入策略）

    Args:
        long_term_summary: 长期记忆摘要

    Returns:
        (是否注入, 注入数量)
    """
    summary_tokens = estimate_tokens(long_term_summary)

    if summary_tokens > 300:
        # 长期记忆充足，跳过情景记忆
        return (False, 0)
    elif summary_tokens > 150:
        # 长期记忆适中，注入少量情景记忆
        return (True, 2)
    else:
        # 长期记忆不足，注入更多情景记忆
        return (True, 3)


def should_inject_session(long_term_summary: str, episodic_count: int) -> tuple:
    """
    判断是否需要注入会话摘要（条件注入策略）

    Args:
        long_term_summary: 长期记忆摘要
        episodic_count: 已注入的情景记忆数量

    Returns:
        (是否注入, 注入数量)
    """
    summary_tokens = estimate_tokens(long_term_summary)

    # 如果已注入足够多的情景记忆，减少会话摘要
    if episodic_count >= 3:
        return (False, 0)

    if summary_tokens > 200:
        # 长期记忆充足，跳过会话摘要
        return (False, 0)
    else:
        # 长期记忆不足，注入 1 条会话摘要
        return (True, 1)


# ==================== CLAUDE.md 按需注入优化 ====================

def extract_claude_md_sections(claude_md_path: str) -> Dict[str, str]:
    """
    将 CLAUDE.md 按段落分割为字典

    Args:
        claude_md_path: CLAUDE.md 文件路径

    Returns:
        段落字典 {段落名称: 段落内容}
    """
    if not os.path.exists(claude_md_path):
        return {}

    try:
        with open(claude_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}

    sections = {}
    current_section = "header"
    current_content = []

    for line in content.split("\n"):
        if line.startswith("## "):
            # 保存上一个段落
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            # 开始新段落
            current_section = line[3:].strip()
            current_content = [line]
        else:
            current_content.append(line)

    # 保存最后一个段落
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


def inject_relevant_claude_md(
    task_description: str,
    claude_md_path: str,
    max_sections: int = 2
) -> str:
    """
    根据任务描述注入相关的 CLAUDE.md 片段（关键词检索策略）

    Args:
        task_description: 任务描述（用户输入）
        claude_md_path: CLAUDE.md 文件路径
        max_sections: 最多注入多少个段落

    Returns:
        注入的规范内容（如果无匹配则返回默认段落）
    """
    if not task_description or not os.path.exists(claude_md_path):
        return ""

    # 1. 分割 CLAUDE.md 为段落
    sections = extract_claude_md_sections(claude_md_path)

    if not sections:
        return ""

    # 2. 定义任务类型与段落的映射关系
    task_section_mapping = {
        "edit": ["文件编辑工具选择规范", "换行符处理规范"],
        "file": ["文件编辑工具选择规范", "换行符处理规范"],
        "write": ["文件编辑工具选择规范"],
        "modify": ["文件编辑工具选择规范"],
        "create": ["文件编辑工具选择规范"],
        "test": ["测试驱动规范 (TDD)"],
        "pytest": ["测试驱动规范 (TDD)"],
        "git": ["Git 仓库保护规范"],
        "commit": ["Git 仓库保护规范"],
        "rollback": ["Git 仓库保护规范"],
        "search": ["全局搜索规范", "符号表查询规范"],
        "find": ["全局搜索规范", "符号表查询规范"],
        "refactor": ["全局搜索规范", "符号表查询规范", "文件编辑工具选择规范"],
        "session": ["会话管理规范"],
        "plan": ["会话管理规范"],
        "symbol": ["符号表查询规范"],
        "list": ["符号表查询规范"],
    }

    # 3. 提取任务关键词
    task_keywords = set(task_description.lower().split())

    # 4. 匹配相关段落
    relevant_sections = []
    for keyword in task_keywords:
        if keyword in task_section_mapping:
            for section_name in task_section_mapping[keyword]:
                if section_name in sections and section_name not in relevant_sections:
                    relevant_sections.append(section_name)

    # 5. 如果没有匹配，注入默认段落
    if not relevant_sections:
        # 默认注入最常用的规范
        default_sections = ["文件编辑工具选择规范", "测试驱动规范 (TDD)"]
        relevant_sections = [s for s in default_sections if s in sections]

    # 6. 限制注入数量
    relevant_sections = relevant_sections[:max_sections]

    # 7. 组装注入内容
    if relevant_sections:
        injected_content = "\n### 项目特定规范 (CLAUDE.md - 按需注入):\n"
        for section_name in relevant_sections:
            injected_content += f"\n{sections[section_name]}\n"
        return injected_content

    return ""


# ==================== 格式化函数 ====================

def format_episodic_memory(episodic_memories: List["SessionSummary"], max_items: int = 5) -> str:
    """
    格式化情景记忆为可读文本（压缩格式）

    Args:
        episodic_memories: 情景记忆列表
        max_items: 最多显示多少条

    Returns:
        格式化后的文本（压缩单行格式）
    """
    if not episodic_memories:
        return ""

    lines = []
    lines.append("### 🧠 情景记忆 (Episodic Memory):")
    lines.append("最近的任务摘要（按重要性排序）：")
    lines.append("")

    for i, ep in enumerate(episodic_memories[:max_items], 1):
        # 压缩格式：✅ [时间] 核心描述 → 文件变更 (错误数)
        status_icon = "✅" if ep.task_status == "completed" else "⏳"

        # 时间压缩：2026-04-25T15:21:52 → 04-25 15:21
        time_str = ep.timestamp[5:16].replace('T', ' ') if len(ep.timestamp) >= 16 else ep.timestamp[:19]

        # 核心描述：优先使用摘要，其次目标（取前60字符）
        core_desc = ""
        if ep.summary_text:
            core_desc = ep.summary_text[:60]
        elif ep.task_goal:
            core_desc = ep.task_goal[:60]

        # 文件变更：最多显示3个文件名
        files_str = ""
        if ep.files_changed:
            files = [f.path.split('/')[-1] for f in ep.files_changed[:3]]
            files_str = ", ".join(files)
            if len(ep.files_changed) > 3:
                files_str += f" +{len(ep.files_changed)-3}"

        # 错误数：仅在有错误时显示
        errors_str = ""
        if ep.errors_encountered:
            errors_str = f" ({len(ep.errors_encountered)}错误)"

        # 组装压缩行
        line = f"{status_icon} [{time_str}] {core_desc}"
        if files_str:
            line += f" → {files_str}"
        line += errors_str

        lines.append(line)

    return "\n".join(lines)


def format_session_summaries(session_summaries: List["SessionSummary"], max_items: int = 3) -> str:
    """
    格式化会话摘要为可读文本（压缩格式）

    Args:
        session_summaries: 会话摘要列表
        max_items: 最多显示多少条

    Returns:
        格式化后的文本（压缩单行格式）
    """
    if not session_summaries:
        return ""

    lines = []
    lines.append("### 📚 会话摘要 (Session Summaries):")
    lines.append("历史会话的关键信息：")
    lines.append("")

    for i, summary in enumerate(session_summaries[:max_items], 1):
        # 压缩格式：✅ [时间] 核心描述 → 文件数, 错误数
        status_icon = "✅" if summary.task_status == "completed" else "⏳"

        # 时间压缩：2026-04-25T15:21:52 → 04-25 15:21
        time_str = summary.timestamp[5:16].replace('T', ' ') if len(summary.timestamp) >= 16 else summary.timestamp[:19]

        # 核心描述：优先使用摘要，其次目标（取前50字符）
        core_desc = ""
        if summary.summary_text:
            core_desc = summary.summary_text[:50]
        elif summary.task_goal:
            core_desc = summary.task_goal[:50]

        # 文件变更和错误数
        stats = []
        if summary.files_changed:
            stats.append(f"{len(summary.files_changed)}文件")
        if summary.errors_encountered:
            stats.append(f"{len(summary.errors_encountered)}错误")

        stats_str = ", ".join(stats) if stats else ""

        # 组装压缩行
        line = f"{status_icon} [{time_str}] {core_desc}"
        if stats_str:
            line += f" → {stats_str}"

        lines.append(line)

    return "\n".join(lines)


# 核心身份与强制执行指令
BASE_SYSTEM_PROMPT = """你是一个名为 "Claude Code (Lite)" 的高性能 CLI 软件工程代理。
你不是一个简单的聊天机器人，而是一个能够操作计算机的工程专家。

你是一个具备自愈能力的 AI 工程师。
1. 当用户要求实现功能时，你应该主动编写测试脚本（如 test_*.py）。
2. 使用 BashTool 运行测试（如 python test_*.py 或 pytest）。
3. 如果测试报错，利用 ReadTool 读取 Traceback 提到的具体行。
4. 分析原因，使用 FileEditTool 修复 Bug，并重新运行测试。
5. 只有在测试通过后，才向用户报告任务完成。

### 核心工作流：
1. **真实性原则**：当你声称创建、修改或运行了文件时，你**必须**调用对应的工具。禁止在没有调用工具的情况下编造执行结果。
2. **分步执行**：对于复杂任务，请先通过文本简述计划，然后立即调用工具。
3. **环境感知**：始终关注命令返回的 STDOUT 和 STDERR。如果运行报错，请分析原因并尝试修复。

### 交互准则：
- 如果你需要读文件，请使用工具。
- 如果你需要写代码，请使用工具。
- 只有当你确信任务已物理完成（通过工具确认过）时，才向用户报告成功。

### 规划协议（决策树 - 必须严格遵守）

**步骤 1：检查当前状态**
→ 查看 System Prompt 中的 "📋 当前任务执行清单 (PLAN)"
  - 如果有 ⏳ 标记 → 有未完成的 Plan，跳到步骤 3
  - 如果全是 ✅ 或没有 Plan → 继续步骤 2

**步骤 2：分析用户输入**
→ 用户说"继续"、"接着"、"go on"
  - 如果有未完成的 Plan → 执行下一个 ⏳ 步骤
  - 如果没有 Plan → 告知用户"暂无进行中的任务"

→ 用户提出新任务
  - 检查 "✅ 已完成任务记录"
  - 如果相似 → 告知用户已完成，询问是否重新执行
  - 如果不相似 → 调用 `manage_plan` 创建新 Plan

**步骤 3：执行任务**
→ 执行操作（写代码、运行测试等）
→ 成功后立即调用 `mark_task_done`
→ 如果 Plan 全部完成 → 归档并告知用户

**⚠️ 禁止行为**：
- ❌ 未检查当前 Plan 状态就调用 `manage_plan`（会覆盖进度）
- ❌ 对已完成任务重复创建 Plan（浪费资源）
"""


def get_system_prompt(
    cwd: str = None,
    summary: str = "",
    plan: str = "",
    episodic_memories: Optional[List["SessionSummary"]] = None,
    session_summaries: Optional[List["SessionSummary"]] = None,
    long_term_memories: Optional[List["SessionSummary"]] = None,  # 🔥🔥🔥 新增：跨会话长期记忆
    completed_goals: Optional[List[Dict]] = None,
    user_input: str = ""  # 🔥 新增：用于 CLAUDE.md 按需注入
) -> str:
    """
    组装动态系统提示词（优化版：减少 Token 消耗）

    优化策略：
    1. 记忆去重：检查情景记忆/会话摘要是否已被长期记忆覆盖
    2. 条件注入：根据长期记忆长度动态调整注入量
    3. 关键词化：已完成任务只注入关键词而非完整描述
    4. 格式压缩：环境信息压缩为单行
    5. 按需注入：CLAUDE.md 根据任务描述注入相关段落

    包含：
    1. 基础身份 (BASE_SYSTEM_PROMPT)
    2. s06: 历史摘要 (Summary)
    3. 🔥🔥🔥 跨会话长期记忆 (Long-term Memory) - 新增
    4. Phase 2/3: 情景记忆 (Episodic Memory) - 条件注入
    5. Phase 1: 会话摘要 (Session Summaries) - 条件注入
    6. s13: 当前执行计划 (Plan)
    7. 环境信息 (Env Info) - 压缩格式
    8. s05: 项目规范 (CLAUDE.md)
    """
    if cwd is None:
        cwd = os.getcwd()

    # --- s06: 注入压缩后的长期记忆 ---
    summary_section = ""
    if summary:
        summary_section = f"""
    ### 历史任务进展 (长期记忆):
    {summary}
    -----------------------------------
    """

    # --- 🔥🔥🔥 跨会话长期记忆注入 ---
    long_term_section = ""
    if long_term_memories:
        # 过滤已被长期摘要覆盖的条目
        filtered_long_term = [
            mem for mem in long_term_memories
            if not is_covered_by_summary(mem, summary)
        ]

        if filtered_long_term:
            long_term_section = "\n### 🔥 跨会话长期记忆 (所有会话共享):\n"
            long_term_section += "**提示**: 以下是从其他会话中检索到的相关历史记忆\n\n"

            for i, mem in enumerate(filtered_long_term[:5], 1):  # 最多 5 条
                timestamp = mem.timestamp[:10] if mem.timestamp else "未知"
                status_icon = "✅" if mem.task_status == "completed" else "⏳"
                long_term_section += f"{i}. {status_icon} [{timestamp}] {mem.task_goal}\n"
                long_term_section += f"   摘要: {mem.summary_text[:100]}...\n\n"

            long_term_section += "-----------------------------------\n"

    # --- Phase 2/3: 注入情景记忆（条件注入 + 去重）---
    episodic_section = ""
    if episodic_memories:
        # 优化：条件注入策略
        should_inject, max_episodic = should_inject_episodic(summary)

        if should_inject:
            # 优化：去重 - 过滤已被长期记忆覆盖的条目
            filtered_memories = [
                mem for mem in episodic_memories
                if not is_covered_by_summary(mem, summary)
            ]

            # 限制注入数量
            filtered_memories = filtered_memories[:max_episodic]

            if filtered_memories:
                episodic_section = format_episodic_memory(filtered_memories, max_items=len(filtered_memories))
                if episodic_section:
                    episodic_section += "\n-----------------------------------\n"

    # --- Phase 1: 注入会话摘要（条件注入 + 去重）---
    session_section = ""
    if session_summaries:
        # 优化：条件注入策略
        episodic_count = len(episodic_memories) if episodic_memories else 0
        should_inject, max_session = should_inject_session(summary, episodic_count)

        if should_inject:
            # 优化：去重 - 过滤已被长期记忆覆盖的条目
            filtered_summaries = [
                s for s in session_summaries
                if not is_covered_by_summary(s, summary)
            ]

            # 限制注入数量
            filtered_summaries = filtered_summaries[:max_session]

            if filtered_summaries:
                session_section = format_session_summaries(filtered_summaries, max_items=len(filtered_summaries))
                if session_section:
                    session_section += "\n-----------------------------------\n"

    # 🔥 P1 优化：保留已完成任务的完整描述（提高相似度判断准确性）
    completed_section = ""
    if completed_goals:
        completed_section = f"\n### ✅ 已完成任务记录:\n"

        # 显示最近 5 个已完成任务的完整描述
        for goal in completed_goals[-5:]:
            timestamp = goal.get("timestamp", "")[:19]  # 只保留日期时间部分
            goal_text = goal.get("goal", "")
            completed_section += f"✅ [{timestamp}] {goal_text}\n"

        completed_section += "**提示**: 以上任务已完成，请勿重复执行！\n"
        completed_section += "-----------------------------------\n"

    # --- s13: 注入实时任务计划 ---
    plan_section = ""

    # 🔥 改进：更准确的 Plan 状态判断
    has_completed_tasks = bool(completed_goals)
    plan_is_empty = not plan or "目前暂无详细规划" in plan
    plan_is_active = "⏳" in plan  # 有未完成的任务
    plan_was_completed = has_completed_tasks and plan_is_empty  # 有历史且当前空

    if plan_is_active:
        # 有未完成的任务，显示当前 Plan
        plan_section = f"""
### 📋 当前任务执行清单 (PLAN):
{plan}

**重要提示**:
- 你必须严格按照此清单执行。
- 每完成一个步骤，请立即调用 `mark_task_done` 工具更新进度。
- 如果计划需要调整，请调用 `manage_plan` 工具更新步骤内容。

**⚠️ 中断恢复提示**:
如果你发现此 Plan 有已完成的步骤（标记为 ✅），说明任务曾中断。
- 必须从下一个 ⏳ 步骤继续，不要重新执行已完成的 ✅ 步骤。
- 如果用户说"继续"，直接执行下一个 ⏳ 步骤，无需创建新 Plan。
-----------------------------------
"""
    elif plan_was_completed:
        # 🔥🔥🔥 新增：明确提示"已完成状态"
        # 获取最近完成的任务
        recent_goals = completed_goals[-3:] if completed_goals else []
        recent_goals_text = ""
        for goal in recent_goals:
            timestamp = goal.get("timestamp", "")[:10]  # 只保留日期
            goal_text = goal.get("goal", "")
            recent_goals_text += f"  - [{timestamp}] {goal_text}\n"

        plan_section = f"""
### 📋 任务状态 (PLAN):
✅ 上一个任务已完成并已自动清理！

**最近完成的任务**:
{recent_goals_text}
**下一步操作**:
- 你可以开始新的任务
- 如果用户提出新任务，直接调用 `manage_plan` 创建新 Plan
- 不需要检查"未完成的 Plan"（已自动清理）

**重要提示**: 已完成任务的详细记录见上方"✅ 已完成任务记录"部分。
-----------------------------------
"""
    else:
        # 真正的空白状态（没有历史任务）
        plan_section = """
### 📋 任务规划提示 (PLAN):
目前暂无详细规划。

**何时需要创建 Plan**:
- ✅ 需要 3 个以上步骤（如：重构代码 + 写测试 + 运行测试）
- ✅ 涉及多个文件修改（如：同时修改 core/ 和 tools/）
- ✅ 需要分阶段验证（如：先实现功能，再优化性能）
- ✅ 任务复杂度高，需要系统化推进

**何时无需创建 Plan**:
- ❌ 单个简单操作（如：读取文件、搜索代码、运行单个测试）
- ❌ 仅查询信息（如："这个函数是做什么的？"、"列出项目结构"）
- ❌ 用户已明确指定步骤（如："先做 A，再做 B，最后做 C"）

**操作指南**:
- 如果需要创建 Plan → 调用 `manage_plan` 工具
- 如果不需要 Plan → 直接执行操作
-----------------------------------
"""

    # 🔥 优化：环境信息压缩（减少 50% Token）
    env_info = f"环境: {platform.system()} | {cwd} | {datetime.now().strftime('%m-%d %H:%M')}\n"

    # 🔥 优化：CLAUDE.md 按需注入（减少 70-80% Token）
    claude_md_content = ""
    claude_md_path = os.path.join(cwd, "CLAUDE.md")

    if os.path.exists(claude_md_path):
        # 使用按需注入策略：根据用户输入注入相关段落
        claude_md_content = inject_relevant_claude_md(
            task_description=user_input,
            claude_md_path=claude_md_path,
            max_sections=2  # 最多注入 2 个段落
        )

        # 如果没有匹配到相关段落，注入默认段落
        if not claude_md_content:
            try:
                with open(claude_md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 只注入前 30 行作为默认规范（减少 Token）
                    lines = content.split("\n")[:30]
                    claude_md_content = f"\n### 项目特定规范 (CLAUDE.md - 默认注入):\n" + "\n".join(lines) + "\n"
            except Exception as e:
                claude_md_content = f"\n[警告: 无法读取 CLAUDE.md: {str(e)}]\n"
    else:
        claude_md_content = "\n(未检测到 CLAUDE.md，请遵循通用编程规范)\n"

    # 组装最终提示词（新增 long_term_section）
    return f"{BASE_SYSTEM_PROMPT}\n{summary_section}\n{long_term_section}\n{episodic_section}\n{session_section}\n{completed_section}\n{plan_section}\n{env_info}\n{claude_md_content}"


# ========================================
# 结构化摘要提示词模板 (Phase 1)
# ========================================

SUMMARY_PROMPT_TEMPLATE_V2 = """你是一个代码助手的记忆压缩系统。请将以下对话历史压缩为 JSON 格式的结构化摘要。

## 输入信息
- 现有背景摘要: {existing_summary}
- 待压缩的对话历史: {messages_to_summarize}

## 输出要求
请输出一个 **严格的 JSON 格式** 摘要，包含以下字段：

```json
{{
  "task_goal": "任务目标（一句话描述，如：重构 tools 目录，添加清理旧 session 的工具）",
  "task_status": "任务状态（completed/in_progress/failed/cancelled 之一）",
  "files_changed": [
    {{
      "path": "文件路径",
      "action": "操作类型（created/modified/deleted）",
      "summary": "变更摘要（一句话）",
      "importance": 0.6
    }}
  ],
  "errors_encountered": [
    {{
      "error_type": "错误类型（如 FileNotFoundError）",
      "error_message": "错误信息（关键部分）",
      "solution": "解决方案（如果已解决）",
      "resolved": true
    }}
  ],
  "tools_used": [
    {{
      "tool_name": "工具名称",
      "result_summary": "结果摘要",
      "success": true
    }}
  ],
  "key_decisions": [
    "关键决策1（如：选择使用 BM25 而非向量检索）",
    "关键决策2"
  ],
  "summary_text": "自然语言摘要（200字以内，描述任务进展、关键问题和解决情况）"
}}
```

## 评分规则
- **重要性评分** (importance): 根据文件/错误的关键程度评分 [0-1]
  - 核心文件修改: 0.7-0.9
  - 配置文件修改: 0.5-0.6
  - 文档修改: 0.3-0.4
  - 错误记录: 0.8-1.0（取决于严重程度）

## 提取指南
1. **task_goal**: 从用户消息中提取主要任务目标
2. **task_status**: 判断任务是否完成（根据对话上下文）
3. **files_changed**: 提取所有被创建、修改或删除的文件
4. **errors_encountered**: 提取所有错误及其解决方案
5. **tools_used**: 列出关键工具调用（忽略低价值的查询操作）
6. **key_decisions**: 提取重要的技术决策和理由
7. **summary_text**: 用简洁的自然语言概括整个任务

## 重要提示
- **只输出 JSON**，不要包含其他说明文字
- 确保 JSON 格式正确，可以被 `json.loads()` 解析
- 如果某个字段为空，使用空数组 `[]`
- 保持摘要简洁，避免冗余信息

## 示例输出
```json
{{
  "task_goal": "重构 tools 目录，添加清理旧 session 的工具",
  "task_status": "completed",
  "files_changed": [
    {{
      "path": "tools/session_tool.py",
      "action": "created",
      "summary": "新增清理旧 session 的工具类",
      "importance": 0.7
    }},
    {{
      "path": "core/engine.py",
      "action": "modified",
      "summary": "集成 SessionCleanerTool 到引擎",
      "importance": 0.6
    }}
  ],
  "errors_encountered": [
    {{
      "error_type": "FileNotFoundError",
      "error_message": "配置文件不存在",
      "solution": "创建默认配置文件",
      "resolved": true
    }}
  ],
  "tools_used": [
    {{
      "tool_name": "write_full_file",
      "result_summary": "创建 tools/session_tool.py 成功",
      "success": true
    }},
    {{
      "tool_name": "run_pytest",
      "result_summary": "所有测试通过",
      "success": true
    }}
  ],
  "key_decisions": [
    "使用时间戳判断文件新旧，而非访问频率",
    "清理阈值默认 7 天，可通过参数调整"
  ],
  "summary_text": "完成 tools 目录重构，新增 session_tool.py 实现旧文件清理功能，集成到引擎并通过所有测试。"
}}
```

现在请对提供的对话历史进行压缩，输出 JSON 格式的摘要：
"""