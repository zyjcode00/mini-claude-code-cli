# core/plan.py
from typing import List, Dict, Tuple
import hashlib
from datetime import datetime

class PlanManager:
    def __init__(self):
        self.tasks: List[Dict] = []  # 格式: {"id": 1, "task": "...", "status": "todo/done"}
        self.current_goal: str = ""
        self.plan_id: str = ""  # Plan 唯一标识符
        # 🔥 新增：已完成任务的历史记录
        self.completed_goals: List[Dict] = []  # 格式: [{"goal": "...", "timestamp": "..."}]

    def update_plan(self, new_tasks: List[str]):
        """覆盖或更新当前计划"""
        self.tasks = [{"id": i+1, "task": t, "status": "todo"} for i, t in enumerate(new_tasks)]
        # 生成 Plan ID：基于 goal 的 hash（前 8 位）
        if self.current_goal:
            goal_hash = hashlib.md5(self.current_goal.encode()).hexdigest()[:8]
            self.plan_id = goal_hash

    def mark_done(self, task_id: int):
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = "done"

    def get_formatted_plan(self) -> str:
        if not self.tasks:
            return "目前暂无详细规划。"
        res = [f"目标: {self.current_goal}"]
        for t in self.tasks:
            icon = "✅" if t["status"] == "done" else "⏳"
            res.append(f"{t['id']}. {icon} {t['task']}")
        return "\n".join(res)

    def has_incomplete_tasks(self) -> bool:
        """检查是否有未完成的任务"""
        return any(t["status"] != "done" for t in self.tasks)

    def is_plan_complete(self) -> bool:
        """检查 Plan 是否全部完成"""
        if not self.tasks:
            return False
        return all(t["status"] == "done" for t in self.tasks)

    def clear_plan(self):
        """清除当前计划（用于任务完成后重置）"""
        # 🔥 新增：将当前 goal 添加到已完成列表
        if self.current_goal:
            self.completed_goals.append({
                "goal": self.current_goal,
                "timestamp": datetime.now().isoformat(),
                "plan_id": self.plan_id
            })
            # 限制已完成任务的历史记录数量
            if len(self.completed_goals) > 10:
                self.completed_goals = self.completed_goals[-10:]

        self.tasks = []
        self.current_goal = ""
        self.plan_id = ""
        print(" [🔄] Plan 已清除，准备接受新任务")

    def get_plan_id(self) -> str:
        """获取当前 Plan 的唯一标识符"""
        return self.plan_id

    def get_completed_goals(self) -> List[Dict]:
        """获取已完成任务列表"""
        return self.completed_goals

    def is_goal_completed(self, goal: str) -> bool:
        """检查某个任务是否已经完成（改进的相似度检测）"""

        def extract_core_keywords(text):
            """提取核心关键词（单字或双字组合）"""
            import re
            keywords = set()

            # 提取英文单词（保留完整单词）
            english_words = re.findall(r'[a-zA-Z]{2,}', text)
            for word in english_words:
                keywords.add(word.lower())

            # 提取中文单字和双字组合（核心概念）
            # 例如："记忆摘要格式压缩" → "记忆", "摘要", "格式", "压缩", "记忆摘要", "摘要格式", "格式压缩"
            chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', text)
            for phrase in chinese_chars:
                # 单字（可能是核心概念）
                for char in phrase:
                    keywords.add(char)
                # 双字组合
                for i in range(len(phrase) - 1):
                    keywords.add(phrase[i:i+2])
                # 三字组合（可选）
                for i in range(len(phrase) - 2):
                    keywords.add(phrase[i:i+3])

            return keywords

        goal_keywords = extract_core_keywords(goal)

        for completed in self.completed_goals:
            completed_keywords = extract_core_keywords(completed["goal"])

            # 方法1：字符串包含（原逻辑）
            if goal in completed["goal"] or completed["goal"] in goal:
                return True

            # 方法2：关键词重叠度（至少匹配30%）
            if goal_keywords and completed_keywords:
                overlap = goal_keywords & completed_keywords
                # 使用 min 作为基数（更宽松）
                similarity = len(overlap) / min(len(goal_keywords), len(completed_keywords))
                if similarity >= 0.3:  # 至少30%相似（宽松阈值）
                    return True

        return False

    def to_dict(self) -> Dict:
        """序列化为字典，用于保存到会话文件"""
        return {
            "current_goal": self.current_goal,
            "tasks": self.tasks,
            "plan_id": self.plan_id,
            "completed_goals": self.completed_goals  # 🔥 新增：保存已完成任务
        }

    def from_dict(self, data: Dict):
        """从字典恢复状态"""
        self.current_goal = data.get("current_goal", "")
        self.tasks = data.get("tasks", [])
        self.plan_id = data.get("plan_id", "")
        self.completed_goals = data.get("completed_goals", [])  # 🔥 新增：恢复已完成任务

    def validate_state(self) -> Tuple[bool, List[str]]:
        """
        🔥🔥🔥 问题 4 修复：验证 Plan 状态一致性
        
        检查以下条件：
        1. current_goal 和 tasks 的对应关系
        2. completed_goals 中不应包含 current_goal
        3. tasks 状态格式正确
        4. 如果 current_goal 为空，tasks 也应为空
        
        Returns:
            (is_valid, issues_list)
            is_valid: 是否通过验证
            issues_list: 问题列表（为空表示无问题）
        """
        issues = []
        
        # 检查 1: current_goal 和 tasks 的对应关系
        has_goal = bool(self.current_goal.strip())
        has_tasks = bool(self.tasks)
        
        if has_goal and not has_tasks:
            issues.append("Goal 存在但 tasks 为空")
        
        if not has_goal and has_tasks:
            issues.append("Tasks 存在但 Goal 为空（数据不一致）")
        
        # 检查 2: completed_goals 中不应包含 current_goal
        if self.current_goal:
            for completed in self.completed_goals:
                if completed.get("goal") == self.current_goal:
                    issues.append(f"Goal '{self.current_goal}' 既在 current 又在 completed 中")
                    break
        
        # 检查 3: tasks 状态格式
        for task in self.tasks:
            if not isinstance(task, dict):
                issues.append(f"Task 格式非法: {task}")
                continue
            
            task_id = task.get("id")
            status = task.get("status")
            
            if status not in ["todo", "done"]:
                issues.append(f"Task {task_id} 状态非法: {status}（应为 'todo' 或 'done'）")
        
        # 检查 4: 空 Goal 时，tasks 也应为空
        if not self.current_goal and self.tasks:
            issues.append("Goal 为空时，tasks 应也为空")
        
        is_valid = len(issues) == 0
        return is_valid, issues

    def auto_fix(self) -> bool:
        """
        🔥🔥🔥 问题 4 修复：自动修复 Plan 状态
        
        尝试修复验证中发现的问题。
        
        Returns:
            是否成功修复
        """
        is_valid, issues = self.validate_state()
        
        if is_valid:
            return True  # 无需修复
        
        print(f"\n[⚠️] 检测到 Plan 状态不一致，尝试自动修复...")
        
        for issue in issues:
            print(f"   问题: {issue}")
        
        # 修复策略：如果问题太多，清除 Plan
        if len(issues) > 2:
            print(f"[🔴] 问题过多，清除当前 Plan")
            self.clear_plan()
            return True
        
        # 修复策略：如果 Goal 为空但有 tasks，清除 tasks
        if not self.current_goal and self.tasks:
            print(f"[🟡] 清除孤立的 tasks")
            self.tasks = []
        
        # 修复策略：如果 completed_goals 中包含 current_goal，移除
        if self.current_goal:
            self.completed_goals = [
                g for g in self.completed_goals 
                if g.get("goal") != self.current_goal
            ]
        
        # 再次验证
        is_valid_after, issues_after = self.validate_state()
        
        if is_valid_after:
            print(f"[✅] Plan 状态已修复")
            return True
        else:
            print(f"[❌] 自动修复失败，剩余问题: {len(issues_after)}")
            print(f"   仍有问题: {issues_after}")
            # 最后的选择：清除 Plan
            self.clear_plan()
            return True