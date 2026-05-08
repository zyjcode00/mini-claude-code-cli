# tools/plan_tool.py
from pydantic import BaseModel, Field
from .base import BaseTool

class PlanArgs(BaseModel):
    goal: str = Field(..., description="本次任务的总目标")
    steps: list[str] = Field(..., description="拆解后的具体执行步骤列表")

class UpdatePlanTool(BaseTool):
    name = "manage_plan"
    description = """【第一步必调用】创建或更新任务规划。

使用时机：
- 当任务涉及多个步骤时，必须在第一轮思考中立即调用此工具
- 当需要调整已有计划时调用

调用后会生成结构化的任务清单，帮助你系统地完成工作。"""
    args_schema = PlanArgs

    def __init__(self, plan_manager):
        super().__init__()
        self.pm = plan_manager

    def run(self, goal: str, steps: list[str], **kwargs) -> str:
        print(f"\n[🔥 TOOL DEBUG] 正在执行 manage_plan! 目标: {goal}, 步骤数: {len(steps)}")

        # 🔥 新增：检查是否已完成相似任务
        if self.pm.is_goal_completed(goal):
            completed = [g for g in self.pm.completed_goals if goal in g["goal"] or g["goal"] in goal]
            if completed:
                last_completed = completed[-1]
                timestamp = last_completed.get("timestamp", "")[:19]
                return (
                    f"⚠️ 该任务已在之前完成！\n\n"
                    f"已完成记录:\n"
                    f"  ✅ [{timestamp}] {last_completed['goal']}\n\n"
                    f"请勿重复执行。如果需要重新执行，请明确说明理由。"
                )

        # 🔥 新增：检查是否有未完成的 Plan
        if self.pm.has_incomplete_tasks():
            current_plan = self.pm.get_formatted_plan()

            # 找到下一个未完成的步骤
            next_task = next((t for t in self.pm.tasks if t["status"] != "done"), None)

            if next_task:
                suggestion = f"📌 下一步建议:\n  执行步骤 {next_task['id']}: {next_task['task']}\n"
            else:
                suggestion = "📌 所有步骤已完成，建议检查是否有遗漏。\n"

            return (
                f"⚠️ 当前有未完成的 Plan！\n\n"
                f"{current_plan}\n\n"
                f"{suggestion}\n"
                f"💡 提示：\n"
                f"  - 用户说'继续'时，应继续执行未完成的步骤\n"
                f"  - 如确需创建新 Plan，请先完成当前 Plan 或使用 /clear 清除"
            )

        # 正常流程：创建新 Plan
        self.pm.current_goal = goal
        self.pm.update_plan(steps)

        result = self.pm.get_formatted_plan()
        print(f"[🔥 TOOL DEBUG] 更新后的计划内容:\n{result}")
        return f"规划已更新：\n{result}"


# --- 新增：标记任务完成的工具 ---
class MarkDoneArgs(BaseModel):
    task_id: int = Field(..., description="要标记为已完成的任务 ID（对应计划列表中的编号）")

class MarkDoneTool(BaseTool):
    name = "mark_task_done"
    description = """【完成即调用】将指定任务标记为已完成。

使用时机：
- 每完成一个计划步骤后，必须立即调用此工具更新进度
- 不能只口头说明完成，必须调用工具记录

参数：task_id 为计划清单中的任务编号"""
    args_schema = MarkDoneArgs

    def __init__(self, plan_manager):
        super().__init__()
        self.pm = plan_manager

    def run(self, task_id: int, **kwargs) -> str:
        # 检查任务是否存在
        task_exists = any(t["id"] == task_id for t in self.pm.tasks)
        if not task_exists:
            return f"错误: 找不到 ID 为 {task_id} 的任务。当前计划:\n{self.pm.get_formatted_plan()}"

        # 检查是否已完成
        task = next(t for t in self.pm.tasks if t["id"] == task_id)
        if task["status"] == "done":
            return f"提示: 任务 {task_id} 已经是完成状态。"

        # 标记完成
        self.pm.mark_done(task_id)
        print(f"\n[🔥 TOOL DEBUG] 任务 {task_id} 已标记完成")

        # 统计进度
        total = len(self.pm.tasks)
        done_count = sum(1 for t in self.pm.tasks if t["status"] == "done")

        result = self.pm.get_formatted_plan()
        return f"✅ 任务 {task_id} 已完成！进度: {done_count}/{total}\n\n当前计划:\n{result}"