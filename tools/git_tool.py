# tools/git_tool.py
# Git 自动化保险与物理回溯系统
import os
import subprocess
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from .base import BaseTool


# ==================== 数据模型 ====================

class GitStatusArgs(BaseModel):
    """获取 Git 状态的参数"""
    pass


class GitCommitArgs(BaseModel):
    """提交快照的参数"""
    message: str = Field(default="", description="提交信息（可选，默认自动生成）")


class GitRollbackArgs(BaseModel):
    """回滚代码的参数"""
    commit_hash: str = Field(default="HEAD~1", description="要回滚到的 commit hash 或引用（默认上一个提交）")


# ==================== Git 状态检查 ====================

class GitStatusTool(BaseTool):
    name = "get_git_status"
    description = "获取当前 Git 仓库状态，包括未提交的修改、暂存区状态等"
    args_schema = GitStatusArgs

    def run(self) -> str:
        """检查 Git 状态"""
        try:
            # 检查是否在 Git 仓库中
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                return "❌ 当前目录不是 Git 仓库"

            # 获取状态
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            # 获取分支名
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            branch = branch_result.stdout.strip() or "HEAD detached"

            # 获取最近的 commit
            log_result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            last_commit = log_result.stdout.strip() or "无提交历史"

            # 解析状态
            lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []

            if not lines or lines == ['']:
                return f"✅ 工作区干净\n📍 分支: {branch}\n📌 最新提交: {last_commit}"

            modified = []
            staged = []
            untracked = []

            for line in lines:
                if not line:
                    continue
                index_status = line[0] if len(line) > 0 else ' '
                work_status = line[1] if len(line) > 1 else ' '
                filepath = line[3:] if len(line) > 3 else ''

                if index_status in 'MADRC':  # 暂存区有变更
                    staged.append(filepath)
                if work_status in 'MD':  # 工作区有变更
                    modified.append(filepath)
                if index_status == '?':  # 未跟踪
                    untracked.append(filepath)

            output = [f"📍 分支: {branch}", f"📌 最新提交: {last_commit}"]

            if staged:
                output.append(f"\n🟢 暂存区 ({len(staged)} 个文件):")
                for f in staged[:10]:
                    output.append(f"  + {f}")
                if len(staged) > 10:
                    output.append(f"  ... 还有 {len(staged) - 10} 个文件")

            if modified:
                output.append(f"\n🟡 工作区修改 ({len(modified)} 个文件):")
                for f in modified[:10]:
                    output.append(f"  ~ {f}")
                if len(modified) > 10:
                    output.append(f"  ... 还有 {len(modified) - 10} 个文件")

            if untracked:
                output.append(f"\n⚪ 未跟踪 ({len(untracked)} 个文件):")
                for f in untracked[:10]:
                    output.append(f"  ? {f}")
                if len(untracked) > 10:
                    output.append(f"  ... 还有 {len(untracked) - 10} 个文件")

            return "\n".join(output)

        except Exception as e:
            return f"❌ 获取 Git 状态失败: {str(e)}"


# ==================== Git 提交快照 ====================

class GitCommitTool(BaseTool):
    name = "commit_snapshot"
    description = "创建 Git 快照提交，自动暂存所有修改并提交。用于在关键步骤后保存进度。"
    args_schema = GitCommitArgs

    def run(self, message: str = "") -> str:
        """创建快照提交"""
        try:
            # 检查 Git 仓库
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                return "❌ 当前目录不是 Git 仓库"

            # 检查是否有修改
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if not status_result.stdout.strip():
                return "ℹ️ 没有需要提交的修改"

            # 生成提交信息
            if not message:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                message = f"🔄 [Auto Snapshot] {timestamp}"

            # 暂存所有修改
            add_result = subprocess.run(
                ["git", "add", "-A"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if add_result.returncode != 0:
                return f"❌ 暂存失败: {add_result.stderr}"

            # 提交
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if commit_result.returncode != 0:
                if "nothing to commit" in commit_result.stdout:
                    return "ℹ️ 没有需要提交的修改"
                return f"❌ 提交失败: {commit_result.stderr}"

            # 获取新的 commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            commit_hash = hash_result.stdout.strip()[:8]

            return f"✅ 快照已保存: {commit_hash}\n📝 {message}"

        except Exception as e:
            return f"❌ 创建快照失败: {str(e)}"


# ==================== Git 回滚 ====================

class GitRollbackTool(BaseTool):
    name = "git_rollback"
    description = "回滚代码到指定提交。用于在连续失败时恢复到安全状态。"
    args_schema = GitRollbackArgs

    def run(self, commit_hash: str = "HEAD~1") -> str:
        """回滚到指定提交"""
        try:
            # 检查 Git 仓库
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                return "❌ 当前目录不是 Git 仓库"

            # 获取当前 commit hash（用于回滚前记录）
            current_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            current_hash = current_result.stdout.strip()[:8]

            # 验证目标 commit 是否存在
            verify_result = subprocess.run(
                ["git", "rev-parse", "--verify", commit_hash],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if verify_result.returncode != 0:
                return f"❌ 无法找到提交: {commit_hash}"

            target_hash = verify_result.stdout.strip()[:8]

            # 执行硬重置
            reset_result = subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            if reset_result.returncode != 0:
                return f"❌ 回滚失败: {reset_result.stderr}"

            # 清理未跟踪的文件（可选）
            clean_result = subprocess.run(
                ["git", "clean", "-fd"],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )

            return f"✅ 已回滚到: {target_hash}\n📍 之前位置: {current_hash}\n⚠️ 所有未提交的修改已丢失"

        except Exception as e:
            return f"❌ 回滚失败: {str(e)}"


# ==================== 辅助函数（供其他模块调用） ====================

def get_git_status_dict() -> Dict[str, any]:
    """获取 Git 状态的字典格式（供外部调用）"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        return {
            "is_repo": True,
            "has_changes": bool(lines and lines != ['']),
            "files": [line[3:] for line in lines if line and len(line) > 3]
        }
    except:
        return {"is_repo": False, "has_changes": False, "files": []}


def has_uncommitted_changes() -> bool:
    """快速检查是否有未提交的修改"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        return bool(result.stdout.strip())
    except:
        return False


def create_snapshot(message: str = "") -> Tuple[bool, str]:
    """创建快照的简化接口（供外部调用）"""
    tool = GitCommitTool()
    result = tool.run(message=message)
    return ("✅" in result, result)


def rollback_to(commit_hash: str = "HEAD~1") -> Tuple[bool, str]:
    """回滚的简化接口（供外部调用）"""
    tool = GitRollbackTool()
    result = tool.run(commit_hash=commit_hash)
    return ("✅" in result, result)


# ==================== 影子分支系统 ====================

def start_plan_branch(plan_id: str) -> Tuple[bool, str]:
    """
    创建并切换到 Plan 级别的影子分支

    Args:
        plan_id: Plan ID（会创建 agent/plan-{id} 分支）

    Returns:
        (success, message): 成功标志和消息
    """
    try:
        # 检查 Git 仓库
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            return (False, "❌ 当前目录不是 Git 仓库")

        # 获取当前分支
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        current_branch = branch_result.stdout.strip()

        # 影子分支名
        shadow_branch = f"agent/plan-{plan_id}"

        # 如果已经在影子分支上，直接返回
        if current_branch == shadow_branch:
            return (True, f"ℹ️ 已在影子分支上: {shadow_branch}")

        # 创建并切换到影子分支（如果已存在则切换）
        checkout_result = subprocess.run(
            ["git", "checkout", "-B", shadow_branch],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        if checkout_result.returncode != 0:
            return (False, f"❌ 创建/切换影子分支失败: {checkout_result.stderr}")

        return (True, f"✅ 已切换到影子分支: {shadow_branch}\n📍 原分支: {current_branch}")

    except Exception as e:
        return (False, f"❌ start_plan_branch 失败: {str(e)}")


def finalize_plan(plan_id: str, description: str = "") -> Tuple[bool, str]:
    """
    将 Plan 影子分支 squash merge 回主分支，生成一条干净的提交记录

    Args:
        plan_id: Plan ID
        description: Plan 描述（用于提交信息）

    Returns:
        (success, message): 成功标志和消息
    """
    try:
        # 检查 Git 仓库
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            return (False, "❌ 当前目录不是 Git 仓库")

        # 影子分支名
        shadow_branch = f"agent/plan-{plan_id}"

        # 获取当前分支
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        current_branch = branch_result.stdout.strip()

        # 如果不在影子分支上，提示错误
        if current_branch != shadow_branch:
            return (False, f"❌ 当前不在影子分支上: {current_branch} (期望: {shadow_branch})")

        # 暂存所有修改
        add_result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if add_result.returncode != 0:
            return (False, f"❌ 暂存失败: {add_result.stderr}")

        # 检查是否有修改
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        # 如果有修改，先提交
        if status_result.stdout.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"🔒 [Plan-{plan_id}] {description}" if description else f"🔒 [Plan-{plan_id}] {timestamp}"

            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
                return (False, f"❌ 提交失败: {commit_result.stderr}")

        # 切换到主分支（尝试 main 或 master）
        main_branch = None
        for branch_name in ["main", "master"]:
            checkout_result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if checkout_result.returncode == 0:
                main_branch = branch_name
                break

        if not main_branch:
            return (False, f"❌ 无法切换到主分支（尝试了 main 和 master）")

        # Squash merge 影子分支
        merge_result = subprocess.run(
            ["git", "merge", "--squash", shadow_branch],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        # 检查是否有合并冲突
        if merge_result.returncode != 0:
            # 回退到影子分支
            subprocess.run(["git", "checkout", shadow_branch], capture_output=True)
            return (False, f"❌ Squash merge 失败（可能有冲突）: {merge_result.stderr}")

        # 检查是否有需要提交的修改
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        if not status_result.stdout.strip():
            # 没有修改，直接删除影子分支
            subprocess.run(["git", "branch", "-D", shadow_branch], capture_output=True)
            return (True, f"ℹ️ Plan 无修改，已删除影子分支: {shadow_branch}")

        # 生成干净的提交信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_msg = f"🎯 [Plan-{plan_id}] {description}" if description else f"🎯 [Plan-{plan_id}] Plan completed {timestamp}"

        commit_result = subprocess.run(
            ["git", "commit", "-m", final_msg],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if commit_result.returncode != 0:
            return (False, f"❌ 最终提交失败: {commit_result.stderr}")

        # 获取 commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        commit_hash = hash_result.stdout.strip()[:8]

        # 删除影子分支
        subprocess.run(["git", "branch", "-D", shadow_branch], capture_output=True)

        return (True, f"✅ Plan 已归档: {commit_hash}\n📝 {final_msg}\n🗑️ 已删除影子分支: {shadow_branch}")

    except Exception as e:
        return (False, f"❌ finalize_plan 失败: {str(e)}")


def start_task_branch(task_id: int) -> Tuple[bool, str]:
    """
    创建并切换到影子分支，实现任务级别的分支隔离

    Args:
        task_id: 任务 ID（会创建 agent/task-{id} 分支）

    Returns:
        (success, message): 成功标志和消息
    """
    try:
        # 检查 Git 仓库
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            return (False, "❌ 当前目录不是 Git 仓库")

        # 获取当前分支
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        current_branch = branch_result.stdout.strip()

        # 影子分支名
        shadow_branch = f"agent/task-{task_id}"

        # 如果已经在影子分支上，直接返回
        if current_branch == shadow_branch:
            return (True, f"ℹ️ 已在影子分支上: {shadow_branch}")

        # 创建并切换到影子分支（如果已存在则切换）
        checkout_result = subprocess.run(
            ["git", "checkout", "-B", shadow_branch],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        if checkout_result.returncode != 0:
            return (False, f"❌ 创建/切换影子分支失败: {checkout_result.stderr}")

        return (True, f"✅ 已切换到影子分支: {shadow_branch}\n📍 原分支: {current_branch}")

    except Exception as e:
        return (False, f"❌ start_task_branch 失败: {str(e)}")


def finalize_task(task_id: int, description: str = "") -> Tuple[bool, str]:
    """
    将影子分支 squash merge 回主分支，生成一条干净的提交记录

    Args:
        task_id: 任务 ID
        description: 任务描述（用于提交信息）

    Returns:
        (success, message): 成功标志和消息
    """
    try:
        # 检查 Git 仓库
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            return (False, "❌ 当前目录不是 Git 仓库")

        # 影子分支名
        shadow_branch = f"agent/task-{task_id}"

        # 获取当前分支
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        current_branch = branch_result.stdout.strip()

        # 如果不在影子分支上，提示错误
        if current_branch != shadow_branch:
            return (False, f"❌ 当前不在影子分支上: {current_branch} (期望: {shadow_branch})")

        # 暂存所有修改
        add_result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if add_result.returncode != 0:
            return (False, f"❌ 暂存失败: {add_result.stderr}")

        # 检查是否有修改
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        # 如果有修改，先提交
        if status_result.stdout.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"🔒 [Task-{task_id}] {description}" if description else f"🔒 [Task-{task_id}] {timestamp}"

            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if commit_result.returncode != 0 and "nothing to commit" not in commit_result.stdout:
                return (False, f"❌ 提交失败: {commit_result.stderr}")

        # 切换到主分支（尝试 main 或 master）
        main_branch = None
        for branch_name in ["main", "master"]:
            checkout_result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            if checkout_result.returncode == 0:
                main_branch = branch_name
                break

        if not main_branch:
            return (False, f"❌ 无法切换到主分支（尝试了 main 和 master）")

        # Squash merge 影子分支
        merge_result = subprocess.run(
            ["git", "merge", "--squash", shadow_branch],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        # 检查是否有合并冲突
        if merge_result.returncode != 0:
            # 回退到影子分支
            subprocess.run(["git", "checkout", shadow_branch], capture_output=True)
            return (False, f"❌ Squash merge 失败（可能有冲突）: {merge_result.stderr}")

        # 检查是否有需要提交的修改
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )

        if not status_result.stdout.strip():
            # 没有修改，直接删除影子分支
            subprocess.run(["git", "branch", "-D", shadow_branch], capture_output=True)
            return (True, f"ℹ️ 任务无修改，已删除影子分支: {shadow_branch}")

        # 生成干净的提交信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_msg = f"🎯 [Task-{task_id}] {description}" if description else f"🎯 [Task-{task_id}] Task completed {timestamp}"

        commit_result = subprocess.run(
            ["git", "commit", "-m", final_msg],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if commit_result.returncode != 0:
            return (False, f"❌ 最终提交失败: {commit_result.stderr}")

        # 获取 commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        commit_hash = hash_result.stdout.strip()[:8]

        # 删除影子分支
        subprocess.run(["git", "branch", "-D", shadow_branch], capture_output=True)

        return (True, f"✅ 任务已归档: {commit_hash}\n📝 {final_msg}\n🗑️ 已删除影子分支: {shadow_branch}")

    except Exception as e:
        return (False, f"❌ finalize_task 失败: {str(e)}")
