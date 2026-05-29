import argparse
import os
import sys
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from core.plan import PlanManager  # 🔥 改为导入类而非全局实例
from core.memory_manager import MemoryManager

# 1. 核心引擎导入
from core.engine import AgentEngine

# 2. 优化后的工具导入
from tools import get_default_tools

# Git 辅助函数导入
from tools.git_tool import has_uncommitted_changes, create_snapshot, get_git_status_dict

console = Console()


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def load_env_file(env_path=None):
    """Load simple KEY=VALUE pairs from a .env file without overriding existing env vars."""
    path = Path(env_path or ".env")
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_agent_config():
    """Return API configuration from environment variables or local .env file."""
    load_env_file()
    api_key = os.getenv("MINI_CLAUDE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MINI_CLAUDE_BASE_URL", DEFAULT_BASE_URL)

    if not api_key:
        raise RuntimeError(
            "未检测到 API Key。请设置环境变量 MINI_CLAUDE_API_KEY，"
            "或在项目根目录创建 .env 文件。可参考 .env.example。"
        )

    return {"api_key": api_key, "base_url": base_url}


def print_task_board(plan_manager):
    """简单的任务看板输出"""
    if not plan_manager.tasks:
        return

    console.print("\n" + "─" * 50)
    console.print("[bold cyan]📋 任务看板[/bold cyan]")
    console.print("─" * 50)

    for task in plan_manager.tasks:
        status_icon = "✅" if task["status"] == "done" else "⏳"

        status_color = "green" if task["status"] == "done" else "dim"

        console.print(f"  {status_icon} [{status_color}]{task['task']}[/{status_color}]")

    console.print("─" * 50 + "\n")


def print_banner():
    banner = """
    [bold cyan]Claude Code (Lite) - Python 研究版[/bold cyan]
    [dim]基于工程化代理架构 | 核心循环 [/dim]
    输入 [bold red]'exit'[/bold red] 退出，输入 [bold yellow]'/clear'[/bold yellow] 清空上下文
    """
    console.print(Panel(banner, border_style="cyan"))


#   main改为异步
async def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Mini-Claude Agent 控制台")
    parser.add_argument("--session", type=str, default="default", help="指定会话 ID")
    parser.add_argument("--model", type=str, default="gpt-5.5", help="指定使用的模型")
    args = parser.parse_args()

    print(f"--- 🚀 正在启动会话: {args.session} ---")

    agent_config = get_agent_config()

    # 🔥 创建独立的 PlanManager 实例（每个 session 独立）
    plan_manager = PlanManager()

    # 3. 初始化共享记忆管理器与工具集
    # tools 和 AgentEngine/ContextManager 共用同一个 MemoryManager，避免启动时重复加载长期记忆索引。
    memory_manager = MemoryManager(plan_manager=plan_manager)
    tools_list = get_default_tools(plan_manager=plan_manager, memory_manager=memory_manager)

    # 4. 初始化核心引擎
    engine = AgentEngine(
        tools=tools_list,
        model=args.model,
        plan_manager=plan_manager,
        session_id=args.session,
        base_url=agent_config["base_url"],
        api_key=agent_config["api_key"],
        max_history=150,
        min_keep=8,
        memory_manager=memory_manager
    )
    engine.plan_manager = plan_manager

    # ========== Git 状态检测 ==========
    if has_uncommitted_changes():
        console.print("\n[yellow]⚠️  检测到未提交的 Git 修改！[/yellow]")
        status = get_git_status_dict()
        if status["files"]:
            console.print(f"[dim]修改的文件 ({len(status['files'])} 个):[/dim]")
            for f in status["files"][:5]:
                console.print(f"  [dim]- {f}[/dim]")
            if len(status["files"]) > 5:
                console.print(f"  [dim]... 还有 {len(status['files']) - 5} 个文件[/dim]")

        console.print("\n[cyan]请选择操作:[/cyan]")
        console.print("  [1] 继续运行 (修改可能在后续被提交)")
        console.print("  [2] 自动保存快照后继续")
        console.print("  [3] 退出 (手动处理后再启动)")

        choice = Prompt.ask("请输入选项", choices=["1", "2", "3"], default="1")

        if choice == "2":
            success, msg = create_snapshot("🚀 [Startup] Pre-session snapshot")
            if success:
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(f"[red]{msg}[/red]")
        elif choice == "3":
            console.print("[dim]已退出，请手动处理 Git 状态后重新启动。[/dim]")
            return
    # ====================================

    print_banner()
    print_task_board(plan_manager)  # 🔥 传入 plan_manager 参数

    # 5. REPL 交互循环
    while True:
        try:
            user_input = console.input("\n[bold green]>>> [/bold green]").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                console.print("[italic white]再见，欢迎下次光临！[/italic white]")
                break

            if user_input == "/clear":
                engine.context.messages = []
                engine.context.history_summary = ""
                engine.plan_manager.clear_plan()  # 🔥 同时清除 Plan 状态
                engine.save_session()
                console.print("[dim]上下文、摘要记忆和任务计划已彻底清空。[/dim]")
                continue

            console.print("\n[dim]Claude 正在思考...[/dim]")

            # --- 核心修改：使用 await 调用异步的 execute_query ---
            final_response = await engine.execute_query(user_input)
            # -----------------------------------------------

            console.print("\n" + "─" * 50)
            console.print(Markdown(final_response))
            console.print("─" * 50 + "\n")

            # 显示更新后的任务看板
            print_task_board(plan_manager)  # 🔥 传入 plan_manager 参数

        except KeyboardInterrupt:
            console.print("\n[yellow]已强制停止当前操作。[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[bold red]发生运行错误:[/bold red] {str(e)}")

if __name__ == "__main__":
    #  使用 asyncio 启动入口
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass