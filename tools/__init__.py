# tools/__init__.py
from .bash_tool import BashTool
from .file_tool import ReadTool, FileEditTool, FileTreeTool, WriteFullFileTool
from .plan_tool import UpdatePlanTool, MarkDoneTool
from .session_tool import SessionCleanerTool
from .pytest_tool import PytestTool
from .search_tool import SearchTool
from .symbol_tool import ListSymbolsTool, FindSymbolTool
from .git_tool import GitStatusTool, GitCommitTool, GitRollbackTool
from .memory_tool import MemorySaveTool, MemoryRecallTool, MemoryFileHistoryTool, MemoryErrorHistoryTool

def get_default_tools(plan_manager=None, memory_manager=None, memory_storage_dir="memory/long_term"):
    shared_memory_manager = memory_manager
    memory_save_tool = MemorySaveTool(memory_manager=shared_memory_manager, long_term_storage_dir=memory_storage_dir)
    shared_memory_manager = memory_save_tool.memory_manager
    memory_recall_tool = MemoryRecallTool(memory_manager=shared_memory_manager)
    memory_file_history_tool = MemoryFileHistoryTool(memory_manager=shared_memory_manager)
    memory_error_history_tool = MemoryErrorHistoryTool(memory_manager=shared_memory_manager)

    tools = [
        BashTool(),
        ReadTool(),
        FileEditTool(),
        WriteFullFileTool(),  # 新增：全量写入工具
        FileTreeTool(),
        SearchTool(),  # 新增：代码搜索工具
        ListSymbolsTool(),  # 新增：符号索引工具
        FindSymbolTool(),  # 新增：符号查找工具
        PytestTool(),
        SessionCleanerTool(),
        # Git 自动化工具
        GitStatusTool(),
        GitCommitTool(),
        GitRollbackTool(),
        # 长期记忆工具
        memory_save_tool,
        memory_recall_tool,
        memory_file_history_tool,
        memory_error_history_tool
    ]

    # 如果提供了管家，就加载规划工具
    if plan_manager:
        tools.append(UpdatePlanTool(plan_manager))
        tools.append(MarkDoneTool(plan_manager))  # 新增：标记完成工具

    return tools