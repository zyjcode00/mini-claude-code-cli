import os
import time
from pydantic import BaseModel, Field
from .base import BaseTool

class CleanArgs(BaseModel):
    days: int = Field(default=7, description="清理多少天之前的旧会话文件")

class SessionCleanerTool(BaseTool):
    name = "clean_old_sessions"
    description = "清理长期未使用的旧会话 JSON 文件，释放磁盘空间。"
    args_schema = CleanArgs

    def run(self, days: int = 7) -> str:
        session_dir = "sessions"
        if not os.path.exists(session_dir):
            return "Session 目录不存在，无需清理。"
        
        now = time.time()
        count = 0
        for f in os.listdir(session_dir):
            path = os.path.join(session_dir, f)
            # 检查文件最后修改时间
            if os.path.getmtime(path) < now - (days * 86400):
                os.remove(path)
                count += 1
        
        return f"清理完成！共删除 {count} 个超过 {days} 天的旧会话文件。"