# ---  实现 Bash 工具 (你的老朋友) ---
import subprocess
import chardet
import os
import platform
from pydantic import BaseModel, Field
from .base import BaseTool

class BashArgs(BaseModel):
    command: str = Field(..., description="要执行的 shell 命令")

class BashTool(BaseTool):
    name = "execute_bash"
    description = "在本地系统执行 bash 命令。小心使用，确保命令安全。"
    args_schema = BashArgs

    def run(self, command: str) -> str:
        try:
            # === Windows 编码修复方案 ===
            # 1. 准备 UTF-8 环境变量（影响 Python 子进程的 stdout/stderr）
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"  # Python 3.7+ 额外保障
            
            # 2. Windows 下先设置控制台代码页为 UTF-8
            if platform.system() == "Windows":
                command = f"chcp 65001 > nul && {command}"
            
            # 3. 执行命令
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                env=env,
            )
            
            # 1. 尝试自动检测编码
            raw_output = result.stdout
            if not raw_output:
                return "Command executed with no output."

            # 优先尝试 UTF-8，失败则尝试 GBK
            try:
                output = raw_output.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    output = raw_output.decode('gbk') # Windows 专用补丁
                except UnicodeDecodeError:
                    # 最后的兜底：chardet 自动识别
                    encoding = chardet.detect(raw_output)['encoding'] or 'utf-8'
                    output = raw_output.decode(encoding, errors='replace')

            return f"STDOUT:\n{output}"
        except Exception as e:
            return f"❌ 运行出错: {str(e)}"
  