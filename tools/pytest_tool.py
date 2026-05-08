import subprocess
from pydantic import BaseModel, Field
from .base import BaseTool

class PytestArgs(BaseModel):
    path: str = Field(..., description="要运行的测试文件或目录路径")

class PytestTool(BaseTool):
    name = "run_pytest"
    description = "运行 pytest 测试。这是验证代码正确性的唯一标准。如果失败，请根据输出的 Traceback 进行修复。"
    args_schema = PytestArgs

    def run(self, path: str) -> str:
        try:
            # 运行 pytest 并捕获输出
            result = subprocess.run(
                ["pytest", "-v", path],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                return f"✅ 测试通过！\n{result.stdout}"
            else:
                # 关键：把报错信息完整传回给 Agent
                return f"❌ 测试失败 (Exit Code {result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e:
            return f"❌ 运行测试工具出错: {str(e)}"