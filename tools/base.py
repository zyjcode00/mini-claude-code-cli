# tools/base.py
# 统一工具输入输出格式
# 使用 pydantic 定义 BaseTool。
# 确保每个工具都能导出符合 Anthropic API 要求的 input_schema。
import os
import subprocess
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Type, Any, Dict

# ---  定义基类 (所有工具的爸爸) ---
class BaseTool(ABC):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = None

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

    def to_anthropic_spec(self) -> Dict[str, Any]:
        """转换为 Anthropic/OpenAI 工具调用协议格式"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_schema.model_json_schema() if self.args_schema else {}
        }
