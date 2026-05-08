import os
from pydantic import BaseModel, Field
from .base import BaseTool


class ReadArgs(BaseModel):
    path: str = Field(..., description="要读取的文件路径")
    start_line: int = Field(1, description="起始行号（从 1 开始）")
    end_line: int = Field(None, description="结束行号（可选，若不传则读到文件末尾）")
    raw_mode: bool = Field(False, description="原始模式：输出无装饰符的内容，方便拷贝")
    
class FileTreeArgs(BaseModel):
    """递归列出文件的参数模型（无参数）"""
    pass

class ReadTool(BaseTool):
    name = "read_file"
    description = "读取文件内容。支持指定行范围，这在处理大文件时非常高效。建议先读取前 100 行了解结构。"
    args_schema = ReadArgs

    def run(self, path: str, start_line: int = 1, end_line: int = None, raw_mode: bool = False) -> str:
        try:
            if not os.path.exists(path):
                return f"错误: 找不到文件 {path}"

            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            total_lines = len(lines)
            s_idx = max(0, start_line - 1)
            e_idx = end_line if end_line is not None else total_lines

            selected_lines = lines[s_idx:e_idx]
            
            # 原始模式：直接输出内容，不带装饰符
            if raw_mode:
                return ''.join(selected_lines)
            
            # 正常模式：带行号和装饰符
            output = []
            for i, line in enumerate(selected_lines):
                output.append(f"{s_idx + i + 1:4d} | {line.rstrip()}")

            header = f"--- 文件: {path} (第 {start_line} 至 {min(e_idx, total_lines)} 行，共 {total_lines} 行) ---\n"
            return header + "\n".join(output) + "\n--- 读取完毕 ---"
        except Exception as e:
            return f"读取失败: {str(e)}"


class EditArgs(BaseModel):
    path: str = Field(..., description="要修改的文件相对或绝对路径")
    old_str: str = Field(..., description="文件中现有的、需要被替换的精确字符串块。必须完全匹配，包括空格和缩进。")
    new_str: str = Field(..., description="替换后的新字符串块。")

class FileEditTool(BaseTool):
    name = "edit_file"
    description = "精准修改文件内容。通过搜索 old_str 并替换为 new_str 实现。比重写整个文件更安全、更省 Token。"
    args_schema = EditArgs

    @staticmethod
    def _normalize_text(text: str) -> str:
        """归一化文本：统一换行符 + 移除行尾空格"""
        # 1. 统一换行符为 \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 2. 移除行尾空格
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        return "\n".join(lines)

    def run(self, path: str, old_str: str, new_str: str, **kwargs) -> str:
        try:
            if not os.path.exists(path):
                return f"错误: 找不到文件 {path}"

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 归一化处理
            content = self._normalize_text(content)
            old_str = self._normalize_text(old_str)
            new_str = self._normalize_text(new_str)

            # 唯一性检查
            count = content.count(old_str)
            if count == 0:
                error_msg = "错误: 在文件中找不到 old_str。\n"
                error_msg += "建议：\n"
                error_msg += "1. 使用 read_file(raw_mode=True) 获取原始内容\n"
                error_msg += "2. 检查是否有多余的空格或换行符\n"
                error_msg += "3. 如果文件较大，考虑使用 write_full_file 全量覆盖"
                return error_msg
            if count > 1:
                return f"错误: 匹配到 {count} 处相同的代码，请提供更具体的 old_str 以确保唯一性。"

            new_content = content.replace(old_str, new_str)
            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(new_content)

            return f"成功: 已更新 {path}。修改已持久化。"
        except Exception as e:
            return f"修改失败: {str(e)}"


class WriteFullFileArgs(BaseModel):
    path: str = Field(..., description="要写入的文件路径")
    content: str = Field(..., description="要写入的完整内容")

class WriteFullFileTool(BaseTool):
    name = "write_full_file"
    description = "全量写入文件内容。用于创建新文件或完全覆盖现有文件。适合复杂修改场景，避免局部匹配失败。"
    args_schema = WriteFullFileArgs

    def run(self, path: str, content: str, **kwargs) -> str:
        try:
            # 确保目录存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)

            return f"成功: 已写入 {path}"
        except Exception as e:
            return f"写入失败: {str(e)}"


class FileTreeTool(BaseTool):
    name = "list_files_recursive"
    description = "递归列出当前项目的所有文件结构。在进入新项目或寻找特定文件时，请优先使用此工具。"
    args_schema = FileTreeArgs

    def run(self, **kwargs) -> str:
        try:
            tree = []
            exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules'}

            for root, dirs, files in os.walk("."):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]

                level = root.replace(".", "").count(os.sep)
                indent = "  " * level
                folder_name = os.path.basename(root) or "."
                tree.append(f"{indent}📁 {folder_name}/")

                sub_indent = "  " * (level + 1)
                for f in files:
                    tree.append(f"{sub_indent}📄 {f}")

            return "\n".join(tree)
        except Exception as e:
            return f"获取目录树失败: {str(e)}"
