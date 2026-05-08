# tools/symbol_tool.py
# 基于 AST 的符号地图系统，提供符号索引和查找功能
# 使用 Python 的 ast 模块实现静态分析
# 修复记录：已验证 edit_file 工具能够正确处理 CRLF 文件 (2026-04-05)
import ast
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .base import BaseTool


class SymbolInfo(BaseModel):
    """符号信息模型"""
    name: str = Field(..., description="符号名称")
    type: str = Field(..., description="符号类型：class 或 function")
    file: str = Field(..., description="文件路径")
    start_line: int = Field(..., description="起始行号")
    end_line: int = Field(..., description="结束行号")
    docstring: Optional[str] = Field(None, description="文档字符串")
    parent: Optional[str] = Field(None, description="父类名（用于嵌套类或方法）")
    is_async: bool = Field(False, description="是否为异步函数")


class SymbolExtractor:
    """符号提取器：遍历项目，提取所有类和函数的定义"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.symbols: List[SymbolInfo] = []
    
    def extract_from_file(self, file_path: Path) -> List[SymbolInfo]:
        """从单个 Python 文件提取符号"""
        symbols = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # 提取类定义
                    docstring = ast.get_docstring(node)
                    symbol = SymbolInfo(
                        name=node.name,
                        type="class",
                        file=str(file_path.relative_to(self.root_path)),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=docstring
                    )
                    symbols.append(symbol)
                    
                    # 提取类中的方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                            method_docstring = ast.get_docstring(item)
                            is_async = isinstance(item, ast.AsyncFunctionDef)
                            method_symbol = SymbolInfo(
                                name=item.name,
                                type="function",
                                file=str(file_path.relative_to(self.root_path)),
                                start_line=item.lineno,
                                end_line=item.end_lineno or item.lineno,
                                docstring=method_docstring,
                                parent=node.name,
                                is_async=is_async
                            )
                            symbols.append(method_symbol)
                
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # 提取顶层函数（排除类中的方法，已在上面处理）
                    if not self._is_method(node, tree):
                        docstring = ast.get_docstring(node)
                        is_async = isinstance(node, ast.AsyncFunctionDef)
                        symbol = SymbolInfo(
                            name=node.name,
                            type="function",
                            file=str(file_path.relative_to(self.root_path)),
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            docstring=docstring,
                            is_async=is_async
                        )
                        symbols.append(symbol)
        
        except (SyntaxError, ValueError, OSError) as e:
            # 忽略解析错误的文件
            pass
        
        return symbols
    
    def _is_method(self, func_node, tree) -> bool:
        """判断函数是否是类的方法"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def extract_from_project(self, exclude_dirs: List[str] = None) -> List[SymbolInfo]:
        """遍历项目，提取所有符号"""
        if exclude_dirs is None:
            exclude_dirs = ['__pycache__', '.git', 'node_modules', 'venv', '.venv', 'env', '.env']
        
        self.symbols = []
        
        for root, dirs, files in os.walk(self.root_path):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    symbols = self.extract_from_file(file_path)
                    self.symbols.extend(symbols)
        
        return self.symbols
    
    def find_symbol(self, name: str) -> List[SymbolInfo]:
        """根据名称查找符号"""
        results = []
        for symbol in self.symbols:
            if name in symbol.name:
                results.append(symbol)
        return results


# --- 工具定义 ---

class ListSymbolsArgs(BaseModel):
    path: str = Field(".", description="项目根路径，默认为当前目录")
    symbol_type: str = Field("all", description="符号类型过滤：all/class/function")


class ListSymbolsTool(BaseTool):
    name = "list_all_symbols"
    description = "返回项目所有 API 的大纲。列出所有类和函数的定义位置、行号和文档字符串，帮助快速理解陌生模块。"
    args_schema = ListSymbolsArgs
    
    def run(self, path: str = ".", symbol_type: str = "all") -> str:
        """
        列出项目中的所有符号
        
        Args:
            path: 项目根路径
            symbol_type: 符号类型过滤（all/class/function）
        
        Returns:
            格式化的符号列表
        """
        try:
            extractor = SymbolExtractor(path)
            symbols = extractor.extract_from_project()
            
            # 过滤符号类型
            if symbol_type != "all":
                symbols = [s for s in symbols if s.type == symbol_type]
            
            if not symbols:
                return "未找到任何符号定义"
            
            # 格式化输出
            output_lines = []
            output_lines.append(f"📊 项目符号索引 (共 {len(symbols)} 个定义)\n")
            output_lines.append("=" * 80 + "\n")
            
            # 按文件分组
            symbols_by_file: Dict[str, List[SymbolInfo]] = {}
            for symbol in symbols:
                if symbol.file not in symbols_by_file:
                    symbols_by_file[symbol.file] = []
                symbols_by_file[symbol.file].append(symbol)
            
            # 输出
            for file, file_symbols in sorted(symbols_by_file.items()):
                output_lines.append(f"\n📁 {file}")
                output_lines.append("-" * 80)
                
                for symbol in sorted(file_symbols, key=lambda s: s.start_line):
                    # 构建符号显示名称
                    if symbol.parent:
                        display_name = f"  {symbol.parent}.{symbol.name}"
                    else:
                        display_name = f"  {symbol.name}"
                    
                    # 类型标识
                    type_icon = "📦" if symbol.type == "class" else "⚡"
                    
                    output_lines.append(f"{type_icon} {display_name} (L{symbol.start_line}-L{symbol.end_line})")
                    
                    # 显示 docstring（截取前两行）
                    if symbol.docstring:
                        doc_lines = symbol.docstring.strip().split('\n')[:2]
                        for line in doc_lines:
                            output_lines.append(f"    💬 {line.strip()}")
            
            return "\n".join(output_lines)
        
        except Exception as e:
            return f"错误: {str(e)}"


class FindSymbolArgs(BaseModel):
    name: str = Field(..., description="要查找的符号名称（支持部分匹配）")
    path: str = Field(".", description="项目根路径，默认为当前目录")


class FindSymbolTool(BaseTool):
    name = "find_symbol_definition"
    description = "根据名称直接定位代码块。返回符号的定义位置（文件、起始/结束行号）、Docstring 和代码片段，跳过盲目的 grep 搜索。"
    args_schema = FindSymbolArgs
    
    def run(self, name: str, path: str = ".") -> str:
        """
        查找符号定义
        
        Args:
            name: 符号名称
            path: 项目根路径
        
        Returns:
            符号定义信息
        """
        try:
            extractor = SymbolExtractor(path)
            symbols = extractor.extract_from_project()
            
            # 查找匹配的符号
            matches = []
            for symbol in symbols:
                if name in symbol.name:
                    matches.append(symbol)
            
            if not matches:
                return f"未找到符号: {name}"
            
            # 格式化输出
            output_lines = []
            output_lines.append(f"🔍 找到 {len(matches)} 个匹配项\n")
            output_lines.append("=" * 80 + "\n")
            
            for idx, symbol in enumerate(matches, 1):
                output_lines.append(f"\n[{idx}] {symbol.type.upper()}: {symbol.name}")
                if symbol.parent:
                    output_lines.append(f"    父类: {symbol.parent}")
                output_lines.append(f"    📁 文件: {symbol.file}")
                output_lines.append(f"    📍 行号: L{symbol.start_line} - L{symbol.end_line}")
                
                if symbol.docstring:
                    output_lines.append(f"\n    💬 文档字符串:")
                    for line in symbol.docstring.strip().split('\n')[:5]:
                        output_lines.append(f"       {line}")
                
                # 尝试读取代码片段
                try:
                    file_path = Path(path) / symbol.file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # 显示代码片段（前 10 行）
                    code_snippet = lines[symbol.start_line-1:min(symbol.start_line+9, len(lines))]
                    output_lines.append(f"\n    📝 代码片段 (前 10 行):")
                    for i, line in enumerate(code_snippet, symbol.start_line):
                        output_lines.append(f"    {i:4d} | {line.rstrip()}")
                
                except Exception:
                    pass
                
                output_lines.append("\n" + "-" * 80)
            
            return "\n".join(output_lines)
        
        except Exception as e:
            return f"错误: {str(e)}"