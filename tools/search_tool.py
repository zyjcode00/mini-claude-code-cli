# tools/search_tool.py
# 高效的代码搜索工具，支持正则搜索、上下文行显示
# 使用 Python 实现，无需外部依赖，跨平台兼容
import os
import re
from pathlib import Path
from pydantic import BaseModel, Field
from .base import BaseTool
from typing import List, Dict, Any


class SearchArgs(BaseModel):
    pattern: str = Field(..., description="要搜索的模式（支持正则表达式）")
    path: str = Field(".", description="搜索路径，默认为当前目录")
    context: int = Field(0, description="显示匹配行前后各多少行的上下文")
    file_pattern: str = Field("*", description="文件名模式（如 *.py），支持 glob 语法")
    case_sensitive: bool = Field(True, description="是否区分大小写")


class SearchTool(BaseTool):
    name = "search_code"
    description = "在项目中搜索代码模式。支持正则表达式、上下文行显示、文件类型过滤。在进行跨文件重构前，必须先使用此工具进行全局影响评估。"
    args_schema = SearchArgs

    def run(self, pattern: str, path: str = ".", context: int = 0, 
            file_pattern: str = "*", case_sensitive: bool = True) -> str:
        """
        执行代码搜索
        
        Args:
            pattern: 正则表达式模式
            path: 搜索路径
            context: 显示上下文行数
            file_pattern: 文件名模式
            case_sensitive: 是否区分大小写
        
        Returns:
            结构化的搜索结果
        """
        try:
            search_path = Path(path)
            if not search_path.exists():
                return f"错误: 路径不存在 {path}"
            
            # 编译正则表达式
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return f"正则表达式错误: {str(e)}"
            
            # 收集所有匹配
            results = []
            total_matches = 0
            files_searched = 0
            
            # 排除的目录
            exclude_dirs = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache'}
            
            # 递归搜索文件
            for file_path in search_path.rglob(file_pattern):
                # 跳过排除的目录
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                
                # 只搜索文本文件
                if not file_path.is_file():
                    continue
                
                # 尝试读取文件
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    files_searched += 1
                except Exception:
                    continue
                
                # 在文件中搜索匹配
                file_matches = []
                for line_num, line in enumerate(lines, start=1):
                    if regex.search(line):
                        file_matches.append({
                            'line_num': line_num,
                            'content': line.rstrip(),
                            'context': self._get_context(lines, line_num, context)
                        })
                        total_matches += 1
                
                # 如果有匹配，添加到结果
                if file_matches:
                    results.append({
                        'file': str(file_path.relative_to(search_path)),
                        'matches': file_matches
                    })
            
            # 格式化输出
            return self._format_results(results, total_matches, files_searched, context)
            
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    def _get_context(self, lines: List[str], line_num: int, context: int) -> Dict[str, List]:
        """获取上下文行"""
        if context == 0:
            return {'before': [], 'after': []}
        
        before = []
        after = []
        
        # 前面的上下文
        start = max(0, line_num - context - 1)
        for i in range(start, line_num - 1):
            before.append(f"{i+1:4d} | {lines[i].rstrip()}")
        
        # 后面的上下文
        end = min(len(lines), line_num + context)
        for i in range(line_num, end):
            after.append(f"{i+1:4d} | {lines[i].rstrip()}")
        
        return {'before': before, 'after': after}
    
    def _format_results(self, results: List[Dict], total_matches: int, 
                       files_searched: int, context: int) -> str:
        """格式化搜索结果"""
        if not results:
            return f"未找到匹配项。已搜索 {files_searched} 个文件。"
        
        output = []
        output.append(f"[SEARCH] 搜索结果: 共找到 {total_matches} 处匹配，分布在 {len(results)} 个文件中\n")
        output.append("=" * 80)
        
        for file_result in results:
            file_path = file_result['file']
            matches = file_result['matches']
            
            output.append(f"\n[FILE] {file_path} ({len(matches)} 处匹配)")
            output.append("-" * 80)
            
            for match in matches:
                line_num = match['line_num']
                content = match['content']
                ctx = match['context']
                
                # 显示前面的上下文
                if ctx['before']:
                    output.append("    [上下文]")
                    for ctx_line in ctx['before']:
                        output.append(f"    {ctx_line}")
                
                # 显示匹配行
                output.append(f">>> {line_num:4d} | {content}")
                
                # 显示后面的上下文
                if ctx['after']:
                    for ctx_line in ctx['after']:
                        output.append(f"    {ctx_line}")
        
        output.append("\n" + "=" * 80)
        output.append(f"[DONE] 搜索完成: {total_matches} 处匹配，{len(results)} 个文件，共搜索 {files_searched} 个文件")
        
        return "\n".join(output)
