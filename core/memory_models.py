# core/memory_models.py
"""
结构化摘要数据模型

用于存储会话摘要的结构化信息，包括：
- 任务目标
- 文件变更列表
- 错误记录
- 工具使用情况
- 关键决策
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class FileChange:
    """文件变更记录"""
    path: str  # 文件路径
    action: str  # 操作类型: created, modified, deleted
    summary: str  # 变更摘要
    importance: float = 0.5  # 重要性分数 [0, 1]
    lines_added: int = 0  # 新增行数
    lines_removed: int = 0  # 删除行数

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "path": self.path,
            "action": self.action,
            "summary": self.summary,
            "importance": self.importance,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileChange":
        """从字典反序列化"""
        return cls(
            path=data["path"],
            action=data["action"],
            summary=data["summary"],
            importance=data.get("importance", 0.5),
            lines_added=data.get("lines_added", 0),
            lines_removed=data.get("lines_removed", 0)
        )


@dataclass
class ErrorRecord:
    """错误记录"""
    error_type: str  # 错误类型（如 FileNotFoundError, SyntaxError）
    error_message: str  # 错误信息
    timestamp: str  # 时间戳
    file_path: Optional[str] = None  # 相关文件路径
    line_number: Optional[int] = None  # 错误行号
    solution: Optional[str] = None  # 解决方案
    resolved: bool = False  # 是否已解决

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "solution": self.solution,
            "resolved": self.resolved
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorRecord":
        """从字典反序列化，支持处理对象输入"""
        from datetime import datetime

        # 如果已经是 ErrorRecord 对象，直接返回
        if isinstance(data, ErrorRecord):
            return data

        # 如果不是字典，尝试转换为字典
        if not isinstance(data, dict):
            # 如果对象有 to_dict 方法，使用它
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            else:
                # 否则使用对象的 __dict__ 属性
                data = data.__dict__ if hasattr(data, '__dict__') else {}

        return cls(
            error_type=data.get("error_type", "UnknownError"),
            error_message=data.get("error_message", "No error message"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            solution=data.get("solution"),
            resolved=data.get("resolved", False)
        )


@dataclass
class ToolUsage:
    """工具使用记录"""
    tool_name: str  # 工具名称
    parameters: Dict[str, Any]  # 调用参数
    result_summary: str  # 结果摘要
    timestamp: str  # 时间戳
    success: bool = True  # 是否成功
    importance: float = 0.3  # 重要性分数 [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
            "success": self.success,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUsage":
        """从字典反序列化，支持处理对象输入"""
        from datetime import datetime

        # 如果已经是 ToolUsage 对象，直接返回
        if isinstance(data, ToolUsage):
            return data

        # 如果不是字典，尝试转换为字典
        if not isinstance(data, dict):
            # 如果对象有 to_dict 方法，使用它
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            else:
                # 否则使用对象的 __dict__ 属性
                data = data.__dict__ if hasattr(data, '__dict__') else {}

        return cls(
            tool_name=data.get("tool_name", "unknown_tool"),
            parameters=data.get("parameters", {}),
            result_summary=data.get("result_summary", "No result summary"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            success=data.get("success", True),
            importance=data.get("importance", 0.3)
        )


@dataclass
class SessionSummary:
    """
    会话摘要（结构化）

    这是 Mini-Claude Code 的核心记忆单元，包含了单个会话的所有关键信息。
    每个 SessionSummary 代表一个压缩后的历史片段。
    """
    # 基本信息
    session_id: str  # 会话 ID
    timestamp: str  # 创建时间戳
    summary_text: str  # 自然语言摘要（200字以内）

    # 任务信息
    task_goal: str  # 任务目标（一句话描述）
    task_status: str  # 任务状态: completed, in_progress, failed, cancelled

    # 结构化信息
    files_changed: List[FileChange] = field(default_factory=list)  # 文件变更列表
    errors_encountered: List[ErrorRecord] = field(default_factory=list)  # 错误记录列表
    tools_used: List[ToolUsage] = field(default_factory=list)  # 工具使用列表
    key_decisions: List[str] = field(default_factory=list)  # 关键决策列表

    # 元数据
    importance: float = 0.5  # 整体重要性分数 [0, 1]
    message_count: int = 0  # 压缩前的消息数量
    token_count: int = 0  # 压缩前的 Token 数量

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "summary_text": self.summary_text,
            "task_goal": self.task_goal,
            "task_status": self.task_status,
            "files_changed": [fc.to_dict() for fc in self.files_changed],
            "errors_encountered": [er.to_dict() for er in self.errors_encountered],
            "tools_used": [tu.to_dict() for tu in self.tools_used],
            "key_decisions": self.key_decisions,
            "importance": self.importance,
            "message_count": self.message_count,
            "token_count": self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSummary":
        """从字典反序列化"""
        return cls(
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            summary_text=data["summary_text"],
            task_goal=data["task_goal"],
            task_status=data["task_status"],
            files_changed=[FileChange.from_dict(fc) for fc in data.get("files_changed", [])],
            errors_encountered=[ErrorRecord.from_dict(er) for er in data.get("errors_encountered", [])],
            tools_used=[ToolUsage.from_dict(tu) for tu in data.get("tools_used", [])],
            key_decisions=data.get("key_decisions", []),
            importance=data.get("importance", 0.5),
            message_count=data.get("message_count", 0),
            token_count=data.get("token_count", 0)
        )

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionSummary":
        """从 JSON 字符串反序列化"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_file_paths(self) -> List[str]:
        """获取所有涉及的文件路径"""
        paths = []
        for fc in self.files_changed:
            paths.append(fc.path)
        for er in self.errors_encountered:
            if er.file_path:
                paths.append(er.file_path)
        return list(set(paths))  # 去重

    def get_keywords(self) -> List[str]:
        """
        提取关键词（用于检索）

        提取规则：
        1. 任务目标中的关键词（中文/英文）
        2. 文件路径
        3. 错误类型
        4. 工具名称
        """
        keywords = []

        # 从任务目标中提取关键词
        # 中文：按字符切分（简单实现）
        # 英文：按空格切分
        import re

        # 提取英文单词（长度 > 2）
        english_words = re.findall(r'[a-zA-Z]{3,}', self.task_goal)
        keywords.extend(english_words)

        # 提取中文词汇（长度 > 1，简单按字切分）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', self.task_goal)
        for word in chinese_chars:
            # 对于中文，保留 2-4 字的词
            if 2 <= len(word) <= 4:
                keywords.append(word)
            # 对于更长的中文短语，拆分为 2 字词组
            elif len(word) > 4:
                for i in range(0, len(word) - 1):
                    keywords.append(word[i:i+2])

        # 添加文件路径
        keywords.extend(self.get_file_paths())

        # 添加错误类型
        for er in self.errors_encountered:
            keywords.append(er.error_type)

        # 添加工具名称
        for tu in self.tools_used:
            keywords.append(tu.tool_name)

        return list(set(keywords))  # 去重

    def __str__(self) -> str:
        """字符串表示（用于日志）"""
        return (
            f"SessionSummary(id={self.session_id}, "
            f"task={self.task_goal[:30]}..., "
            f"status={self.task_status}, "
            f"files={len(self.files_changed)}, "
            f"errors={len(self.errors_encountered)}, "
            f"importance={self.importance:.2f})"
        )


def create_summary_from_messages(
    messages: List[Dict[str, Any]],
    session_id: str,
    task_goal: str = "",
    task_status: str = "in_progress",
    importance_scorer=None
) -> SessionSummary:
    """
    从消息列表创建会话摘要（简化版）

    这是一个辅助函数，用于快速创建基本的 SessionSummary。
    更高级的摘要生成应该使用 LLM。

    Args:
        messages: 消息列表
        session_id: 会话 ID
        task_goal: 任务目标
        task_status: 任务状态
        importance_scorer: 重要性评分器（可选）

    Returns:
        SessionSummary 实例
    """
    # 计算重要性分数
    if importance_scorer:
        scores = [importance_scorer.score(msg) for msg in messages]
        avg_importance = sum(scores) / len(scores) if scores else 0.5
    else:
        avg_importance = 0.5

    # 创建基本摘要
    summary = SessionSummary(
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        summary_text=f"包含 {len(messages)} 条消息的会话",
        task_goal=task_goal or "未知任务",
        task_status=task_status,
        importance=avg_importance,
        message_count=len(messages),
        token_count=sum(len(str(msg)) for msg in messages)
    )

    return summary