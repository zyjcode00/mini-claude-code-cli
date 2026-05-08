# tests/test_memory_models.py
"""
测试 Phase 1 数据模型

测试内容：
- FileChange 序列化/反序列化
- ErrorRecord 序列化/反序列化
- ToolUsage 序列化/反序列化
- SessionSummary 序列化/反序列化
- 辅助方法测试
"""

import pytest
from datetime import datetime
from core.memory_models import (
    FileChange,
    ErrorRecord,
    ToolUsage,
    SessionSummary,
    create_summary_from_messages
)


def test_file_change_to_dict():
    """测试 FileChange 序列化"""
    fc = FileChange(
        path="tools/session_tool.py",
        action="created",
        summary="新增清理旧 session 的工具类",
        importance=0.7,
        lines_added=100,
        lines_removed=0
    )

    data = fc.to_dict()

    assert data["path"] == "tools/session_tool.py"
    assert data["action"] == "created"
    assert data["summary"] == "新增清理旧 session 的工具类"
    assert data["importance"] == 0.7
    assert data["lines_added"] == 100
    assert data["lines_removed"] == 0


def test_file_change_from_dict():
    """测试 FileChange 反序列化"""
    data = {
        "path": "core/context.py",
        "action": "modified",
        "summary": "集成结构化摘要",
        "importance": 0.6,
        "lines_added": 50,
        "lines_removed": 10
    }

    fc = FileChange.from_dict(data)

    assert fc.path == "core/context.py"
    assert fc.action == "modified"
    assert fc.summary == "集成结构化摘要"
    assert fc.importance == 0.6
    assert fc.lines_added == 50
    assert fc.lines_removed == 10


def test_error_record_to_dict():
    """测试 ErrorRecord 序列化"""
    er = ErrorRecord(
        error_type="FileNotFoundError",
        error_message="配置文件不存在",
        timestamp="2026-04-23T10:00:00",
        file_path="config/settings.json",
        line_number=42,
        solution="创建默认配置文件",
        resolved=True
    )

    data = er.to_dict()

    assert data["error_type"] == "FileNotFoundError"
    assert data["error_message"] == "配置文件不存在"
    assert data["timestamp"] == "2026-04-23T10:00:00"
    assert data["file_path"] == "config/settings.json"
    assert data["line_number"] == 42
    assert data["solution"] == "创建默认配置文件"
    assert data["resolved"] is True


def test_error_record_from_dict():
    """测试 ErrorRecord 反序列化"""
    data = {
        "error_type": "SyntaxError",
        "error_message": "无效的语法",
        "timestamp": "2026-04-23T11:00:00",
        "file_path": "main.py",
        "line_number": 10,
        "solution": None,
        "resolved": False
    }

    er = ErrorRecord.from_dict(data)

    assert er.error_type == "SyntaxError"
    assert er.error_message == "无效的语法"
    assert er.timestamp == "2026-04-23T11:00:00"
    assert er.file_path == "main.py"
    assert er.line_number == 10
    assert er.solution is None
    assert er.resolved is False


def test_tool_usage_to_dict():
    """测试 ToolUsage 序列化"""
    tu = ToolUsage(
        tool_name="write_full_file",
        parameters={"path": "test.py", "content": "# test"},
        result_summary="创建文件成功",
        timestamp="2026-04-23T12:00:00",
        success=True,
        importance=0.5
    )

    data = tu.to_dict()

    assert data["tool_name"] == "write_full_file"
    assert data["parameters"]["path"] == "test.py"
    assert data["result_summary"] == "创建文件成功"
    assert data["timestamp"] == "2026-04-23T12:00:00"
    assert data["success"] is True
    assert data["importance"] == 0.5


def test_tool_usage_from_dict():
    """测试 ToolUsage 反序列化"""
    data = {
        "tool_name": "execute_bash",
        "parameters": {"command": "ls"},
        "result_summary": "列出文件成功",
        "timestamp": "2026-04-23T13:00:00",
        "success": True,
        "importance": 0.3
    }

    tu = ToolUsage.from_dict(data)

    assert tu.tool_name == "execute_bash"
    assert tu.parameters["command"] == "ls"
    assert tu.result_summary == "列出文件成功"
    assert tu.timestamp == "2026-04-23T13:00:00"
    assert tu.success is True
    assert tu.importance == 0.3


def test_session_summary_to_dict():
    """测试 SessionSummary 序列化"""
    summary = SessionSummary(
        session_id="session_001",
        timestamp="2026-04-23T14:00:00",
        summary_text="完成工具重构",
        task_goal="重构 tools 目录",
        task_status="completed",
        files_changed=[
            FileChange("tools/new_tool.py", "created", "新增工具", 0.7)
        ],
        errors_encountered=[
            ErrorRecord("ImportError", "模块未找到", "2026-04-23T14:30:00")
        ],
        tools_used=[
            ToolUsage("write_file", {}, "成功", "2026-04-23T14:15:00")
        ],
        key_decisions=["使用类继承模式"],
        importance=0.8,
        message_count=20,
        token_count=5000
    )

    data = summary.to_dict()

    assert data["session_id"] == "session_001"
    assert data["task_goal"] == "重构 tools 目录"
    assert data["task_status"] == "completed"
    assert len(data["files_changed"]) == 1
    assert len(data["errors_encountered"]) == 1
    assert len(data["tools_used"]) == 1
    assert data["importance"] == 0.8
    assert data["message_count"] == 20


def test_session_summary_from_dict():
    """测试 SessionSummary 反序列化"""
    data = {
        "session_id": "session_002",
        "timestamp": "2026-04-23T15:00:00",
        "summary_text": "修复测试失败",
        "task_goal": "修复测试用例",
        "task_status": "completed",
        "files_changed": [
            {
                "path": "tests/test_example.py",
                "action": "modified",
                "summary": "修复断言",
                "importance": 0.6
            }
        ],
        "errors_encountered": [],
        "tools_used": [],
        "key_decisions": ["增加边界检查"],
        "importance": 0.7,
        "message_count": 10,
        "token_count": 2000
    }

    summary = SessionSummary.from_dict(data)

    assert summary.session_id == "session_002"
    assert summary.task_goal == "修复测试用例"
    assert summary.task_status == "completed"
    assert len(summary.files_changed) == 1
    assert summary.files_changed[0].path == "tests/test_example.py"
    assert len(summary.errors_encountered) == 0
    assert summary.importance == 0.7


def test_session_summary_to_json():
    """测试 SessionSummary JSON 序列化"""
    summary = SessionSummary(
        session_id="session_003",
        timestamp="2026-04-23T16:00:00",
        summary_text="测试摘要",
        task_goal="测试目标",
        task_status="in_progress"
    )

    json_str = summary.to_json()

    assert "session_003" in json_str
    assert "测试摘要" in json_str
    assert "测试目标" in json_str


def test_session_summary_from_json():
    """测试 SessionSummary JSON 反序列化"""
    json_str = '''{
        "session_id": "session_004",
        "timestamp": "2026-04-23T17:00:00",
        "summary_text": "JSON测试",
        "task_goal": "JSON反序列化测试",
        "task_status": "completed",
        "files_changed": [],
        "errors_encountered": [],
        "tools_used": [],
        "key_decisions": [],
        "importance": 0.5,
        "message_count": 5,
        "token_count": 1000
    }'''

    summary = SessionSummary.from_json(json_str)

    assert summary.session_id == "session_004"
    assert summary.summary_text == "JSON测试"
    assert summary.task_status == "completed"


def test_get_file_paths():
    """测试获取文件路径"""
    summary = SessionSummary(
        session_id="session_005",
        timestamp="2026-04-23T18:00:00",
        summary_text="测试文件路径提取",
        task_goal="测试",
        task_status="completed",
        files_changed=[
            FileChange("file1.py", "modified", "修改1", 0.5),
            FileChange("file2.py", "created", "创建2", 0.6)
        ],
        errors_encountered=[
            ErrorRecord("Error", "错误", "2026-04-23T18:00:00", file_path="file3.py")
        ]
    )

    paths = summary.get_file_paths()

    assert len(paths) == 3
    assert "file1.py" in paths
    assert "file2.py" in paths
    assert "file3.py" in paths


def test_get_keywords():
    """测试关键词提取"""
    summary = SessionSummary(
        session_id="session_006",
        timestamp="2026-04-23T19:00:00",
        summary_text="测试关键词提取",
        task_goal="重构工具模块",
        task_status="completed",
        files_changed=[
            FileChange("tools/helper.py", "modified", "修改", 0.5)
        ],
        errors_encountered=[
            ErrorRecord("ImportError", "导入错误", "2026-04-23T19:00:00")
        ],
        tools_used=[
            ToolUsage("write_file", {}, "成功", "2026-04-23T19:00:00")
        ]
    )

    keywords = summary.get_keywords()

    # 应该包含任务目标关键词
    assert "重构" in keywords or "工具" in keywords or "模块" in keywords

    # 应该包含文件路径
    assert "tools/helper.py" in keywords

    # 应该包含错误类型
    assert "ImportError" in keywords

    # 应该包含工具名称
    assert "write_file" in keywords


def test_create_summary_from_messages():
    """测试从消息列表创建摘要"""
    messages = [
        {"role": "user", "content": "帮我重构代码"},
        {"role": "assistant", "content": "好的，我来分析..."},
        {"role": "tool", "content": "File read successfully", "tool_name": "read_file"}
    ]

    summary = create_summary_from_messages(
        messages=messages,
        session_id="session_007",
        task_goal="重构代码",
        task_status="in_progress"
    )

    assert summary.session_id == "session_007"
    assert summary.task_goal == "重构代码"
    assert summary.task_status == "in_progress"
    assert summary.message_count == 3
    assert summary.token_count > 0


def test_session_summary_str():
    """测试 SessionSummary 字符串表示"""
    summary = SessionSummary(
        session_id="session_009",
        timestamp="2026-04-23T20:00:00",
        summary_text="测试字符串表示",
        task_goal="这是一个很长的任务目标用于测试截断功能",
        task_status="completed",
        files_changed=[FileChange("f1.py", "m", "修改", 0.5)],
        errors_encountered=[ErrorRecord("E", "错误", "2026-04-23T20:00:00")]
    )

    str_repr = str(summary)

    assert "session_009" in str_repr
    assert "completed" in str_repr
    assert "files=1" in str_repr
    assert "errors=1" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])