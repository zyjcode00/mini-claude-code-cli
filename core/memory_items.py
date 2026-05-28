"""统一长期记忆数据模型。

阶段 2 引入的模型用于把“原始事件”和“长期知识条目”从旧的
SessionSummary 中拆出来：
- RawObservation：记录未加工的原始事件来源；
- MemoryItem：记录可长期复用的知识型记忆；
- MemoryRecallResult：记录召回结果、分数和来源说明。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


class ObservationType(str, Enum):
    """原始观察事件类型。"""

    PROMPT_SUBMIT = "prompt_submit"
    ASSISTANT_MESSAGE = "assistant_message"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    TOOL_FAILURE = "tool_failure"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TEST_RESULT = "test_result"
    FILE_CHANGE = "file_change"
    OTHER = "other"


class MemoryKind(str, Enum):
    """长期记忆条目类型。"""

    FACT = "fact"
    ARCHITECTURE = "architecture"
    PREFERENCE = "preference"
    BUG = "bug"
    WORKFLOW = "workflow"
    DECISION = "decision"
    PROCEDURAL = "procedural"
    TASK = "task"
    SUMMARY = "summary"
    OTHER = "other"


class MemoryStatus(str, Enum):
    """MemoryItem 生命周期状态。"""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass
class RawObservation:
    """原始事件记录。

    RawObservation 不直接承担“知识”的语义，它保留事件的原貌，方便后续
    追溯 MemoryItem 来源、排查错误或重新提炼记忆。
    """

    id: str = field(default_factory=lambda: _new_id("obs"))
    session_id: str = ""
    project: str = ""
    cwd: str = ""
    timestamp: str = field(default_factory=_now_iso)
    event_type: str = ObservationType.OTHER.value
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    user_prompt: Optional[str] = None
    assistant_message: Optional[str] = None
    error: Optional[str] = None
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_type = _normalize_enum_value(self.event_type, ObservationType, ObservationType.OTHER.value)
        self.files = [str(item) for item in _as_list(self.files)]
        self.metadata = dict(self.metadata or {})
        if self.tool_input is not None:
            self.tool_input = dict(self.tool_input)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project": self.project,
            "cwd": self.cwd,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "user_prompt": self.user_prompt,
            "assistant_message": self.assistant_message,
            "error": self.error,
            "files": list(self.files),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawObservation":
        return cls(
            id=data.get("id") or _new_id("obs"),
            session_id=data.get("session_id", ""),
            project=data.get("project", ""),
            cwd=data.get("cwd", ""),
            timestamp=data.get("timestamp") or _now_iso(),
            event_type=data.get("event_type", ObservationType.OTHER.value),
            tool_name=data.get("tool_name"),
            tool_input=data.get("tool_input"),
            tool_output=data.get("tool_output"),
            user_prompt=data.get("user_prompt"),
            assistant_message=data.get("assistant_message"),
            error=data.get("error"),
            files=data.get("files", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "RawObservation":
        return cls.from_dict(json.loads(json_str))


@dataclass
class MemoryItem:
    """可长期保存和召回的知识型记忆条目。"""

    id: str = field(default_factory=lambda: _new_id("mem"))
    kind: str = MemoryKind.OTHER.value
    title: str = ""
    content: str = ""
    project: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    concepts: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    source_observation_ids: List[str] = field(default_factory=list)
    source_session_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.8
    status: str = MemoryStatus.ACTIVE.value
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 兼容直接传 Enum 的调用方式。
        self.kind = _normalize_enum_value(self.kind, MemoryKind, MemoryKind.OTHER.value)
        self.status = _normalize_enum_value(self.status, MemoryStatus, MemoryStatus.ACTIVE.value)
        self.concepts = [str(item) for item in _as_list(self.concepts)]
        self.files = [str(item) for item in _as_list(self.files)]
        self.source_observation_ids = [str(item) for item in _as_list(self.source_observation_ids)]
        self.source_session_ids = [str(item) for item in _as_list(self.source_session_ids)]
        self.importance = _clamp_float(self.importance, default=0.5)
        self.confidence = _clamp_float(self.confidence, default=0.8)
        self.version = int(self.version or 1)
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            # type 是早期路线文档中的字段名，导出时保留别名，方便外部工具兼容。
            "type": self.kind,
            "title": self.title,
            "content": self.content,
            "project": self.project,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "concepts": list(self.concepts),
            "files": list(self.files),
            "source_observation_ids": list(self.source_observation_ids),
            "source_session_ids": list(self.source_session_ids),
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data.get("id") or _new_id("mem"),
            kind=data.get("kind") or data.get("type") or MemoryKind.OTHER.value,
            title=data.get("title", ""),
            content=data.get("content", ""),
            project=data.get("project", ""),
            created_at=data.get("created_at") or _now_iso(),
            updated_at=data.get("updated_at") or data.get("created_at") or _now_iso(),
            concepts=data.get("concepts", []),
            files=data.get("files", []),
            source_observation_ids=data.get("source_observation_ids", []),
            source_session_ids=data.get("source_session_ids", []),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 0.8),
            status=data.get("status", MemoryStatus.ACTIVE.value),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryItem":
        return cls.from_dict(json.loads(json_str))

    def searchable_text(self) -> str:
        """返回用于关键词检索的文本。"""
        parts = [
            self.kind,
            self.title,
            self.content,
            self.project,
            " ".join(self.concepts),
            " ".join(self.files),
            " ".join(self.source_session_ids),
        ]
        return "\n".join(part for part in parts if part)


@dataclass
class MemoryRecallResult:
    """统一召回结果。"""

    item: MemoryItem
    score: float = 0.0
    source: str = "long_term"
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.item, dict):
            self.item = MemoryItem.from_dict(self.item)
        self.score = float(self.score or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecallResult":
        return cls(
            item=MemoryItem.from_dict(data.get("item", {})),
            score=data.get("score", 0.0),
            source=data.get("source", "long_term"),
            reason=data.get("reason", ""),
        )


def _normalize_enum_value(value: Any, enum_cls: Any, default: str) -> str:
    if isinstance(value, enum_cls):
        return value.value
    if isinstance(value, str):
        normalized = value.strip().lower()
        valid_values = {item.value for item in enum_cls}
        if normalized in valid_values:
            return normalized
    return default


def _clamp_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
