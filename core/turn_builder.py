# core/turn_builder.py
"""Conversation turn builder for provider-safe context compression.

This module converts a flat provider message list into semantic turns.  The
important invariant is that OpenAI-style ``assistant.tool_calls`` plus their
immediately following ``tool`` responses are represented as one atomic turn so
compression strategies can avoid cutting the pair in half.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class ConversationTurn:
    """A semantic, provider-safe unit of conversation history."""

    id: str
    start_index: int
    end_index: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    user_message: Optional[Dict[str, Any]] = None
    assistant_messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_messages: List[Dict[str, Any]] = field(default_factory=list)
    has_tool_calls: bool = False
    is_tool_pair_complete: bool = True
    missing_tool_call_ids: List[str] = field(default_factory=list)
    orphan_tool_call_ids: List[str] = field(default_factory=list)
    token_estimate: int = 0
    importance: float = 0.0
    categories: List[str] = field(default_factory=list)
    files_touched: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)

    @property
    def is_valid_openai_tool_turn(self) -> bool:
        """Whether this turn can be sent to OpenAI-compatible APIs as-is."""
        return self.is_tool_pair_complete and not self.orphan_tool_call_ids


class TurnBuilder:
    """Build complete conversation turns from a flat message list.

    Rules implemented for OpenAI-compatible tool calls:
    - an ``assistant`` message with ``tool_calls`` starts an atomic tool turn;
    - only immediately following contiguous ``tool`` messages can satisfy that
      assistant's tool calls;
    - missing tool responses mark the turn incomplete;
    - ``tool`` messages without an immediately preceding assistant tool-call
      turn are emitted as orphan/incomplete turns instead of being attached to
      unrelated context.
    """

    ERROR_KEYWORDS = ("error", "exception", "traceback", "failed", "failure", "失败", "错误", "报错")
    TEST_KEYWORDS = ("pytest", "unittest", "test", "测试", "passed", "failed", "assertionerror")
    PLAN_KEYWORDS = ("plan", "todo", "步骤", "继续", "完成", "已完成", "待办")
    FILE_EDIT_KEYWORDS = ("edit_file", "write_full_file", "created", "modified", "deleted", "创建", "修改", "删除")
    FILE_READ_KEYWORDS = ("read_file", "cat ", "读取", "查看文件")
    SEARCH_KEYWORDS = ("search_code", "grep", "ripgrep", "搜索")
    COMMAND_KEYWORDS = ("execute_bash", "pytest", "python ", "npm ", "命令", "运行")
    SUCCESS_KEYWORDS = ("passed", "success", "成功", "测试通过", "✅")
    FAILURE_KEYWORDS = ("failed", "failure", "error", "traceback", "失败", "错误", "❌")

    FILE_PATTERN = re.compile(
        r"(?<![\w.-])(?:[A-Za-z]:)?[\w.@+-]+(?:[/\\][\w.@+-]+)+\.[A-Za-z0-9]{1,8}(?![\w.-])"
        r"|(?<![\w.-])[\w.@+-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|yaml|yml|toml|ini|cfg|java|cpp|c|h|rs|go|sh|bat|ps1)(?![\w.-])"
    )

    def build(self, messages: List[Dict[str, Any]]) -> List[ConversationTurn]:
        """Return semantic turns while preserving original message order."""
        turns: List[ConversationTurn] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if not isinstance(msg, dict):
                i += 1
                continue

            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                turn, next_index = self._build_tool_call_turn(messages, i)
                turns.append(turn)
                i = next_index
                continue

            if role == "tool":
                turn, next_index = self._build_orphan_tool_turn(messages, i)
                turns.append(turn)
                i = next_index
                continue

            turn = self._new_turn(
                turn_index=len(turns),
                start_index=i,
                end_index=i,
                messages=[msg],
            )
            turns.append(turn)
            i += 1

        return turns

    def flatten(self, turns: Iterable[ConversationTurn]) -> List[Dict[str, Any]]:
        """Flatten turns back to a message list."""
        flattened: List[Dict[str, Any]] = []
        for turn in turns:
            flattened.extend(turn.messages)
        return flattened

    def complete_turns_only(self, turns: Iterable[ConversationTurn]) -> List[ConversationTurn]:
        """Filter out turns that contain incomplete OpenAI tool-call pairs."""
        return [turn for turn in turns if turn.is_valid_openai_tool_turn]

    def build_complete_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build turns and flatten only provider-safe turns."""
        return self.flatten(self.complete_turns_only(self.build(messages)))

    def _build_tool_call_turn(
        self,
        messages: List[Dict[str, Any]],
        assistant_index: int,
    ) -> tuple[ConversationTurn, int]:
        assistant = messages[assistant_index]
        expected_ids = self._tool_call_ids(assistant)
        turn_messages = [assistant]
        tool_messages: List[Dict[str, Any]] = []
        found_ids: Set[str] = set()

        j = assistant_index + 1
        while j < len(messages):
            candidate = messages[j]
            if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                break
            turn_messages.append(candidate)
            tool_messages.append(candidate)
            tool_call_id = candidate.get("tool_call_id")
            if tool_call_id:
                found_ids.add(tool_call_id)
            j += 1

        missing_ids = sorted(expected_ids - found_ids)
        orphan_ids = sorted(found_ids - expected_ids)
        turn = self._new_turn(
            turn_index=0,
            start_index=assistant_index,
            end_index=j - 1,
            messages=turn_messages,
            is_tool_pair_complete=not missing_ids and not orphan_ids and bool(expected_ids),
            missing_tool_call_ids=missing_ids,
            orphan_tool_call_ids=orphan_ids,
        )
        return turn, j

    def _build_orphan_tool_turn(
        self,
        messages: List[Dict[str, Any]],
        tool_index: int,
    ) -> tuple[ConversationTurn, int]:
        turn_messages: List[Dict[str, Any]] = []
        orphan_ids: List[str] = []
        j = tool_index
        while j < len(messages):
            candidate = messages[j]
            if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                break
            turn_messages.append(candidate)
            if candidate.get("tool_call_id"):
                orphan_ids.append(candidate.get("tool_call_id"))
            j += 1

        turn = self._new_turn(
            turn_index=0,
            start_index=tool_index,
            end_index=j - 1,
            messages=turn_messages,
            is_tool_pair_complete=False,
            orphan_tool_call_ids=sorted(set(orphan_ids)),
        )
        return turn, j

    def _new_turn(
        self,
        turn_index: int,
        start_index: int,
        end_index: int,
        messages: List[Dict[str, Any]],
        is_tool_pair_complete: bool = True,
        missing_tool_call_ids: Optional[List[str]] = None,
        orphan_tool_call_ids: Optional[List[str]] = None,
    ) -> ConversationTurn:
        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        tool_messages = [msg for msg in messages if msg.get("role") == "tool"]
        has_tool_calls = any(msg.get("role") == "assistant" and msg.get("tool_calls") for msg in messages)

        categories = self._categorize(messages)
        files_touched = self._extract_files(messages)
        errors = self._extract_matching_lines(messages, self.ERROR_KEYWORDS)
        tests = self._extract_matching_lines(messages, self.TEST_KEYWORDS)
        token_estimate = self._estimate_tokens(messages)
        importance = self._estimate_importance(
            messages,
            categories,
            is_tool_pair_complete,
            token_estimate,
            files_touched,
            errors,
            tests,
        )
        return ConversationTurn(
            id=f"turn_{start_index}_{end_index}_{turn_index}",
            start_index=start_index,
            end_index=end_index,
            messages=[dict(msg) for msg in messages],
            user_message=dict(user_message) if user_message else None,
            assistant_messages=[dict(msg) for msg in assistant_messages],
            tool_messages=[dict(msg) for msg in tool_messages],
            has_tool_calls=has_tool_calls,
            is_tool_pair_complete=is_tool_pair_complete,
            missing_tool_call_ids=missing_tool_call_ids or [],
            orphan_tool_call_ids=orphan_tool_call_ids or [],
            token_estimate=token_estimate,
            importance=importance,
            categories=categories,
            files_touched=files_touched,
            errors=errors,
            tests=tests,
        )

    def _tool_call_ids(self, message: Dict[str, Any]) -> Set[str]:
        return {tc.get("id") for tc in message.get("tool_calls", []) if tc.get("id")}

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        # Lightweight approximation: four chars per token, minimum one token.
        return max(1, sum(len(str(msg)) for msg in messages) // 4)

    def _estimate_importance(
        self,
        messages: List[Dict[str, Any]],
        categories: List[str],
        is_tool_pair_complete: bool,
        token_estimate: int,
        files_touched: List[str],
        errors: List[str],
        tests: List[str],
    ) -> float:
        score = 0.08
        text = self._messages_text(messages).lower()

        if any(msg.get("role") == "user" for msg in messages):
            score += 0.22
        if any(msg.get("tool_calls") for msg in messages):
            score += 0.10
        if files_touched:
            score += min(0.20, 0.06 * len(files_touched))
        if errors or "error" in categories:
            score += 0.30
        if tests or "test" in categories:
            score += 0.20
        if "planning" in categories:
            score += 0.16
        if "code_edit" in categories:
            score += 0.22
        if "command" in categories:
            score += 0.08
        if any(keyword in text for keyword in self.SUCCESS_KEYWORDS):
            score += 0.06
        if any(keyword in text for keyword in self.FAILURE_KEYWORDS):
            score += 0.08
        if "bulk_output" in categories:
            score -= 0.20
        if not is_tool_pair_complete:
            score -= 0.25

        # Very large turns are often file dumps/stdout; keep their metadata but
        # avoid letting size alone dominate keyframe selection.
        if token_estimate > 1200 and not ({"error", "test", "code_edit"} & set(categories)):
            score -= 0.12

        return max(0.0, min(1.0, round(score, 4)))

    def _categorize(self, messages: List[Dict[str, Any]]) -> List[str]:
        text = self._messages_text(messages).lower()
        categories: List[str] = []

        if any(msg.get("tool_calls") or msg.get("role") == "tool" for msg in messages):
            categories.append("tool")
        if any(keyword in text for keyword in self.ERROR_KEYWORDS):
            categories.append("error")
        if any(keyword in text for keyword in self.TEST_KEYWORDS):
            categories.append("test")
        if any(keyword in text for keyword in self.PLAN_KEYWORDS):
            categories.append("planning")
        if any(keyword in text for keyword in self.FILE_EDIT_KEYWORDS):
            categories.append("code_edit")
        if any(keyword in text for keyword in self.FILE_READ_KEYWORDS):
            categories.append("file_read")
        if any(keyword in text for keyword in self.SEARCH_KEYWORDS):
            categories.append("search")
        if any(keyword in text for keyword in self.COMMAND_KEYWORDS):
            categories.append("command")
        if self._looks_like_bulk_output(text):
            categories.append("bulk_output")

        return categories

    def _extract_files(self, messages: List[Dict[str, Any]]) -> List[str]:
        files: List[str] = []
        for msg in messages:
            self._extract_files_from_obj(msg, files)
        normalized = [path.replace("\\", "/").strip("`'\"(),:;[]{}<>") for path in files]
        return list(dict.fromkeys(path for path in normalized if path))[:20]

    def _extract_files_from_obj(self, obj: Any, files: List[str]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in {"path", "file", "filename", "filepath"} and isinstance(value, str):
                    files.extend(match.group(0) for match in self.FILE_PATTERN.finditer(value))
                else:
                    self._extract_files_from_obj(value, files)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_files_from_obj(item, files)
        elif isinstance(obj, str):
            files.extend(match.group(0) for match in self.FILE_PATTERN.finditer(obj))

    def _extract_matching_lines(
        self,
        messages: List[Dict[str, Any]],
        keywords: Iterable[str],
    ) -> List[str]:
        lowered_keywords = tuple(keyword.lower() for keyword in keywords)
        matches: List[str] = []
        for msg in messages:
            content = str(msg.get("content", ""))
            for line in content.splitlines() or [content]:
                if any(keyword in line.lower() for keyword in lowered_keywords):
                    matches.append(line[:300])
        return matches[:10]

    def _messages_text(self, messages: List[Dict[str, Any]]) -> str:
        return "\n".join(str(msg.get("content", "")) for msg in messages)

    def _looks_like_bulk_output(self, text: str) -> bool:
        line_count = text.count("\n") + 1
        return len(text) > 4000 or line_count > 80
