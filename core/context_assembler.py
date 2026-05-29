# core/context_assembler.py
"""Provider-safe context assembly with explicit budgets.

Phase 5 introduces this module as the single place that assembles the final
LLM request context.  It keeps high-priority prompt sections, compressed state,
recent complete turns and the current user request within a rough token budget
without breaking OpenAI-style assistant/tool pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.turn_builder import TurnBuilder, ConversationTurn


@dataclass
class ContextBudget:
    """Approximate token budgets for each context layer."""

    total: int = 32000
    system: int = 8000
    plan: int = 1200
    memory: int = 1500
    compressed_state: int = 2500
    recent_turns: int = 12000
    current_user: int = 2000
    emergency_buffer_ratio: float = 0.12

    @property
    def available_total(self) -> int:
        reserved = int(self.total * self.emergency_buffer_ratio)
        return max(1, self.total - reserved)


@dataclass
class AssembledContext:
    """Result returned by :class:`ContextAssembler`."""

    system_prompt: str
    messages: List[Dict[str, Any]]
    token_estimate: int
    dropped_turn_count: int = 0
    dropped_message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def openai_messages(self) -> List[Dict[str, Any]]:
        """Return a complete OpenAI-compatible message list."""
        return [{"role": "system", "content": self.system_prompt}] + self.messages


class ContextAssembler:
    """Assemble system/memory/state/recent messages under budget.

    The assembler is intentionally deterministic and provider agnostic.  OpenAI
    tool-call safety is achieved by selecting only complete turns from
    :class:`TurnBuilder`; injected prompt sections are only appended to the
    system prompt and are therefore never inserted between ``assistant`` tool
    calls and their ``tool`` responses.
    """

    def __init__(self, budget: Optional[ContextBudget] = None, turn_builder: Optional[TurnBuilder] = None):
        self.budget = budget or ContextBudget()
        self.turn_builder = turn_builder or TurnBuilder()

    def assemble(
        self,
        *,
        base_system_prompt: str,
        plan_context: str = "",
        memory_context: str = "",
        compressed_state: str = "",
        messages: Optional[Sequence[Dict[str, Any]]] = None,
        current_user_message: Optional[Dict[str, Any]] = None,
        provider: str = "openai",
        budget: Optional[ContextBudget] = None,
    ) -> AssembledContext:
        """Build final prompt and recent messages.

        Args:
            base_system_prompt: Already-rendered base system prompt.
            plan_context: Optional explicit plan section if not already included
                in ``base_system_prompt``.
            memory_context: Relevant long-term memory section.
            compressed_state: Current session compressed state/summary section.
            messages: Conversation history snapshot.
            current_user_message: Optional current request.  If omitted and the
                last message is a user message, that message is treated as the
                current request and is protected from recent-turn truncation.
            provider: Currently used for metadata; OpenAI-compatible safety is
                always applied because it is stricter for tool pairs.
            budget: Per-call budget override.
        """
        active_budget = budget or self.budget
        source_messages = [dict(msg) for msg in (messages or []) if isinstance(msg, dict)]
        current_user = self._resolve_current_user_message(source_messages, current_user_message)
        history_messages = self._remove_current_user_from_history(source_messages, current_user)

        system_prompt = self._assemble_system_prompt(
            base_system_prompt=base_system_prompt,
            plan_context=plan_context,
            memory_context=memory_context,
            compressed_state=compressed_state,
            budget=active_budget,
        )

        complete_turns = self.turn_builder.complete_turns_only(self.turn_builder.build(history_messages))
        recent_messages, kept_turns = self._select_recent_turns(complete_turns, active_budget.recent_turns)

        if current_user:
            # Current request is mandatory.  If total budget is tight, trim more
            # recent turns instead of dropping the request.
            recent_messages = self._fit_messages_with_current_user(
                recent_messages,
                current_user,
                active_budget,
            )
            final_messages = recent_messages + [dict(current_user)]
        else:
            final_messages = recent_messages

        final_messages = self._validate_provider_messages(final_messages)
        token_estimate = self._estimate_text_tokens(system_prompt) + self._estimate_messages_tokens(final_messages)
        dropped_turn_count = max(0, len(complete_turns) - kept_turns)
        dropped_message_count = max(0, len(history_messages) - len(recent_messages))

        return AssembledContext(
            system_prompt=system_prompt,
            messages=final_messages,
            token_estimate=token_estimate,
            dropped_turn_count=dropped_turn_count,
            dropped_message_count=dropped_message_count,
            metadata={
                "provider": provider,
                "kept_turns": kept_turns,
                "source_messages": len(source_messages),
                "final_messages": len(final_messages),
                "system_tokens": self._estimate_text_tokens(system_prompt),
                "message_tokens": self._estimate_messages_tokens(final_messages),
            },
        )

    def _assemble_system_prompt(
        self,
        *,
        base_system_prompt: str,
        plan_context: str,
        memory_context: str,
        compressed_state: str,
        budget: ContextBudget,
    ) -> str:
        sections = [self._truncate_text(base_system_prompt, budget.system)]
        if plan_context:
            sections.append(self._truncate_text(plan_context, budget.plan))
        if memory_context:
            sections.append(self._truncate_text(memory_context, budget.memory))
        if compressed_state:
            sections.append(self._truncate_text(compressed_state, budget.compressed_state))

        prompt = "\n".join(section.rstrip() for section in sections if section and section.strip())
        max_system_tokens = max(1, budget.available_total - budget.current_user - budget.recent_turns)
        return self._truncate_text(prompt, max(budget.system, max_system_tokens))

    def _select_recent_turns(
        self,
        turns: Sequence[ConversationTurn],
        token_budget: int,
    ) -> tuple[List[Dict[str, Any]], int]:
        if token_budget <= 0:
            return [], 0

        selected_reversed: List[ConversationTurn] = []
        used = 0
        for turn in reversed(turns):
            cost = max(1, turn.token_estimate or self._estimate_messages_tokens(turn.messages))
            if selected_reversed and used + cost > token_budget:
                break
            if not selected_reversed and cost > token_budget:
                # Oversized atomic turn: keep it whole rather than cutting a
                # tool pair in half.  The emergency buffer absorbs this case.
                selected_reversed.append(turn)
                used += cost
                break
            selected_reversed.append(turn)
            used += cost

        selected = list(reversed(selected_reversed))
        return self.turn_builder.flatten(selected), len(selected)

    def _fit_messages_with_current_user(
        self,
        recent_messages: List[Dict[str, Any]],
        current_user: Dict[str, Any],
        budget: ContextBudget,
    ) -> List[Dict[str, Any]]:
        current_cost = self._estimate_messages_tokens([current_user])
        target = max(0, budget.available_total - budget.system - budget.memory - budget.compressed_state - current_cost)
        target = min(target, budget.recent_turns)
        if target <= 0:
            return []

        turns = self.turn_builder.complete_turns_only(self.turn_builder.build(recent_messages))
        fitted, _ = self._select_recent_turns(turns, target)
        return fitted

    def _validate_provider_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final provider-safe validation.

        Uses the stricter OpenAI tool-pair invariant for all providers because it
        also prevents accidental memory injection inside tool pairs.
        """
        return self.turn_builder.build_complete_messages(messages)

    @staticmethod
    def _resolve_current_user_message(
        messages: List[Dict[str, Any]],
        current_user_message: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if current_user_message:
            return dict(current_user_message)
        if messages and messages[-1].get("role") == "user":
            return dict(messages[-1])
        return None

    @staticmethod
    def _remove_current_user_from_history(
        messages: List[Dict[str, Any]],
        current_user: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not current_user or not messages:
            return messages
        if messages[-1] == current_user:
            return messages[:-1]
        return [msg for msg in messages if msg is not current_user]

    @classmethod
    def _truncate_text(cls, text: str, token_budget: int) -> str:
        if not text or token_budget <= 0:
            return ""
        char_budget = token_budget * 4
        value = str(text)
        if len(value) <= char_budget:
            return value
        marker = "\n...[truncated by ContextAssembler budget]"
        return value[: max(0, char_budget - len(marker))].rstrip() + marker

    @staticmethod
    def _estimate_text_tokens(text: Any) -> int:
        return max(0, len(str(text or "")) // 4)

    @classmethod
    def _estimate_messages_tokens(cls, messages: Iterable[Dict[str, Any]]) -> int:
        return sum(max(1, len(str(msg)) // 4) for msg in messages)
