"""MemoryItem 生命周期治理：去重、版本流转、归档与访问追踪。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .memory_index import BM25MemoryIndex
from .memory_items import MemoryItem, MemoryStatus, _now_iso


@dataclass
class MaintenanceDecision:
    """保存前生命周期治理决策。"""

    action: str
    item: MemoryItem
    matched_item: Optional[MemoryItem] = None
    reason: str = ""


class MemoryMaintenance:
    """MemoryItem 生命周期治理策略集合。"""

    duplicate_threshold = 0.92
    supersede_threshold = 0.55
    low_confidence_archive_threshold = 0.2

    def prepare_for_save(self, item: MemoryItem, existing_items: List[MemoryItem]) -> MaintenanceDecision:
        """保存前执行归档、去重和 supersede 判断。"""
        now = _now_iso()
        item.updated_at = now
        item.metadata.setdefault("normalized_content_hash", self.normalized_hash(item))
        item.metadata.setdefault("exact_content_hash", self.exact_hash(item))
        if "quality_score" not in item.metadata:
            item.quality_score = max(item.quality_score, item._default_quality_score())

        if item.status != MemoryStatus.ACTIVE.value:
            item.is_latest = False
            item.metadata.setdefault("normalized_content_hash", self.normalized_hash(item))
            item.metadata.setdefault("exact_content_hash", self.exact_hash(item))
            return MaintenanceDecision("status_update", item, reason="显式非 active 状态更新")

        if item.confidence <= self.low_confidence_archive_threshold or item.is_expired():
            item.status = MemoryStatus.ARCHIVED.value
            item.is_latest = False
            return MaintenanceDecision("archive", item, reason="低置信度或已过期，直接归档")

        active_items = [candidate for candidate in existing_items if candidate.status == MemoryStatus.ACTIVE.value]
        duplicate = self.find_duplicate(item, active_items)
        if duplicate:
            merged = self.merge_duplicate(duplicate, item)
            return MaintenanceDecision("merge_duplicate", merged, duplicate, "重复 MemoryItem 已合并")

        superseded = self.find_superseded(item, active_items)
        if superseded:
            item.parent_id = superseded.id
            item.supersedes = list(dict.fromkeys([*item.supersedes, superseded.id]))
            item.version = max(item.version, superseded.version + 1)
            item.is_latest = True
            item.status = MemoryStatus.ACTIVE.value
            return MaintenanceDecision("supersede", item, superseded, "新 MemoryItem supersede 旧版本")

        item.status = MemoryStatus.ACTIVE.value
        item.is_latest = True
        return MaintenanceDecision("create", item, reason="新增 MemoryItem")

    def apply_archive_policy(self, item: MemoryItem) -> bool:
        """按 TTL、低置信度/质量和长期未命中策略归档；返回是否发生状态变化。"""
        if item.status != MemoryStatus.ACTIVE.value:
            return False
        stale_without_access = item.access_count == 0 and self._age_days(item.created_at) >= 180
        should_archive = (
            item.is_expired()
            or item.confidence <= self.low_confidence_archive_threshold
            or item.quality_score <= 0.15
            or stale_without_access
        )
        if not should_archive:
            return False
        item.status = MemoryStatus.ARCHIVED.value
        item.is_latest = False
        item.updated_at = _now_iso()
        item.metadata["archive_reason"] = "expired_or_low_quality_or_stale"
        return True

    def find_duplicate(self, item: MemoryItem, candidates: List[MemoryItem]) -> Optional[MemoryItem]:
        exact = self.exact_hash(item)
        normalized = self.normalized_hash(item)
        for candidate in candidates:
            if candidate.id == item.id:
                continue
            if candidate.project != item.project or candidate.kind != item.kind:
                continue
            candidate_exact = candidate.metadata.get("exact_content_hash") or self.exact_hash(candidate)
            candidate_normalized = candidate.metadata.get("normalized_content_hash") or self.normalized_hash(candidate)
            if exact == candidate_exact or normalized == candidate_normalized:
                return candidate
            if self.similarity(item, candidate) >= self.duplicate_threshold:
                return candidate
        return None

    def find_superseded(self, item: MemoryItem, candidates: List[MemoryItem]) -> Optional[MemoryItem]:
        best: Tuple[float, Optional[MemoryItem]] = (0.0, None)
        for candidate in candidates:
            if candidate.id == item.id:
                continue
            if not candidate.is_latest or candidate.project != item.project or candidate.kind != item.kind:
                continue
            overlap = self.title_content_overlap(item, candidate)
            concept_overlap = self.jaccard(set(item.concepts), set(candidate.concepts))
            file_overlap = self.jaccard(set(item.files), set(candidate.files))
            score = max(overlap, concept_overlap * 0.8 + file_overlap * 0.2)
            if score > best[0]:
                best = (score, candidate)
        if best[1] and best[0] >= self.supersede_threshold:
            return best[1]
        return None

    def merge_duplicate(self, existing: MemoryItem, incoming: MemoryItem) -> MemoryItem:
        """把重复 MemoryItem 的来源、标签和质量信号合并到已有条目。"""
        existing.concepts = list(dict.fromkeys([*existing.concepts, *incoming.concepts]))
        existing.files = list(dict.fromkeys([*existing.files, *incoming.files]))
        existing.source_observation_ids = list(dict.fromkeys([*existing.source_observation_ids, *incoming.source_observation_ids]))
        existing.source_session_ids = list(dict.fromkeys([*existing.source_session_ids, *incoming.source_session_ids]))
        existing.importance = max(existing.importance, incoming.importance)
        existing.confidence = max(existing.confidence, incoming.confidence)
        existing.quality_score = min(1.0, max(existing.quality_score, incoming.quality_score) + 0.03)
        existing.related_ids = list(dict.fromkeys([*existing.related_ids, incoming.id]))
        existing.updated_at = _now_iso()
        existing.metadata.setdefault("deduplicated_count", 0)
        existing.metadata["deduplicated_count"] += 1
        existing.metadata["normalized_content_hash"] = self.normalized_hash(existing)
        existing.metadata["exact_content_hash"] = self.exact_hash(existing)
        return existing

    def mark_superseded(self, old_item: MemoryItem, new_item: MemoryItem) -> MemoryItem:
        old_item.status = MemoryStatus.SUPERSEDED.value
        old_item.is_latest = False
        old_item.related_ids = list(dict.fromkeys([*old_item.related_ids, new_item.id]))
        old_item.updated_at = _now_iso()
        old_item.metadata["superseded_by"] = new_item.id
        return old_item

    def record_access(self, item: MemoryItem, injected: bool = False) -> MemoryItem:
        item.mark_accessed(injected=injected)
        return item

    @staticmethod
    def exact_hash(item: MemoryItem) -> str:
        return hashlib.sha256(f"{item.title}\n{item.content}".encode("utf-8")).hexdigest()

    @classmethod
    def normalized_hash(cls, item: MemoryItem) -> str:
        normalized = cls.normalize_text(f"{item.title}\n{item.content}")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def normalize_text(cls, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @classmethod
    def similarity(cls, left: MemoryItem, right: MemoryItem) -> float:
        return max(
            cls.jaccard(set(BM25MemoryIndex.tokenize(left.content)), set(BM25MemoryIndex.tokenize(right.content))),
            cls.title_content_overlap(left, right),
        )

    @classmethod
    def title_content_overlap(cls, left: MemoryItem, right: MemoryItem) -> float:
        left_tokens = set(BM25MemoryIndex.tokenize(f"{left.title} {left.content}"))
        right_tokens = set(BM25MemoryIndex.tokenize(f"{right.title} {right.content}"))
        return cls.jaccard(left_tokens, right_tokens)

    @staticmethod
    def jaccard(left: set, right: set) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _age_days(timestamp: str) -> int:
        try:
            return max((datetime.now() - datetime.fromisoformat(timestamp)).days, 0)
        except Exception:
            return 0
