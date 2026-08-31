"""Deterministic context budgeting while preserving the full transcript."""

from __future__ import annotations

import json
from typing import Any


DEFAULT_MAX_CONTEXT_CHARS = 60_000
MIN_CONTEXT_CHARS = 8_000


class ContextManager:
    def __init__(self, max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> None:
        if max_chars < MIN_CONTEXT_CHARS:
            raise ValueError(f"max context chars must be at least {MIN_CONTEXT_CHARS}")
        self.max_chars = max_chars
        self.compaction_count = 0

    def prepare(
        self,
        history: list[dict[str, Any]],
        state_snapshot: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
        original_chars = self._size(history)
        if original_chars <= self.max_chars:
            return list(history), None

        head = list(history[:2])
        tail_groups = self._message_groups(history[2:])
        snapshot_budget = max(1_000, self.max_chars - self._size(head) - 500)
        snapshot = self._snapshot_message(state_snapshot, snapshot_budget)
        selected_groups: list[list[dict[str, Any]]] = []
        used = self._size(head + [snapshot])
        omitted_messages = len(history) - len(head)

        for group in reversed(tail_groups):
            group_size = self._size(group)
            if used + group_size > self.max_chars:
                break
            selected_groups.append(group)
            used += group_size
            omitted_messages -= len(group)

        selected_groups.reverse()
        compacted = head + [snapshot]
        for group in selected_groups:
            compacted.extend(group)
        self.compaction_count += 1
        metadata = {
            "original_chars": original_chars,
            "sent_chars": self._size(compacted),
            "omitted_messages": max(0, omitted_messages),
        }
        return compacted, metadata

    @classmethod
    def _snapshot_message(cls, state: dict[str, Any], budget: int) -> dict[str, str]:
        prefix = (
            "Controller context snapshot: older conversation messages were compacted. "
            "Treat this structured state as authoritative and reread files when omitted "
            "content is needed.\n"
        )
        bounded = cls._bounded_state(state)
        content = prefix + json.dumps(bounded, ensure_ascii=False, sort_keys=True)
        if len(content) > budget:
            plan = state.get("task_plan") or []
            failure = state.get("last_verification_failure") or {}
            fallback = {
                "task_plan": [
                    {"id": item.get("id"), "status": item.get("status")}
                    for item in plan[:8]
                    if isinstance(item, dict)
                ],
                "plan_items_omitted": max(0, len(plan) - 8),
                "change_version": state.get("change_version", 0),
                "changed_file_count": len(state.get("changed_files") or []),
                "verification_status": state.get("verification_status"),
                "verification_evidence": state.get("verification_evidence") or [],
                "test_provenance_risks": state.get("test_provenance_risks") or [],
                "last_failure": (
                    {
                        "command": str(failure.get("command") or "")[:160],
                        "exit_code": failure.get("exit_code"),
                        "failure_reason": failure.get("failure_reason"),
                        "tests_collected": failure.get("tests_collected"),
                        "test_provenance_risk": failure.get("test_provenance_risk"),
                        "excerpt": str(failure.get("excerpt") or "")[-200:],
                    }
                    if failure
                    else None
                ),
            }
            content = prefix + json.dumps(fallback, ensure_ascii=False, sort_keys=True)
        if len(content) > budget:
            minimal = {
                "plan_item_count": len(state.get("task_plan") or []),
                "change_version": state.get("change_version", 0),
                "changed_file_count": len(state.get("changed_files") or []),
                "verification_status": state.get("verification_status"),
            }
            content = prefix + json.dumps(minimal, ensure_ascii=False, sort_keys=True)
        return {"role": "user", "content": content}

    @staticmethod
    def _bounded_state(state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("task_plan") or []
        changed_files = list(state.get("changed_files") or [])
        file_versions = state.get("file_versions") or {}
        failure = state.get("last_verification_failure") or {}
        bounded_plan = [
            {
                "id": item.get("id"),
                "description": str(item.get("description") or "")[:100],
                "status": item.get("status"),
                "evidence_count": len(item.get("evidence") or []),
            }
            for item in plan[:20]
            if isinstance(item, dict)
        ]
        return {
            "task_plan": bounded_plan,
            "change_version": state.get("change_version", 0),
            "changed_files": changed_files[:10],
            "changed_files_omitted": max(0, len(changed_files) - 10),
            "file_versions": dict(list(file_versions.items())[:10]),
            "file_versions_omitted": max(0, len(file_versions) - 10),
            "verification_status": state.get("verification_status"),
            "verification_evidence": state.get("verification_evidence") or [],
            "test_provenance_risks": state.get("test_provenance_risks") or [],
            "last_verification_failure": (
                {
                    "command": str(failure.get("command") or "")[:300],
                    "exit_code": failure.get("exit_code"),
                    "failure_reason": failure.get("failure_reason"),
                    "tests_collected": failure.get("tests_collected"),
                    "test_provenance_risk": failure.get("test_provenance_risk"),
                    "excerpt": str(failure.get("excerpt") or "")[-500:],
                }
                if failure
                else None
            ),
        }

    @staticmethod
    def _message_groups(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            group = [message]
            index += 1
            if message.get("role") == "assistant" and message.get("tool_calls"):
                while index < len(messages) and messages[index].get("role") == "tool":
                    group.append(messages[index])
                    index += 1
            groups.append(group)
        return groups

    @staticmethod
    def _size(messages: list[dict[str, Any]]) -> int:
        return len(json.dumps(messages, ensure_ascii=False, default=str))
