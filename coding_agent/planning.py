"""Controller-owned structured task planning."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


PLAN_STATUSES = {"pending", "in_progress", "completed", "blocked"}
MAX_PLAN_ITEMS = 20
_PLAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_ALLOWED_TRANSITIONS = {
    "pending": {"pending", "in_progress", "blocked"},
    "in_progress": {"in_progress", "completed", "blocked"},
    "blocked": {"blocked", "pending", "in_progress"},
    "completed": {"completed"},
}


class PlanError(ValueError):
    """A model-proposed plan update violated controller rules."""


@dataclass
class PlanItem:
    id: str
    description: str
    status: str
    evidence: list[str] = field(default_factory=list)
    started_action_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("started_action_version")
        return result


class TaskPlan:
    """A plan whose state transitions are validated by deterministic Python logic."""

    def __init__(self) -> None:
        self.items: list[PlanItem] = []
        self.action_version = 0
        self._successful_actions: list[tuple[int, str]] = []

    @property
    def exists(self) -> bool:
        return bool(self.items)

    @property
    def is_complete(self) -> bool:
        return self.exists and all(item.status == "completed" for item in self.items)

    @property
    def unfinished_items(self) -> list[PlanItem]:
        return [item for item in self.items if item.status != "completed"]

    def record_action(self, tool_name: str, result: str) -> None:
        """Record successful non-plan work as controller-owned completion evidence."""
        self.action_version += 1
        first_line = next((line.strip() for line in result.splitlines() if line.strip()), "success")
        evidence = f"action-{self.action_version}:{tool_name}:{first_line[:120]}"
        self._successful_actions.append((self.action_version, evidence))

    def apply_proposal(self, raw_items: Any) -> str:
        proposed = self._validate_items(raw_items)
        if not self.items:
            if any(item["status"] == "completed" for item in proposed):
                raise PlanError("new plan items cannot start as completed")
            self.items = [
                PlanItem(
                    id=item["id"],
                    description=item["description"],
                    status=item["status"],
                    started_action_version=self.action_version,
                )
                for item in proposed
            ]
            return self.render()

        current_by_id = {item.id: item for item in self.items}
        proposed_ids = [item["id"] for item in proposed]
        current_ids = [item.id for item in self.items]
        if proposed_ids[: len(current_ids)] != current_ids:
            raise PlanError("existing plan item IDs and order are immutable; new items may only be appended")

        updated: list[PlanItem] = []
        for index, proposal in enumerate(proposed):
            if index >= len(self.items):
                if proposal["status"] == "completed":
                    raise PlanError("new plan items cannot start as completed")
                updated.append(
                    PlanItem(
                        id=proposal["id"],
                        description=proposal["description"],
                        status=proposal["status"],
                        started_action_version=self.action_version,
                    )
                )
                continue

            current = current_by_id[proposal["id"]]
            if proposal["description"] != current.description:
                raise PlanError(f"description is immutable for plan item {current.id}")
            next_status = proposal["status"]
            if next_status not in _ALLOWED_TRANSITIONS[current.status]:
                raise PlanError(
                    f"invalid transition for {current.id}: {current.status} -> {next_status}"
                )

            evidence = list(current.evidence)
            started_version = current.started_action_version
            if next_status == "in_progress" and current.status != "in_progress":
                started_version = self.action_version
            if next_status == "completed" and current.status == "in_progress":
                new_evidence = [
                    description
                    for version, description in self._successful_actions
                    if version > current.started_action_version
                ]
                if not new_evidence:
                    raise PlanError(
                        f"plan item {current.id} cannot complete without a successful tool action"
                    )
                evidence.extend(item for item in new_evidence if item not in evidence)

            updated.append(
                PlanItem(
                    id=current.id,
                    description=current.description,
                    status=next_status,
                    evidence=evidence,
                    started_action_version=started_version,
                )
            )

        self.items = updated
        return self.render()

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]

    def render(self) -> str:
        lines = ["Task plan accepted by controller:"]
        for item in self.items:
            evidence = f" evidence={len(item.evidence)}" if item.evidence else ""
            lines.append(f"- [{item.status}] {item.id}: {item.description}{evidence}")
        return "\n".join(lines)

    @staticmethod
    def _validate_items(raw_items: Any) -> list[dict[str, str]]:
        if not isinstance(raw_items, list) or not raw_items:
            raise PlanError("items must be a non-empty list")
        if len(raw_items) > MAX_PLAN_ITEMS:
            raise PlanError(f"a plan cannot contain more than {MAX_PLAN_ITEMS} items")

        normalized: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise PlanError("each plan item must be an object")
            item_id = raw.get("id")
            description = raw.get("description")
            status = raw.get("status")
            if not isinstance(item_id, str) or not _PLAN_ID_PATTERN.fullmatch(item_id):
                raise PlanError("plan item id must use 1-40 letters, digits, '_' or '-'")
            if item_id in seen_ids:
                raise PlanError(f"duplicate plan item id: {item_id}")
            if not isinstance(description, str) or not description.strip():
                raise PlanError(f"plan item {item_id} needs a description")
            description = description.strip()
            if len(description) > 300:
                raise PlanError(f"description is too long for plan item {item_id}")
            if status not in PLAN_STATUSES:
                raise PlanError(f"invalid status for plan item {item_id}: {status}")
            seen_ids.add(item_id)
            normalized.append({"id": item_id, "description": description, "status": status})

        if sum(item["status"] == "in_progress" for item in normalized) > 1:
            raise PlanError("at most one plan item may be in_progress")
        return normalized
