"""The model/tool orchestration loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .context import ContextManager, DEFAULT_MAX_CONTEXT_CHARS
from .planning import TaskPlan
from .tools import TOOL_SCHEMAS, ToolError, ToolExecutor, decode_arguments


DEFAULT_MAX_TOOL_RESULT_CHARS = 20_000


class ModelClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


SYSTEM_PROMPT = """You are a careful coding agent.
Work only inside the supplied workspace. Inspect existing files before changing them.
Use search_text and bounded read_file calls to gather only the context you need.
Prefer replace_in_file for a precise edit when the old text occurs exactly once; use write_file for new files or full rewrites.
Existing files are revision-guarded: read them before editing and re-read after any stale-observation error.
For multi-step tasks, use update_plan before acting. Keep at most one item in_progress, update the plan after real work, and do not treat plan completion as a substitute for verification.
Use tools for all file and command operations. Explain what you changed and verify it by running a relevant test.
Treat command execution as potentially destructive and use it only when it helps the task.
When you believe the task is complete, stop and give a concise completion proposal; the controller decides whether the evidence is sufficient.
Never include or request secrets."""

VERIFICATION_POLICY_LEVELS = {"none": 0, "syntax": 1, "test": 2, "full": 3}


@dataclass(frozen=True)
class CompletionDecision:
    accepted: bool
    reason: str
    verification_status: str
    evidence: list[str]


class CompletionPolicy:
    """Deterministic controller policy; the model cannot waive these requirements."""

    def __init__(self, minimum: str = "test") -> None:
        if minimum not in VERIFICATION_POLICY_LEVELS:
            choices = ", ".join(VERIFICATION_POLICY_LEVELS)
            raise ValueError(f"verification policy must be one of: {choices}")
        self.minimum = minimum

    def evaluate(
        self,
        executor: ToolExecutor,
        task_plan: TaskPlan | None = None,
    ) -> CompletionDecision:
        status = executor.verification_status
        evidence = executor.verification_evidence
        if executor.has_unresolved_verification_failure:
            return CompletionDecision(False, "verification_failed", status, evidence)
        if task_plan is not None and task_plan.exists and not task_plan.is_complete:
            return CompletionDecision(False, "plan_incomplete", status, evidence)
        if not executor.changed_files:
            return CompletionDecision(True, "completed_no_changes", status, evidence)

        required_level = VERIFICATION_POLICY_LEVELS[self.minimum]
        if executor.strongest_current_verification_level >= required_level:
            reason = "completed_by_policy" if required_level == 0 else "completed_verified"
            return CompletionDecision(True, reason, status, evidence)
        reason = "partial_verification" if evidence else "verification_required"
        return CompletionDecision(False, reason, status, evidence)


@dataclass
class AgentResult:
    answer: str
    steps: int
    stop_reason: str
    tool_calls: int
    verification_status: str
    verification_policy: str
    verification_evidence: list[str]
    verification_records: list[dict[str, Any]]
    changed_files: list[str]
    task_plan: list[dict[str, Any]] = field(default_factory=list)
    context_compactions: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        client: ModelClient,
        executor: ToolExecutor,
        *,
        max_steps: int = 12,
        max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
        max_repeated_tool_batches: int = 3,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        verification_policy: str = "test",
        approve_incomplete: Callable[[dict[str, Any]], bool] | None = None,
        on_event: Any = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if max_tool_result_chars < 1:
            raise ValueError("max_tool_result_chars must be positive")
        if max_repeated_tool_batches < 2:
            raise ValueError("max_repeated_tool_batches must be at least 2")
        self.client = client
        self.executor = executor
        self.max_steps = max_steps
        self.max_tool_result_chars = max_tool_result_chars
        self.max_repeated_tool_batches = max_repeated_tool_batches
        self.max_context_chars = max_context_chars
        self.completion_policy = CompletionPolicy(verification_policy)
        self.approve_incomplete = approve_incomplete or (lambda _review: False)
        self.on_event = on_event or (lambda _event: None)

    def _emit(self, event: dict[str, Any]) -> None:
        self.on_event(event)

    def run(self, task: str) -> AgentResult:
        if not task.strip():
            raise ValueError("task is empty")
        self.task_plan = TaskPlan()
        self.context_manager = ContextManager(self.max_context_chars)
        history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Workspace: {self.executor.workspace}\nTask: {task}",
            },
        ]
        tool_call_count = 0
        last_reminded_state = ""
        previous_tool_batch = ""
        repeated_tool_batches = 0
        handled_failure_version = self.executor.verification_failure_version
        for step in range(1, self.max_steps + 1):
            self._emit({"type": "model_request", "step": step})
            model_history, compaction = self.context_manager.prepare(
                history,
                self._context_state(),
            )
            if compaction is not None:
                self._emit({"type": "context_compacted", "step": step, **compaction})
            message = self.client.complete(model_history, TOOL_SCHEMAS)
            assistant = {
                "role": "assistant",
                "content": message.get("content"),
            }
            raw_tool_calls = message.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                raw_tool_calls = []
            tool_calls = [call for call in raw_tool_calls if isinstance(call, dict)]
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            history.append(assistant)
            if not tool_calls:
                decision = self.completion_policy.evaluate(self.executor, self.task_plan)
                if decision.accepted:
                    return self._finish(
                        answer=str(message.get("content") or "(model returned no final answer)"),
                        step=step,
                        stop_reason=decision.reason,
                        tool_call_count=tool_call_count,
                        history=history,
                    )
                current_state = self._completion_state_signature()
                if current_state != last_reminded_state and step < self.max_steps:
                    changed_files = sorted(self.executor.changed_files)
                    evidence = ", ".join(decision.evidence) or "none"
                    reminder = self._completion_reminder(decision, changed_files, evidence)
                    history.append({"role": "user", "content": reminder})
                    last_reminded_state = current_state
                    event_type = (
                        "plan_required" if decision.reason == "plan_incomplete" else "verification_required"
                    )
                    self._emit(
                        {
                            "type": event_type,
                            "step": step,
                            "reason": decision.reason,
                            "policy": self.completion_policy.minimum,
                            "evidence": decision.evidence,
                            "task_plan": self.task_plan.snapshot(),
                            "changed_files": changed_files,
                        }
                    )
                    previous_tool_batch = ""
                    repeated_tool_batches = 0
                    continue
                answer = str(message.get("content") or "(model returned no final answer)")
                review = {
                    "answer": answer,
                    "reason": decision.reason,
                    "verification_policy": self.completion_policy.minimum,
                    "verification_status": decision.verification_status,
                    "verification_evidence": decision.evidence,
                    "changed_files": sorted(self.executor.changed_files),
                    "task_plan": self.task_plan.snapshot(),
                }
                self._emit(
                    {
                        "type": "user_review_required",
                        "step": step,
                        **review,
                    }
                )
                if self.approve_incomplete(review):
                    history.append(
                        {
                            "role": "user",
                            "content": "User explicitly accepted completion with insufficient evidence.",
                        }
                    )
                    return self._finish(
                        answer=answer,
                        step=step,
                        stop_reason="user_accepted_incomplete",
                        tool_call_count=tool_call_count,
                        history=history,
                    )
                self._emit({"type": "stopped", "reason": decision.reason, "step": step})
                return self._result(
                    answer=answer,
                    steps=step,
                    stop_reason=decision.reason,
                    tool_call_count=tool_call_count,
                    history=history,
                )

            tool_batch = self._tool_batch_signature(tool_calls)
            if tool_batch == previous_tool_batch:
                repeated_tool_batches += 1
            else:
                previous_tool_batch = tool_batch
                repeated_tool_batches = 1
            if repeated_tool_batches >= self.max_repeated_tool_batches:
                reason = (
                    f"the model requested the same tool batch {repeated_tool_batches} times in a row"
                )
                for call_index, call in enumerate(tool_calls, start=1):
                    tool_call_count += 1
                    result = f"ERROR: {reason}; execution skipped to prevent a loop"
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"step-{step}-call-{call_index}"),
                            "content": result,
                        }
                    )
                    self._emit(
                        {
                            "type": "tool_result",
                            "step": step,
                            "name": self._call_name(call),
                            "result": result,
                        }
                    )
                answer = f"Stopped because {reason}. Rephrase the task or inspect the repeated call."
                history.append({"role": "assistant", "content": answer})
                self._emit({"type": "stopped", "reason": "repeated_tool_call", "step": step})
                return self._result(
                    answer=answer,
                    steps=step,
                    stop_reason="repeated_tool_call",
                    tool_call_count=tool_call_count,
                    history=history,
                )

            for call_index, call in enumerate(tool_calls, start=1):
                tool_call_count += 1
                result = self._run_tool_call(call)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", f"step-{step}-call-{call_index}"),
                    "content": result,
                }
                history.append(tool_message)
                self._emit({"type": "tool_result", "step": step, "name": self._call_name(call), "result": result})

            if (
                self.executor.verification_failure_version > handled_failure_version
                and self.executor.last_verification_failure is not None
            ):
                failure = self.executor.last_verification_failure
                changed_files = ", ".join(failure["changed_files"]) or "(none tracked)"
                reflection = (
                    "Reflection checkpoint: the latest verification failed. Base the next action on "
                    "the concrete evidence below instead of guessing or repeating the same patch. "
                    "Inspect the relevant code before editing, form a specific failure hypothesis, "
                    "then rerun verification.\n"
                    f"Command: {failure['command']}\n"
                    f"Exit: {failure['exit_code']}\n"
                    f"Changed files: {changed_files}\n"
                    f"Diagnostic excerpt:\n{failure['excerpt']}"
                )
                history.append({"role": "user", "content": reflection})
                handled_failure_version = self.executor.verification_failure_version
                previous_tool_batch = ""
                repeated_tool_batches = 0
                self._emit(
                    {
                        "type": "reflection_required",
                        "step": step,
                        "command": failure["command"],
                    }
                )

        answer = (
            f"Reached the maximum number of steps ({self.max_steps}); stopped to avoid an infinite loop. "
            "Inspect the workspace and continue with a new task if needed."
        )
        history.append({"role": "assistant", "content": answer})
        self._emit({"type": "stopped", "reason": "max_steps", "step": self.max_steps})
        return self._result(
            answer=answer,
            steps=self.max_steps,
            stop_reason="max_steps",
            tool_call_count=tool_call_count,
            history=history,
        )

    def _result(
        self,
        *,
        answer: str,
        steps: int,
        stop_reason: str,
        tool_call_count: int,
        history: list[dict[str, Any]],
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            steps=steps,
            stop_reason=stop_reason,
            tool_calls=tool_call_count,
            verification_status=self.executor.verification_status,
            verification_policy=self.completion_policy.minimum,
            verification_evidence=self.executor.verification_evidence,
            verification_records=[record.to_dict() for record in self.executor.verification_records],
            changed_files=sorted(self.executor.changed_files),
            task_plan=self.task_plan.snapshot(),
            context_compactions=self.context_manager.compaction_count,
            history=history,
        )

    def _finish(
        self,
        *,
        answer: str,
        step: int,
        stop_reason: str,
        tool_call_count: int,
        history: list[dict[str, Any]],
    ) -> AgentResult:
        self._emit(
            {
                "type": "final",
                "step": step,
                "answer": answer,
                "stop_reason": stop_reason,
                "verification_status": self.executor.verification_status,
            }
        )
        return self._result(
            answer=answer,
            steps=step,
            stop_reason=stop_reason,
            tool_call_count=tool_call_count,
            history=history,
        )

    @staticmethod
    def _call_name(call: dict[str, Any]) -> str:
        function = call.get("function") or {}
        return str(function.get("name") or call.get("name") or "unknown")

    @classmethod
    def _tool_batch_signature(cls, tool_calls: list[dict[str, Any]]) -> str:
        normalized = []
        for call in tool_calls:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments", call.get("arguments", {}))
            try:
                arguments: Any = decode_arguments(raw_arguments)
            except ToolError:
                arguments = raw_arguments
            normalized.append({"name": cls._call_name(call), "arguments": arguments})
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)

    def _run_tool_call(self, call: dict[str, Any]) -> str:
        name = self._call_name(call)
        function = call.get("function") or {}
        try:
            arguments = decode_arguments(function.get("arguments", call.get("arguments", {})))
            self._emit({"type": "tool_call", "name": name, "arguments": arguments})
            if name == "update_plan":
                result = self.task_plan.apply_proposal(arguments.get("items"))
                self._emit({"type": "plan_updated", "task_plan": self.task_plan.snapshot()})
            else:
                result = self.executor.call(name, arguments)
                if self._tool_result_succeeded(name, result):
                    self.task_plan.record_action(name, result)
            if len(result) <= self.max_tool_result_chars:
                return result
            omitted = len(result) - self.max_tool_result_chars
            return result[: self.max_tool_result_chars] + f"\n... ({omitted} characters omitted)"
        except (ToolError, ValueError, TypeError) as exc:
            return f"ERROR: {exc}"

    @staticmethod
    def _tool_result_succeeded(name: str, result: str) -> bool:
        if result.startswith("ERROR:"):
            return False
        if name == "run_command":
            return "exit_code=0" in result and "verification=failed" not in result
        return True

    def _context_state(self) -> dict[str, Any]:
        return {
            "task_plan": self.task_plan.snapshot(),
            "change_version": self.executor.change_version,
            "changed_files": sorted(self.executor.changed_files),
            "file_versions": dict(self.executor.file_versions),
            "verification_status": self.executor.verification_status,
            "verification_evidence": self.executor.verification_evidence,
            "last_verification_failure": self.executor.last_verification_failure,
        }

    def _completion_state_signature(self) -> str:
        return json.dumps(self._context_state(), ensure_ascii=False, sort_keys=True)

    def _completion_reminder(
        self,
        decision: CompletionDecision,
        changed_files: list[str],
        evidence: str,
    ) -> str:
        if decision.reason == "plan_incomplete":
            unfinished = ", ".join(
                f"{item.id}={item.status}" for item in self.task_plan.unfinished_items
            )
            return (
                "Controller rejected the completion proposal because the structured task plan "
                f"is incomplete: {unfinished}. Continue the current item, perform real tool work, "
                "then call update_plan with valid state transitions."
            )
        return (
            "Controller rejected the completion proposal because its deterministic "
            f"verification policy is '{self.completion_policy.minimum}'. "
            f"Reason: {decision.reason}; current evidence: {evidence}. "
            "Run a sufficiently strong relevant check if possible. If that is impossible, "
            "explain the limitation in the next completion proposal. Changed files: "
            + (", ".join(changed_files) or "(none)")
        )


def save_transcript(path: str, result: AgentResult) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        for message in result.history:
            stream.write(json.dumps(message, ensure_ascii=False) + "\n")
        summary = {
            "type": "run_summary",
            "steps": result.steps,
            "tool_calls": result.tool_calls,
            "stop_reason": result.stop_reason,
            "verification_status": result.verification_status,
            "verification_policy": result.verification_policy,
            "verification_evidence": result.verification_evidence,
            "verification_records": result.verification_records,
            "changed_files": result.changed_files,
            "task_plan": result.task_plan,
            "context_compactions": result.context_compactions,
        }
        stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
