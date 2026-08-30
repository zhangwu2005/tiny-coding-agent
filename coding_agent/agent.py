"""The model/tool orchestration loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import TOOL_SCHEMAS, ToolError, ToolExecutor, decode_arguments


DEFAULT_MAX_TOOL_RESULT_CHARS = 20_000


class ModelClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


SYSTEM_PROMPT = """You are a careful coding agent.
Work only inside the supplied workspace. Inspect existing files before changing them.
Use search_text and bounded read_file calls to gather only the context you need.
Prefer replace_in_file for a precise edit when the old text occurs exactly once; use write_file for new files or full rewrites.
Existing files are revision-guarded: read them before editing and re-read after any stale-observation error.
Use tools for all file and command operations. Explain what you changed and verify it by running a relevant test.
Treat command execution as potentially destructive and use it only when it helps the task.
When the task is complete, stop and give a concise summary. Never include or request secrets."""


@dataclass
class AgentResult:
    answer: str
    steps: int
    stop_reason: str
    tool_calls: int
    verification_status: str
    changed_files: list[str]
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
        self.on_event = on_event or (lambda _event: None)

    def _emit(self, event: dict[str, Any]) -> None:
        self.on_event(event)

    def run(self, task: str) -> AgentResult:
        if not task.strip():
            raise ValueError("task is empty")
        history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Workspace: {self.executor.workspace}\nTask: {task}",
            },
        ]
        tool_call_count = 0
        last_reminded_change_version = 0
        previous_tool_batch = ""
        repeated_tool_batches = 0
        handled_failure_version = self.executor.verification_failure_version
        for step in range(1, self.max_steps + 1):
            self._emit({"type": "model_request", "step": step})
            message = self.client.complete(history, TOOL_SCHEMAS)
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
                if (
                    self.executor.has_unverified_changes
                    and self.executor.change_version > last_reminded_change_version
                    and step + 1 < self.max_steps
                ):
                    changed_files = sorted(self.executor.changed_files)
                    reminder = (
                        "You changed files but have no successful verification after the latest edit. "
                        "Run a relevant test or build command if possible. If verification is not possible, "
                        "explain the limitation explicitly in the final answer. Changed files: "
                        + ", ".join(changed_files)
                    )
                    history.append({"role": "user", "content": reminder})
                    last_reminded_change_version = self.executor.change_version
                    self._emit(
                        {
                            "type": "verification_required",
                            "step": step,
                            "changed_files": changed_files,
                        }
                    )
                    previous_tool_batch = ""
                    repeated_tool_batches = 0
                    continue
                answer = str(message.get("content") or "(model returned no final answer)")
                verification_status = self._verification_status()
                self._emit(
                    {
                        "type": "final",
                        "step": step,
                        "answer": answer,
                        "verification_status": verification_status,
                    }
                )
                return AgentResult(
                    answer=answer,
                    steps=step,
                    stop_reason="completed",
                    tool_calls=tool_call_count,
                    verification_status=verification_status,
                    changed_files=sorted(self.executor.changed_files),
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
                return AgentResult(
                    answer=answer,
                    steps=step,
                    stop_reason="repeated_tool_call",
                    tool_calls=tool_call_count,
                    verification_status=self._verification_status(),
                    changed_files=sorted(self.executor.changed_files),
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
        return AgentResult(
            answer=answer,
            steps=self.max_steps,
            stop_reason="max_steps",
            tool_calls=tool_call_count,
            verification_status=self._verification_status(),
            changed_files=sorted(self.executor.changed_files),
            history=history,
        )

    def _verification_status(self) -> str:
        if not self.executor.changed_files:
            return "not_needed"
        return "unverified" if self.executor.has_unverified_changes else "verified"

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
            result = self.executor.call(name, arguments)
            if len(result) <= self.max_tool_result_chars:
                return result
            omitted = len(result) - self.max_tool_result_chars
            return result[: self.max_tool_result_chars] + f"\n... ({omitted} characters omitted)"
        except (ToolError, ValueError, TypeError) as exc:
            return f"ERROR: {exc}"


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
            "changed_files": result.changed_files,
        }
        stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
