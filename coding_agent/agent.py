"""The model/tool orchestration loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tools import TOOL_SCHEMAS, ToolError, ToolExecutor, decode_arguments


class ModelClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


SYSTEM_PROMPT = """You are a careful coding agent.
Work only inside the supplied workspace. Inspect existing files before changing them.
Use tools for all file and command operations. Explain what you changed and verify it by running a relevant test.
Treat command execution as potentially destructive and use it only when it helps the task.
When the task is complete, stop and give a concise summary. Never include or request secrets."""


@dataclass
class AgentResult:
    answer: str
    steps: int
    history: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        client: ModelClient,
        executor: ToolExecutor,
        *,
        max_steps: int = 12,
        on_event: Any = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.client = client
        self.executor = executor
        self.max_steps = max_steps
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
                answer = str(message.get("content") or "(model returned no final answer)")
                self._emit({"type": "final", "step": step, "answer": answer})
                return AgentResult(answer=answer, steps=step, history=history)

            for call_index, call in enumerate(tool_calls, start=1):
                result = self._run_tool_call(call)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", f"step-{step}-call-{call_index}"),
                    "content": result,
                }
                history.append(tool_message)
                self._emit({"type": "tool_result", "step": step, "name": self._call_name(call), "result": result})

        answer = (
            f"Reached the maximum number of steps ({self.max_steps}); stopped to avoid an infinite loop. "
            "Inspect the workspace and continue with a new task if needed."
        )
        history.append({"role": "assistant", "content": answer})
        self._emit({"type": "stopped", "reason": "max_steps", "step": self.max_steps})
        return AgentResult(answer=answer, steps=self.max_steps, history=history)

    @staticmethod
    def _call_name(call: dict[str, Any]) -> str:
        function = call.get("function") or {}
        return str(function.get("name") or call.get("name") or "unknown")

    def _run_tool_call(self, call: dict[str, Any]) -> str:
        name = self._call_name(call)
        function = call.get("function") or {}
        try:
            arguments = decode_arguments(function.get("arguments", call.get("arguments", {})))
            self._emit({"type": "tool_call", "name": name, "arguments": arguments})
            return self.executor.call(name, arguments)
        except (ToolError, ValueError, TypeError) as exc:
            return f"ERROR: {exc}"


def save_transcript(path: str, result: AgentResult) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        for message in result.history:
            stream.write(json.dumps(message, ensure_ascii=False) + "\n")
