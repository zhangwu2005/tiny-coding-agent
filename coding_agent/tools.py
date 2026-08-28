"""Local, deliberately small tools exposed to the model."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable


MAX_FILE_BYTES = 120_000


class ToolError(RuntimeError):
    """A safe, user-facing tool failure."""


class ToolExecutor:
    def __init__(
        self,
        workspace: str | Path,
        *,
        approve_command: Callable[[str], bool] | None = None,
        command_timeout: float = 30.0,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.approve_command = approve_command or (lambda _command: False)
        self.command_timeout = command_timeout

    def _path(self, raw: str) -> Path:
        if not raw or Path(raw).is_absolute():
            raise ToolError("path must be a non-empty relative path")
        candidate = (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError("path escapes the workspace") from exc
        return candidate

    def list_files(self, path: str = ".") -> str:
        directory = self._path(path)
        if not directory.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries: list[str] = []
        for item in sorted(directory.rglob("*")):
            if ".git" in item.parts or "__pycache__" in item.parts:
                continue
            relative = item.relative_to(self.workspace).as_posix()
            entries.append(relative + ("/" if item.is_dir() else ""))
            if len(entries) >= 200:
                entries.append("... (truncated)")
                break
        return "\n".join(entries) or "(empty)"

    def read_file(self, path: str) -> str:
        file_path = self._path(path)
        if not file_path.is_file():
            raise ToolError(f"not a file: {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
        return file_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        file_path = self._path(path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"content is larger than {MAX_FILE_BYTES} bytes")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {file_path.relative_to(self.workspace).as_posix()}"

    def run_command(self, command: str) -> str:
        if not command.strip():
            raise ToolError("command is empty")
        if not self.approve_command(command):
            raise ToolError("command was not approved by the user")
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return f"command timed out after {self.command_timeout:g}s\n{output[-6000:]}"
        output = (completed.stdout + completed.stderr)[-12_000:]
        return f"exit_code={completed.returncode}\n{output}".rstrip()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        functions = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_command": self.run_command,
        }
        function = functions.get(name)
        if function is None:
            raise ToolError(f"unknown tool: {name}")
        try:
            return str(function(**arguments))
        except TypeError as exc:
            raise ToolError(f"invalid arguments for {name}: {exc}") from exc
        except (OSError, UnicodeError) as exc:
            raise ToolError(f"tool {name} failed: {exc}") from exc


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a relative workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the workspace. The user must approve it.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def decode_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ToolError("tool arguments are not JSON")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"tool arguments are invalid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ToolError("tool arguments must be a JSON object")
    return decoded
