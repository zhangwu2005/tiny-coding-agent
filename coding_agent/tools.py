"""Local, deliberately small tools exposed to the model."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable


MAX_FILE_BYTES = 120_000
DEFAULT_READ_LINES = 200
MAX_READ_LINES = 500
MAX_SEARCH_MATCHES = 100
MAX_SEARCH_FILES = 1_000
IGNORED_DIRECTORY_NAMES = {".git", ".venv", "__pycache__", "node_modules"}


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

    def _is_ignored(self, path: Path) -> bool:
        relative_parts = path.relative_to(self.workspace).parts
        return any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts)

    def _iter_files(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            if not self._is_ignored(root):
                yield root
            return
        for current, directory_names, file_names in os.walk(root):
            directory_names[:] = sorted(
                name for name in directory_names if name not in IGNORED_DIRECTORY_NAMES
            )
            current_path = Path(current)
            for file_name in sorted(file_names):
                yield current_path / file_name

    def list_files(self, path: str = ".") -> str:
        directory = self._path(path)
        if not directory.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries: list[str] = []
        for current, directory_names, file_names in os.walk(directory):
            directory_names[:] = sorted(
                name for name in directory_names if name not in IGNORED_DIRECTORY_NAMES
            )
            current_path = Path(current)
            items = [(name, True) for name in directory_names]
            items.extend((name, False) for name in sorted(file_names))
            for name, is_directory in items:
                item = current_path / name
                relative = item.relative_to(self.workspace).as_posix()
                entries.append(relative + ("/" if is_directory else ""))
                if len(entries) >= 200:
                    entries.append("... (truncated)")
                    return "\n".join(entries)
        return "\n".join(entries) or "(empty)"

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        file_path = self._path(path)
        if not file_path.is_file():
            raise ToolError(f"not a file: {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
        if start_line < 1:
            raise ToolError("start_line must be at least 1")
        if end_line is None:
            end_line = start_line + DEFAULT_READ_LINES - 1
        if end_line < start_line:
            raise ToolError("end_line must be greater than or equal to start_line")
        if end_line - start_line + 1 > MAX_READ_LINES:
            raise ToolError(f"cannot read more than {MAX_READ_LINES} lines at once")

        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines:
            return f"{path}: empty file"
        if start_line > len(lines):
            raise ToolError(f"start_line {start_line} is beyond the file's {len(lines)} lines")
        actual_end = min(end_line, len(lines))
        content = "".join(lines[start_line - 1 : actual_end])
        return f"{path}: lines {start_line}-{actual_end} of {len(lines)}\n{content}"

    def search_text(self, query: str, path: str = ".", case_sensitive: bool = False) -> str:
        if not isinstance(query, str) or not query:
            raise ToolError("query must be a non-empty string")
        if len(query) > 200:
            raise ToolError("query is longer than 200 characters")
        if not isinstance(case_sensitive, bool):
            raise ToolError("case_sensitive must be a boolean")

        root = self._path(path)
        if not root.exists():
            raise ToolError(f"path does not exist: {path}")
        needle = query if case_sensitive else query.casefold()
        matches: list[str] = []
        files_checked = 0

        for file_path in self._iter_files(root):
            if files_checked >= MAX_SEARCH_FILES:
                break
            files_checked += 1
            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
                lines = file_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                relative = file_path.relative_to(self.workspace).as_posix()
                preview = line.strip()
                if len(preview) > 300:
                    preview = preview[:297] + "..."
                matches.append(f"{relative}:{line_number}: {preview}")
                if len(matches) >= MAX_SEARCH_MATCHES:
                    matches.append("... (match limit reached)")
                    return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def write_file(self, path: str, content: str) -> str:
        file_path = self._path(path)
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"content is larger than {MAX_FILE_BYTES} bytes")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {file_path.relative_to(self.workspace).as_posix()}"

    def replace_in_file(self, path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            raise ToolError("old_text must not be empty")
        file_path = self._path(path)
        if not file_path.is_file():
            raise ToolError(f"not a file: {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
        content = file_path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ToolError(f"old_text must match exactly once; found {occurrences} matches")
        updated = content.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"updated file would be larger than {MAX_FILE_BYTES} bytes")
        file_path.write_text(updated, encoding="utf-8")
        relative = file_path.relative_to(self.workspace).as_posix()
        return f"replaced 1 occurrence in {relative}"

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
            "search_text": self.search_text,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
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
            "description": "Read a bounded line range from a UTF-8 text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search UTF-8 workspace files for literal text and return matching file names and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "case_sensitive": {"type": "boolean", "default": False},
                },
                "required": ["query"],
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
            "name": "replace_in_file",
            "description": "Safely edit an existing UTF-8 file by replacing text that must occur exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
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
