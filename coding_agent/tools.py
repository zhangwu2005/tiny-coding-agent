"""Local, deliberately small tools exposed to the model."""

from __future__ import annotations

import hashlib
import json
import os
import re
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

HIGH_RISK_COMMAND_PATTERNS = (
    (
        re.compile(
            r"\b(?:curl|wget|invoke-webrequest|irm)\b[^\r\n|]*\|\s*"
            r"(?:bash|sh|powershell|pwsh|iex|invoke-expression)\b",
            re.IGNORECASE,
        ),
        "downloads and executes remote content",
    ),
    (re.compile(r"\brm\b[^\r\n]*(?:-rf|-fr|--recursive)", re.IGNORECASE), "recursive deletion"),
    (re.compile(r"\bremove-item\b[^\r\n]*\s-recurse\b", re.IGNORECASE), "recursive deletion"),
    (re.compile(r"\brmdir\b[^\r\n]*\s/[sq]\b", re.IGNORECASE), "recursive directory deletion"),
    (re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE), "destructive Git reset"),
    (re.compile(r"\bgit\s+clean\s+-[^\s]*[fd][^\s]*", re.IGNORECASE), "destructive Git clean"),
    (re.compile(r"\bgit\s+push\b[^\r\n]*\s--force(?:-with-lease)?\b", re.IGNORECASE), "forced remote update"),
    (re.compile(r"\b(?:format|diskpart|shutdown)\b", re.IGNORECASE), "system-level operation"),
)
MEDIUM_RISK_COMMAND_PATTERNS = (
    (re.compile(r"\b(?:pip|uv)\s+(?:install|uninstall)\b", re.IGNORECASE), "changes Python packages"),
    (re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:install|add|remove)\b", re.IGNORECASE), "changes packages"),
    (re.compile(r"\bgit\s+push\b", re.IGNORECASE), "changes a remote repository"),
    (re.compile(r"\b(?:curl|wget|invoke-webrequest)\b", re.IGNORECASE), "uses the network"),
)
VERIFICATION_COMMAND_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)(?:python\s+-m\s+)?pytest\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)python\s+-m\s+unittest\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)python\s+-m\s+tests(?:\.[\w.-]+)*\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)python\s+[^;&|\r\n]*test[^;&|\r\n]*\.py\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)python\s+-m\s+compileall\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:npm|pnpm|yarn)\s+(?:run\s+)?(?:test|lint|build)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)cargo\s+(?:test|check|clippy)\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:go|dotnet|mvn|gradle)\s+test\b", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)(?:ruff\s+check|mypy|pyright|eslint|tsc)\b", re.IGNORECASE),
)


class ToolError(RuntimeError):
    """A safe, user-facing tool failure."""


def assess_command_risk(command: str) -> tuple[str, str]:
    """Return a small, explainable risk hint; this is not a security sandbox."""
    for pattern, reason in HIGH_RISK_COMMAND_PATTERNS:
        if pattern.search(command):
            return "high", reason
    for pattern, reason in MEDIUM_RISK_COMMAND_PATTERNS:
        if pattern.search(command):
            return "medium", reason
    if is_verification_command(command):
        return "low", "recognized verification command"
    return "medium", "unrecognized shell command"


def is_verification_command(command: str) -> bool:
    """Recognize common test/build checks without claiming that arbitrary commands verify code."""
    return any(pattern.search(command) for pattern in VERIFICATION_COMMAND_PATTERNS)


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
        self.change_version = 0
        self.last_verified_version = 0
        self.changed_files: set[str] = set()
        self.observed_revisions: dict[str, str] = {}
        self.fully_observed_files: set[str] = set()
        self.verification_failure_version = 0
        self.last_verification_failure: dict[str, Any] | None = None

    @property
    def has_unverified_changes(self) -> bool:
        return bool(self.changed_files) and (
            self.change_version > self.last_verified_version
            or self.last_verification_failure is not None
        )

    def _record_change(self, file_path: Path) -> None:
        self.change_version += 1
        self.changed_files.add(file_path.relative_to(self.workspace).as_posix())

    @staticmethod
    def _file_revision(file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def _remember_revision(self, file_path: Path, *, full_content_seen: bool = False) -> str:
        revision = self._file_revision(file_path)
        relative = file_path.relative_to(self.workspace).as_posix()
        self.observed_revisions[relative] = revision
        if full_content_seen:
            self.fully_observed_files.add(relative)
        return revision

    def _require_fresh_observation(self, file_path: Path, *, require_full: bool = False) -> None:
        relative = file_path.relative_to(self.workspace).as_posix()
        observed = self.observed_revisions.get(relative)
        if observed is None:
            raise ToolError(f"read_file must be called before editing existing file: {relative}")
        current = self._file_revision(file_path)
        if current != observed:
            self.observed_revisions.pop(relative, None)
            self.fully_observed_files.discard(relative)
            raise ToolError(
                f"stale observation for {relative}; the file changed since it was read, so read it again"
            )
        if require_full and relative not in self.fully_observed_files:
            raise ToolError(
                f"write_file requires a full read of existing file {relative}; "
                "use read_file for the whole file or make a precise replace_in_file edit"
            )

    def _record_verification_failure(self, command: str, exit_code: str, output: str) -> None:
        nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
        excerpt = "\n".join(nonempty_lines[-8:])[-1_200:] or "(no diagnostic output)"
        self.verification_failure_version += 1
        self.last_verification_failure = {
            "command": command,
            "exit_code": exit_code,
            "excerpt": excerpt,
            "changed_files": sorted(self.changed_files),
        }

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
            revision = self._remember_revision(file_path, full_content_seen=True)
            return f"{path}: empty file (revision={revision[:12]})"
        if start_line > len(lines):
            raise ToolError(f"start_line {start_line} is beyond the file's {len(lines)} lines")
        actual_end = min(end_line, len(lines))
        revision = self._remember_revision(
            file_path,
            full_content_seen=start_line == 1 and actual_end == len(lines),
        )
        content = "".join(lines[start_line - 1 : actual_end])
        return (
            f"{path}: lines {start_line}-{actual_end} of {len(lines)} "
            f"(revision={revision[:12]})\n{content}"
        )

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
        encoded_content = content.encode("utf-8")
        if len(encoded_content) > MAX_FILE_BYTES:
            raise ToolError(f"content is larger than {MAX_FILE_BYTES} bytes")
        previous_content = file_path.read_text(encoding="utf-8") if file_path.is_file() else None
        relative = file_path.relative_to(self.workspace).as_posix()
        if previous_content == content:
            return f"unchanged: {relative} already contains the requested {len(encoded_content)} bytes"
        if previous_content is not None:
            self._require_fresh_observation(file_path, require_full=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        revision = self._remember_revision(file_path, full_content_seen=True)
        self._record_change(file_path)
        action = "created" if previous_content is None else "updated"
        return (
            f"{action} {relative} ({len(encoded_content)} bytes, revision={revision[:12]}); "
            "verification is now stale"
        )

    def replace_in_file(self, path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            raise ToolError("old_text must not be empty")
        file_path = self._path(path)
        if not file_path.is_file():
            raise ToolError(f"not a file: {path}")
        if file_path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
        self._require_fresh_observation(file_path)
        content = file_path.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise ToolError(f"old_text must match exactly once; found {occurrences} matches")
        updated = content.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError(f"updated file would be larger than {MAX_FILE_BYTES} bytes")
        relative = file_path.relative_to(self.workspace).as_posix()
        if updated == content:
            return f"unchanged: replacement has no effect in {relative}"
        file_path.write_text(updated, encoding="utf-8")
        revision = self._remember_revision(file_path)
        self._record_change(file_path)
        return (
            f"replaced 1 occurrence in {relative} (revision={revision[:12]}); "
            "verification is now stale"
        )

    def run_command(self, command: str) -> str:
        if not command.strip():
            raise ToolError("command is empty")
        risk_level, risk_reason = assess_command_risk(command)
        verification = is_verification_command(command)
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
            if verification:
                self._record_verification_failure(command, "timeout", output)
            return (
                f"risk={risk_level} ({risk_reason})\n"
                f"verification={'failed' if verification else 'not_recorded'}\n"
                f"command timed out after {self.command_timeout:g}s\n{output[-6000:]}"
            )
        if completed.returncode == 0 and verification:
            self.last_verified_version = self.change_version
            self.last_verification_failure = None
        output = (completed.stdout + completed.stderr)[-12_000:]
        if completed.returncode != 0 and verification:
            self._record_verification_failure(command, str(completed.returncode), output)
        if verification:
            verification_status = "passed" if completed.returncode == 0 else "failed"
        else:
            verification_status = "not_recorded"
        return (
            f"risk={risk_level} ({risk_reason})\n"
            f"verification={verification_status}\n"
            f"exit_code={completed.returncode}\n{output}"
        ).rstrip()

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
