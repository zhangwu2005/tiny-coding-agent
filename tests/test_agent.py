from pathlib import Path

from coding_agent.agent import Agent
from coding_agent.tools import ToolError, ToolExecutor, decode_arguments


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"answer.py","content":"print(42)\\n"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Task completed."}


class LoopingClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"loop-{self.calls}",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": '{"path":"."}'},
                }
            ],
        }


class ReadThenFinishClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"large.txt"}'},
                    }
                ],
            }
        return {"role": "assistant", "content": "done"}


def test_agent_executes_tool_then_stops(tmp_path: Path):
    events = []
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    result = Agent(FakeClient(), executor, on_event=events.append).run("create answer.py")
    assert result.steps == 2
    assert result.stop_reason == "completed"
    assert result.tool_calls == 1
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "print(42)\n"
    assert any(event["type"] == "tool_call" for event in events)
    assert result.answer == "Task completed."


def test_workspace_blocks_path_escape(tmp_path: Path):
    executor = ToolExecutor(tmp_path)
    try:
        executor.read_file("../secret.txt")
    except ToolError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape was not blocked")


def test_command_requires_approval(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: False)
    try:
        executor.run_command("echo should-not-run")
    except ToolError as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("unapproved command was run")


def test_decode_arguments_rejects_non_object():
    try:
        decode_arguments("[]")
    except ToolError as exc:
        assert "object" in str(exc)
    else:
        raise AssertionError("array accepted as arguments")


def test_read_file_supports_bounded_line_ranges(tmp_path: Path):
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    result = ToolExecutor(tmp_path).read_file("sample.txt", start_line=2, end_line=3)
    assert "lines 2-3 of 4" in result
    assert result.endswith("two\nthree\n")
    assert "one" not in result


def test_search_text_returns_paths_and_line_numbers(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "alpha.py").write_text("first\nNeedle here\n", encoding="utf-8")
    (source / "beta.py").write_text("nothing\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "ignored.py").write_text("Needle hidden\n", encoding="utf-8")
    result = ToolExecutor(tmp_path).search_text("needle")
    assert "src/alpha.py:2: Needle here" in result
    assert "beta.py" not in result
    assert ".venv" not in result


def test_replace_in_file_requires_a_unique_match(tmp_path: Path):
    target = tmp_path / "config.py"
    target.write_text("value = 1\n", encoding="utf-8")
    result = ToolExecutor(tmp_path).replace_in_file("config.py", "value = 1", "value = 2")
    assert result == "replaced 1 occurrence in config.py"
    assert target.read_text(encoding="utf-8") == "value = 2\n"

    target.write_text("x x\n", encoding="utf-8")
    try:
        ToolExecutor(tmp_path).replace_in_file("config.py", "x", "y")
    except ToolError as exc:
        assert "found 2 matches" in str(exc)
    else:
        raise AssertionError("ambiguous replacement was accepted")


def test_agent_reports_max_steps(tmp_path: Path):
    result = Agent(LoopingClient(), ToolExecutor(tmp_path), max_steps=2).run("keep looking")
    assert result.stop_reason == "max_steps"
    assert result.steps == 2
    assert result.tool_calls == 2


def test_agent_truncates_large_tool_results(tmp_path: Path):
    (tmp_path / "large.txt").write_text("x" * 200, encoding="utf-8")
    result = Agent(
        ReadThenFinishClient(),
        ToolExecutor(tmp_path),
        max_tool_result_chars=40,
    ).run("read the file")
    tool_messages = [message for message in result.history if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert "characters omitted" in tool_messages[0]["content"]


if __name__ == "__main__":
    # The tests are pytest-compatible, but this runner has no third-party dependency.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        temporary_path = Path(directory)
        test_agent_executes_tool_then_stops(temporary_path)
        test_workspace_blocks_path_escape(temporary_path)
        test_command_requires_approval(temporary_path)
        test_read_file_supports_bounded_line_ranges(temporary_path)
        test_search_text_returns_paths_and_line_numbers(temporary_path)
        test_replace_in_file_requires_a_unique_match(temporary_path)
        test_agent_reports_max_steps(temporary_path)
        test_agent_truncates_large_tool_results(temporary_path)
    test_decode_arguments_rejects_non_object()
    print("9 offline checks passed")
