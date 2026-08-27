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
                        "function": {"name": "write_file", "arguments": '{"path":"answer.py","content":"print(42)\\n"}'},
                    }
                ],
            }
        return {"role": "assistant", "content": "已创建并完成任务。"}


def test_agent_executes_tool_then_stops(tmp_path: Path):
    events = []
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    result = Agent(FakeClient(), executor, on_event=events.append).run("创建 answer.py")
    assert result.steps == 2
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "print(42)\n"
    assert any(event["type"] == "tool_call" for event in events)
    assert result.answer == "已创建并完成任务。"


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


if __name__ == "__main__":
    # The tests are pytest-compatible, but this tiny runner keeps the demo dependency-free.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_agent_executes_tool_then_stops(Path(directory))
        test_workspace_blocks_path_escape(Path(directory))
        test_command_requires_approval(Path(directory))
    test_decode_arguments_rejects_non_object()
    print("4 offline checks passed")
