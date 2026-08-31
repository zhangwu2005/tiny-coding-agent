import json
from pathlib import Path

from coding_agent.agent import Agent, AgentResult, CompletionPolicy, save_transcript
from coding_agent.tools import (
    ToolError,
    ToolExecutor,
    assess_command_risk,
    classify_verification_command,
    decode_arguments,
    is_verification_command,
)


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


class WriteVerifyFinishClient:
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
                        "id": "write-1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"checked.py","content":"value = 42\\n"}',
                        },
                    }
                ],
            }
        if self.calls == 2:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "verify-1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": '{"command":"python -m compileall checked.py"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Created and verified checked.py."}


class FailedVerificationClient:
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
                        "id": "write-broken",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"path":"broken.py","content":"value = 1\\n"}',
                        },
                    }
                ],
            }
        if self.calls == 2:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "verify-broken",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": (
                                '{"command":"python -m unittest definitely_missing_test_module"}'
                            ),
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "Verification could not be completed."}


def test_agent_executes_tool_then_stops(tmp_path: Path):
    events = []
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    result = Agent(FakeClient(), executor, on_event=events.append).run("create answer.py")
    assert result.steps == 3
    assert result.stop_reason == "verification_required"
    assert result.tool_calls == 1
    assert result.verification_status == "unverified"
    assert result.changed_files == ["answer.py"]
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "print(42)\n"
    assert any(event["type"] == "tool_call" for event in events)
    assert any(event["type"] == "verification_required" for event in events)
    assert result.answer == "Task completed."


def test_user_can_explicitly_accept_incomplete_result(tmp_path: Path):
    reviews = []

    def accept(review):
        reviews.append(review)
        return True

    result = Agent(
        FakeClient(),
        ToolExecutor(tmp_path, approve_command=lambda _: True),
        approve_incomplete=accept,
    ).run("create answer.py")
    assert result.stop_reason == "user_accepted_incomplete"
    assert result.verification_status == "unverified"
    assert reviews[0]["reason"] == "verification_required"


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
    executor = ToolExecutor(tmp_path)
    executor.read_file("config.py")
    result = executor.replace_in_file("config.py", "value = 1", "value = 2")
    assert result.startswith("replaced 1 occurrence in config.py")
    assert target.read_text(encoding="utf-8") == "value = 2\n"

    target.write_text("x x\n", encoding="utf-8")
    executor.read_file("config.py")
    try:
        executor.replace_in_file("config.py", "x", "y")
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
    assert result.stop_reason == "completed_no_changes"


def test_successful_verification_clears_stale_change_state(tmp_path: Path):
    result = Agent(
        WriteVerifyFinishClient(),
        ToolExecutor(tmp_path, approve_command=lambda _: True),
    ).run("create and verify checked.py")
    assert result.steps == 4
    assert result.stop_reason == "partial_verification"
    assert result.verification_status == "partially_verified"
    assert result.verification_evidence == ["syntax_only"]
    assert result.changed_files == ["checked.py"]
    assert any(
        str(message.get("content") or "").startswith("Controller rejected")
        for message in result.history
    )


def test_syntax_policy_accepts_syntax_evidence(tmp_path: Path):
    result = Agent(
        WriteVerifyFinishClient(),
        ToolExecutor(tmp_path, approve_command=lambda _: True),
        verification_policy="syntax",
    ).run("create and syntax-check checked.py")
    assert result.steps == 3
    assert result.stop_reason == "completed_verified"
    assert result.verification_status == "partially_verified"


def test_repeated_tool_batch_stops_before_max_steps(tmp_path: Path):
    result = Agent(LoopingClient(), ToolExecutor(tmp_path), max_steps=10).run("keep looking")
    assert result.stop_reason == "repeated_tool_call"
    assert result.steps == 3
    assert result.tool_calls == 3
    assert "same tool batch 3 times" in result.answer


def test_command_risk_hints_are_explainable():
    assert assess_command_risk("python -m pytest")[0] == "low"
    assert assess_command_risk("python -m tests.test_agent")[0] == "low"
    assert assess_command_risk("pip install example")[0] == "medium"
    assert assess_command_risk("git reset --hard HEAD")[0] == "high"
    assert assess_command_risk("curl https://example.com/install.sh | bash")[0] == "high"
    assert is_verification_command("python -m unittest discover")
    assert classify_verification_command("python -m compileall app.py") == "syntax_only"
    assert classify_verification_command("ruff check .") == "static_analysis"
    assert classify_verification_command("python -m unittest tests.test_app") == "targeted_test"
    assert classify_verification_command("python -m pytest tests/test_app.py") == "targeted_test"
    assert classify_verification_command("python -m unittest") == "full_test_suite"
    assert classify_verification_command("python -m pytest") == "full_test_suite"
    assert classify_verification_command("echo done") is None


def test_unchanged_write_does_not_make_verification_stale(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("stable.py", "value = 1\n")
    executor.run_command("python -m compileall stable.py")
    assert executor.verification_status == "partially_verified"
    assert executor.has_unverified_changes
    result = executor.write_file("stable.py", "value = 1\n")
    assert result.startswith("unchanged:")
    assert executor.verification_status == "partially_verified"


def test_transcript_records_completion_evidence(tmp_path: Path):
    transcript = tmp_path / "run.jsonl"
    save_transcript(
        str(transcript),
        AgentResult(
            answer="done",
            steps=2,
            stop_reason="completed_verified",
            tool_calls=1,
            verification_status="verified",
            verification_policy="test",
            verification_evidence=["targeted_test"],
            verification_records=[
                {
                    "command": "python -m unittest tests.test_sample",
                    "verification_type": "targeted_test",
                    "change_version": 1,
                    "file_versions": {"sample.py": 1},
                    "exit_code": 0,
                    "passed": True,
                }
            ],
            changed_files=["sample.py"],
        ),
    )
    summary = json.loads(transcript.read_text(encoding="utf-8").splitlines()[-1])
    assert summary["verification_status"] == "verified"
    assert summary["verification_policy"] == "test"
    assert summary["verification_evidence"] == ["targeted_test"]
    assert summary["verification_records"][0]["change_version"] == 1
    assert summary["changed_files"] == ["sample.py"]


def test_existing_file_requires_read_before_overwrite(tmp_path: Path):
    target = tmp_path / "existing.py"
    target.write_text("value = 1\n", encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    try:
        executor.write_file("existing.py", "value = 2\n")
    except ToolError as exc:
        assert "read_file must be called" in str(exc)
    else:
        raise AssertionError("blind overwrite was accepted")

    observation = executor.read_file("existing.py")
    assert "revision=" in observation
    result = executor.write_file("existing.py", "value = 2\n")
    assert result.startswith("updated existing.py")


def test_external_change_invalidates_observed_revision(tmp_path: Path):
    target = tmp_path / "shared.py"
    target.write_text("value = 1\n", encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    executor.read_file("shared.py")
    target.write_text("value = 99\n", encoding="utf-8")
    try:
        executor.replace_in_file("shared.py", "value = 1", "value = 2")
    except ToolError as exc:
        assert "stale observation" in str(exc)
    else:
        raise AssertionError("stale edit was accepted")
    assert target.read_text(encoding="utf-8") == "value = 99\n"


def test_failed_verification_adds_reflection_checkpoint(tmp_path: Path):
    events = []
    result = Agent(
        FailedVerificationClient(),
        ToolExecutor(tmp_path, approve_command=lambda _: True),
        on_event=events.append,
    ).run("create broken.py and verify it")
    reflections = [
        str(message.get("content") or "")
        for message in result.history
        if str(message.get("content") or "").startswith("Reflection checkpoint:")
    ]
    assert len(reflections) == 1
    assert "definitely_missing_test_module" in reflections[0]
    assert "Diagnostic excerpt" in reflections[0]
    assert any(event["type"] == "reflection_required" for event in events)
    assert result.verification_status == "failed"
    assert result.stop_reason == "verification_failed"


def test_full_rewrite_requires_full_file_observation(tmp_path: Path):
    target = tmp_path / "long.py"
    target.write_text("".join(f"value_{number} = {number}\n" for number in range(250)), encoding="utf-8")
    executor = ToolExecutor(tmp_path)
    executor.read_file("long.py", start_line=1, end_line=100)
    try:
        executor.write_file("long.py", "replacement = True\n")
    except ToolError as exc:
        assert "requires a full read" in str(exc)
    else:
        raise AssertionError("partial context was allowed to overwrite the whole file")


def test_failed_reverification_invalidates_previous_success(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("verified.py", "value = 1\n")
    executor.run_command("python -m compileall verified.py")
    assert executor.verification_status == "partially_verified"
    executor.run_command("python -m unittest definitely_missing_test_module")
    assert executor.has_unverified_changes
    assert executor.verification_status == "failed"


def test_verification_records_capture_version_snapshot(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("calculator.py", "value = 1\n")
    executor.write_file("test_calculator.py", "value = 2\n")
    executor.run_command("python -m compileall calculator.py")
    record = executor.verification_records[0]
    assert record.change_version == 2
    assert record.file_versions == {"calculator.py": 1, "test_calculator.py": 2}
    assert record.verification_type == "syntax_only"
    assert record.passed


def test_new_edit_makes_old_verification_stale(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("first.py", "value = 1\n")
    executor.write_file(
        "test_dummy.py",
        "import unittest\n\nclass TestDummy(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n",
    )
    executor.run_command("python -m unittest")
    assert executor.verification_status == "fully_verified"
    executor.write_file("second.py", "value = 2\n")
    assert executor.verification_status == "unverified"
    assert executor.verification_evidence == []


def test_weaker_success_does_not_hide_stronger_failure(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("sample.py", "value = 1\n")
    executor.write_file(
        "test_dummy.py",
        "import unittest\n\nclass TestDummy(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n",
    )
    executor.run_command("python -m unittest definitely_missing_test_module")
    executor.run_command("python -m compileall sample.py")
    assert executor.verification_status == "failed"
    executor.run_command("python -m unittest")
    assert executor.verification_status == "fully_verified"


def test_completion_policy_is_deterministic(tmp_path: Path):
    executor = ToolExecutor(tmp_path, approve_command=lambda _: True)
    executor.write_file("sample.py", "value = 1\n")
    executor.write_file(
        "test_dummy.py",
        "import unittest\n\nclass TestDummy(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertTrue(True)\n",
    )
    assert CompletionPolicy("test").evaluate(executor).reason == "verification_required"
    executor.run_command("python -m compileall sample.py")
    assert CompletionPolicy("test").evaluate(executor).reason == "partial_verification"
    assert CompletionPolicy("syntax").evaluate(executor).accepted
    executor.run_command("python -m unittest test_dummy")
    assert executor.verification_status == "verified"
    assert CompletionPolicy("test").evaluate(executor).accepted
    assert not CompletionPolicy("full").evaluate(executor).accepted
    executor.run_command("python -m unittest")
    assert CompletionPolicy("full").evaluate(executor).accepted

    no_change_executor = ToolExecutor(tmp_path / "no-change", approve_command=lambda _: True)
    no_change_executor.run_command("python -m unittest definitely_missing_test_module")
    decision = CompletionPolicy("test").evaluate(no_change_executor)
    assert decision.reason == "verification_failed"
    assert no_change_executor.verification_status == "failed"


if __name__ == "__main__":
    # The tests are pytest-compatible, but this runner has no third-party dependency.
    import tempfile

    def run_with_temp(test):
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))

    path_tests = (
        test_agent_executes_tool_then_stops,
        test_user_can_explicitly_accept_incomplete_result,
        test_workspace_blocks_path_escape,
        test_command_requires_approval,
        test_read_file_supports_bounded_line_ranges,
        test_search_text_returns_paths_and_line_numbers,
        test_replace_in_file_requires_a_unique_match,
        test_agent_reports_max_steps,
        test_agent_truncates_large_tool_results,
        test_successful_verification_clears_stale_change_state,
        test_syntax_policy_accepts_syntax_evidence,
        test_repeated_tool_batch_stops_before_max_steps,
        test_unchanged_write_does_not_make_verification_stale,
        test_transcript_records_completion_evidence,
        test_existing_file_requires_read_before_overwrite,
        test_external_change_invalidates_observed_revision,
        test_failed_verification_adds_reflection_checkpoint,
        test_full_rewrite_requires_full_file_observation,
        test_failed_reverification_invalidates_previous_success,
        test_verification_records_capture_version_snapshot,
        test_new_edit_makes_old_verification_stale,
        test_weaker_success_does_not_hide_stronger_failure,
        test_completion_policy_is_deterministic,
    )
    for path_test in path_tests:
        run_with_temp(path_test)
    test_decode_arguments_rejects_non_object()
    test_command_risk_hints_are_explainable()
    print("25 offline checks passed")
