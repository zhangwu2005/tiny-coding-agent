"""Offline demonstrations for the controller-owned safety features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from .agent import Agent, CompletionPolicy
from .console import configure_utf8_stdio
from .context import ContextManager
from .planning import PlanError, TaskPlan
from .tools import ToolError, ToolExecutor


def _executor(directory: str | Path) -> ToolExecutor:
    return ToolExecutor(directory, approve_command=lambda _command: True)


INVENTORY_BUGGY = '''"""Inventory reservation example with intentional bugs."""


def reserve_stock(stock, requests):
    """Reserve requested quantities and return remaining stock."""
    for sku, quantity in requests:
        if sku not in stock or stock[sku] < quantity:
            raise ValueError("insufficient stock")
        stock[sku] -= quantity
    return stock
'''

INVENTORY_FIXED = '''"""Inventory reservation with validation and atomic updates."""


def reserve_stock(stock, requests):
    """Return remaining stock without mutating the caller's mapping."""
    totals = {}
    for sku, quantity in requests:
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        if sku not in stock:
            raise KeyError(sku)
        totals[sku] = totals.get(sku, 0) + quantity

    for sku, quantity in totals.items():
        if stock[sku] < quantity:
            raise ValueError(f"insufficient stock for {sku}")

    remaining = dict(stock)
    for sku, quantity in totals.items():
        remaining[sku] -= quantity
    return remaining
'''

INVENTORY_ACCEPTANCE_TESTS = '''import unittest

from inventory import reserve_stock


class TestReserveStock(unittest.TestCase):
    def test_aggregates_duplicates_without_mutating_input(self):
        stock = {"A": 5, "B": 3}
        result = reserve_stock(stock, [("A", 2), ("A", 1), ("B", 2)])
        self.assertEqual(result, {"A": 2, "B": 1})
        self.assertEqual(stock, {"A": 5, "B": 3})
        self.assertIsNot(result, stock)

    def test_failure_is_atomic(self):
        stock = {"A": 5, "B": 1}
        with self.assertRaises(ValueError):
            reserve_stock(stock, [("A", 2), ("B", 2)])
        self.assertEqual(stock, {"A": 5, "B": 1})

    def test_missing_sku_raises_key_error(self):
        stock = {"A": 5}
        with self.assertRaises(KeyError):
            reserve_stock(stock, [("B", 1)])
        self.assertEqual(stock, {"A": 5})

    def test_rejects_invalid_quantities(self):
        for quantity in [0, -1, 1.5, True]:
            with self.subTest(quantity=quantity):
                with self.assertRaises(ValueError):
                    reserve_stock({"A": 5}, [("A", quantity)])

    def test_empty_request_returns_a_copy(self):
        stock = {"A": 5}
        result = reserve_stock(stock, [])
        self.assertEqual(result, stock)
        self.assertIsNot(result, stock)
'''


class ScriptedInventoryClient:
    """Deterministic model substitute used to demonstrate the real controller loop offline."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }

    def complete(self, _messages, _tools):
        self.calls += 1
        plans = {
            "initial": [
                {"id": "inspect", "description": "复现失败并检查代码", "status": "in_progress"},
                {"id": "implement", "description": "实现原子库存预占", "status": "pending"},
                {"id": "verify", "description": "重跑独立验收测试", "status": "pending"},
            ],
            "implement": [
                {"id": "inspect", "description": "复现失败并检查代码", "status": "completed"},
                {"id": "implement", "description": "实现原子库存预占", "status": "in_progress"},
                {"id": "verify", "description": "重跑独立验收测试", "status": "pending"},
            ],
            "verify": [
                {"id": "inspect", "description": "复现失败并检查代码", "status": "completed"},
                {"id": "implement", "description": "实现原子库存预占", "status": "completed"},
                {"id": "verify", "description": "重跑独立验收测试", "status": "in_progress"},
            ],
            "complete": [
                {"id": "inspect", "description": "复现失败并检查代码", "status": "completed"},
                {"id": "implement", "description": "实现原子库存预占", "status": "completed"},
                {"id": "verify", "description": "重跑独立验收测试", "status": "completed"},
            ],
        }
        actions = {
            1: ("plan-1", "update_plan", {"items": plans["initial"]}),
            2: ("read-code", "read_file", {"path": "inventory.py"}),
            3: ("read-tests", "read_file", {"path": "test_inventory_acceptance.py"}),
            4: (
                "test-failing",
                "run_command",
                {"command": "python -m unittest test_inventory_acceptance"},
            ),
            5: ("plan-2", "update_plan", {"items": plans["implement"]}),
            6: (
                "fix-code",
                "replace_in_file",
                {"path": "inventory.py", "old_text": INVENTORY_BUGGY, "new_text": INVENTORY_FIXED},
            ),
            7: ("plan-3", "update_plan", {"items": plans["verify"]}),
            8: (
                "test-passing",
                "run_command",
                {"command": "python -m unittest test_inventory_acceptance"},
            ),
            9: ("plan-4", "update_plan", {"items": plans["complete"]}),
        }
        action = actions.get(self.calls)
        if action is None:
            return {
                "role": "assistant",
                "content": "库存预占逻辑已修复；5项独立验收测试通过。",
            }
        return self._call(*action)


def demo_verification() -> None:
    """Show shell-composition rejection and the zero-test rule."""
    print("\n=== 1. 单一验证命令保护 ===")
    commands = (
        "python -m unittest; echo forged",
        "python -m unittest && echo forged",
        "python -m unittest || exit 0",
        "python -m unittest > result.txt",
    )
    with TemporaryDirectory() as directory:
        executor = _executor(directory)
        for command in commands:
            print(f"测试命令：{command}")
            try:
                executor.run_command(command)
            except ToolError as exc:
                print(f"控制器已拒绝：{exc}")
            else:
                raise AssertionError("复合验证命令没有被阻止")

    print("\n=== 2. 零测试不能算成功 ===")
    with TemporaryDirectory() as directory:
        executor = _executor(directory)
        executor.write_file("test_empty.py", "# This file intentionally contains no tests.\n")
        print(executor.run_command("python -m unittest test_empty"))
        print("验证记录：")
        print(
            json.dumps(
                executor.verification_records[-1].to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"最终验证状态：{executor.verification_status}")


def demo_retry() -> None:
    """Show that only the same command can resolve its earlier failure."""
    print("\n=== 相同命令重跑规则 ===")
    with TemporaryDirectory() as directory:
        executor = _executor(directory)
        executor.write_file(
            "test_retry.py",
            """import unittest
from pathlib import Path


class TestRetry(unittest.TestCase):
    def test_retry(self):
        marker = Path("retry.marker")
        if not marker.exists():
            marker.write_text("ready", encoding="utf-8")
            self.fail("intentional first failure")
""",
        )
        targeted_command = "python -m unittest test_retry"

        executor.run_command(targeted_command)
        print(f"第一次运行原命令（故意失败）：{executor.verification_status}")

        executor.run_command("python -m unittest")
        print(f"不同命令运行成功后：{executor.verification_status}")

        executor.run_command(targeted_command)
        print(f"相同命令成功重跑后：{executor.verification_status}")
        print("命令退出码：", [record.exit_code for record in executor.verification_records])
        print(
            "命令指纹关系：",
            executor.verification_records[0].command_fingerprint
            == executor.verification_records[2].command_fingerprint,
        )


def demo_provenance() -> None:
    """Compare pre-existing tests with tests modified during this agent run."""
    print("\n=== 测试来源风险 ===")
    original_test = """import unittest
from calculator import add


class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
    updated_test = original_test + """
    def test_add_negative(self):
        self.assertEqual(add(-1, -2), -3)
"""
    with TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "test_calculator.py").write_text(original_test, encoding="utf-8")
        executor = _executor(workspace)
        executor.write_file("calculator.py", "def add(a, b):\n    return a + b\n")

        executor.run_command("python -m unittest test_calculator")
        first = executor.verification_records[-1]
        print(f"运行原有测试：{first.test_provenance_risk}")
        print(f"Agent 修改的测试文件：{first.agent_modified_test_files}")

        executor.read_file("test_calculator.py")
        executor.write_file("test_calculator.py", updated_test)
        executor.run_command("python -m unittest test_calculator")
        second = executor.verification_records[-1]
        print(f"Agent 修改测试后：{second.test_provenance_risk}")
        print(f"Agent 修改的测试文件：{second.agent_modified_test_files}")
        print(f"收集到的测试数量：{first.tests_collected} -> {second.tests_collected}")
        print(f"文件角色：{executor.file_roles}")
        warn_decision = CompletionPolicy("test", "warn").evaluate(executor)
        independent_decision = CompletionPolicy("test", "independent").evaluate(executor)
        print(f"warn 策略允许结束：{warn_decision.accepted}")
        print(
            "independent 策略允许结束："
            f"{independent_decision.accepted} ({independent_decision.reason})"
        )


def demo_stale() -> None:
    """Show that an edit invalidates evidence for the previous workspace version."""
    print("\n=== 修改后旧验证失效 ===")
    with TemporaryDirectory() as directory:
        executor = _executor(directory)
        executor.write_file("calculator.py", "def add(a, b):\n    return a + b\n")
        executor.write_file(
            "test_calculator.py",
            """import unittest
from calculator import add


class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
""",
        )
        executor.run_command("python -m unittest")
        print(f"完整测试后：version={executor.change_version}, status={executor.verification_status}")
        print(f"验证证据：{executor.verification_evidence}")

        executor.write_file("new_feature.py", "value = 42\n")
        print(f"再次修改后：version={executor.change_version}, status={executor.verification_status}")
        print(f"验证证据：{executor.verification_evidence}")


def demo_planning() -> None:
    """Show controller validation of model-proposed task-plan transitions."""
    print("\n=== 结构化 To-do list ===")
    plan = TaskPlan()
    initial = [
        {"id": "inspect", "description": "检查现有代码", "status": "in_progress"},
        {"id": "implement", "description": "实现功能", "status": "pending"},
        {"id": "verify", "description": "运行测试", "status": "pending"},
    ]
    print(plan.apply_proposal(initial))

    updated = [
        {"id": "inspect", "description": "检查现有代码", "status": "completed"},
        {"id": "implement", "description": "实现功能", "status": "in_progress"},
        {"id": "verify", "description": "运行测试", "status": "pending"},
    ]
    try:
        plan.apply_proposal(updated)
    except PlanError as exc:
        print(f"没有工具证据时，控制器拒绝：{exc}")

    plan.record_action("read_file", "calculator.py: lines 1-20")
    print("加入真实工具证据后：")
    print(plan.apply_proposal(updated))
    print(json.dumps(plan.snapshot(), ensure_ascii=False, indent=2))


def demo_context() -> None:
    """Show bounded model context while retaining authoritative state."""
    print("\n=== 对话历史与上下文压缩 ===")
    history: list[dict[str, object]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "给 calculator.py 增加 divide。"},
    ]
    for number in range(20):
        history.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{number}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{number}",
                    "content": "x" * 700,
                },
            ]
        )
    history.append({"role": "assistant", "content": "最新结论必须被保留"})

    manager = ContextManager(max_chars=8_000)
    compacted, metadata = manager.prepare(
        history,
        {
            "task_plan": [
                {
                    "id": "implement",
                    "description": "实现 divide",
                    "status": "in_progress",
                    "evidence": [],
                }
            ],
            "change_version": 2,
            "changed_files": ["calculator.py"],
            "file_versions": {"calculator.py": 2},
            "verification_status": "unverified",
            "verification_evidence": [],
        },
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"原始历史消息数：{len(history)}")
    print(f"压缩后消息数：{len(compacted)}")
    print(f"原始任务是否保留：{compacted[1]['content']}")
    print(f"结构化快照是否存在：{'Controller context snapshot' in str(compacted[2]['content'])}")
    print(f"最新结论是否保留：{compacted[-1]['content']}")
    print(f"压缩后字符数：{len(json.dumps(compacted, ensure_ascii=False))}")


def demo_inventory_agent() -> None:
    """Run a medium-difficulty repair through the complete offline Agent loop."""
    print("\n=== 案例：库存原子预占 ===")
    print(
        "控制关系：大模型提出行动建议；Agent协调行动；TaskPlan校验计划；"
        "ToolExecutor执行文件与测试；"
        "CompletionPolicy裁决完成"
    )
    print(
        "业务验收规则：合并重复SKU、校验数量、失败原子性、不修改输入"
    )

    with TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "inventory.py").write_text(INVENTORY_BUGGY, encoding="utf-8")
        (workspace / "test_inventory_acceptance.py").write_text(
            INVENTORY_ACCEPTANCE_TESTS,
            encoding="utf-8",
        )
        original_test_revision = (workspace / "test_inventory_acceptance.py").read_bytes()

        def event(event_data: dict[str, object]) -> None:
            kind = event_data.get("type")
            if kind == "plan_updated":
                rendered = " | ".join(
                    f"{item['id']}={item['status']}"
                    for item in event_data["task_plan"]  # type: ignore[index]
                )
                print(f"[计划已接受] {rendered}")
            elif kind == "tool_call":
                name = str(event_data["name"])
                arguments = event_data["arguments"]  # type: ignore[index]
                if name == "replace_in_file":
                    arguments = {"path": arguments["path"]}  # type: ignore[index]
                print(f"[工具建议] {name} {json.dumps(arguments, ensure_ascii=False)}")
            elif kind == "tool_result":
                name = str(event_data["name"])
                result = str(event_data["result"])
                if name == "update_plan":
                    return
                if name == "run_command":
                    prefixes = (
                        "verification=",
                        "verification_type=",
                        "tests_collected=",
                        "test_provenance_risk=",
                        "agent_modified_test_files=",
                        "failure_reason=",
                        "change_version=",
                        "changed_file_roles=",
                        "exit_code=",
                    )
                    selected = [
                        line
                        for line in result.splitlines()
                        if line.startswith(prefixes) or line == "OK" or line.startswith("FAILED")
                    ]
                    print("[测试结果]\n" + "\n".join(selected))
                else:
                    first_line = next(
                        (line for line in result.splitlines() if line.strip()),
                        "(empty result)",
                    )
                    print(f"[工具结果] {first_line}")
            elif kind == "reflection_required":
                print(f"[失败反思] 相同命令必须修复后重跑：{event_data['command']}")
            elif kind == "final":
                print(f"[模型完成建议] {event_data['answer']}")

        executor = _executor(workspace)
        result = Agent(
            ScriptedInventoryClient(),
            executor,
            max_steps=12,
            verification_policy="test",
            test_provenance_policy="independent",
            on_event=event,
        ).run(
            "修复 inventory.py 的 reserve_stock；先复现失败，只修改业务代码，"
            "最后用用户预先提供的 test_inventory_acceptance.py 验收。"
        )
        records = executor.verification_records
        same_command_retried = (
            len(records) == 2
            and records[0].command_fingerprint == records[1].command_fingerprint
            and not records[0].passed
            and records[1].passed
        )
        test_file_unchanged = (
            workspace / "test_inventory_acceptance.py"
        ).read_bytes() == original_test_revision

        print("\n=== CompletionPolicy最终裁决 ===")
        print(f"stop_reason={result.stop_reason}")
        print(f"verification_status={result.verification_status}")
        print(f"verification_evidence={result.verification_evidence}")
        print(f"eligible_evidence={result.eligible_verification_evidence}")
        print(f"test_provenance_policy={result.test_provenance_policy}")
        print(f"changed_files={result.changed_files}")
        print(f"changed_file_roles={result.changed_file_roles}")
        print(f"same_failed_command_retried={same_command_retried}")
        print(f"acceptance_test_unchanged={test_file_unchanged}")
        print(f"task_plan_complete={all(item['status'] == 'completed' for item in result.task_plan)}")

        if not (
            result.stop_reason == "completed_verified"
            and result.eligible_verification_evidence == ["targeted_test"]
            and result.changed_file_roles == {"inventory.py": "implementation"}
            and same_command_retried
            and test_file_unchanged
        ):
            raise AssertionError("inventory demonstration did not satisfy its controller invariants")


DEMOS: dict[str, Callable[[], None]] = {
    "inventory": demo_inventory_agent,
    "verification": demo_verification,
    "retry": demo_retry,
    "provenance": demo_provenance,
    "stale": demo_stale,
    "planning": demo_planning,
    "context": demo_context,
}


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run offline Tiny Coding Agent demonstrations")
    parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=("all", *DEMOS),
        help="demonstration to run (default: all)",
    )
    args = parser.parse_args(argv)
    selected = DEMOS.values() if args.demo == "all" else (DEMOS[args.demo],)
    for demonstration in selected:
        demonstration()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
