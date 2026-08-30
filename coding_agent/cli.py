"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Agent, save_transcript
from .provider import OpenAICompatibleClient, ProviderError
from .tools import ToolExecutor, assess_command_risk


def main() -> int:
    parser = argparse.ArgumentParser(description="A small coding agent using an OpenAI-compatible model")
    parser.add_argument("task", nargs="?", help="the programming task")
    parser.add_argument("--workspace", default=".", help="workspace directory (default: current directory)")
    parser.add_argument("--max-steps", type=int, default=12, help="maximum model/tool rounds (default: 12)")
    parser.add_argument(
        "--repeat-limit",
        type=int,
        default=3,
        help="stop after this many identical tool batches (default: 3)",
    )
    parser.add_argument("--auto-approve", action="store_true", help="approve run_command calls automatically")
    parser.add_argument("--transcript", help="write the conversation as JSONL")
    args = parser.parse_args()
    task = args.task or input("Task: ").strip()

    def approve(command: str) -> bool:
        risk_level, risk_reason = assess_command_risk(command)
        print(f"[risk] {risk_level}: {risk_reason}")
        if args.auto_approve and risk_level != "high":
            print(f"[approval] auto-approved ({risk_level} risk): {command}")
            return True
        if args.auto_approve:
            print("[approval] high-risk commands still require explicit confirmation")
        if risk_level == "high":
            answer = input(
                f"\nAgent requests HIGH-RISK command:\n  {command}\nType 'approve' to continue: "
            ).strip().lower()
            return answer == "approve"
        answer = input(f"\nAgent requests command:\n  {command}\nApprove? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def event(event_data: dict[str, object]) -> None:
        kind = event_data.get("type")
        if kind == "tool_call":
            print(f"\n[tool] {event_data['name']} {json.dumps(event_data['arguments'], ensure_ascii=False)}")
        elif kind == "model_request":
            print(f"\n[model] requesting completion (step {event_data['step']})...")
        elif kind == "tool_result":
            print(f"[result]\n{event_data['result']}\n")
        elif kind == "verification_required":
            files = ", ".join(str(path) for path in event_data["changed_files"])
            print(f"\n[self-check] verification requested for: {files}")
        elif kind == "reflection_required":
            print(f"\n[reflection] verification failed: {event_data['command']}")
        elif kind == "stopped":
            print(f"\n[guard] stopped: {event_data['reason']}")
        elif kind == "final":
            print(f"\n[final]\n{event_data['answer']}")

    try:
        client = OpenAICompatibleClient.from_environment()
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.repeat_limit < 2:
        parser.error("--repeat-limit must be at least 2")
    executor = ToolExecutor(Path(args.workspace), approve_command=approve)
    try:
        result = Agent(
            client,
            executor,
            max_steps=args.max_steps,
            max_repeated_tool_batches=args.repeat_limit,
            on_event=event,
        ).run(task)
    except ProviderError as exc:
        parser.exit(1, f"Model request failed: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Check the API endpoint or use a compatible gateway, then try again.\n")
    print(
        f"\n[summary] stop_reason={result.stop_reason} "
        f"steps={result.steps} tool_calls={result.tool_calls} "
        f"verification={result.verification_status} "
        f"changed_files={len(result.changed_files)}"
    )
    if args.transcript:
        save_transcript(args.transcript, result)
        print(f"[transcript] {args.transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
