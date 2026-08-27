"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import Agent, save_transcript
from .provider import OpenAICompatibleClient
from .tools import ToolExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description="A small coding agent using an OpenAI-compatible model")
    parser.add_argument("task", nargs="?", help="the programming task")
    parser.add_argument("--workspace", default=".", help="workspace directory (default: current directory)")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--auto-approve", action="store_true", help="approve run_command calls automatically")
    parser.add_argument("--transcript", help="write the conversation as JSONL")
    args = parser.parse_args()
    task = args.task or input("Task: ").strip()

    def approve(command: str) -> bool:
        if args.auto_approve:
            print(f"[approval] auto-approved: {command}")
            return True
        answer = input(f"\nAgent requests command:\n  {command}\nApprove? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def event(event_data: dict[str, object]) -> None:
        kind = event_data.get("type")
        if kind == "tool_call":
            print(f"\n[tool] {event_data['name']} {json.dumps(event_data['arguments'], ensure_ascii=False)}")
        elif kind == "tool_result":
            print(f"[result]\n{event_data['result']}\n")
        elif kind == "final":
            print(f"\n[final]\n{event_data['answer']}")

    try:
        client = OpenAICompatibleClient(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
    except ValueError as exc:
        parser.error(str(exc))
    executor = ToolExecutor(Path(args.workspace), approve_command=approve)
    result = Agent(client, executor, max_steps=args.max_steps, on_event=event).run(task)
    if args.transcript:
        save_transcript(args.transcript, result)
        print(f"[transcript] {args.transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
