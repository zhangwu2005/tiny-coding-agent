"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Agent, save_transcript
from .console import configure_utf8_stdio
from .provider import OpenAICompatibleClient, ProviderError
from .tools import ToolExecutor, assess_command_risk


def main() -> int:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="A small coding agent using an OpenAI-compatible model")
    parser.add_argument("task", nargs="?", help="the programming task")
    parser.add_argument("--workspace", default=".", help="workspace directory (default: current directory)")
    parser.add_argument("--max-steps", type=int, default=12, help="maximum model/tool rounds (default: 12)")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=60_000,
        help="maximum approximate context characters sent to the model (default: 60000)",
    )
    parser.add_argument(
        "--repeat-limit",
        type=int,
        default=3,
        help="stop after this many identical tool batches (default: 3)",
    )
    parser.add_argument("--auto-approve", action="store_true", help="approve run_command calls automatically")
    parser.add_argument(
        "--verification-policy",
        choices=("none", "syntax", "test", "full"),
        default="test",
        help="minimum evidence required by the controller (default: test)",
    )
    parser.add_argument(
        "--test-provenance-policy",
        choices=("allow", "warn", "independent"),
        default="warn",
        help=(
            "how controller treats tests changed by this agent run: allow, warn, "
            "or require independent evidence (default: warn)"
        ),
    )
    parser.add_argument(
        "--accept-incomplete",
        action="store_true",
        help="explicitly accept a completion proposal that does not satisfy the verification policy",
    )
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

    def approve_incomplete(review: dict[str, object]) -> bool:
        if args.accept_incomplete:
            print("[acceptance] incomplete result accepted by command-line user policy")
            return True
        evidence = ", ".join(str(item) for item in review["verification_evidence"]) or "none"
        eligible_evidence = (
            ", ".join(str(item) for item in review["eligible_verification_evidence"]) or "none"
        )
        print(
            "\n[user acceptance] The controller cannot approve this completion proposal.\n"
            f"  reason: {review['reason']}\n"
            f"  required policy: {review['verification_policy']}\n"
            f"  current status: {review['verification_status']}\n"
            f"  successful evidence: {evidence}\n"
            f"  provenance policy: {review['test_provenance_policy']}\n"
            f"  eligible evidence: {eligible_evidence}"
        )
        answer = input("Type 'accept' to accept it anyway; anything else rejects: ").strip().lower()
        return answer == "accept"

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
            print(
                f"\n[controller] completion rejected ({event_data['reason']}), "
                f"policy={event_data['policy']}, files={files}"
            )
        elif kind == "plan_required":
            unfinished = [
                f"{item['id']}={item['status']}"
                for item in event_data["task_plan"]
                if item["status"] != "completed"
            ]
            print(f"\n[controller] completion rejected; unfinished plan: {', '.join(unfinished)}")
        elif kind == "plan_updated":
            print("\n[plan]")
            for item in event_data["task_plan"]:
                print(f"  [{item['status']}] {item['id']}: {item['description']}")
        elif kind == "context_compacted":
            print(
                f"\n[context] compacted {event_data['original_chars']} -> "
                f"{event_data['sent_chars']} chars; omitted={event_data['omitted_messages']} messages"
            )
        elif kind == "reflection_required":
            print(f"\n[reflection] verification failed: {event_data['command']}")
        elif kind == "user_review_required":
            print("\n[controller] insufficient evidence; handing the final decision to the user")
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
    if args.context_limit < 8_000:
        parser.error("--context-limit must be at least 8000")
    executor = ToolExecutor(Path(args.workspace), approve_command=approve)
    try:
        result = Agent(
            client,
            executor,
            max_steps=args.max_steps,
            max_repeated_tool_batches=args.repeat_limit,
            max_context_chars=args.context_limit,
            verification_policy=args.verification_policy,
            test_provenance_policy=args.test_provenance_policy,
            approve_incomplete=approve_incomplete,
            on_event=event,
        ).run(task)
    except ProviderError as exc:
        parser.exit(1, f"Model request failed: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Check the API endpoint or use a compatible gateway, then try again.\n")
    test_risks = sorted(
        {
            str(record["test_provenance_risk"])
            for record in result.verification_records
            if record.get("test_provenance_risk") not in {None, "not_applicable"}
        }
    )
    role_counts: dict[str, int] = {}
    for role in result.changed_file_roles.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    rendered_roles = ",".join(
        f"{role}:{count}" for role, count in sorted(role_counts.items())
    ) or "none"
    print(
        f"\n[summary] stop_reason={result.stop_reason} "
        f"steps={result.steps} tool_calls={result.tool_calls} "
        f"verification={result.verification_status} "
        f"policy={result.verification_policy} "
        f"evidence={','.join(result.verification_evidence) or 'none'} "
        f"eligible_evidence={','.join(result.eligible_verification_evidence) or 'none'} "
        f"provenance_policy={result.test_provenance_policy} "
        f"records={len(result.verification_records)} "
        f"test_risk={','.join(test_risks) or 'none'} "
        f"file_roles={rendered_roles} "
        f"plan_items={len(result.task_plan)} "
        f"context_compactions={result.context_compactions} "
        f"changed_files={len(result.changed_files)}"
    )
    if (
        result.test_provenance_policy == "warn"
        and "elevated_agent_modified_tests" in test_risks
    ):
        print(
            "[warning] passing evidence includes tests changed by this agent run; "
            "review those tests or use --test-provenance-policy independent"
        )
    if args.transcript:
        save_transcript(args.transcript, result)
        print(f"[transcript] {args.transcript}")
    success_reasons = {
        "completed_no_changes",
        "completed_by_policy",
        "completed_verified",
        "user_accepted_incomplete",
    }
    return 0 if result.stop_reason in success_reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
