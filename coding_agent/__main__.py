"""Allow both ``python -m coding_agent`` and direct execution of this file."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    # Direct execution has no package context, so add the project root that
    # contains the coding_agent package before importing the CLI.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from coding_agent.cli import main
else:
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
