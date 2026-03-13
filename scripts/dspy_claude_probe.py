#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dspy.clients.claude import probe_claude_runtime
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/dspy_claude_probe.py`."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Claude CLI probe.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="claude/default")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = probe_claude_runtime(
        repo_root=Path(args.repo_root).resolve(),
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.content)
        print(f"transport={result.transport}")
        print(f"resolved_model={result.resolved_model or 'unknown'}")
        if result.session_id:
            print(f"session_id={result.session_id}")
        if result.usage:
            print(f"usage={json.dumps(result.usage, sort_keys=True)}")
        if result.cost_usd is not None:
            print(f"cost_usd={result.cost_usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
