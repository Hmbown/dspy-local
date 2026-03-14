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
    from dspy.clients.codex import probe_codex_runtime
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/dspy_local_codex_probe.py`."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Codex probe through the selected transport.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="codex/default")
    parser.add_argument("--transport", choices=["auto", "cli", "mcp"], default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = probe_codex_runtime(
        repo_root=Path(args.repo_root).resolve(),
        model=args.model,
        transport=args.transport,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.content)
        print(f"transport={result.transport}")
        print(f"resolved_model={result.resolved_model or 'unknown'}")
        if result.thread_id:
            print(f"thread_id={result.thread_id}")
        if result.fallback_from:
            print(f"fallback_from={result.fallback_from}")
        if result.usage:
            print(f"usage={json.dumps(result.usage, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
