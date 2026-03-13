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
    from dspy.clients.claude import inspect_claude_runtime, probe_claude_runtime
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/dspy_claude_doctor.py`."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local Claude CLI runtime readiness.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="claude/default")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = inspect_claude_runtime()
    payload = {"runtime": runtime.as_dict()}
    if args.live:
        payload["probe"] = probe_claude_runtime(
            repo_root=Path(args.repo_root).resolve(),
            model=args.model,
            timeout_seconds=args.timeout_seconds,
        ).as_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Claude CLI: {runtime.cli_path or 'missing'}")
        print(f"Claude home: {runtime.claude_home}")
        print(f"Credentials configured: {runtime.credentials_configured}")
        if runtime.credential_sources:
            print(f"Credential sources: {', '.join(runtime.credential_sources)}")
        if runtime.settings_file:
            print(f"Settings file: {runtime.settings_file}")
        if args.live:
            print(f"Probe model: {payload['probe'].get('resolved_model') or 'unknown'}")
            print(f"Probe answer: {payload['probe'].get('content')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
