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
    from dspy.clients.codex import inspect_codex_runtime, probe_codex_runtime
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/dspy_local_codex_doctor.py`."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local Codex runtime availability for DSPy.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="codex/default")
    parser.add_argument("--transport", choices=["auto", "cli", "mcp"], default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run a live readiness probe after the static inspection.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    runtime = inspect_codex_runtime(model=args.model, transport=args.transport)
    probe = None
    if args.live:
        probe = probe_codex_runtime(
            repo_root=Path(args.repo_root).resolve(),
            model=args.model,
            transport=args.transport,
            timeout_seconds=args.timeout_seconds,
        )

    if args.json:
        payload = {
            "mode": "static+live" if probe else "static",
            "static_only": probe is None,
            "runtime": runtime.as_dict(),
        }
        if probe is not None:
            payload["probe"] = probe.as_dict()
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Repo root: {REPO_ROOT}")
    print("Mode: static inspection only" if probe is None else "Mode: static inspection plus live probe")
    print(f"Requested transport: {runtime.requested_transport}")
    print(f"Preferred transport: {runtime.preferred_transport}")
    print(f"Available transports: {', '.join(runtime.available_transports) or 'none'}")
    print(f"Codex CLI: {runtime.cli_path or 'missing'}")
    print(f"MCP SDK available: {'yes' if runtime.mcp_sdk_available else 'no'}")
    print(f"Credential source: {runtime.credential_source or 'missing'}")
    print(f"CODEX_HOME: {runtime.codex_home}")
    print(f"Default model: {runtime.default_model or 'unset'}")
    if runtime.available_models:
        preview = ", ".join(runtime.available_models[:6])
        print(f"Known models: {preview}")
    if probe is not None:
        print(f"Probe content: {probe.content}")
        print(f"Probe transport: {probe.transport}")
        print(f"Probe resolved model: {probe.resolved_model or 'unknown'}")
        if probe.thread_id:
            print(f"Probe thread_id: {probe.thread_id}")
        if probe.fallback_from:
            print(f"Probe fallback_from: {probe.fallback_from}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
