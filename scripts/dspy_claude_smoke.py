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
    import dspy
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/dspy_claude_smoke.py`."
    ) from exc


class SmokeSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal DSPy Predict call through dspy.ClaudeLM."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="claude/default")
    parser.add_argument("--question", default="Reply with exactly one word: ready")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lm = dspy.ClaudeLM(
        model=args.model,
        repo_root=Path(args.repo_root).resolve(),
        timeout_seconds=args.timeout_seconds,
    )
    dspy.configure(lm=lm)
    predict = dspy.Predict(SmokeSignature)
    result = predict(question=args.question)

    payload = {
        "prediction_type": type(result).__name__,
        "answer": result.answer,
        "transport": None,
        "response_model": None,
        "session_id": None,
        "stderr": None,
        "usage": None,
        "cost_usd": None,
    }

    last_entry = lm.history[-1] if lm.history else {}
    response = last_entry.get("response") if last_entry else None
    hidden = getattr(response, "_hidden_params", {}) if last_entry else {}
    payload["transport"] = hidden.get("transport")
    payload["response_model"] = last_entry.get("response_model") if last_entry else None
    payload["session_id"] = hidden.get("session_id")
    payload["stderr"] = hidden.get("stderr")
    payload["usage"] = last_entry.get("usage") if last_entry else None
    payload["cost_usd"] = hidden.get("cost_usd")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"prediction_type={payload['prediction_type']}")
        print(f"answer={payload['answer']}")
        print(f"transport={payload['transport']}")
        if payload["response_model"]:
            print(f"response_model={payload['response_model']}")
        if payload["session_id"]:
            print(f"session_id={payload['session_id']}")
        if payload["usage"]:
            print(f"usage={json.dumps(payload['usage'], sort_keys=True)}")
        if payload["cost_usd"] is not None:
            print(f"cost_usd={payload['cost_usd']}")
        if payload["stderr"]:
            print(f"stderr={payload['stderr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
