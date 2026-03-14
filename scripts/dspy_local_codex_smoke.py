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
        "`uv run python scripts/dspy_local_codex_smoke.py`."
    ) from exc


class SmokeSignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal DSPy Predict call through dspy.CodexLM."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="codex/default")
    parser.add_argument("--transport", choices=["auto", "cli", "mcp"], default=None)
    parser.add_argument("--question", default="Reply with exactly one word: ready")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lm = dspy.CodexLM(
        model=args.model,
        repo_root=Path(args.repo_root).resolve(),
        transport=args.transport,
    )
    dspy.configure(lm=lm)
    predict = dspy.Predict(SmokeSignature)
    result = predict(
        question=args.question,
        config={"timeout_seconds": args.timeout_seconds},
    )

    payload = {
        "prediction_type": type(result).__name__,
        "answer": result.answer,
        "transport": None,
        "response_model": None,
        "thread_id": None,
        "fallback_from": None,
        "stderr": None,
        "usage": None,
    }

    last_entry = lm.history[-1] if lm.history else {}
    response = last_entry.get("response") if last_entry else None
    hidden = getattr(response, "_hidden_params", {}) if last_entry else {}
    payload["transport"] = hidden.get("transport")
    payload["response_model"] = last_entry.get("response_model") if last_entry else None
    payload["thread_id"] = hidden.get("thread_id")
    payload["fallback_from"] = hidden.get("fallback_from")
    payload["stderr"] = hidden.get("stderr")
    payload["usage"] = last_entry.get("usage") if last_entry else None

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"prediction_type={payload['prediction_type']}")
        print(f"answer={payload['answer']}")
        print(f"transport={payload['transport']}")
        if payload["response_model"]:
            print(f"response_model={payload['response_model']}")
        if payload["thread_id"]:
            print(f"thread_id={payload['thread_id']}")
        if payload["fallback_from"]:
            print(f"fallback_from={payload['fallback_from']}")
        if payload["usage"]:
            print(f"usage={json.dumps(payload['usage'], sort_keys=True)}")
        if payload["stderr"]:
            print(f"stderr={payload['stderr']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
