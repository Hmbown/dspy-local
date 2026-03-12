#!/usr/bin/env python3
"""Minimal GEPA optimization using dspy.CodexLM.

Demonstrates DSPy's GEPA optimizer (reflective prompt evolution) running
end-to-end on a local Codex runtime.  Both the student module and the
reflection LM use CodexLM so nothing leaves the machine.

Usage:
    uv run python scripts/dspy_codex_gepa.py --json
    uv run python scripts/dspy_codex_gepa.py --transport cli --json
    uv run python scripts/dspy_codex_gepa.py --max-metric-calls 10 --json
"""
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
    from dspy.teleprompt.gepa import GEPA
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script "
        "with `uv run python scripts/dspy_codex_gepa.py`."
    ) from exc


TRAINSET = [
    dspy.Example(
        question="Translate to French: 'The cat sat on the mat.'",
        answer="Le chat s'est assis sur le tapis.",
    ).with_inputs("question"),
    dspy.Example(
        question="Translate to French: 'Good morning, how are you?'",
        answer="Bonjour, comment allez-vous ?",
    ).with_inputs("question"),
    dspy.Example(
        question="Translate to French: 'The weather is nice today.'",
        answer="Le temps est beau aujourd'hui.",
    ).with_inputs("question"),
    dspy.Example(
        question="Translate to French: 'I would like a coffee, please.'",
        answer="Je voudrais un café, s'il vous plaît.",
    ).with_inputs("question"),
]

VALSET = [
    dspy.Example(
        question="Translate to French: 'Where is the nearest train station?'",
        answer="Où est la gare la plus proche ?",
    ).with_inputs("question"),
    dspy.Example(
        question="Translate to French: 'She reads a book every evening.'",
        answer="Elle lit un livre chaque soir.",
    ).with_inputs("question"),
]


def translation_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
) -> dspy.Prediction:
    """Score translation quality and provide feedback for GEPA reflection."""
    predicted = (pred.answer or "").strip().lower()
    expected = (gold.answer or "").strip().lower()

    if predicted == expected:
        return dspy.Prediction(score=1.0, feedback="Perfect translation.")

    # Partial credit: check key content words overlap
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    if expected_words and predicted_words:
        overlap = len(expected_words & predicted_words) / len(expected_words)
    else:
        overlap = 0.0

    score = round(min(0.9, overlap), 2)
    feedback = (
        f"Translation is {'partially' if score > 0.3 else 'not'} correct. "
        f"Expected: '{gold.answer}' but got: '{pred.answer}'. "
        f"Word overlap: {score:.0%}."
    )
    return dspy.Prediction(score=score, feedback=feedback)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal GEPA optimization using dspy.CodexLM.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="codex/default")
    parser.add_argument("--transport", choices=["auto", "cli", "mcp"], default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=5,
        help="GEPA budget: total metric evaluations (default: 5, very small).",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    lm = dspy.CodexLM(
        model=args.model,
        repo_root=Path(args.repo_root).resolve(),
        transport=args.transport,
        timeout_seconds=args.timeout_seconds,
    )
    dspy.configure(lm=lm)

    student = dspy.Predict("question -> answer")

    optimizer = GEPA(
        metric=translation_metric,
        reflection_lm=lm,
        max_metric_calls=args.max_metric_calls,
        num_threads=1,
    )

    with dspy.track_usage() as tracker:
        optimized = optimizer.compile(student, trainset=TRAINSET, valset=VALSET)

    # Run the optimized program on validation examples
    results = []
    for example in VALSET:
        pred = optimized(question=example.question)
        score = translation_metric(example, pred)
        results.append({
            "question": example.question,
            "expected": example.answer,
            "predicted": pred.answer,
            "score": score.score,
        })

    payload = {
        "optimizer": "GEPA",
        "max_metric_calls": args.max_metric_calls,
        "model": args.model,
        "transport": args.transport,
        "val_results": results,
        "avg_score": round(sum(r["score"] for r in results) / len(results), 3),
        "usage": {
            model: [dict(u) for u in usages]
            for model, usages in tracker.usage_data.items()
        } if tracker.usage_data else {},
        "history_len": len(lm.history),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Optimizer: GEPA (max_metric_calls={args.max_metric_calls})")
        print(f"Model: {args.model}")
        for r in results:
            print(f"  Q: {r['question']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Got:      {r['predicted']}")
            print(f"  Score:    {r['score']}")
            print()
        print(f"Avg score: {payload['avg_score']}")
        print(f"LM calls:  {payload['history_len']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
