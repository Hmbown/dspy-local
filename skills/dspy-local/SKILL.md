---
name: dspy-local
description: Configure or troubleshoot dspy-local integrations that use local Codex runtimes as the language model backend. Use when a repo needs a `dspy.BaseLM` adapter for Codex, when DSPy should auto-route between `codex exec` and `codex mcp-server`, when the bundled `dspy-local` skill must be installed into `CODEX_HOME`, or when local Codex auth/model discovery needs to be verified before compile or eval runs.
---

# DSPy Local

## Overview

Use the repo-local `dspy.CodexLM` client instead of open-coding new Codex subprocess wrappers.
Prefer the shipped helper scripts and the existing runtime-selection logic before changing transport behavior.

## First Pass

1. From the repo root, run `uv run python scripts/dspy_local_codex_doctor.py`.
2. If the task is a live transport check, run `uv run python scripts/dspy_local_codex_probe.py --transport auto`.
3. If the task is a real DSPy smoke check, run `uv run python scripts/dspy_local_codex_smoke.py --transport auto`.
4. Read `skills/dspy-local/references/runtime-selection.md` before changing aliases, env vars, or fallback order.

## Python Integration

Use `dspy.CodexLM` directly:

```python
from pathlib import Path

import dspy

lm = dspy.CodexLM(
    model="codex/default",
    repo_root=Path.cwd(),
)
dspy.configure(lm=lm)
```

Model aliases:

- `codex/default` selects the preferred configured transport.
- `codex-exec/default` forces direct `codex exec`.
- `codex-mcp/default` forces the local `codex mcp-server` path.

## Skill Installation

Install the bundled skill into Codex:

```bash
uv run python scripts/install_dspy_local_skill.py
```

For a non-default home:

```bash
uv run python scripts/install_dspy_local_skill.py --codex-home /path/to/codex-home
```

## GEPA Optimization

GEPA (reflective prompt evolution) works with CodexLM out of the box. Use the
same CodexLM instance as both the student LM and the reflection LM:

```python
from pathlib import Path

import dspy
from dspy.teleprompt.gepa import GEPA

lm = dspy.CodexLM(model="codex/default", repo_root=Path.cwd())
dspy.configure(lm=lm)

def my_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(pred.answer.strip().lower() == gold.answer.strip().lower())
    return dspy.Prediction(score=score, feedback="Check the answer.")

optimizer = GEPA(metric=my_metric, reflection_lm=lm, max_metric_calls=5)
optimized = optimizer.compile(dspy.Predict("question -> answer"), trainset=trainset)
```

Or run the bundled example:

```bash
uv run python scripts/dspy_local_codex_gepa.py --json
```

## Runtime Rules

- Use `codex/default` unless there is a concrete reason to force a transport.
- Use `DSPY_CODEX_TRANSPORT=auto|cli|mcp` only with the generic `codex/...` alias.
- Keep the default sandbox at `read-only` unless the user explicitly wants Codex itself to write files.
- Expect `auto` to prefer MCP when the local MCP SDK is installed and to fall back to CLI once if the MCP call fails.
- Expect structured non-text content and native tool-calling requests to fail loudly until CodexLM grows real support for them.

## Bundled Scripts

- `scripts/dspy_local_codex_doctor.py`: run the repo-local environment check. Add `--live` to append a readiness probe.
- `scripts/dspy_local_codex_probe.py`: run a live `ready` probe through the selected transport.
- `scripts/dspy_local_codex_smoke.py`: run a minimal `dspy.Predict(...)` call through `dspy.CodexLM`.
- `scripts/dspy_local_codex_gepa.py`: run a minimal GEPA optimization with `dspy.CodexLM` as both student and reflection LM.
- `scripts/install_dspy_local_skill.py`: install the bundled skill into `CODEX_HOME/skills`.

## Guardrails

- Do not replace the built-in runtime-selection logic with ad hoc `subprocess` calls unless the shipped client is actually insufficient.
- Do not debug transport failures by immediately switching to `danger-full-access`; inspect the doctor output first.
- Do not assume the user-level `CODEX_HOME` config is safe for automation. The client intentionally isolates auth and model cache into a temporary runtime home.

## Verification

Use this checklist:

1. `uv sync --extra mcp --extra dev`
2. `uv run python scripts/dspy_local_codex_doctor.py --json`
3. `uv run python scripts/dspy_local_codex_probe.py --transport auto --json`
4. `uv run python scripts/dspy_local_codex_probe.py --transport cli --json`
5. `uv run python scripts/dspy_local_codex_probe.py --transport mcp --json`
6. `uv run python scripts/dspy_local_codex_smoke.py --transport auto --json`
7. `uv run python scripts/dspy_local_codex_smoke.py --transport cli --json`
8. `uv run python scripts/dspy_local_codex_smoke.py --transport mcp --json`
9. `uv run python scripts/dspy_local_codex_gepa.py --json`
10. `uv run python -m pytest tests/clients/test_codex.py`

## References

- `references/runtime-selection.md`
