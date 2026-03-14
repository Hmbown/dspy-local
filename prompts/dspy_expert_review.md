You are a DSPy expert reviewing a fork of DSPy that adds local CLI runtime support.

Your job is to determine two things:

1. Does this repo actually function as a first-class DSPy fork with `dspy.CodexLM`, `dspy.QwenLM`, and `dspy.ClaudeLM`?
2. Is the setup and validation flow simple enough for a new user to follow from a fresh clone?

Repository goals:

- The repo is a fork of DSPy, not a sidecar package.
- It adds three `dspy.BaseLM` backends that route through locally-installed CLI tools:
  - `dspy.CodexLM` — `codex exec` and `codex mcp-server` (CLI + MCP transports, auto-fallback)
  - `dspy.QwenLM` — `qwen` CLI
  - `dspy.ClaudeLM` — `claude -p --output-format json`
- All three backends are compatible with GEPA, LabeledFewShot, Evaluate, and BootstrapFewShot.
- The repo ships doctor, probe, smoke, and GEPA scripts for each backend, plus a Codex skill installer.

What to review:

- API fit with DSPy:
  - Do all three backends behave like valid `dspy.BaseLM` implementations?
  - Do they work with normal DSPy flows like `dspy.configure(...)`, `dspy.Predict(...)`, and history inspection?
  - Is the response shape correct for DSPy internals?
  - Are there hidden incompatibilities with common DSPy modules or adapters?
- Transport behavior:
  - Is CodexLM transport resolution coherent and predictable?
  - Does `auto` prefer MCP and fall back to CLI safely?
  - Are model aliases and environment overrides reasonable?
  - Are auth/model discovery choices sound across all three backends?
- Config isolation:
  - Does each backend correctly isolate user config into a temporary runtime home?
  - Could interactive hooks or local side effects leak into automated DSPy runs?
- Setup and onboarding:
  - From a fresh clone, are the required steps obvious and minimal?
  - Do the README and skill instructions match what actually works?
  - Are there confusing edge cases for Python environment setup, auth, or MCP availability?
  - Is there a better default setup or smoke-test flow?
- Code quality:
  - Look for behavioral bugs, bad assumptions, weak error handling, and missing verification.
  - Prioritize real breakage or onboarding friction over style comments.

Start by reading at least these files:

- `README.md`
- `pyproject.toml`
- `dspy/clients/codex.py`
- `dspy/clients/claude.py`
- `dspy/clients/qwen.py`
- `dspy/clients/__init__.py`
- `scripts/dspy_local_codex_doctor.py`
- `scripts/dspy_local_codex_smoke.py`
- `scripts/dspy_claude_doctor.py`
- `scripts/dspy_claude_smoke.py`
- `scripts/install_dspy_local_skill.py`
- `skills/dspy-local/SKILL.md`
- `tests/clients/test_codex.py`
- `tests/clients/test_claude.py`

If your environment allows it, run these commands:

```bash
uv sync --extra mcp --extra dev
uv run python scripts/dspy_local_codex_doctor.py --json
uv run python scripts/dspy_local_codex_smoke.py --transport auto --json
uv run python scripts/dspy_claude_doctor.py --json
uv run python scripts/dspy_claude_smoke.py --json
uv run python -m pytest tests/clients/test_codex.py tests/clients/test_claude.py
```

If the CLIs are installed and authenticated, also verify a real DSPy path:

```python
from pathlib import Path
import dspy

class Answer(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# Test with any available backend
lm = dspy.ClaudeLM(model="claude/sonnet", repo_root=Path.cwd())
dspy.configure(lm=lm)
predict = dspy.Predict(Answer)
result = predict(question="Reply with exactly one word: ready")
print(result.answer)
```

Deliverables:

- First, list findings ordered by severity.
- For each finding, include:
  - severity: critical | high | medium | low
  - file path
  - why it matters for DSPy correctness or user setup
  - the concrete fix you recommend
- Then provide:
  - a short verdict on whether the integration is production-usable
  - a short verdict on whether setup is easy enough for a new user
  - the top 3 changes that would most improve confidence and usability

Important review rules:

- Do not spend time on cosmetic nits unless they affect setup clarity or API trust.
- Treat onboarding friction as important if it would realistically stop a new user.
- Prefer direct, technical judgment over generic praise.
- If something seems correct but unproven, say exactly what still needs validation.
