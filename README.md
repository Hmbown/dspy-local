# dspy-local

Local CLI runtime adapters for [DSPy](https://dspy.ai/). Run DSPy programs through Codex, Qwen, and Claude command-line tools.

| Backend | CLI | GEPA |
|---|---|---|
| `dspy.CodexLM` | `codex exec` / `codex mcp-server` | Yes |
| `dspy.QwenLM` | `qwen` | Yes |
| `dspy.ClaudeLM` | `claude -p --output-format json` | Yes |

## Setup

```bash
git clone https://github.com/Hmbown/dspy-local.git && cd dspy-local
uv sync --extra mcp --extra dev
```

## Usage

```python
from pathlib import Path
import dspy

lm = dspy.CodexLM(model="codex/default", repo_root=Path.cwd())
# lm = dspy.QwenLM(model="qwen/default", repo_root=Path.cwd())
# lm = dspy.ClaudeLM(model="claude/sonnet", repo_root=Path.cwd())
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
print(predict(question="What is the capital of France?").answer)
```

## GEPA

All three backends work with [GEPA](https://arxiv.org/abs/2507.19457):

```bash
uv run python scripts/dspy_claude_gepa.py --model claude/sonnet --max-metric-calls 3 --json
uv run python scripts/dspy_local_codex_gepa.py --json
uv run python scripts/dspy_qwen_gepa.py --json
```

## Validation

```bash
uv run python scripts/dspy_local_codex_doctor.py --json   # check Codex
uv run python scripts/dspy_claude_doctor.py --json         # check Claude
uv run python scripts/dspy_qwen_doctor.py --json           # check Qwen
uv run python -m pytest tests/clients/ -q                  # run tests
```

## How It Works

Each backend wraps a local CLI binary and implements `BaseLM.forward()`. Config isolation copies auth files into a temp directory so DSPy runs stay reproducible. Unsupported features (tool calling, structured content, sampling controls) raise explicit errors.

---

Built on [DSPy](https://dspy.ai/) by Stanford NLP. See [GEPA paper](https://arxiv.org/abs/2507.19457), [DSPy paper](https://arxiv.org/abs/2310.03714).
