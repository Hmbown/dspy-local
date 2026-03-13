## dspy-codex: DSPy With Local LM Runtimes

This fork keeps the `dspy-codex` repo name, but it now ships three repo-local
DSPy backends for local CLI runtimes:

- `dspy.CodexLM` for `codex exec` and `codex mcp-server`
- `dspy.QwenLM` for the `qwen` CLI
- `dspy.ClaudeLM` for `claude -p --output-format json`

`pip install dspy` does not include these fork-specific clients. Use this repo
checkout and its `uv` environment.

## Prerequisites

1. Install the Python environment from this repo root:

```bash
uv sync --extra mcp --extra dev
```

2. Install and authenticate whichever local CLI you want to use:

- Codex: `codex --help` works and `~/.codex/auth.json` exists or `OPENAI_API_KEY` is set.
- Qwen: `qwen --help` works and your `~/.qwen` credentials are configured.
- Claude: `claude --help` works and `~/.claude/credentials.json` exists or `ANTHROPIC_API_KEY` is set.

## Fresh-Clone Quickstart

```bash
git clone <this-fork>
cd dspy-codex
uv sync --extra mcp --extra dev
```

Run the backend you want to validate:

```bash
uv run python scripts/dspy_codex_doctor.py --json
uv run python scripts/dspy_codex_probe.py --transport auto --json
uv run python scripts/dspy_codex_smoke.py --transport auto --json
```

```bash
uv run python scripts/dspy_qwen_doctor.py --json
uv run python scripts/dspy_qwen_probe.py --json
uv run python scripts/dspy_qwen_smoke.py --json
```

```bash
uv run python scripts/dspy_claude_doctor.py --json
uv run python scripts/dspy_claude_probe.py --json
uv run python scripts/dspy_claude_smoke.py --json
```

## Model Aliases

Codex aliases:

- `codex/default`: prefer MCP when available and fall back once to CLI if MCP fails.
- `codex-exec/default`: force direct `codex exec`.
- `codex-mcp/default`: force `codex mcp-server`.

Optional environment override for the generic Codex alias:

```bash
export DSPY_CODEX_TRANSPORT=auto  # auto | cli | mcp
```

Qwen aliases:

- `qwen/default`
- `qwen/qwen-max`

Claude aliases:

- `claude/default`
- `claude/sonnet`
- `claude/opus`
- `claude/claude-sonnet-4-6`

## Quick Examples

Codex:

```python
from pathlib import Path

import dspy

lm = dspy.CodexLM(model="codex/default", repo_root=Path.cwd())
dspy.configure(lm=lm)
```

Qwen:

```python
from pathlib import Path

import dspy

lm = dspy.QwenLM(model="qwen/default", repo_root=Path.cwd())
dspy.configure(lm=lm)
```

Claude:

```python
from pathlib import Path

import dspy

lm = dspy.ClaudeLM(model="claude/sonnet", repo_root=Path.cwd())
dspy.configure(lm=lm)
```

Minimal DSPy usage with Claude:

```python
from pathlib import Path

import dspy

class AnswerQuestion(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

lm = dspy.ClaudeLM(model="claude/claude-sonnet-4-6", repo_root=Path.cwd())
dspy.configure(lm=lm)

predict = dspy.Predict(AnswerQuestion)
result = predict(question="Reply with exactly one word: ready")
print(result.answer)
```

## Runtime Notes

- `dspy.CodexLM` supports both CLI and MCP transports.
- `dspy.QwenLM` is CLI-only in v1 and runs with `--approval-mode plan`.
- `dspy.ClaudeLM` is CLI-only in v1 and runs through `claude -p --output-format json`.
- All three backends support text-only DSPy flows and reject structured non-text inputs, native tool calls, and predicted outputs explicitly.

## Isolation Behavior

The local runtime wrappers isolate user config before invoking the CLIs so DSPy
runs stay reproducible:

- `dspy.CodexLM` copies the needed `CODEX_HOME` files into a temporary runtime home.
- `dspy.QwenLM` copies the needed `~/.qwen` files into a temporary runtime home.
- `dspy.ClaudeLM` copies the needed `~/.claude` files into a temporary runtime home.

That preserves authentication and model discovery while avoiding unrelated
interactive hooks and local side effects from leaking into automated DSPy runs.

## Optimizer Compatibility

All three local backends work through the standard text-only DSPy `lm(prompt)`
and `lm.forward()` paths.

- **GEPA**: compatible with CodexLM, QwenLM, and ClaudeLM.
- **LabeledFewShot**: compatible.
- **Evaluate**: compatible.
- **BootstrapFewShot**: compatible with `max_rounds=1`; cache-busting kwargs such as `temperature` and `rollout_id` are intentionally stripped because the local CLIs do not expose those controls in a meaningful way.

Bundled GEPA demos:

```bash
uv run python scripts/dspy_codex_gepa.py --json
uv run python scripts/dspy_qwen_gepa.py --json
uv run python scripts/dspy_claude_gepa.py --json
```

Claude GEPA live example:

```bash
uv run python scripts/dspy_claude_gepa.py --model claude/sonnet --max-metric-calls 3 --json
```

## Troubleshooting

- Missing Codex runtime: `uv run python scripts/dspy_codex_doctor.py --json`
- Missing Qwen runtime: `uv run python scripts/dspy_qwen_doctor.py --json`
- Missing Claude runtime: `uv run python scripts/dspy_claude_doctor.py --json`
- Missing Python deps: `uv sync --extra mcp --extra dev`
- Unexpected Codex transport: inspect `requested_transport`, `preferred_transport`, `available_transports`, and `resolved_model` in the doctor or probe output.
- Codex fallback behavior: `codex/default` records `fallback_from=...` when it has to drop from MCP to CLI.

<p align="center">
  <img align="center" src="docs/docs/static/img/dspy_logo.png" width="460px" />
</p>
<p align="left">

## DSPy: _Programming_—not prompting—Foundation Models

**Documentation:** [DSPy Docs](https://dspy.ai/)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dspy?period=monthly)](https://pepy.tech/projects/dspy)


----

DSPy is the framework for _programming—rather than prompting—language models_. It allows you to iterate fast on **building modular AI systems** and offers algorithms for **optimizing their prompts and weights**, whether you're building simple classifiers, sophisticated RAG pipelines, or Agent loops.

DSPy stands for Declarative Self-improving Python. Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**. Learn more via our [official documentation site](https://dspy.ai/) or meet the community, seek help, or start contributing via this GitHub repo and our [Discord server](https://discord.gg/XCGy2WDCQB).


## Documentation: [dspy.ai](https://dspy.ai)


**Please go to the [DSPy Docs at dspy.ai](https://dspy.ai)**


## Installation

If you want this fork’s `dspy.CodexLM`, stay in this checkout and install the
repo environment instead:

```bash
uv sync --extra mcp --extra dev
```

The upstream install commands below do not include this fork-specific Codex
client.

```bash
pip install dspy
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/stanfordnlp/dspy.git
````




## 📜 Citation & Reading More

If you're looking to understand the framework, please go to the [DSPy Docs at dspy.ai](https://dspy.ai).

If you're looking to understand the underlying research, this is a set of our papers:

**[Jul'25] [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)**       
**[Jun'24] [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)**       
**[Oct'23] [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)**     
[Jul'24] [Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/abs/2407.10930)     
[Jun'24] [Prompts as Auto-Optimized Training Hyperparameters](https://arxiv.org/abs/2406.11706)    
[Feb'24] [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207)         
[Jan'24] [In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178)       
[Dec'23] [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382)   
[Dec'22] [Demonstrate-Search-Predict: Composing Retrieval & Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024.pdf)

To stay up to date or learn more, follow [@DSPyOSS](https://twitter.com/DSPyOSS) on Twitter or the DSPy page on LinkedIn.

The **DSPy** logo is designed by **Chuyi Zhang**.

If you use DSPy or DSP in a research paper, please cite our work as follows:

```
@inproceedings{khattab2024dspy,
  title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines},
  author={Khattab, Omar and Singhvi, Arnav and Maheshwari, Paridhi and Zhang, Zhiyuan and Santhanam, Keshav and Vardhamanan, Sri and Haq, Saiful and Sharma, Ashutosh and Joshi, Thomas T. and Moazam, Hanna and Miller, Heather and Zaharia, Matei and Potts, Christopher},
  journal={The Twelfth International Conference on Learning Representations},
  year={2024}
}
@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
```

<!-- You can also read more about the evolution of the framework from Demonstrate-Search-Predict to DSPy:

* [**DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines**](https://arxiv.org/abs/2312.13382)   (Academic Paper, Dec 2023) 
* [**DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**](https://arxiv.org/abs/2310.03714) (Academic Paper, Oct 2023) 
* [**Releasing DSPy, the latest iteration of the framework**](https://twitter.com/lateinteraction/status/1694748401374490946) (Twitter Thread, Aug 2023)
* [**Releasing the DSP Compiler (v0.1)**](https://twitter.com/lateinteraction/status/1625231662849073160)  (Twitter Thread, Feb 2023)
* [**Introducing DSP**](https://twitter.com/lateinteraction/status/1617953413576425472)  (Twitter Thread, Jan 2023)
* [**Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP**](https://arxiv.org/abs/2212.14024.pdf) (Academic Paper, Dec 2022) -->
