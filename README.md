## dspy-codex: DSPy With Local Codex Runtimes

This fork adds a repo-local `dspy.CodexLM` client that routes DSPy calls through
the local Codex CLI or the local `codex mcp-server`.

`pip install dspy` does not include this fork’s `dspy.CodexLM`. Use this repo
checkout and its `uv` environment.

## Prerequisites

1. Install the Codex CLI and make sure `codex --help` works.
2. Authenticate Codex so `~/.codex/auth.json` exists or `OPENAI_API_KEY` is set.
3. From this repo root, install the Python environment:

```bash
uv sync --extra mcp --extra dev
```

## Fresh-Clone Quickstart

```bash
git clone <this-fork>
cd dspy-codex
uv sync --extra mcp --extra dev
uv run python scripts/dspy_codex_doctor.py --json
uv run python scripts/dspy_codex_probe.py --transport auto --json
uv run python scripts/dspy_codex_smoke.py --transport auto --json
```

## Model Aliases

- `codex/default`: prefer MCP when available and fall back once to CLI if MCP fails.
- `codex-exec/default`: force direct `codex exec`.
- `codex-mcp/default`: force `codex mcp-server`.

Optional environment override for the generic alias:

```bash
export DSPY_CODEX_TRANSPORT=auto  # auto | cli | mcp
```

## First Real Check

Run these in order after a fresh clone:

```bash
uv run python scripts/dspy_codex_doctor.py --json
uv run python scripts/dspy_codex_probe.py --transport auto --json
uv run python scripts/dspy_codex_smoke.py --transport auto --json
```

If you want the doctor to do a live probe too:

```bash
uv run python scripts/dspy_codex_doctor.py --live --json
```

## Quick Example

```python
from pathlib import Path

import dspy

lm = dspy.CodexLM(
    model="codex/default",
    repo_root=Path.cwd(),
)
dspy.configure(lm=lm)
```

Force the transport with the alias when needed:

```python
dspy.CodexLM(model="codex/default", repo_root=Path.cwd())
dspy.CodexLM(model="codex-exec/default", repo_root=Path.cwd())
dspy.CodexLM(model="codex-mcp/default", repo_root=Path.cwd())
```

## Why `CODEX_HOME` Is Isolated

`dspy.CodexLM` copies `auth.json`, `models_cache.json`, and `version.json` into
a temporary `CODEX_HOME` before launching Codex. That preserves authentication
and model discovery while avoiding unrelated user-level hooks, MCP servers, and
other interactive Codex config from leaking into automated DSPy runs.

## Optimizer Compatibility

`dspy.CodexLM` works with DSPy optimizers that run through the standard
`lm(prompt)` and `lm.forward()` paths:

- **GEPA** — fully compatible. Use CodexLM as both the student and reflection
  LM. See `scripts/dspy_codex_gepa.py` for a minimal working example.
- **LabeledFewShot** — fully compatible.
- **BootstrapFewShot** — compatible with `max_rounds=1` (the default). Multi-
  round bootstrap uses `lm.copy(temperature=..., rollout_id=...)` for cache
  busting; CodexLM silently strips these since the local transports don't
  expose sampling controls.
- **Evaluate** — fully compatible with both CLI and MCP transports.

```bash
uv run python scripts/dspy_codex_gepa.py --json
```

## Current Capability Boundaries

- Text-only prompts are supported for sync and async flows.
- Structured non-text message content such as images, audio, files, and tool
  call messages is rejected explicitly instead of being silently flattened.
- Native tool calling is not supported yet.
- MCP usage data is currently unknown, so MCP calls report empty usage rather
  than synthetic zero-token usage.

## Troubleshooting

- Missing CLI: `scripts/dspy_codex_doctor.py` should show `Codex CLI: missing`.
- Missing auth: doctor should show `Credential source: missing`.
- Missing MCP dependency: install the repo env with `uv sync --extra mcp --extra dev`.
- Unexpected transport: inspect `requested_transport`, `preferred_transport`,
  `available_transports`, and `resolved_model` in doctor/probe output.
- Fallback behavior: `codex/default` prefers MCP and records `fallback_from=...`
  when it has to drop back to CLI.

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
