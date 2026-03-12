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
    from dspy.clients.codex import install_codex_skill
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python dependency: {missing}. "
        "Run `uv sync --extra mcp --extra dev` once, or invoke this script with "
        "`uv run python scripts/install_codex_skill.py`."
    ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the bundled dspy-codex skill into CODEX_HOME/skills.")
    parser.add_argument("--codex-home", default=None)
    parser.add_argument("--force-relink", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = install_codex_skill(codex_home=args.codex_home, force_relink=args.force_relink)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"status={result['status']}")
        print(f"source={result['source']}")
        print(f"destination={result['destination']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
