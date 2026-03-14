# Runtime Selection

`dspy-local` supports three model aliases:

- `codex/default`
- `codex-exec/default`
- `codex-mcp/default`

Use a concrete Codex model slug after the slash to force a specific model, for
example `codex/gpt-5.3-codex`.

## Resolution Order

1. If the model alias is `codex-exec/...`, force the direct `codex exec` path.
2. If the model alias is `codex-mcp/...`, force the `codex mcp-server` path.
3. Otherwise read `DSPY_CODEX_TRANSPORT` if it is set to `auto`, `cli`, or `mcp`.
4. If the transport is still `auto`, prefer MCP when the Python `mcp` package is
   installed and the local Codex CLI is available.
5. If the MCP attempt fails during `auto`, fall back once to direct CLI.

## Why The Runtime Uses An Isolated `CODEX_HOME`

Both transport paths copy `auth.json` and `models_cache.json` into a temporary
`CODEX_HOME` before invoking Codex. This keeps local authentication and model
discovery intact without pulling unrelated Codex config, hooks, or external MCP
server definitions into DSPy runs.

That matters in practice because a user-level Codex config can include project
hooks or MCP servers that are useful interactively but noisy or brittle during
automated DSPy compile and eval loops.

## Recommended Defaults

- Use `codex/default` for normal repo code.
- Use `codex-exec/default` when you want the simplest, most debuggable path.
- Use `codex-mcp/default` when you explicitly want the session to go through the
  Codex MCP surface.
- Keep the default sandbox at `read-only` unless the task genuinely needs file
  writes inside Codex itself.
