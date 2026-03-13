from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Literal, cast

from dspy.dsp.utils.settings import settings as dspy_settings

from .base_lm import BaseLM

DEFAULT_CLAUDE_MODEL = "claude/default"
DEFAULT_PROBE_PROMPT = "Reply with exactly one word: ready"
DEFAULT_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
CLAUDE_MODEL_PREFIX = "claude/"

logger = logging.getLogger(__name__)


class ClaudeTransportError(RuntimeError):
    """Raised when the local Claude CLI cannot complete a request."""


class ClaudeUnsupportedFeatureError(ValueError):
    """Raised when DSPy asks ClaudeLM to use unsupported features."""


@dataclass(frozen=True)
class ClaudeModelSpec:
    raw: str
    claude_model: str | None


@dataclass(frozen=True)
class ClaudeRuntimeInfo:
    cli_path: str | None
    credentials_configured: bool
    credential_sources: tuple[str, ...]
    claude_home: str
    settings_file: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaudeResult:
    content: str
    usage: dict[str, Any]
    session_id: str | None = None
    stderr: str = ""
    resolved_model: str | None = None
    cost_usd: float | None = None
    transport: str = "cli"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaudeCallOptions:
    prompt: str
    timeout_seconds: int
    permission_mode: str
    isolate_home: bool


_TEXT_CONTENT_TYPES = {None, "text", "input_text"}
_UNSUPPORTED_KWARG_MESSAGES = {
    "temperature": "ClaudeLM does not support `temperature`; the local Claude CLI does not expose sampling controls through this adapter.",
    "max_tokens": "ClaudeLM does not support `max_tokens`; the local Claude CLI does not expose output token limits through this adapter.",
    "cache": "ClaudeLM does not implement DSPy cache integration yet. Omit `cache` or pass `cache=False`.",
    "rollout_id": "ClaudeLM does not support `rollout_id`.",
    "n": "ClaudeLM only supports one completion per call. `n` is not supported.",
    "num_generations": "ClaudeLM only supports one completion per call. `num_generations` is not supported.",
    "tools": "ClaudeLM does not support native tool calling through DSPy yet.",
    "tool_choice": "ClaudeLM does not support native tool calling through DSPy yet.",
    "parallel_tool_calls": "ClaudeLM does not support native tool calling through DSPy yet.",
    "prediction": "ClaudeLM does not support predicted outputs yet.",
    "response_format": "ClaudeLM does not support native structured response formats yet.",
    "logprobs": "ClaudeLM does not expose logprobs.",
}
_COPIED_CLAUDE_FILES = (
    "settings.json",
    "credentials.json",
)


def is_claude_model(model: str | None) -> bool:
    if model is None:
        return False
    return model.startswith(CLAUDE_MODEL_PREFIX)


def claude_cli_path() -> str | None:
    return shutil.which("claude")


def resolve_claude_home(home_dir: Path | None = None) -> Path:
    configured = os.environ.get("CLAUDE_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = home_dir.expanduser() if home_dir else Path.home()
    return base / ".claude"


def parse_claude_model(model: str | None) -> ClaudeModelSpec:
    raw = (model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
    if not raw.startswith(CLAUDE_MODEL_PREFIX):
        raise ValueError(f"Claude models must start with claude/. Received {raw!r}.")
    configured = raw.split("/", 1)[1].strip()
    if configured in {"", "default", "auto"}:
        configured = ""
    return ClaudeModelSpec(raw=raw, claude_model=configured or None)


def _detect_claude_credentials(claude_home: Path) -> tuple[bool, tuple[str, ...]]:
    sources: list[str] = []
    for key in ("ANTHROPIC_API_KEY",):
        if os.environ.get(key, "").strip():
            sources.append(f"env:{key}")

    creds_file = claude_home / "credentials.json"
    if creds_file.exists():
        sources.append(str(creds_file))

    return bool(sources), tuple(sources)


def inspect_claude_runtime(home_dir: Path | None = None) -> ClaudeRuntimeInfo:
    claude_home = resolve_claude_home(home_dir)
    credentials_configured, credential_sources = _detect_claude_credentials(claude_home)
    settings_file = claude_home / "settings.json"
    return ClaudeRuntimeInfo(
        cli_path=claude_cli_path(),
        credentials_configured=credentials_configured,
        credential_sources=credential_sources,
        claude_home=str(claude_home),
        settings_file=str(settings_file) if settings_file.exists() else None,
    )


@contextmanager
def _isolated_claude_home() -> Iterator[Path]:
    source_home = resolve_claude_home()
    with tempfile.TemporaryDirectory(prefix="dspy-claude-home-") as tmpdir:
        overlay_home = Path(tmpdir)
        overlay_claude = overlay_home / ".claude"
        overlay_claude.mkdir(parents=True, exist_ok=True)
        for filename in _COPIED_CLAUDE_FILES:
            source = source_home / filename
            if source.exists():
                shutil.copy2(source, overlay_claude / filename)
        yield overlay_home


@contextmanager
def _claude_environment(isolate_home: bool = True) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    if not isolate_home:
        yield env
        return
    with _isolated_claude_home() as overlay_home:
        env["HOME"] = str(overlay_home)
        env.pop("USERPROFILE", None)
        yield env


def _coerce_text_part(part: dict[str, Any], *, role: str) -> str:
    part_type = part.get("type")
    if part_type not in _TEXT_CONTENT_TYPES and "text" not in part:
        raise ClaudeUnsupportedFeatureError(
            "ClaudeLM does not support structured non-text content yet. "
            f"Received part type {part_type!r} in a {role!r} message."
        )

    if part_type == "input_text":
        text = part.get("input_text")
    else:
        text = part.get("text")

    if text is None:
        return ""
    return str(text)


def _coerce_content(content: Any, *, role: str) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = _coerce_text_part(item, role=role)
                if text:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("type") in _TEXT_CONTENT_TYPES or "text" in content:
            return _coerce_text_part(content, role=role)
        if any(key in content for key in ("image_url", "input_audio", "file")):
            raise ClaudeUnsupportedFeatureError(
                "ClaudeLM does not support structured non-text content yet. "
                f"Received keys {sorted(content)} in a {role!r} message."
            )
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def build_prompt(
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    if messages:
        chunks: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            if message.get("tool_calls"):
                raise ClaudeUnsupportedFeatureError("ClaudeLM does not support tool-call messages yet.")
            if message.get("tool_call_id"):
                raise ClaudeUnsupportedFeatureError("ClaudeLM does not support tool result messages yet.")
            raw_role = str(message.get("role", "user")).lower()
            if raw_role == "tool":
                raise ClaudeUnsupportedFeatureError("ClaudeLM does not support tool result messages yet.")
            content = _coerce_content(message.get("content", ""), role=raw_role)
            if content:
                chunks.append(f"{role}:\n{content}")
        if prompt:
            chunks.append(f"PROMPT:\n{prompt}")
        return "\n\n".join(chunks).strip()
    return (prompt or "").strip()


def _coerce_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}

    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    total_tokens = input_tokens + output_tokens
    normalized: dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    cached_tokens = usage.get("cache_read_input_tokens")
    if isinstance(cached_tokens, (int, float)) and int(cached_tokens) > 0:
        normalized["prompt_tokens_details"] = {"cached_tokens": int(cached_tokens)}
    return normalized


def _parse_claude_result(stdout: str) -> tuple[str, dict[str, Any], str | None, str | None, float | None]:
    """Parse the JSON output from `claude -p --output-format json`.

    Returns (text, usage, session_id, resolved_model, cost_usd).
    """
    data = json.loads(stdout)
    if not isinstance(data, dict):
        raise ClaudeTransportError("claude CLI returned a non-object JSON payload.")

    result_type = data.get("type")
    subtype = data.get("subtype")
    if result_type == "result" and subtype != "success":
        error_text = data.get("result", "") or data.get("error", "")
        raise ClaudeTransportError(f"claude CLI returned error: {error_text}")

    text = data.get("result")
    if not text or not str(text).strip():
        raise ClaudeTransportError("claude CLI returned no result text.")
    text = str(text).strip()

    parsed_usage = _coerce_usage(data.get("usage"))

    session_id = data.get("session_id")
    if session_id:
        session_id = str(session_id)

    # Extract resolved model from modelUsage keys
    resolved_model: str | None = None
    model_usage = data.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        resolved_model = next(iter(model_usage))

    cost_usd = data.get("total_cost_usd")
    if isinstance(cost_usd, (int, float)):
        cost_usd = float(cost_usd)
    else:
        cost_usd = None

    return text, parsed_usage, session_id, resolved_model, cost_usd


PermissionMode = Literal["default", "plan", "acceptEdits", "bypassPermissions", "dontAsk", "auto"]


def _build_claude_command(
    *,
    claude_model: str | None,
    permission_mode: str,
) -> list[str]:
    cli = claude_cli_path()
    if not cli:
        raise ClaudeTransportError("claude CLI is not installed or not on PATH.")
    command = [
        cli,
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        permission_mode,
        "--no-session-persistence",
    ]
    if claude_model:
        command.extend(["--model", claude_model])
    return command


def run_claude_cli(
    *,
    prompt: str,
    repo_root: Path,
    claude_model: str | None = None,
    timeout_seconds: int = 120,
    permission_mode: str = "plan",
    isolate_home: bool = True,
) -> ClaudeResult:
    command = _build_claude_command(claude_model=claude_model, permission_mode=permission_mode)
    command.append(prompt)

    with _claude_environment(isolate_home=isolate_home) as env:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise ClaudeTransportError(message or f"claude exited with code {completed.returncode}.")

    text, usage, session_id, resolved_model, cost_usd = _parse_claude_result(completed.stdout)
    return ClaudeResult(
        content=text,
        usage=usage,
        session_id=session_id,
        stderr=completed.stderr,
        resolved_model=resolved_model or claude_model,
        cost_usd=cost_usd,
    )


async def arun_claude_cli(
    *,
    prompt: str,
    repo_root: Path,
    claude_model: str | None = None,
    timeout_seconds: int = 120,
    permission_mode: str = "plan",
    isolate_home: bool = True,
) -> ClaudeResult:
    command = _build_claude_command(claude_model=claude_model, permission_mode=permission_mode)
    command.append(prompt)

    with _claude_environment(isolate_home=isolate_home) as env:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(repo_root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except TimeoutError as exc:
            process.kill()
            stdout, stderr = await process.communicate()
            raise subprocess.TimeoutExpired(command, timeout_seconds, output=stdout, stderr=stderr) from exc

    stdout_text = stdout.decode()
    stderr_text = stderr.decode()
    if process.returncode != 0:
        message = stderr_text.strip() or stdout_text.strip()
        raise ClaudeTransportError(message or f"claude exited with code {process.returncode}.")

    text, usage, session_id, resolved_model, cost_usd = _parse_claude_result(stdout_text)
    return ClaudeResult(
        content=text,
        usage=usage,
        session_id=session_id,
        stderr=stderr_text,
        resolved_model=resolved_model or claude_model,
        cost_usd=cost_usd,
    )


def run_claude(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_CLAUDE_MODEL,
    timeout_seconds: int = 120,
    permission_mode: str = "plan",
    isolate_home: bool = True,
) -> ClaudeResult:
    spec = parse_claude_model(model)
    return run_claude_cli(
        prompt=prompt,
        repo_root=repo_root,
        claude_model=spec.claude_model,
        timeout_seconds=timeout_seconds,
        permission_mode=permission_mode,
        isolate_home=isolate_home,
    )


async def arun_claude(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_CLAUDE_MODEL,
    timeout_seconds: int = 120,
    permission_mode: str = "plan",
    isolate_home: bool = True,
) -> ClaudeResult:
    spec = parse_claude_model(model)
    return await arun_claude_cli(
        prompt=prompt,
        repo_root=repo_root,
        claude_model=spec.claude_model,
        timeout_seconds=timeout_seconds,
        permission_mode=permission_mode,
        isolate_home=isolate_home,
    )


def probe_claude_runtime(
    *,
    repo_root: Path,
    model: str = DEFAULT_CLAUDE_MODEL,
    timeout_seconds: int = 120,
) -> ClaudeResult:
    return run_claude(
        prompt=DEFAULT_PROBE_PROMPT,
        repo_root=repo_root,
        model=model,
        timeout_seconds=timeout_seconds,
    )


class ClaudeLM(BaseLM):
    """DSPy BaseLM wrapper that routes requests through the local Claude CLI."""

    _CACHE_BUSTING_KWARGS = frozenset({"temperature", "rollout_id"})
    _UNSET_UNSUPPORTED_KWARGS = frozenset({"temperature", "max_tokens", "rollout_id"})

    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        *,
        repo_root: str | Path,
        permission_mode: str = "plan",
        isolate_home: bool = True,
        timeout_seconds: int = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.repo_root = Path(repo_root).resolve()
        self.permission_mode = permission_mode
        self.isolate_home = isolate_home
        self.model_spec = parse_claude_model(model)
        self._validate_runtime_kwargs(dict(self.kwargs), cache=self.cache)
        self.kwargs = self._canonicalize_kwargs(self.kwargs)

    @staticmethod
    def _validate_runtime_kwargs(kwargs: dict[str, Any], *, cache: bool) -> None:
        errors: list[str] = []

        if cache:
            errors.append(_UNSUPPORTED_KWARG_MESSAGES["cache"])

        for key, value in kwargs.items():
            if key in {"timeout_seconds", "permission_mode", "isolate_home"}:
                continue
            if key == "temperature":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "max_tokens":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "rollout_id":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "n":
                if value not in (None, 1):
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "num_generations":
                if value not in (None, 1):
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "tools":
                if value not in (None, [], ()):
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "tool_choice":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "parallel_tool_calls":
                if value not in (None, False):
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "prediction":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "response_format":
                if value is not None:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            if key == "logprobs":
                if value:
                    errors.append(_UNSUPPORTED_KWARG_MESSAGES[key])
                continue
            errors.append(f"ClaudeLM does not support the `{key}` kwarg.")

        if errors:
            raise ClaudeUnsupportedFeatureError(" ".join(dict.fromkeys(errors)))

    @classmethod
    def _canonicalize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(kwargs)
        for key in cls._UNSET_UNSUPPORTED_KWARGS:
            if canonical.get(key) is None:
                canonical.pop(key, None)
        return canonical

    def copy(self, **kwargs: Any) -> "ClaudeLM":
        stripped = {k: kwargs.pop(k) for k in list(kwargs) if k in self._CACHE_BUSTING_KWARGS}
        if stripped:
            logger.debug("ClaudeLM.copy(): stripped cache-busting kwargs %s (no effect on Claude)", stripped)
        copied = cast(ClaudeLM, super().copy(**kwargs))
        copied.kwargs = self._canonicalize_kwargs(copied.kwargs)
        self._validate_runtime_kwargs(dict(copied.kwargs), cache=copied.cache)
        return copied

    def _prepare_call(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> ClaudeCallOptions:
        call_kwargs = dict(kwargs)
        cache_value = call_kwargs.pop("cache", self.cache)
        permission_mode = str(call_kwargs.pop("permission_mode", self.permission_mode))
        isolate_home = bool(call_kwargs.pop("isolate_home", self.isolate_home))
        merged_kwargs = {**self.kwargs, **call_kwargs}
        self._validate_runtime_kwargs(dict(merged_kwargs), cache=bool(cache_value))
        timeout_seconds = int(merged_kwargs.pop("timeout_seconds"))
        return ClaudeCallOptions(
            prompt=build_prompt(prompt=prompt, messages=messages),
            timeout_seconds=timeout_seconds,
            permission_mode=permission_mode,
            isolate_home=isolate_home,
        )

    def _record_usage(self, usage: dict[str, Any]) -> None:
        if usage and dspy_settings.usage_tracker:
            dspy_settings.usage_tracker.add_usage(self.model, dict(usage))

    def _to_response(self, result: ClaudeResult) -> Any:
        self._record_usage(result.usage)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result.content))],
            usage=dict(result.usage),
            model=result.resolved_model or self.model,
            _hidden_params={
                "transport": result.transport,
                "session_id": result.session_id,
                "stderr": result.stderr,
                "cost_usd": result.cost_usd,
            },
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        options = self._prepare_call(prompt=prompt, messages=messages, kwargs=kwargs)
        result = run_claude(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            timeout_seconds=options.timeout_seconds,
            permission_mode=options.permission_mode,
            isolate_home=options.isolate_home,
        )
        return self._to_response(result)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        options = self._prepare_call(prompt=prompt, messages=messages, kwargs=kwargs)
        result = await arun_claude(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            timeout_seconds=options.timeout_seconds,
            permission_mode=options.permission_mode,
            isolate_home=options.isolate_home,
        )
        return self._to_response(result)


__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_PROBE_PROMPT",
    "ClaudeLM",
    "ClaudeModelSpec",
    "ClaudeResult",
    "ClaudeRuntimeInfo",
    "ClaudeTransportError",
    "ClaudeUnsupportedFeatureError",
    "arun_claude",
    "arun_claude_cli",
    "build_prompt",
    "claude_cli_path",
    "inspect_claude_runtime",
    "is_claude_model",
    "parse_claude_model",
    "probe_claude_runtime",
    "resolve_claude_home",
    "run_claude",
    "run_claude_cli",
]
