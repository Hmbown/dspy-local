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
from typing import Any, Iterator, cast

from dspy.dsp.utils.settings import settings as dspy_settings

from .base_lm import BaseLM

DEFAULT_QWEN_MODEL = "qwen/default"
DEFAULT_PROBE_PROMPT = "Reply with exactly one word: ready"
DEFAULT_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
QWEN_MODEL_PREFIX = "qwen/"

logger = logging.getLogger(__name__)


class QwenTransportError(RuntimeError):
    """Raised when the local Qwen CLI cannot complete a request."""


class QwenUnsupportedFeatureError(ValueError):
    """Raised when DSPy asks QwenLM to use unsupported features."""


@dataclass(frozen=True)
class QwenModelSpec:
    raw: str
    qwen_model: str | None


@dataclass(frozen=True)
class QwenRuntimeInfo:
    cli_path: str | None
    credentials_configured: bool
    credential_sources: tuple[str, ...]
    qwen_home: str
    settings_file: str | None
    oauth_file: str | None
    mcp_oauth_file: str | None
    installation_id_file: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QwenResult:
    content: str
    usage: dict[str, Any]
    session_id: str | None = None
    stderr: str = ""
    resolved_model: str | None = None
    transport: str = "cli"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QwenCallOptions:
    prompt: str
    timeout_seconds: int
    approval_mode: str
    isolate_home: bool


_TEXT_CONTENT_TYPES = {None, "text", "input_text"}
_UNSUPPORTED_KWARG_MESSAGES = {
    "temperature": "QwenLM does not support `temperature`; the local Qwen CLI path does not expose sampling controls through this adapter.",
    "max_tokens": "QwenLM does not support `max_tokens`; the local Qwen CLI path does not expose output token limits through this adapter.",
    "cache": "QwenLM does not implement DSPy cache integration yet. Omit `cache` or pass `cache=False`.",
    "rollout_id": "QwenLM does not support `rollout_id`; the local Qwen CLI path does not expose a meaningful rollout or cache-bypass control.",
    "n": "QwenLM only supports one completion per call. `n` is not supported.",
    "num_generations": "QwenLM only supports one completion per call. `num_generations` is not supported.",
    "tools": "QwenLM does not support native tool calling through DSPy yet.",
    "tool_choice": "QwenLM does not support native tool calling through DSPy yet.",
    "parallel_tool_calls": "QwenLM does not support native tool calling through DSPy yet.",
    "prediction": "QwenLM does not support predicted outputs yet.",
    "response_format": "QwenLM does not support native structured response formats yet.",
    "logprobs": "QwenLM does not expose logprobs.",
}
_COPIED_QWEN_FILES = (
    "settings.json",
    "oauth_creds.json",
    "mcp-oauth-tokens.json",
    "installation_id",
    "output-language.md",
)


def is_qwen_model(model: str | None) -> bool:
    if model is None:
        return False
    return model.startswith(QWEN_MODEL_PREFIX)


def qwen_cli_path() -> str | None:
    return shutil.which("qwen")


def resolve_qwen_home(home_dir: Path | None = None) -> Path:
    base = home_dir.expanduser() if home_dir else Path.home()
    return base / ".qwen"


def parse_qwen_model(model: str | None) -> QwenModelSpec:
    raw = (model or DEFAULT_QWEN_MODEL).strip() or DEFAULT_QWEN_MODEL
    if not raw.startswith(QWEN_MODEL_PREFIX):
        raise ValueError(f"Qwen models must start with qwen/. Received {raw!r}.")
    configured = raw.split("/", 1)[1].strip()
    if configured in {"", "default", "auto"}:
        configured = ""
    return QwenModelSpec(raw=raw, qwen_model=configured or None)


def _detect_qwen_credentials(qwen_home: Path) -> tuple[bool, tuple[str, ...]]:
    sources: list[str] = []
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "QWEN_API_KEY",
        "OPENAI_BASE_URL",
    ):
        if os.environ.get(key, "").strip():
            sources.append(f"env:{key}")

    oauth_file = qwen_home / "oauth_creds.json"
    if oauth_file.exists():
        sources.append(str(oauth_file))

    settings_file = qwen_home / "settings.json"
    if settings_file.exists():
        sources.append(str(settings_file))

    return bool(sources), tuple(sources)


def inspect_qwen_runtime(home_dir: Path | None = None) -> QwenRuntimeInfo:
    qwen_home = resolve_qwen_home(home_dir)
    credentials_configured, credential_sources = _detect_qwen_credentials(qwen_home)
    settings_file = qwen_home / "settings.json"
    oauth_file = qwen_home / "oauth_creds.json"
    mcp_oauth_file = qwen_home / "mcp-oauth-tokens.json"
    installation_id_file = qwen_home / "installation_id"
    return QwenRuntimeInfo(
        cli_path=qwen_cli_path(),
        credentials_configured=credentials_configured,
        credential_sources=credential_sources,
        qwen_home=str(qwen_home),
        settings_file=str(settings_file) if settings_file.exists() else None,
        oauth_file=str(oauth_file) if oauth_file.exists() else None,
        mcp_oauth_file=str(mcp_oauth_file) if mcp_oauth_file.exists() else None,
        installation_id_file=str(installation_id_file) if installation_id_file.exists() else None,
    )


@contextmanager
def _isolated_qwen_home() -> Iterator[Path]:
    source_home = resolve_qwen_home()
    with tempfile.TemporaryDirectory(prefix="dspy-qwen-home-") as tmpdir:
        overlay_home = Path(tmpdir)
        overlay_qwen = overlay_home / ".qwen"
        overlay_qwen.mkdir(parents=True, exist_ok=True)
        for filename in _COPIED_QWEN_FILES:
            source = source_home / filename
            if source.exists():
                shutil.copy2(source, overlay_qwen / filename)
        yield overlay_home


@contextmanager
def _qwen_environment(isolate_home: bool = True) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    if not isolate_home:
        yield env
        return
    with _isolated_qwen_home() as overlay_home:
        env["HOME"] = str(overlay_home)
        env.pop("USERPROFILE", None)
        yield env


def _coerce_text_part(part: dict[str, Any], *, role: str) -> str:
    part_type = part.get("type")
    if part_type not in _TEXT_CONTENT_TYPES and "text" not in part:
        raise QwenUnsupportedFeatureError(
            "QwenLM does not support structured non-text content yet. "
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
            raise QwenUnsupportedFeatureError(
                "QwenLM does not support structured non-text content yet. "
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
                raise QwenUnsupportedFeatureError("QwenLM does not support tool-call messages yet.")
            if message.get("tool_call_id"):
                raise QwenUnsupportedFeatureError("QwenLM does not support tool result messages yet.")
            raw_role = str(message.get("role", "user")).lower()
            if raw_role == "tool":
                raise QwenUnsupportedFeatureError("QwenLM does not support tool result messages yet.")
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

    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    normalized: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    cached_tokens = usage.get("cache_read_input_tokens")
    if isinstance(cached_tokens, (int, float)) and int(cached_tokens) > 0:
        normalized["prompt_tokens_details"] = {"cached_tokens": int(cached_tokens)}
    return normalized


def _extract_text_from_assistant(event: dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(part for part in parts if part).strip()


def _parse_qwen_events(stdout: str) -> tuple[str, dict[str, Any], str | None, str | None]:
    data = json.loads(stdout)
    if not isinstance(data, list):
        raise QwenTransportError("qwen CLI returned a non-list JSON payload.")

    final_text: str | None = None
    fallback_text: str | None = None
    parsed_usage: dict[str, Any] = {}
    session_id: str | None = None
    resolved_model: str | None = None

    for event in data:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "system" and event.get("subtype") == "init":
            session_id = str(event.get("session_id")) if event.get("session_id") else session_id
            resolved_model = str(event.get("model")) if event.get("model") else resolved_model
        elif event_type == "assistant":
            text = _extract_text_from_assistant(event)
            if text:
                fallback_text = text
        elif event_type == "result" and event.get("subtype") == "success":
            value = event.get("result")
            if value is not None:
                final_text = str(value).strip()
            parsed_usage = _coerce_usage(event.get("usage"))

    content = final_text or fallback_text
    if not content:
        raise QwenTransportError("qwen CLI returned no final assistant text.")

    return content, parsed_usage, session_id, resolved_model


def _build_qwen_command(
    *,
    qwen_model: str | None,
    approval_mode: str,
) -> list[str]:
    cli = qwen_cli_path()
    if not cli:
        raise QwenTransportError("qwen CLI is not installed or not on PATH.")
    command = [
        cli,
        "--approval-mode",
        approval_mode,
        "--output-format",
        "json",
    ]
    if qwen_model:
        command.extend(["--model", qwen_model])
    return command


def run_qwen_cli(
    *,
    prompt: str,
    repo_root: Path,
    qwen_model: str | None = None,
    timeout_seconds: int = 120,
    approval_mode: str = "plan",
    isolate_home: bool = True,
) -> QwenResult:
    command = _build_qwen_command(qwen_model=qwen_model, approval_mode=approval_mode)
    command.append(prompt)

    with _qwen_environment(isolate_home=isolate_home) as env:
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
        raise QwenTransportError(message or f"qwen exited with code {completed.returncode}.")

    text, usage, session_id, resolved_model = _parse_qwen_events(completed.stdout)
    return QwenResult(
        content=text,
        usage=usage,
        session_id=session_id,
        stderr=completed.stderr,
        resolved_model=resolved_model or qwen_model,
    )


async def arun_qwen_cli(
    *,
    prompt: str,
    repo_root: Path,
    qwen_model: str | None = None,
    timeout_seconds: int = 120,
    approval_mode: str = "plan",
    isolate_home: bool = True,
) -> QwenResult:
    command = _build_qwen_command(qwen_model=qwen_model, approval_mode=approval_mode)
    command.append(prompt)

    with _qwen_environment(isolate_home=isolate_home) as env:
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
        raise QwenTransportError(message or f"qwen exited with code {process.returncode}.")

    text, usage, session_id, resolved_model = _parse_qwen_events(stdout_text)
    return QwenResult(
        content=text,
        usage=usage,
        session_id=session_id,
        stderr=stderr_text,
        resolved_model=resolved_model or qwen_model,
    )


def run_qwen(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_QWEN_MODEL,
    timeout_seconds: int = 120,
    approval_mode: str = "plan",
    isolate_home: bool = True,
) -> QwenResult:
    spec = parse_qwen_model(model)
    return run_qwen_cli(
        prompt=prompt,
        repo_root=repo_root,
        qwen_model=spec.qwen_model,
        timeout_seconds=timeout_seconds,
        approval_mode=approval_mode,
        isolate_home=isolate_home,
    )


async def arun_qwen(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_QWEN_MODEL,
    timeout_seconds: int = 120,
    approval_mode: str = "plan",
    isolate_home: bool = True,
) -> QwenResult:
    spec = parse_qwen_model(model)
    return await arun_qwen_cli(
        prompt=prompt,
        repo_root=repo_root,
        qwen_model=spec.qwen_model,
        timeout_seconds=timeout_seconds,
        approval_mode=approval_mode,
        isolate_home=isolate_home,
    )


def probe_qwen_runtime(
    *,
    repo_root: Path,
    model: str = DEFAULT_QWEN_MODEL,
    timeout_seconds: int = 120,
) -> QwenResult:
    return run_qwen(
        prompt=DEFAULT_PROBE_PROMPT,
        repo_root=repo_root,
        model=model,
        timeout_seconds=timeout_seconds,
    )


class QwenLM(BaseLM):
    """DSPy BaseLM wrapper that routes requests through the local Qwen CLI."""

    _CACHE_BUSTING_KWARGS = frozenset({"temperature", "rollout_id"})
    _UNSET_UNSUPPORTED_KWARGS = frozenset({"temperature", "max_tokens", "rollout_id"})

    def __init__(
        self,
        model: str = DEFAULT_QWEN_MODEL,
        *,
        repo_root: str | Path,
        approval_mode: str = "plan",
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
        self.approval_mode = approval_mode
        self.isolate_home = isolate_home
        self.model_spec = parse_qwen_model(model)
        self._validate_runtime_kwargs(dict(self.kwargs), cache=self.cache)
        self.kwargs = self._canonicalize_kwargs(self.kwargs)

    @staticmethod
    def _validate_runtime_kwargs(kwargs: dict[str, Any], *, cache: bool) -> None:
        errors: list[str] = []

        if cache:
            errors.append(_UNSUPPORTED_KWARG_MESSAGES["cache"])

        for key, value in kwargs.items():
            if key in {"timeout_seconds", "approval_mode", "isolate_home"}:
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
            errors.append(f"QwenLM does not support the `{key}` kwarg.")

        if errors:
            raise QwenUnsupportedFeatureError(" ".join(dict.fromkeys(errors)))

    @classmethod
    def _canonicalize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(kwargs)
        for key in cls._UNSET_UNSUPPORTED_KWARGS:
            if canonical.get(key) is None:
                canonical.pop(key, None)
        return canonical

    def copy(self, **kwargs: Any) -> "QwenLM":
        stripped = {k: kwargs.pop(k) for k in list(kwargs) if k in self._CACHE_BUSTING_KWARGS}
        if stripped:
            logger.debug("QwenLM.copy(): stripped cache-busting kwargs %s (no effect on Qwen)", stripped)
        copied = cast(QwenLM, super().copy(**kwargs))
        copied.kwargs = self._canonicalize_kwargs(copied.kwargs)
        self._validate_runtime_kwargs(dict(copied.kwargs), cache=copied.cache)
        return copied

    def _prepare_call(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> QwenCallOptions:
        call_kwargs = dict(kwargs)
        cache_value = call_kwargs.pop("cache", self.cache)
        approval_mode = str(call_kwargs.pop("approval_mode", self.approval_mode))
        isolate_home = bool(call_kwargs.pop("isolate_home", self.isolate_home))
        merged_kwargs = {**self.kwargs, **call_kwargs}
        self._validate_runtime_kwargs(dict(merged_kwargs), cache=bool(cache_value))
        timeout_seconds = int(merged_kwargs.pop("timeout_seconds"))
        return QwenCallOptions(
            prompt=build_prompt(prompt=prompt, messages=messages),
            timeout_seconds=timeout_seconds,
            approval_mode=approval_mode,
            isolate_home=isolate_home,
        )

    def _record_usage(self, usage: dict[str, Any]) -> None:
        if usage and dspy_settings.usage_tracker:
            dspy_settings.usage_tracker.add_usage(self.model, dict(usage))

    def _to_response(self, result: QwenResult) -> Any:
        self._record_usage(result.usage)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result.content))],
            usage=dict(result.usage),
            model=result.resolved_model or self.model,
            _hidden_params={
                "transport": result.transport,
                "session_id": result.session_id,
                "stderr": result.stderr,
            },
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        options = self._prepare_call(prompt=prompt, messages=messages, kwargs=kwargs)
        result = run_qwen(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            timeout_seconds=options.timeout_seconds,
            approval_mode=options.approval_mode,
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
        result = await arun_qwen(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            timeout_seconds=options.timeout_seconds,
            approval_mode=options.approval_mode,
            isolate_home=options.isolate_home,
        )
        return self._to_response(result)


__all__ = [
    "DEFAULT_PROBE_PROMPT",
    "DEFAULT_QWEN_MODEL",
    "QwenLM",
    "QwenModelSpec",
    "QwenResult",
    "QwenRuntimeInfo",
    "QwenTransportError",
    "QwenUnsupportedFeatureError",
    "arun_qwen",
    "arun_qwen_cli",
    "build_prompt",
    "inspect_qwen_runtime",
    "is_qwen_model",
    "parse_qwen_model",
    "probe_qwen_runtime",
    "qwen_cli_path",
    "resolve_qwen_home",
    "run_qwen",
    "run_qwen_cli",
]
