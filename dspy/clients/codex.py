from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Literal, cast

from dspy.dsp.utils.settings import settings as dspy_settings

from .base_lm import BaseLM

TransportName = Literal["auto", "cli", "mcp"]

DEFAULT_CODEX_MODEL = "codex/default"
DEFAULT_PROBE_PROMPT = "Reply with exactly one word: ready"
DEFAULT_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}
MODEL_PREFIXES: tuple[tuple[str, TransportName], ...] = (
    ("codex-exec/", "cli"),
    ("codex-mcp/", "mcp"),
    ("codex/", "auto"),
)
DEFAULT_CODEX_MODELS = [
    "gpt-5.4",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
]

logger = logging.getLogger(__name__)


class CodexTransportError(RuntimeError):
    """Raised when no Codex transport can complete a request."""


class CodexUnsupportedFeatureError(ValueError):
    """Raised when DSPy asks CodexLM to use a feature the local transports do not support."""


@dataclass(frozen=True)
class CodexModelSpec:
    raw: str
    codex_model: str | None
    transport_hint: TransportName


@dataclass(frozen=True)
class CodexRuntimeInfo:
    requested_transport: TransportName
    preferred_transport: TransportName | None
    available_transports: tuple[TransportName, ...]
    cli_path: str | None
    mcp_sdk_available: bool
    credential_source: str | None
    codex_home: str
    auth_file: str | None
    models_cache: str | None
    default_model: str | None
    available_models: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodexResult:
    content: str
    transport: TransportName
    usage: dict[str, Any]
    thread_id: str | None = None
    stderr: str = ""
    fallback_from: TransportName | None = None
    resolved_model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CodexCallOptions:
    prompt: str
    transport: TransportName | None
    timeout_seconds: int
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"]
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"]
    isolate_home: bool


_TEXT_CONTENT_TYPES = {None, "text", "input_text"}
_UNSUPPORTED_KWARG_MESSAGES = {
    "temperature": "CodexLM does not support `temperature`; the local Codex CLI and MCP tool do not expose sampling controls.",
    "max_tokens": "CodexLM does not support `max_tokens`; the local Codex CLI and MCP tool do not expose output token limits.",
    "cache": "CodexLM does not implement DSPy cache integration yet. Omit `cache` or pass `cache=False`.",
    "rollout_id": "CodexLM does not support `rollout_id`; the local Codex transports do not expose a meaningful rollout or cache-bypass control.",
    "n": "CodexLM only supports one completion per call. `n` is not supported.",
    "num_generations": "CodexLM only supports one completion per call. `num_generations` is not supported.",
    "tools": "CodexLM does not support native tool calling yet.",
    "tool_choice": "CodexLM does not support native tool calling yet.",
    "parallel_tool_calls": "CodexLM does not support native tool calling yet.",
    "prediction": "CodexLM does not support predicted outputs yet.",
    "response_format": "CodexLM does not support native structured response formats yet.",
    "logprobs": "CodexLM does not expose logprobs.",
}


def is_codex_model(model: str | None) -> bool:
    if model is None:
        return False
    return model.startswith(tuple(prefix for prefix, _ in MODEL_PREFIXES))


def codex_cli_path() -> str | None:
    return shutil.which("codex")


def resolve_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def _normalize_codex_model(model: str | None) -> str | None:
    if model is None:
        return None
    normalized = model.strip()
    if not normalized or normalized in {"auto", "default"}:
        return None
    return normalized


def parse_codex_model(model: str | None) -> CodexModelSpec:
    raw = (model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL
    for prefix, transport in MODEL_PREFIXES:
        if raw.startswith(prefix):
            return CodexModelSpec(
                raw=raw,
                codex_model=_normalize_codex_model(raw.split("/", 1)[1]),
                transport_hint=transport,
            )
    raise ValueError(
        "Codex models must start with codex/, codex-exec/, or codex-mcp/. "
        f"Received {raw!r}."
    )


def resolve_requested_transport(
    model: str | None = None,
    transport: TransportName | None = None,
) -> TransportName:
    if transport is not None:
        return _normalize_transport(transport)
    spec = parse_codex_model(model)
    if spec.transport_hint != "auto":
        return spec.transport_hint
    configured = os.environ.get("DSPY_CODEX_TRANSPORT", "").strip()
    if configured:
        return _normalize_transport(configured)
    return "auto"


def _normalize_transport(value: str) -> TransportName:
    normalized = value.strip().lower()
    if normalized not in {"auto", "cli", "mcp"}:
        raise ValueError("Transport must be one of: auto, cli, mcp.")
    return normalized  # type: ignore[return-value]


def _read_default_model(codex_home: Path) -> str | None:
    config_path = codex_home / "config.toml"
    if not config_path.exists():
        return None
    try:
        import tomllib

        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _read_cache_models(codex_home: Path) -> list[str]:
    cache_path = codex_home / "models_cache.json"
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    ordered: list[tuple[int, str]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            continue
        if item.get("supported_in_api") is False:
            continue
        visibility = item.get("visibility")
        if isinstance(visibility, str) and visibility.strip().lower() == "hidden":
            continue
        priority = item.get("priority")
        rank = int(priority) if isinstance(priority, (int, float)) else 10_000
        ordered.append((rank, slug.strip()))

    ordered.sort(key=lambda item: (item[0], item[1]))
    deduped: list[str] = []
    for _, slug in ordered:
        if slug not in deduped:
            deduped.append(slug)
    return deduped


def available_codex_models() -> list[str]:
    codex_home = resolve_codex_home()
    ordered: list[str] = []
    default_model = _read_default_model(codex_home)
    if default_model:
        ordered.append(default_model)
    for model in _read_cache_models(codex_home):
        if model not in ordered:
            ordered.append(model)
    for model in DEFAULT_CODEX_MODELS:
        if model not in ordered:
            ordered.append(model)
    return ordered


def credential_source() -> str | None:
    for env_key in ("OPENAI_API_KEY",):
        if os.environ.get(env_key, "").strip():
            return f"env:{env_key}"
    auth_path = resolve_codex_home() / "auth.json"
    if auth_path.exists():
        return str(auth_path)
    return None


def _mcp_sdk_available() -> bool:
    try:
        import anyio  # noqa: F401
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


def inspect_codex_runtime(
    model: str | None = None,
    transport: TransportName | None = None,
) -> CodexRuntimeInfo:
    requested_transport = resolve_requested_transport(model=model, transport=transport)
    cli = codex_cli_path()
    available: list[TransportName] = []
    if cli:
        available.append("cli")
    mcp_available = bool(cli) and _mcp_sdk_available()
    if mcp_available:
        available.append("mcp")

    preferred: TransportName | None
    if requested_transport == "auto":
        preferred = "mcp" if "mcp" in available else ("cli" if "cli" in available else None)
    else:
        preferred = requested_transport if requested_transport in available else None

    codex_home = resolve_codex_home()
    auth_path = codex_home / "auth.json"
    cache_path = codex_home / "models_cache.json"
    default_model = _read_default_model(codex_home)

    return CodexRuntimeInfo(
        requested_transport=requested_transport,
        preferred_transport=preferred,
        available_transports=tuple(available),
        cli_path=cli,
        mcp_sdk_available=mcp_available,
        credential_source=credential_source(),
        codex_home=str(codex_home),
        auth_file=str(auth_path) if auth_path.exists() else None,
        models_cache=str(cache_path) if cache_path.exists() else None,
        default_model=default_model,
        available_models=tuple(available_codex_models()),
    )


@contextmanager
def _isolated_codex_home() -> Iterator[Path]:
    source_home = resolve_codex_home()
    with tempfile.TemporaryDirectory(prefix="dspy-codex-home-") as tmpdir:
        overlay = Path(tmpdir)
        for filename in ("auth.json", "models_cache.json", "version.json"):
            source = source_home / filename
            if source.exists():
                shutil.copy2(source, overlay / filename)
        yield overlay


@contextmanager
def _codex_environment(isolate_home: bool = True) -> Iterator[dict[str, str]]:
    env = os.environ.copy()
    if not isolate_home:
        yield env
        return
    with _isolated_codex_home() as overlay:
        env["CODEX_HOME"] = str(overlay)
        yield env


@contextmanager
def _quiet_root_logging(level: int = logging.ERROR) -> Iterator[None]:
    root = logging.getLogger()
    original = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(original)


def _coerce_text_part(part: dict[str, Any], *, role: str) -> str:
    part_type = part.get("type")
    if part_type not in _TEXT_CONTENT_TYPES and "text" not in part:
        raise CodexUnsupportedFeatureError(
            "CodexLM does not support structured non-text content yet. "
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
            raise CodexUnsupportedFeatureError(
                "CodexLM does not support structured non-text content yet. "
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
                raise CodexUnsupportedFeatureError("CodexLM does not support tool-call messages yet.")
            if message.get("tool_call_id"):
                raise CodexUnsupportedFeatureError("CodexLM does not support tool result messages yet.")
            raw_role = str(message.get("role", "user")).lower()
            if raw_role == "tool":
                raise CodexUnsupportedFeatureError("CodexLM does not support tool result messages yet.")
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
    normalized: dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    cached_input_tokens = usage.get("cached_input_tokens")
    if isinstance(cached_input_tokens, (int, float)) and int(cached_input_tokens) > 0:
        normalized["prompt_tokens_details"] = {"cached_tokens": int(cached_input_tokens)}
    return normalized


def _parse_codex_exec_events(stdout: str) -> tuple[str, dict[str, Any], str | None]:
    final_text = ""
    parsed_usage: dict[str, Any] = {}
    thread_id: str | None = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        event = json.loads(line)
        event_type = event.get("type")
        if event_type == "thread.started":
            value = event.get("thread_id")
            thread_id = str(value) if value else None
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                final_text = str(item.get("text", ""))
        elif event_type == "turn.completed":
            parsed_usage = _coerce_usage(event.get("usage"))

    if not final_text.strip():
        raise CodexTransportError("codex exec returned no final agent message.")

    return final_text, parsed_usage, thread_id


def _extract_mcp_text(result: Any) -> tuple[str, str | None, dict[str, Any]]:
    usage = _coerce_usage(getattr(result, "usage", None))
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        content = structured.get("content")
        thread_id = structured.get("threadId")
        usage = usage or _coerce_usage(structured.get("usage"))
        if content:
            return str(content), str(thread_id) if thread_id else None, usage

    contents = getattr(result, "content", []) or []
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))

    content = "\n".join(part for part in parts if part).strip()
    if not content:
        raise CodexTransportError("codex MCP tool returned no text content.")
    return content, None, usage


def _build_codex_exec_command(
    *,
    repo_root: Path,
    codex_model: str | None,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"],
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"],
) -> list[str]:
    cli = codex_cli_path()
    if not cli:
        raise CodexTransportError("codex CLI is not installed or not on PATH.")

    command = [
        cli,
        "-a",
        approval_policy,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
        "--ephemeral",
        "-C",
        str(repo_root),
    ]
    if codex_model:
        command.extend(["-m", codex_model])
    return command


def _resolve_run_attempts(runtime: CodexRuntimeInfo) -> list[TransportName]:
    if runtime.requested_transport != "auto":
        return [runtime.requested_transport]

    attempts = list(runtime.available_transports)
    if "mcp" in attempts:
        attempts.remove("mcp")
        attempts.insert(0, "mcp")
    return attempts


def _resolved_response_model(spec: CodexModelSpec, runtime: CodexRuntimeInfo) -> str | None:
    if spec.codex_model:
        return spec.codex_model
    if runtime.default_model:
        return runtime.default_model
    if runtime.available_models:
        return runtime.available_models[0]
    return None


def run_codex_cli(
    *,
    prompt: str,
    repo_root: Path,
    codex_model: str | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    command = _build_codex_exec_command(
        repo_root=repo_root,
        codex_model=codex_model,
        sandbox=sandbox,
        approval_policy=approval_policy,
    )
    command.append(prompt)

    with _codex_environment(isolate_home=isolate_home) as env:
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
        raise CodexTransportError(message or f"codex exec exited with code {completed.returncode}.")

    text, usage, thread_id = _parse_codex_exec_events(completed.stdout)
    return CodexResult(
        content=text,
        transport="cli",
        usage=usage,
        thread_id=thread_id,
        stderr=completed.stderr,
    )


async def arun_codex_cli(
    *,
    prompt: str,
    repo_root: Path,
    codex_model: str | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    command = _build_codex_exec_command(
        repo_root=repo_root,
        codex_model=codex_model,
        sandbox=sandbox,
        approval_policy=approval_policy,
    )
    command.append(prompt)

    with _codex_environment(isolate_home=isolate_home) as env:
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
        raise CodexTransportError(message or f"codex exec exited with code {process.returncode}.")

    text, usage, thread_id = _parse_codex_exec_events(stdout_text)
    return CodexResult(
        content=text,
        transport="cli",
        usage=usage,
        thread_id=thread_id,
        stderr=stderr_text,
    )


async def arun_codex_mcp(
    *,
    prompt: str,
    repo_root: Path,
    codex_model: str | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    cli = codex_cli_path()
    if not cli:
        raise CodexTransportError("codex CLI is not installed or not on PATH.")

    try:
        import anyio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise CodexTransportError(
            "MCP transport requires the optional dspy[mcp] dependency set."
        ) from exc

    with _codex_environment(isolate_home=isolate_home) as env, _quiet_root_logging():
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "cwd": str(repo_root),
            "sandbox": sandbox,
            "approval-policy": approval_policy,
        }
        if codex_model:
            arguments["model"] = codex_model

        server = StdioServerParameters(command=cli, args=["mcp-server"], env=env)
        with anyio.fail_after(timeout_seconds):
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("codex", arguments=arguments)
                    text, thread_id, usage = _extract_mcp_text(result)
                    return CodexResult(
                        content=text,
                        transport="mcp",
                        usage=usage,
                        thread_id=thread_id,
                    )


def run_codex_mcp(
    *,
    prompt: str,
    repo_root: Path,
    codex_model: str | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    try:
        import anyio
    except ImportError as exc:
        raise CodexTransportError(
            "MCP transport requires the optional dspy[mcp] dependency set."
        ) from exc

    async def _runner() -> CodexResult:
        return await arun_codex_mcp(
            prompt=prompt,
            repo_root=repo_root,
            codex_model=codex_model,
            timeout_seconds=timeout_seconds,
            sandbox=sandbox,
            approval_policy=approval_policy,
            isolate_home=isolate_home,
        )

    return anyio.run(_runner)


def run_codex(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_CODEX_MODEL,
    transport: TransportName | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    spec = parse_codex_model(model)
    runtime = inspect_codex_runtime(model=model, transport=transport)

    if runtime.preferred_transport is None:
        raise CodexTransportError(
            "No usable Codex transport is configured. Run scripts/dspy_codex_doctor.py first."
        )

    requested = runtime.requested_transport
    attempts = _resolve_run_attempts(runtime)
    resolved_model = _resolved_response_model(spec, runtime)

    errors: list[str] = []
    failed_transport: TransportName | None = None
    for attempt in attempts:
        try:
            if attempt == "mcp":
                result = run_codex_mcp(
                    prompt=prompt,
                    repo_root=repo_root,
                    codex_model=spec.codex_model,
                    timeout_seconds=timeout_seconds,
                    sandbox=sandbox,
                    approval_policy=approval_policy,
                    isolate_home=isolate_home,
                )
            else:
                result = run_codex_cli(
                    prompt=prompt,
                    repo_root=repo_root,
                    codex_model=spec.codex_model,
                    timeout_seconds=timeout_seconds,
                    sandbox=sandbox,
                    approval_policy=approval_policy,
                    isolate_home=isolate_home,
                )
            result = replace(result, resolved_model=resolved_model)
            if failed_transport:
                return replace(result, fallback_from=failed_transport)
            return result
        except Exception as exc:
            errors.append(f"{attempt}: {exc}")
            failed_transport = attempt
            if requested != "auto":
                break

    raise CodexTransportError(" ; ".join(errors))


async def arun_codex(
    *,
    prompt: str,
    repo_root: Path,
    model: str = DEFAULT_CODEX_MODEL,
    transport: TransportName | None = None,
    timeout_seconds: int = 120,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
    approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
    isolate_home: bool = True,
) -> CodexResult:
    spec = parse_codex_model(model)
    runtime = inspect_codex_runtime(model=model, transport=transport)

    if runtime.preferred_transport is None:
        raise CodexTransportError(
            "No usable Codex transport is configured. Run scripts/dspy_codex_doctor.py first."
        )

    requested = runtime.requested_transport
    attempts = _resolve_run_attempts(runtime)
    resolved_model = _resolved_response_model(spec, runtime)

    errors: list[str] = []
    failed_transport: TransportName | None = None
    for attempt in attempts:
        try:
            if attempt == "mcp":
                result = await arun_codex_mcp(
                    prompt=prompt,
                    repo_root=repo_root,
                    codex_model=spec.codex_model,
                    timeout_seconds=timeout_seconds,
                    sandbox=sandbox,
                    approval_policy=approval_policy,
                    isolate_home=isolate_home,
                )
            else:
                result = await arun_codex_cli(
                    prompt=prompt,
                    repo_root=repo_root,
                    codex_model=spec.codex_model,
                    timeout_seconds=timeout_seconds,
                    sandbox=sandbox,
                    approval_policy=approval_policy,
                    isolate_home=isolate_home,
                )
            result = replace(result, resolved_model=resolved_model)
            if failed_transport:
                return replace(result, fallback_from=failed_transport)
            return result
        except Exception as exc:
            errors.append(f"{attempt}: {exc}")
            failed_transport = attempt
            if requested != "auto":
                break

    raise CodexTransportError(" ; ".join(errors))


def probe_codex_runtime(
    *,
    repo_root: Path,
    model: str = DEFAULT_CODEX_MODEL,
    transport: TransportName | None = None,
    timeout_seconds: int = 120,
) -> CodexResult:
    return run_codex(
        prompt=DEFAULT_PROBE_PROMPT,
        repo_root=repo_root,
        model=model,
        transport=transport,
        timeout_seconds=timeout_seconds,
    )


class CodexLM(BaseLM):
    """DSPy `BaseLM` wrapper that routes requests through local Codex runtimes."""

    _CACHE_BUSTING_KWARGS = frozenset({"temperature", "rollout_id"})
    _UNSET_UNSUPPORTED_KWARGS = frozenset({"temperature", "max_tokens", "rollout_id"})

    def __init__(
        self,
        model: str = DEFAULT_CODEX_MODEL,
        *,
        repo_root: str | Path,
        transport: TransportName | None = None,
        sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "read-only",
        approval_policy: Literal["untrusted", "on-failure", "on-request", "never"] = "never",
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
        self.transport = transport
        self.sandbox = sandbox
        self.approval_policy = approval_policy
        self.isolate_home = isolate_home
        self.model_spec = parse_codex_model(model)
        self._validate_runtime_kwargs(dict(self.kwargs), cache=self.cache)
        self.kwargs = self._canonicalize_kwargs(self.kwargs)

    @staticmethod
    def _validate_runtime_kwargs(kwargs: dict[str, Any], *, cache: bool) -> None:
        errors: list[str] = []

        if cache:
            errors.append(_UNSUPPORTED_KWARG_MESSAGES["cache"])

        for key, value in kwargs.items():
            if key == "timeout_seconds":
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
            errors.append(f"CodexLM does not support the `{key}` kwarg.")

        if errors:
            raise CodexUnsupportedFeatureError(" ".join(dict.fromkeys(errors)))

    @classmethod
    def _canonicalize_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        canonical = dict(kwargs)
        for key in cls._UNSET_UNSUPPORTED_KWARGS:
            if canonical.get(key) is None:
                canonical.pop(key, None)
        return canonical

    def copy(self, **kwargs: Any) -> "CodexLM":
        stripped = {k: kwargs.pop(k) for k in list(kwargs) if k in self._CACHE_BUSTING_KWARGS}
        if stripped:
            logger.debug("CodexLM.copy(): stripped cache-busting kwargs %s (no effect on Codex)", stripped)
        copied = cast(CodexLM, super().copy(**kwargs))
        copied.kwargs = self._canonicalize_kwargs(copied.kwargs)
        self._validate_runtime_kwargs(dict(copied.kwargs), cache=copied.cache)
        return copied

    def _prepare_call(
        self,
        *,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> CodexCallOptions:
        call_kwargs = dict(kwargs)
        cache_value = call_kwargs.pop("cache", self.cache)
        transport = cast(TransportName | None, call_kwargs.pop("transport", self.transport))
        sandbox = call_kwargs.pop("sandbox", self.sandbox)
        approval_policy = call_kwargs.pop("approval_policy", self.approval_policy)
        isolate_home = bool(call_kwargs.pop("isolate_home", self.isolate_home))
        merged_kwargs = {**self.kwargs, **call_kwargs}
        self._validate_runtime_kwargs(dict(merged_kwargs), cache=bool(cache_value))

        timeout_seconds = int(merged_kwargs.pop("timeout_seconds"))
        return CodexCallOptions(
            prompt=build_prompt(prompt=prompt, messages=messages),
            transport=transport,
            timeout_seconds=timeout_seconds,
            sandbox=sandbox,
            approval_policy=approval_policy,
            isolate_home=isolate_home,
        )

    def _record_usage(self, usage: dict[str, Any]) -> None:
        if usage and dspy_settings.usage_tracker:
            dspy_settings.usage_tracker.add_usage(self.model, dict(usage))

    def _to_response(self, result: CodexResult) -> Any:
        self._record_usage(result.usage)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=result.content))],
            usage=dict(result.usage),
            model=result.resolved_model or self.model,
            _hidden_params={
                "transport": result.transport,
                "thread_id": result.thread_id,
                "stderr": result.stderr,
                "fallback_from": result.fallback_from,
            },
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        options = self._prepare_call(prompt=prompt, messages=messages, kwargs=kwargs)
        result = run_codex(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            transport=options.transport,
            timeout_seconds=options.timeout_seconds,
            sandbox=options.sandbox,
            approval_policy=options.approval_policy,
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
        result = await arun_codex(
            prompt=options.prompt,
            repo_root=self.repo_root,
            model=self.model,
            transport=options.transport,
            timeout_seconds=options.timeout_seconds,
            sandbox=options.sandbox,
            approval_policy=options.approval_policy,
            isolate_home=options.isolate_home,
        )
        return self._to_response(result)


def install_codex_skill(
    *,
    codex_home: str | Path | None = None,
    force_relink: bool = False,
) -> dict[str, str]:
    source = Path(__file__).resolve().parents[2] / "skills" / "dspy-codex"
    if not source.exists():
        raise FileNotFoundError(f"Skill source not found: {source}")

    home = Path(codex_home).expanduser() if codex_home else resolve_codex_home()
    destination_root = home / "skills"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "dspy-codex"

    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return {
                "status": "unchanged",
                "source": str(source),
                "destination": str(destination),
            }
        if not force_relink:
            raise FileExistsError(
                f"{destination} already points to {destination.resolve()}. "
                "Pass force_relink=True to replace it."
            )
        destination.unlink()

    elif destination.exists():
        raise FileExistsError(
            f"{destination} already exists and is not a symlink. Move it aside before installing."
        )

    try:
        destination.symlink_to(source, target_is_directory=True)
        status = "linked"
    except OSError:
        shutil.copytree(source, destination)
        status = "copied"

    return {
        "status": status,
        "source": str(source),
        "destination": str(destination),
    }


__all__ = [
    "CodexLM",
    "CodexModelSpec",
    "CodexRuntimeInfo",
    "CodexResult",
    "CodexTransportError",
    "CodexUnsupportedFeatureError",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_PROBE_PROMPT",
    "arun_codex",
    "arun_codex_cli",
    "arun_codex_mcp",
    "available_codex_models",
    "build_prompt",
    "codex_cli_path",
    "credential_source",
    "inspect_codex_runtime",
    "install_codex_skill",
    "is_codex_model",
    "parse_codex_model",
    "probe_codex_runtime",
    "resolve_codex_home",
    "resolve_requested_transport",
    "run_codex",
    "run_codex_cli",
    "run_codex_mcp",
]
