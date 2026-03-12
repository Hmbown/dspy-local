from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients.codex import (
    DEFAULT_USAGE,
    CodexLM,
    CodexResult,
    CodexUnsupportedFeatureError,
    _extract_mcp_text,
    _parse_codex_exec_events,
    arun_codex,
    available_codex_models,
    build_prompt,
    install_codex_skill,
    parse_codex_model,
    resolve_requested_transport,
    run_codex,
    run_codex_cli,
)


def test_parse_codex_model_aliases() -> None:
    assert parse_codex_model("codex/default").transport_hint == "auto"
    assert parse_codex_model("codex-exec/default").transport_hint == "cli"
    assert parse_codex_model("codex-mcp/gpt-5.3-codex").codex_model == "gpt-5.3-codex"


def test_resolve_requested_transport_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSPY_CODEX_TRANSPORT", "mcp")
    assert resolve_requested_transport(model="codex/default") == "mcp"
    assert resolve_requested_transport(model="codex-exec/default") == "cli"
    assert resolve_requested_transport(model="codex/default", transport="cli") == "cli"


def test_available_codex_models_reads_local_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.toml").write_text('model = "gpt-5.3-codex"\n', encoding="utf-8")
    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-5.4", "priority": 1},
                    {"slug": "gpt-5.3-codex", "priority": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    models = available_codex_models()
    assert models[:2] == ["gpt-5.3-codex", "gpt-5.4"]


def test_parse_codex_exec_events_extracts_text_usage_and_thread() -> None:
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "ready"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 11, "cached_input_tokens": 3, "output_tokens": 7},
                }
            ),
        ]
    )
    text, usage, thread_id = _parse_codex_exec_events(stdout)
    assert text == "ready"
    assert thread_id == "thread-1"
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


def test_extract_mcp_text_prefers_structured_content() -> None:
    result = SimpleNamespace(
        structuredContent={"threadId": "thread-1", "content": "ready"},
        content=[],
        usage=None,
    )
    text, thread_id, usage = _extract_mcp_text(result)
    assert text == "ready"
    assert thread_id == "thread-1"
    assert usage == {}


def test_build_prompt_rejects_structured_non_text_content() -> None:
    with pytest.raises(CodexUnsupportedFeatureError, match="image_url"):
        build_prompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    ],
                }
            ]
        )


def test_build_prompt_rejects_tool_messages() -> None:
    with pytest.raises(CodexUnsupportedFeatureError, match="tool"):
        build_prompt(messages=[{"role": "assistant", "content": "x", "tool_calls": [{"id": "call_1"}]}])


def test_run_codex_auto_falls_back_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.inspect_codex_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            requested_transport="auto",
            preferred_transport="mcp",
            available_transports=("mcp", "cli"),
            default_model="gpt-5.4",
            available_models=("gpt-5.4",),
        ),
    )
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex_mcp",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("mcp failed")),
    )
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex_cli",
        lambda **kwargs: CodexResult(content="ready", transport="cli", usage=dict(DEFAULT_USAGE)),
    )

    result = run_codex(prompt="ready", repo_root=Path.cwd(), model="codex/default")
    assert result.transport == "cli"
    assert result.fallback_from == "mcp"
    assert result.resolved_model == "gpt-5.4"


@pytest.mark.asyncio
async def test_arun_codex_auto_falls_back_to_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.inspect_codex_runtime",
        lambda *args, **kwargs: SimpleNamespace(
            requested_transport="auto",
            preferred_transport="mcp",
            available_transports=("mcp", "cli"),
            default_model="gpt-5.4",
            available_models=("gpt-5.4",),
        ),
    )

    async def fail_mcp(**kwargs):
        raise RuntimeError("mcp failed")

    async def ok_cli(**kwargs):
        return CodexResult(content="ready", transport="cli", usage=dict(DEFAULT_USAGE))

    monkeypatch.setattr("dspy.clients.codex.arun_codex_mcp", fail_mcp)
    monkeypatch.setattr("dspy.clients.codex.arun_codex_cli", ok_cli)

    result = await arun_codex(prompt="ready", repo_root=Path.cwd(), model="codex/default")
    assert result.transport == "cli"
    assert result.fallback_from == "mcp"
    assert result.resolved_model == "gpt-5.4"


def test_run_codex_cli_passes_approval_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ready"}}),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}}),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("dspy.clients.codex.codex_cli_path", lambda: "/tmp/codex")
    monkeypatch.setattr("dspy.clients.codex.subprocess.run", fake_run)

    result = run_codex_cli(
        prompt="ready",
        repo_root=Path.cwd(),
        approval_policy="on-request",
    )

    assert result.thread_id == "thread-1"
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("-a") + 1] == "on-request"


def test_codex_lm_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: CodexResult(
            content="answer",
            transport="cli",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            thread_id="thread-1",
            stderr="stderr text",
            resolved_model="gpt-5.4",
        ),
    )
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    response = lm.forward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.usage["total_tokens"] == 5
    assert response.model == "gpt-5.4"
    assert response._hidden_params["thread_id"] == "thread-1"
    assert response._hidden_params["stderr"] == "stderr text"


@pytest.mark.asyncio
async def test_codex_lm_aforward_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun_codex(**kwargs):
        return CodexResult(
            content="answer",
            transport="mcp",
            usage={},
            thread_id="thread-1",
            fallback_from="cli",
            resolved_model="gpt-5.4",
        )

    monkeypatch.setattr("dspy.clients.codex.arun_codex", fake_arun_codex)
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    response = await lm.aforward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.model == "gpt-5.4"
    assert response._hidden_params["transport"] == "mcp"
    assert response._hidden_params["fallback_from"] == "cli"


@pytest.mark.asyncio
async def test_codex_lm_acall_works_and_records_history(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun_codex(**kwargs):
        return CodexResult(
            content="answer",
            transport="cli",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="gpt-5.4",
        )

    monkeypatch.setattr("dspy.clients.codex.arun_codex", fake_arun_codex)
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    outputs = await lm.acall(prompt="question")
    assert outputs == ["answer"]
    assert lm.history[-1]["outputs"] == ["answer"]
    assert lm.history[-1]["response_model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_codex_lm_supports_async_predict_path(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun_codex(**kwargs):
        return CodexResult(
            content="[[ ## answer ## ]]\nready\n\n[[ ## completed ## ]]\n",
            transport="cli",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="gpt-5.4",
        )

    monkeypatch.setattr("dspy.clients.codex.arun_codex", fake_arun_codex)
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    predict = dspy.Predict("question -> answer")
    with dspy.context(lm=lm):
        result = await predict.acall(question="Reply with exactly one word: ready")
    assert result.answer == "ready"
    assert lm.history[-1]["response_model"] == "gpt-5.4"


def test_track_usage_records_codex_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: CodexResult(
            content="answer",
            transport="cli",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="gpt-5.4",
        ),
    )
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())

    with dspy.track_usage() as tracker:
        assert lm(prompt="question") == ["answer"]

    assert tracker.usage_data["codex/default"][0]["total_tokens"] == 5


def test_track_usage_skips_unknown_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: CodexResult(
            content="answer",
            transport="mcp",
            usage={},
            resolved_model="gpt-5.4",
        ),
    )
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())

    with dspy.track_usage() as tracker:
        assert lm(prompt="question") == ["answer"]

    assert tracker.usage_data == {}


def test_codex_lm_supported_runtime_kwargs_are_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_codex(**kwargs):
        captured.update(kwargs)
        return CodexResult(content="answer", transport="cli", usage=dict(DEFAULT_USAGE))

    monkeypatch.setattr("dspy.clients.codex.run_codex", fake_run_codex)
    lm = CodexLM(
        model="codex/default",
        repo_root=Path.cwd(),
        transport="cli",
        sandbox="workspace-write",
        approval_policy="on-request",
        isolate_home=False,
        timeout_seconds=45,
    )

    lm.forward(
        prompt="question",
        transport="mcp",
        sandbox="read-only",
        approval_policy="never",
        isolate_home=True,
        timeout_seconds=7,
    )

    assert captured["transport"] == "mcp"
    assert captured["sandbox"] == "read-only"
    assert captured["approval_policy"] == "never"
    assert captured["isolate_home"] is True
    assert captured["timeout_seconds"] == 7


@pytest.mark.parametrize(
    ("constructor_kwargs", "call_kwargs", "match"),
    [
        ({"cache": True}, {}, "cache"),
        ({}, {"temperature": 0.7}, "temperature"),
        ({}, {"max_tokens": 32}, "max_tokens"),
        ({}, {"rollout_id": 3}, "rollout_id"),
        ({}, {"tools": [{"name": "search"}]}, "tool calling"),
    ],
)
def test_codex_lm_unsupported_kwargs_error_clearly(
    monkeypatch: pytest.MonkeyPatch,
    constructor_kwargs: dict[str, object],
    call_kwargs: dict[str, object],
    match: str,
) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_codex should not be called")),
    )

    if constructor_kwargs:
        with pytest.raises(CodexUnsupportedFeatureError, match=match):
            CodexLM(model="codex/default", repo_root=Path.cwd(), **constructor_kwargs)
        return

    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    with pytest.raises(CodexUnsupportedFeatureError, match=match):
        lm.forward(prompt="question", **call_kwargs)


def test_codex_lm_rejects_non_text_messages_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_codex should not be called")),
    )
    lm = CodexLM(model="codex/default", repo_root=Path.cwd())

    with pytest.raises(CodexUnsupportedFeatureError, match="image_url"):
        lm.forward(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                    ],
                }
            ]
        )


def test_codex_lm_copy_strips_cache_busting_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.codex.run_codex",
        lambda **kwargs: CodexResult(
            content="answer",
            transport="cli",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="gpt-5.4",
        ),
    )

    lm = CodexLM(model="codex/default", repo_root=Path.cwd())
    copied = lm.copy(rollout_id=1, temperature=1.0)
    assert "temperature" not in copied.kwargs or copied.kwargs.get("temperature") is None
    assert "rollout_id" not in copied.kwargs

    response = copied.forward(prompt="question")
    assert response.choices[0].message.content == "answer"


def test_install_codex_skill_links_into_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    result = install_codex_skill()
    destination = Path(result["destination"])
    assert destination.exists()
    assert destination.is_symlink() or destination.is_dir()
