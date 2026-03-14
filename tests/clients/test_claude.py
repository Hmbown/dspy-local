from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients.claude import (
    ClaudeLM,
    ClaudeResult,
    ClaudeUnsupportedFeatureError,
    PromptParts,
    _parse_claude_result,
    build_prompt,
    inspect_claude_runtime,
    parse_claude_model,
    run_claude,
    run_claude_cli,
)


def test_parse_claude_model_aliases() -> None:
    assert parse_claude_model("claude/default").claude_model is None
    assert parse_claude_model("claude/sonnet").claude_model == "sonnet"
    assert parse_claude_model("claude/opus").claude_model == "opus"
    assert parse_claude_model("claude/claude-sonnet-4-6").claude_model == "claude-sonnet-4-6"


def test_inspect_claude_runtime_reads_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text("{}", encoding="utf-8")
    (claude_dir / ".credentials.json").write_text("{}", encoding="utf-8")

    runtime = inspect_claude_runtime(home_dir=tmp_path)
    assert runtime.settings_file is not None
    assert runtime.credentials_configured is True
    assert str(claude_dir / ".credentials.json") in runtime.credential_sources


def test_inspect_claude_runtime_accepts_legacy_credentials_filename(tmp_path: Path) -> None:
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "credentials.json").write_text("{}", encoding="utf-8")

    runtime = inspect_claude_runtime(home_dir=tmp_path)
    assert runtime.credentials_configured is True
    assert str(claude_dir / "credentials.json") in runtime.credential_sources


def test_inspect_claude_runtime_does_not_treat_settings_file_as_credentials(tmp_path: Path) -> None:
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text("{}", encoding="utf-8")

    runtime = inspect_claude_runtime(home_dir=tmp_path)
    assert runtime.settings_file is not None
    assert runtime.credentials_configured is False
    assert runtime.credential_sources == ()


def test_parse_claude_result_extracts_text_usage_session_and_model() -> None:
    payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "duration_ms": 2227,
            "num_turns": 1,
            "result": "ready",
            "session_id": "session-1",
            "total_cost_usd": 0.085,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
            },
            "modelUsage": {
                "claude-opus-4-6[1m]": {
                    "inputTokens": 11,
                    "outputTokens": 7,
                }
            },
        }
    )
    text, usage, session_id, model, cost_usd = _parse_claude_result(payload)
    assert text == "ready"
    assert session_id == "session-1"
    assert model == "claude-opus-4-6[1m]"
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}
    assert cost_usd == 0.085


def test_parse_claude_result_raises_on_error() -> None:
    payload = json.dumps(
        {
            "type": "result",
            "subtype": "error",
            "result": "something went wrong",
        }
    )
    from dspy.clients.claude import ClaudeTransportError

    with pytest.raises(ClaudeTransportError, match="something went wrong"):
        _parse_claude_result(payload)


def test_run_claude_cli_sets_permission_mode_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    home_dir = tmp_path / "home"
    claude_dir = home_dir / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.json").write_text("{}", encoding="utf-8")
    (claude_dir / ".credentials.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "ready",
                    "session_id": "session-1",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                    "modelUsage": {"claude-sonnet-4-6": {"inputTokens": 2, "outputTokens": 1}},
                    "total_cost_usd": 0.01,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("dspy.clients.claude.claude_cli_path", lambda: "/tmp/claude")
    monkeypatch.setattr("dspy.clients.claude.subprocess.run", fake_run)

    result = run_claude_cli(
        prompt="ready",
        repo_root=Path.cwd(),
        claude_model="sonnet",
        isolate_home=True,
    )

    assert result.session_id == "session-1"
    assert result.resolved_model == "claude-sonnet-4-6"
    command = captured["command"]
    assert isinstance(command, list)
    assert "--permission-mode" in command
    assert "--model" in command
    assert "sonnet" in command
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["HOME"] != str(home_dir)


def test_claude_lm_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.claude.run_claude",
        lambda **kwargs: ClaudeResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            session_id="session-1",
            stderr="stderr text",
            resolved_model="claude-sonnet-4-6",
            cost_usd=0.01,
        ),
    )
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())
    response = lm.forward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.usage["total_tokens"] == 5
    assert response.model == "claude-sonnet-4-6"
    assert response._hidden_params["session_id"] == "session-1"
    assert response._hidden_params["stderr"] == "stderr text"
    assert response._hidden_params["cost_usd"] == 0.01


@pytest.mark.asyncio
async def test_claude_lm_aforward_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun_claude(**kwargs):
        return ClaudeResult(
            content="answer",
            usage={},
            session_id="session-1",
            resolved_model="claude-sonnet-4-6",
        )

    monkeypatch.setattr("dspy.clients.claude.arun_claude", fake_arun_claude)
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())
    response = await lm.aforward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.model == "claude-sonnet-4-6"
    assert response._hidden_params["transport"] == "cli"


def test_track_usage_records_claude_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.claude.run_claude",
        lambda **kwargs: ClaudeResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="claude-sonnet-4-6",
        ),
    )
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())

    with dspy.track_usage() as tracker:
        assert lm(prompt="question") == ["answer"]

    assert tracker.usage_data["claude/default"][0]["total_tokens"] == 5


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
def test_claude_lm_unsupported_kwargs_error_clearly(
    monkeypatch: pytest.MonkeyPatch,
    constructor_kwargs: dict[str, object],
    call_kwargs: dict[str, object],
    match: str,
) -> None:
    monkeypatch.setattr(
        "dspy.clients.claude.run_claude",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_claude should not be called")),
    )

    if constructor_kwargs:
        with pytest.raises(ClaudeUnsupportedFeatureError, match=match):
            ClaudeLM(model="claude/default", repo_root=Path.cwd(), **constructor_kwargs)
        return

    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())
    with pytest.raises(ClaudeUnsupportedFeatureError, match=match):
        lm.forward(prompt="question", **call_kwargs)


def test_claude_lm_rejects_non_text_messages_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.claude.run_claude",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_claude should not be called")),
    )
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())

    with pytest.raises(ClaudeUnsupportedFeatureError, match="image_url"):
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


def test_claude_lm_does_not_store_unset_unsupported_kwargs() -> None:
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())
    assert "temperature" not in lm.kwargs
    assert "max_tokens" not in lm.kwargs
    assert lm.kwargs == {"timeout_seconds": 120}


def test_claude_lm_copy_strips_cache_busting_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.claude.run_claude",
        lambda **kwargs: ClaudeResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="claude-sonnet-4-6",
        ),
    )

    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())
    copied = lm.copy(rollout_id=1, temperature=1.0)
    assert "temperature" not in copied.kwargs
    assert "max_tokens" not in copied.kwargs
    assert "rollout_id" not in copied.kwargs
    assert copied.kwargs == {"timeout_seconds": 120}

    response = copied.forward(prompt="question")
    assert response.choices[0].message.content == "answer"


def test_run_claude_uses_alias_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_claude_cli(**kwargs):
        captured.update(kwargs)
        return ClaudeResult(content="ready", usage={})

    monkeypatch.setattr("dspy.clients.claude.run_claude_cli", fake_run_claude_cli)
    result = run_claude(prompt="ready", repo_root=Path.cwd(), model="claude/sonnet")
    assert result.content == "ready"
    assert captured["claude_model"] == "sonnet"


# ---------- build_prompt / PromptParts tests ----------


def test_build_prompt_returns_prompt_parts() -> None:
    parts = build_prompt(prompt="hello")
    assert isinstance(parts, PromptParts)
    assert parts.system == ""
    assert parts.prompt == "hello"


def test_build_prompt_separates_system_messages() -> None:
    messages = [
        {"role": "system", "content": "You are a translator."},
        {"role": "user", "content": "Translate this."},
    ]
    parts = build_prompt(messages=messages)
    assert parts.system == "You are a translator."
    assert parts.prompt == "Translate this."
    assert "SYSTEM" not in parts.prompt
    assert "USER" not in parts.prompt


def test_build_prompt_no_role_prefixes_in_prompt() -> None:
    """Verify the prompt never contains suspicious ROLE: markers."""
    messages = [
        {"role": "system", "content": "System instructions here."},
        {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"},
        {"role": "assistant", "content": "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]"},
        {"role": "user", "content": "[[ ## question ## ]]\nWhat is 3+3?"},
    ]
    parts = build_prompt(messages=messages)
    assert parts.system == "System instructions here."
    # No role markers that could trigger injection detection
    assert "SYSTEM:" not in parts.prompt
    assert "USER:" not in parts.prompt
    assert "ASSISTANT:" not in parts.prompt
    # Content is preserved
    assert "[[ ## question ## ]]" in parts.prompt
    assert "[[ ## answer ## ]]" in parts.prompt
    assert "What is 3+3?" in parts.prompt


def test_build_prompt_multiple_system_messages_merged() -> None:
    messages = [
        {"role": "system", "content": "Part 1."},
        {"role": "system", "content": "Part 2."},
        {"role": "user", "content": "Go."},
    ]
    parts = build_prompt(messages=messages)
    assert "Part 1." in parts.system
    assert "Part 2." in parts.system
    assert parts.prompt == "Go."


def test_build_prompt_with_prompt_and_messages() -> None:
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Input data."},
    ]
    parts = build_prompt(prompt="Extra context.", messages=messages)
    assert parts.system == "Be helpful."
    assert "Input data." in parts.prompt
    assert "Extra context." in parts.prompt


# ---------- system_prompt threading tests ----------


def test_system_prompt_passed_to_cli_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that system messages end up as --system-prompt in the CLI command."""
    captured: dict[str, object] = {}

    home_dir = tmp_path / "home"
    claude_dir = home_dir / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / ".credentials.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "answer",
                    "session_id": "s-1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "modelUsage": {"claude-sonnet-4-6": {}},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("dspy.clients.claude.claude_cli_path", lambda: "/tmp/claude")
    monkeypatch.setattr("dspy.clients.claude.subprocess.run", fake_run)

    run_claude_cli(
        prompt="user text",
        repo_root=Path.cwd(),
        system_prompt="system instructions",
        isolate_home=True,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--system-prompt" in command
    idx = command.index("--system-prompt")
    assert command[idx + 1] == "system instructions"


def test_no_system_prompt_flag_when_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify --system-prompt is omitted when there's no system content."""
    captured: dict[str, object] = {}

    home_dir = tmp_path / "home"
    claude_dir = home_dir / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / ".credentials.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "result": "answer",
                    "session_id": "s-1",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "modelUsage": {"claude-sonnet-4-6": {}},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("dspy.clients.claude.claude_cli_path", lambda: "/tmp/claude")
    monkeypatch.setattr("dspy.clients.claude.subprocess.run", fake_run)

    run_claude_cli(
        prompt="user text",
        repo_root=Path.cwd(),
        isolate_home=True,
    )

    command = captured["command"]
    assert "--system-prompt" not in command


def test_claude_lm_forward_threads_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify end-to-end: system messages in DSPy chat format → --system-prompt on CLI."""
    captured: dict[str, object] = {}

    def fake_run_claude(**kwargs):
        captured.update(kwargs)
        return ClaudeResult(
            content="[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            resolved_model="claude-sonnet-4-6",
        )

    monkeypatch.setattr("dspy.clients.claude.run_claude", fake_run_claude)
    lm = ClaudeLM(model="claude/default", repo_root=Path.cwd())

    lm.forward(
        messages=[
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"},
        ]
    )

    assert captured["system_prompt"] == "You are a math tutor."
    assert "SYSTEM" not in captured["prompt"]
    assert "[[ ## question ## ]]" in captured["prompt"]
