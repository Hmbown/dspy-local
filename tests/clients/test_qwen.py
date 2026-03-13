from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients.qwen import (
    QwenLM,
    QwenResult,
    QwenUnsupportedFeatureError,
    _parse_qwen_events,
    inspect_qwen_runtime,
    parse_qwen_model,
    run_qwen,
    run_qwen_cli,
)


def test_parse_qwen_model_aliases() -> None:
    assert parse_qwen_model("qwen/default").qwen_model is None
    assert parse_qwen_model("qwen/qwen-max").qwen_model == "qwen-max"


def test_inspect_qwen_runtime_reads_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    qwen_dir = tmp_path / ".qwen"
    qwen_dir.mkdir()
    (qwen_dir / "settings.json").write_text("{}", encoding="utf-8")
    (qwen_dir / "oauth_creds.json").write_text("{}", encoding="utf-8")
    (qwen_dir / "installation_id").write_text("x", encoding="utf-8")

    runtime = inspect_qwen_runtime()
    assert runtime.settings_file is not None
    assert runtime.oauth_file is not None
    assert runtime.installation_id_file is not None
    assert runtime.credentials_configured is True


def test_parse_qwen_events_extracts_text_usage_session_and_model() -> None:
    payload = json.dumps(
        [
            {
                "type": "system",
                "subtype": "init",
                "session_id": "session-1",
                "model": "coder-model",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "noop"},
                        {"type": "text", "text": "fallback"},
                    ]
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "result": "ready",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "cache_read_input_tokens": 3,
                },
            },
        ]
    )
    text, usage, session_id, model = _parse_qwen_events(payload)
    assert text == "ready"
    assert session_id == "session-1"
    assert model == "coder-model"
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 7
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


def test_run_qwen_cli_sets_home_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    home_dir = tmp_path / "home"
    qwen_dir = home_dir / ".qwen"
    qwen_dir.mkdir(parents=True)
    (qwen_dir / "settings.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home_dir))

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {"type": "system", "subtype": "init", "session_id": "session-1", "model": "coder-model"},
                    {"type": "result", "subtype": "success", "result": "ready", "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}},
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("dspy.clients.qwen.qwen_cli_path", lambda: "/tmp/qwen")
    monkeypatch.setattr("dspy.clients.qwen.subprocess.run", fake_run)

    result = run_qwen_cli(
        prompt="ready",
        repo_root=Path.cwd(),
        qwen_model="qwen-max",
        isolate_home=True,
    )

    assert result.session_id == "session-1"
    assert result.resolved_model == "coder-model"
    command = captured["command"]
    assert isinstance(command, list)
    assert "--approval-mode" in command
    assert "--model" in command
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["HOME"] != str(home_dir)


def test_qwen_lm_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.qwen.run_qwen",
        lambda **kwargs: QwenResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            session_id="session-1",
            stderr="stderr text",
            resolved_model="coder-model",
        ),
    )
    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())
    response = lm.forward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.usage["total_tokens"] == 5
    assert response.model == "coder-model"
    assert response._hidden_params["session_id"] == "session-1"
    assert response._hidden_params["stderr"] == "stderr text"


@pytest.mark.asyncio
async def test_qwen_lm_aforward_returns_openai_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_arun_qwen(**kwargs):
        return QwenResult(
            content="answer",
            usage={},
            session_id="session-1",
            resolved_model="coder-model",
        )

    monkeypatch.setattr("dspy.clients.qwen.arun_qwen", fake_arun_qwen)
    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())
    response = await lm.aforward(prompt="question")
    assert response.choices[0].message.content == "answer"
    assert response.model == "coder-model"
    assert response._hidden_params["transport"] == "cli"


def test_track_usage_records_qwen_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.qwen.run_qwen",
        lambda **kwargs: QwenResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="coder-model",
        ),
    )
    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())

    with dspy.track_usage() as tracker:
        assert lm(prompt="question") == ["answer"]

    assert tracker.usage_data["qwen/default"][0]["total_tokens"] == 5


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
def test_qwen_lm_unsupported_kwargs_error_clearly(
    monkeypatch: pytest.MonkeyPatch,
    constructor_kwargs: dict[str, object],
    call_kwargs: dict[str, object],
    match: str,
) -> None:
    monkeypatch.setattr(
        "dspy.clients.qwen.run_qwen",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_qwen should not be called")),
    )

    if constructor_kwargs:
        with pytest.raises(QwenUnsupportedFeatureError, match=match):
            QwenLM(model="qwen/default", repo_root=Path.cwd(), **constructor_kwargs)
        return

    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())
    with pytest.raises(QwenUnsupportedFeatureError, match=match):
        lm.forward(prompt="question", **call_kwargs)


def test_qwen_lm_rejects_non_text_messages_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.qwen.run_qwen",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("run_qwen should not be called")),
    )
    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())

    with pytest.raises(QwenUnsupportedFeatureError, match="image_url"):
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


def test_qwen_lm_does_not_store_unset_unsupported_kwargs() -> None:
    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())
    assert "temperature" not in lm.kwargs
    assert "max_tokens" not in lm.kwargs
    assert lm.kwargs == {"timeout_seconds": 120}


def test_qwen_lm_copy_strips_cache_busting_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dspy.clients.qwen.run_qwen",
        lambda **kwargs: QwenResult(
            content="answer",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            resolved_model="coder-model",
        ),
    )

    lm = QwenLM(model="qwen/default", repo_root=Path.cwd())
    copied = lm.copy(rollout_id=1, temperature=1.0)
    assert "temperature" not in copied.kwargs
    assert "max_tokens" not in copied.kwargs
    assert "rollout_id" not in copied.kwargs
    assert copied.kwargs == {"timeout_seconds": 120}

    response = copied.forward(prompt="question")
    assert response.choices[0].message.content == "answer"


def test_run_qwen_uses_alias_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_qwen_cli(**kwargs):
        captured.update(kwargs)
        return QwenResult(content="ready", usage={})

    monkeypatch.setattr("dspy.clients.qwen.run_qwen_cli", fake_run_qwen_cli)
    result = run_qwen(prompt="ready", repo_root=Path.cwd(), model="qwen/qwen-max")
    assert result.content == "ready"
    assert captured["qwen_model"] == "qwen-max"
