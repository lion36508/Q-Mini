import argparse
from types import SimpleNamespace

from qmini import cli


def test_cli_main_returns_subcommand_status(monkeypatch):
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(func=lambda _args: 7),
    )

    assert cli.main() == 7


import pytest

from qmini.cli import _validate_thinking_steps


def test_cli_thinking_steps_follow_model_limit_two():
    _validate_thinking_steps(1, 2)
    _validate_thinking_steps(2, 2)
    with pytest.raises(ValueError, match="1..2"):
        _validate_thinking_steps(3, 2)


def test_cli_thinking_steps_allow_custom_model_limit_above_four():
    _validate_thinking_steps(5, 5)
    with pytest.raises(ValueError, match="1..5"):
        _validate_thinking_steps(6, 5)


def test_cli_thinking_steps_accept_model_default():
    _validate_thinking_steps(None, 2)


def test_tokenizer_cli_rejects_zero_validation_fraction_at_public_boundary(
    tmp_path, monkeypatch
):
    import sys
    from qmini.cli import main

    corpus = tmp_path / "cli-corpus.jsonl"
    corpus.write_text(
        '{"text":"First independent tokenizer document."}\n'
        '{"text":"Second independent tokenizer document."}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "tokenizer",
            "--data-dir", str(corpus),
            "--output", str(tmp_path / "tokenizer.json"),
            "--vocab-size", "300",
            "--min-frequency", "1",
            "--config-output", str(tmp_path / "config.json"),
            "--valid-fraction", "0",
        ],
    )
    with pytest.raises(ValueError, match="greater than 0 and below 0.5"):
        main()


def test_eval_conversations_cli_routes_through_structured_evaluator(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    import qmini.cli as cli

    model = SimpleNamespace(
        config=SimpleNamespace(max_thinking_steps=3),
    )
    tokenizer = object()
    expected_report = {"format": "qmini-conversation-evaluation-v1"}
    called = {}

    monkeypatch.setattr(
        cli, "load_model", lambda model_dir, device=None: (
            model, tokenizer, "cpu"
        )
    )

    def fake_evaluate(*args):
        called["args"] = args
        return expected_report

    monkeypatch.setattr(cli, "evaluate_conversation_file", fake_evaluate)
    monkeypatch.setattr(
        cli, "emit_report",
        lambda report, output: called.update({"report": report, "output": output}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "eval-conversations",
            "--model-dir", str(tmp_path / "model"),
            "--dataset", str(tmp_path / "conversations.jsonl"),
            "--thinking-steps", "3",
            "--output", str(tmp_path / "report.json"),
        ],
    )
    assert cli.main() == 0
    assert called["args"][0] is model
    assert called["args"][1] is tokenizer
    assert called["args"][2] == str(tmp_path / "conversations.jsonl")
    assert called["report"] is expected_report
    assert called["output"] == str(tmp_path / "report.json")
