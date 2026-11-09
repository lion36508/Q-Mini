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
