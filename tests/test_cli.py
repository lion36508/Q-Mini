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
