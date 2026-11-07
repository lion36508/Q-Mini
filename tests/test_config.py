import json
import pytest

from qmini.config import QMiniConfig


def test_config_round_trip(tmp_path):
    original = QMiniConfig(
        vocab_size=512, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
    )
    path = original.save(tmp_path / "config.json")
    restored = QMiniConfig.from_json(path)
    assert restored.to_dict() == original.to_dict()


def test_unknown_config_keys_fail_loudly():
    with pytest.raises(ValueError, match="Unknown Q-Mini config keys"):
        QMiniConfig.from_dict({"vocab_size": 128, "hidden_szie": 256})


def test_dilation_count_must_match_layers():
    with pytest.raises(ValueError, match="dilations"):
        QMiniConfig.from_dict({"n_layers": 3, "dilations": [1, 2]})
