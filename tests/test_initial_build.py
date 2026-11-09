import json

import torch

from qmini.export import verify_packed_export
from qmini.initial_build import build_initial_brain, iter_initial_state_tensors
from qmini.model import QMiniMoE
from qmini.packed_runtime import load_packed_model


def test_streamed_initial_state_matches_full_model_initialization(tiny_config):
    torch.manual_seed(tiny_config.seed)
    full_model = QMiniMoE(tiny_config)
    expected = full_model.state_dict()
    streamed = dict(iter_initial_state_tensors(tiny_config))

    assert set(streamed) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(streamed[name], tensor), name


def test_initial_build_emits_verified_q2_v2_artifact(tiny_config, tmp_path):
    target_dir = tmp_path / "fresh-q2-initial"
    manifest = build_initial_brain(
        tiny_config, target_dir, source_revision="test-revision"
    )

    assert manifest["format"] == "qmini-initial-brain-v1"
    assert manifest["status"] == "untrained_initialization"
    assert manifest["training_steps"] == 0
    assert manifest["optimizer_steps"] == 0
    assert manifest["tokens_seen"] == 0
    assert manifest["matrix_weight_format"] == "qmini-packed-v2"
    assert manifest["matrix_weight_bits"] == 2
    assert manifest["parameters"] == tiny_config.estimated_parameter_count()
    assert not (target_dir / "model.pt").exists()

    report = verify_packed_export(target_dir / "model.qmini.npz")
    assert report["format"] == "qmini-packed-v2"
    assert report["q2_version"] == 2
    assert report["parameter_count"] == tiny_config.estimated_parameter_count()

    with (target_dir / "model_manifest.json").open(encoding="utf-8") as handle:
        saved_manifest = json.load(handle)
    assert saved_manifest == manifest

    packed_model = load_packed_model(
        target_dir / "model.qmini.npz", expected_config=tiny_config
    )
    assert getattr(packed_model, "_qmini_packed_runtime", False)
    assert all(parameter.ndim < 2 for parameter in packed_model.parameters())


def test_initial_build_refuses_to_overwrite_existing_directory(tiny_config, tmp_path):
    target_dir = tmp_path / "existing"
    target_dir.mkdir()
    sentinel = target_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    import pytest
    with pytest.raises(FileExistsError):
        build_initial_brain(tiny_config, target_dir)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
