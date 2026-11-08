import numpy as np
import pytest
import torch

from qmini.export import verify_packed_export
from qmini.train import train_model


def _write_stream(path, vocab_size, length, seed):
    rng = np.random.default_rng(seed)
    np.asarray(rng.integers(0, vocab_size, size=length), dtype=np.uint16).tofile(path)


def test_smoke_training_checkpoint_resume_and_export(tiny_config, tmp_path):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 2026)
    _write_stream(valid_bin, tiny_config.vocab_size, 192, 2027)
    run = tmp_path / "run"

    final = train_model(
        tiny_config, train_bin, valid_bin, run,
        steps=2, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
    )
    checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["format"] == "qmini-checkpoint-v2"
    assert checkpoint["resume_contract"]["train"]["sha256"]
    assert checkpoint["tokens_seen"] == 16

    train_model(
        tiny_config, train_bin, valid_bin, run,
        steps=3, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        resume=True, extend_schedule=True,
    )
    resumed = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["step"] == 3
    assert resumed["schedule_total_steps"] == 3

    report = verify_packed_export(final / "model.qmini.npz")
    assert report["tensor_count"] == len(resumed["model"])
    assert report["parameter_count"] == sum(value.numel() for value in resumed["model"].values())
    assert (final / "model_manifest.json").is_file()


def test_resume_rejects_changed_training_data(tiny_config, tmp_path):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 10)
    _write_stream(valid_bin, tiny_config.vocab_size, 256, 11)
    run = tmp_path / "run"

    train_model(
        tiny_config, train_bin, valid_bin, run,
        steps=2, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
    )
    _write_stream(train_bin, tiny_config.vocab_size, 256, 99)
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        train_model(
            tiny_config, train_bin, valid_bin, run,
            steps=3, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            resume=True, extend_schedule=True,
        )


def test_training_rejects_out_of_vocab_token_ids(tiny_config, tmp_path):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 1)
    _write_stream(valid_bin, tiny_config.vocab_size, 256, 2)
    values = np.memmap(train_bin, mode="r+", dtype=np.uint16)
    values[0] = tiny_config.vocab_size + 1
    del values

    with pytest.raises(ValueError, match="outside the configured vocabulary"):
        train_model(
            tiny_config, train_bin, valid_bin, tmp_path / "run",
            steps=1, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
        )
