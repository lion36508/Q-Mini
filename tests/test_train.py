import numpy as np
import pytest
import torch

from qmini.export import verify_packed_export
from qmini.train import _masked_cross_entropy, train_model


def _write_stream(path, vocab_size, length, seed):
    rng = np.random.default_rng(seed)
    np.asarray(rng.integers(0, vocab_size, size=length), dtype=np.uint16).tofile(path)


def test_masked_cross_entropy_ignores_context_positions():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]]])
    targets = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[0.0, 1.0, 0.0]])
    first = _masked_cross_entropy(logits, targets, mask)
    changed = logits.clone()
    changed[0, 0] = torch.tensor([-100.0, 100.0])
    changed[0, 2] = torch.tensor([-100.0, 100.0])
    second = _masked_cross_entropy(changed, targets, mask)
    assert torch.allclose(first, second)


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


def test_training_uses_response_masks_and_tracks_supervised_tokens(tiny_config, tmp_path):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    train_mask = tmp_path / "train.mask.bin"
    valid_mask = tmp_path / "valid.mask.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 12)
    _write_stream(valid_bin, tiny_config.vocab_size, 192, 13)
    rng = np.random.default_rng(14)
    np.asarray((rng.random(256) < 0.35).astype(np.uint8)).tofile(train_mask)
    np.asarray((rng.random(192) < 0.35).astype(np.uint8)).tofile(valid_mask)

    final = train_model(
        tiny_config, train_bin, valid_bin, tmp_path / "masked-run",
        steps=1, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        train_mask_bin=train_mask, valid_mask_bin=valid_mask,
    )
    report = torch.load(
        tmp_path / "masked-run" / "latest.pt", map_location="cpu", weights_only=False
    )
    assert report["supervised_tokens_seen"] > 0
    assert report["supervised_tokens_seen"] <= report["tokens_seen"]
    assert (final / "model_manifest.json").is_file()


def test_resume_rejects_changed_loss_mask(tiny_config, tmp_path):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    train_mask = tmp_path / "train.mask.bin"
    valid_mask = tmp_path / "valid.mask.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 32)
    _write_stream(valid_bin, tiny_config.vocab_size, 256, 33)
    np.ones(256, dtype=np.uint8).tofile(train_mask)
    np.ones(256, dtype=np.uint8).tofile(valid_mask)
    run = tmp_path / "masked-resume"

    train_model(
        tiny_config, train_bin, valid_bin, run,
        steps=2, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        train_mask_bin=train_mask, valid_mask_bin=valid_mask,
    )
    changed = np.ones(256, dtype=np.uint8)
    changed[3] = 0
    changed.tofile(train_mask)
    with pytest.raises(ValueError, match="Resume contract mismatch"):
        train_model(
            tiny_config, train_bin, valid_bin, run,
            steps=3, batch_size=1, seq_len=8, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            resume=True, extend_schedule=True,
            train_mask_bin=train_mask, valid_mask_bin=valid_mask,
        )
