import numpy as np
import torch
from qmini.export import verify_packed_export
from qmini.train import train_model


def test_smoke_training_checkpoint_resume_and_export(tiny_config, tmp_path):
    rng = np.random.default_rng(2026)
    train_bin, valid_bin = tmp_path / "train.bin", tmp_path / "valid.bin"
    np.asarray(rng.integers(0, tiny_config.vocab_size, size=256), dtype=np.uint16).tofile(train_bin)
    np.asarray(rng.integers(0, tiny_config.vocab_size, size=192), dtype=np.uint16).tofile(valid_bin)
    run = tmp_path / "run"
    final = train_model(
        tiny_config, train_bin, valid_bin, run, steps=2, batch_size=1,
        seq_len=8, grad_accum=1, eval_every=1, save_every=1, eval_batches=1,
    )
    checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    train_model(
        tiny_config, train_bin, valid_bin, run, steps=3, batch_size=1,
        seq_len=8, grad_accum=1, eval_every=1, save_every=1,
        eval_batches=1, resume=True,
    )
    resumed = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["step"] == 3
    report = verify_packed_export(final / "model.qmini.npz")
    assert report["tensor_count"] == len(resumed["model"])
    assert report["parameter_count"] == sum(value.numel() for value in resumed["model"].values())
