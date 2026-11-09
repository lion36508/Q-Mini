import hashlib
import json
import numpy as np
import pytest
import torch

from qmini.export import verify_packed_export
from qmini.train import (
    SupervisedWindowIndex, _evaluate, _masked_cross_entropy,
    _masked_cross_entropy_sum, _sample_batch, train_model,
)
from qmini.data import prepare_token_streams
from qmini.tokenizer import train_tokenizer


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
    assert sum(checkpoint["thinking_step_counts"].values()) == 2
    metrics = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(metrics) == 2
    for record in metrics:
        assert set(record["eval_by_thinking_steps"]) == {
            str(value) for value in range(1, tiny_config.max_thinking_steps + 1)
        }
        assert record["eval_primary_thinking_steps"] == tiny_config.thinking_steps
        assert record["eval_thinking_steps"] == tiny_config.thinking_steps
        assert record["eval_supervised_tokens"] > 0
        assert sum(record["thinking_step_sampling_counts"].values()) == record["step"]

    final = train_model(
        tiny_config, train_bin, valid_bin, run,
        steps=3, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        resume=True, extend_schedule=True,
    )
    resumed = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert resumed["step"] == 3
    assert resumed["schedule_total_steps"] == 3
    assert sum(resumed["thinking_step_counts"].values()) == 3

    report = verify_packed_export(final / "model.qmini.npz")
    assert report["tensor_count"] == len(resumed["model"])
    assert report["parameter_count"] == sum(value.numel() for value in resumed["model"].values())
    assert (final / "model_manifest.json").is_file()
    manifest = json.loads(
        (final / "model_manifest.json").read_text(encoding="utf-8")
    )
    packed_path = final / manifest["packed_export"]
    dense_path = final / "model.pt"
    assert manifest["format"] == "qmini-final-model-v2"
    assert manifest["packed_export_bytes"] == packed_path.stat().st_size
    assert manifest["packed_export_sha256"] == hashlib.sha256(packed_path.read_bytes()).hexdigest()
    assert manifest["dense_snapshot_bytes"] == dense_path.stat().st_size
    assert manifest["dense_snapshot_sha256"] == hashlib.sha256(dense_path.read_bytes()).hexdigest()


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



def _prepare_chat_dataset(tmp_path, valid_fraction=0.25):
    corpus = tmp_path / "chat.jsonl"
    records = [
        {"messages": [
            {"role": "user", "content": f"Question number {index}: explain a scientific idea in detail."},
            {"role": "assistant", "content": f"Answer number {index}: this verified response contains enough text to create a training sequence."},
        ]}
        for index in range(8)
    ]
    corpus.write_text(
        "\n".join(__import__("json").dumps(item) for item in records),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_tokenizer(
        [corpus], tokenizer_path, vocab_size=300, min_frequency=1,
        validation_fraction=valid_fraction,
    )
    token_dir = tmp_path / "prepared"
    report = prepare_token_streams(
        [corpus], tokenizer_path, token_dir,
        valid_fraction=valid_fraction, loss_mask_mode="auto",
    )
    return tokenizer, tokenizer_path, token_dir, report


def test_training_requires_manifest_assistant_masks(tiny_config, tmp_path):
    tokenizer, tokenizer_path, token_dir, report = _prepare_chat_dataset(tmp_path)
    tiny_config.vocab_size = tokenizer.get_vocab_size()
    tiny_config.validate()
    assert report["loss_mask_mode"] == "auto"

    with pytest.raises(ValueError, match="both train and validation loss-mask files are mandatory"):
        train_model(
            tiny_config,
            token_dir / "train.bin",
            token_dir / "valid.bin",
            tmp_path / "run",
            steps=1, batch_size=1, seq_len=2, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            tokenizer_path=tokenizer_path,
        )


def test_training_rejects_tokenizer_manifest_mismatch(tiny_config, tmp_path):
    tokenizer, tokenizer_path, token_dir, report = _prepare_chat_dataset(tmp_path)
    tiny_config.vocab_size = tokenizer.get_vocab_size()
    tiny_config.validate()
    wrong_tokenizer = tmp_path / "wrong-tokenizer.json"
    wrong_tokenizer.write_bytes(tokenizer_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Tokenizer SHA-256"):
        train_model(
            tiny_config,
            token_dir / "train.bin",
            token_dir / "valid.bin",
            tmp_path / "run",
            steps=1, batch_size=1, seq_len=2, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            tokenizer_path=wrong_tokenizer,
            train_mask_bin=token_dir / "train.mask.bin",
            valid_mask_bin=token_dir / "valid.mask.bin",
        )


def test_training_rejects_modified_prepared_token_stream(tiny_config, tmp_path):
    tokenizer, tokenizer_path, token_dir, report = _prepare_chat_dataset(tmp_path)
    tiny_config.vocab_size = tokenizer.get_vocab_size()
    tiny_config.validate()
    values = np.memmap(token_dir / "train.bin", mode="r+", dtype=np.uint16)
    values[0] = (int(values[0]) + 1) % tiny_config.vocab_size
    del values

    with pytest.raises(ValueError, match="Training token stream SHA-256"):
        train_model(
            tiny_config,
            token_dir / "train.bin",
            token_dir / "valid.bin",
            tmp_path / "run",
            steps=1, batch_size=1, seq_len=2, grad_accum=1,
            eval_every=1, save_every=1, eval_batches=1,
            tokenizer_path=tokenizer_path,
            train_mask_bin=token_dir / "train.mask.bin",
            valid_mask_bin=token_dir / "valid.mask.bin",
        )

def test_masked_cross_entropy_uses_token_weight_across_microbatches():
    first_logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    first_targets = torch.tensor([[0, 1]])
    first_mask = torch.tensor([[1.0, 0.0]])
    second_logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]])
    second_targets = torch.tensor([[1, 0, 1]])
    second_mask = torch.tensor([[1.0, 1.0, 1.0]])

    first_sum, first_count = _masked_cross_entropy_sum(
        first_logits, first_targets, first_mask
    )
    second_sum, second_count = _masked_cross_entropy_sum(
        second_logits, second_targets, second_mask
    )
    weighted = (first_sum + second_sum) / (first_count + second_count)
    expected = (
        _masked_cross_entropy(first_logits, first_targets, first_mask)
        + 3 * _masked_cross_entropy(second_logits, second_targets, second_mask)
    ) / 4
    assert torch.allclose(weighted, expected)


def test_finetuning_initializes_weights_and_resumes_with_same_source(tmp_path, tiny_config):
    train_bin = tmp_path / "train.bin"
    valid_bin = tmp_path / "valid.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 501)
    _write_stream(valid_bin, tiny_config.vocab_size, 256, 502)

    pretrained_dir = train_model(
        tiny_config, train_bin, valid_bin, tmp_path / "pretrain",
        steps=1, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
    )
    initial_weights = pretrained_dir / "model.pt"
    sft_run = tmp_path / "sft"
    train_model(
        tiny_config, train_bin, valid_bin, sft_run,
        steps=1, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        init_from=initial_weights,
    )
    manifest = json.loads((sft_run / "run.json").read_text(encoding="utf-8"))
    assert manifest["initial_model"]["sha256"]

    train_model(
        tiny_config, train_bin, valid_bin, sft_run,
        steps=2, batch_size=1, seq_len=8, grad_accum=1,
        eval_every=1, save_every=1, eval_batches=1,
        resume=True, extend_schedule=True,
    )
    checkpoint = torch.load(
        sft_run / "latest.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["step"] == 2
    assert checkpoint["resume_contract"]["initial_model"]["sha256"]


def test_periodic_validation_loss_is_weighted_by_supervised_tokens(monkeypatch):
    import qmini.train as train_module

    from types import SimpleNamespace

    class DummyModel:
        def __init__(self):
            self.training = True
            self.config = SimpleNamespace(thinking_steps=1)
            self.logits_by_marker = {
                0: torch.tensor([[[2.0, 0.0]] * 4]),
                1: torch.tensor([[[0.0, 2.0]] * 4]),
            }

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def __call__(self, input_ids):
            marker = int(input_ids[0, 0].item())
            return {"logits": self.logits_by_marker[marker]}

    batches = iter([
        (
            torch.zeros((1, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ),
        (
            torch.ones((1, 4), dtype=torch.long),
            torch.zeros((1, 4), dtype=torch.long),
            torch.ones((1, 4)),
        ),
    ])
    monkeypatch.setattr(
        train_module, "_sample_batch",
        lambda *args, **kwargs: next(batches),
    )
    model = DummyModel()
    report = _evaluate(
        model,
        np.zeros(32, dtype=np.uint16),
        None,
        batch_size=1,
        seq_len=4,
        batches=2,
        device=torch.device("cpu"),
        seed=123,
    )

    first_sum, first_count = _masked_cross_entropy_sum(
        model.logits_by_marker[0],
        torch.zeros((1, 4), dtype=torch.long),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    second_sum, second_count = _masked_cross_entropy_sum(
        model.logits_by_marker[1],
        torch.zeros((1, 4), dtype=torch.long),
        torch.ones((1, 4)),
    )
    expected = float(
        ((first_sum + second_sum) / (first_count + second_count)).item()
    )
    unweighted_batch_mean = float(
        (first_sum / first_count + second_sum / second_count).item() / 2.0
    )

    assert report["eval_loss"] == pytest.approx(expected, rel=1e-7)
    assert report["eval_perplexity"] == pytest.approx(np.exp(expected), rel=1e-7)
    assert abs(report["eval_loss"] - unweighted_batch_mean) > 0.5
    assert model.training is True


def test_periodic_validation_uses_all_tokens_for_plain_text(monkeypatch):
    import qmini.train as train_module

    from types import SimpleNamespace

    class FixedModel:
        training = False
        config = SimpleNamespace(thinking_steps=1)

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = bool(mode)
            return self

        def __call__(self, input_ids):
            return {"logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])}

    monkeypatch.setattr(
        train_module, "_sample_batch",
        lambda *args, **kwargs: (
            torch.zeros((1, 2), dtype=torch.long),
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.ones((1, 2)),
        ),
    )
    report = _evaluate(
        FixedModel(),
        np.zeros(16, dtype=np.uint16),
        None,
        batch_size=1,
        seq_len=2,
        batches=1,
        device=torch.device("cpu"),
        seed=456,
    )
    expected = float(
        torch.nn.functional.cross_entropy(
            torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]).reshape(-1, 2),
            torch.tensor([[0, 1]]).reshape(-1),
            reduction="mean",
        ).item()
    )
    assert report["eval_loss"] == pytest.approx(expected, rel=1e-7)


def test_sparse_assistant_masks_always_sample_legal_windows(tmp_path):
    token_count = 5000
    seq_len = 4
    tokens = np.arange(token_count, dtype=np.uint16)
    masks = np.zeros(token_count, dtype=np.uint8)
    # Position zero is not a label; only the first and last legal label positions
    # should be sampled, even when positives are extremely sparse.
    masks[0] = 1
    masks[1] = 1
    masks[-1] = 1
    index = SupervisedWindowIndex(
        masks, token_count=token_count, seq_len=seq_len, block_size=128
    )
    assert index.total_valid_windows == 2

    rng = np.random.default_rng(20261109)
    seen_starts = set()
    for _ in range(80):
        x, y, loss_mask = _sample_batch(
            tokens, masks, batch_size=4, seq_len=seq_len,
            rng=rng, device=torch.device("cpu"), window_index=index,
        )
        assert torch.all(loss_mask.sum(dim=1) > 0)
        starts = x[:, 0].cpu().numpy().astype(int)
        seen_starts.update(starts.tolist())
        for row_index, (start, row_mask) in enumerate(
            zip(starts, loss_mask.cpu().numpy())
        ):
            expected = masks[start + 1:start + seq_len + 1]
            assert np.array_equal(row_mask.astype(np.uint8), expected)
            assert expected.sum() > 0
            assert np.array_equal(
                y[row_index].cpu().numpy(),
                tokens[start + 1:start + seq_len + 1].astype(np.int64),
            )
    assert seen_starts == {0, token_count - seq_len - 1}


def test_sparse_assistant_mask_with_no_legal_label_window_fails_clearly():
    token_count = 128
    masks = np.zeros(token_count, dtype=np.uint8)
    masks[0] = 1  # BOS/context position is never a next-token label.
    with pytest.raises(ValueError, match="no supervised target inside any legal"):
        SupervisedWindowIndex(
            masks, token_count=token_count, seq_len=8, block_size=16
        )


def test_supervised_window_index_samples_uniformly_from_valid_start_positions():
    token_count = 64
    seq_len = 8
    masks = np.zeros(token_count, dtype=np.uint8)
    masks[20] = 1
    index = SupervisedWindowIndex(
        masks, token_count=token_count, seq_len=seq_len, block_size=8
    )
    valid_starts = set(range(20 - seq_len, 20))
    rng = np.random.default_rng(91)
    observed = {index.sample_start(rng) for _ in range(1000)}
    assert observed == valid_starts



def test_finetuning_initializes_from_new_q2_v2_packed_artifact(
    tmp_path, tiny_config
):
    from qmini.export import export_packed_state
    from qmini.model import QMiniMoE

    train_bin = tmp_path / "packed-train.bin"
    valid_bin = tmp_path / "packed-valid.bin"
    _write_stream(train_bin, tiny_config.vocab_size, 256, 601)
    _write_stream(valid_bin, tiny_config.vocab_size, 256, 602)
    initial = QMiniMoE(tiny_config)
    with torch.no_grad():
        next(initial.parameters()).zero_()
    packed_path = export_packed_state(
        initial.state_dict(), tiny_config, tmp_path / "initial-v2.qmini.npz"
    )

    run_dir = tmp_path / "packed-sft"
    train_model(
        tiny_config,
        train_bin,
        valid_bin,
        run_dir,
        steps=1,
        batch_size=1,
        seq_len=8,
        grad_accum=1,
        eval_every=1,
        save_every=1,
        eval_batches=1,
        init_from=packed_path,
    )
    run_manifest = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )
    assert run_manifest["initial_model"]["sha256"]
    assert run_manifest["initial_model"]["path"] == str(packed_path.resolve())

    checkpoint = torch.load(
        run_dir / "latest.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["resume_contract"]["initial_model"]["sha256"] == (
        run_manifest["initial_model"]["sha256"]
    )
