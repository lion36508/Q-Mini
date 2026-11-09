import hashlib
import pytest
import numpy as np
import torch
from qmini.quantization import (
    fake_quantize_int8, pack_quaternary, pack_ternary,
    pack_int4_groupwise, quantization_error_metrics,
    quantize_quaternary, quantize_ternary_rowwise,
    quantize_int8_rowwise, dequantize_int8_rowwise,
    unpack_quaternary, unpack_ternary, unpack_int4_groupwise,
)

def test_packed_quaternary_roundtrip():
    w = torch.tensor([[-4., -1., 1., 4., 2.], [4., 1., -1., -4., -2.]])
    packed, scales = pack_quaternary(w); restored = unpack_quaternary(packed, scales, 5)
    expected = quantize_quaternary(w).detach().numpy()
    assert np.allclose(restored, expected, atol=1e-6)

def test_quaternary_ste_gradient():
    w = torch.randn(3, 7, requires_grad=True); quantize_quaternary(w).square().mean().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0

def test_int8_ste_gradient():
    x = torch.randn(2, 5, 16, requires_grad=True); y = fake_quantize_int8(x)
    assert torch.isfinite(y).all(); y.sum().backward()
    assert x.grad is not None and torch.all(x.grad == 1)


def test_ternary_quantization_preserves_exact_zeros_and_round_trips_odd_width():
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.01, 0.0, 0.01, 0.0, -0.01, 0.0],
        [0.0, -2.0, 0.0, 1.0, 0.0, -0.5, 0.0],
    ])
    quantized, scales = quantize_ternary_rowwise(weight)
    assert set(quantized.unique().tolist()).issubset({-1, 0, 1})
    assert torch.all(quantized[weight == 0] == 0)
    assert torch.all(quantized[0] == 0)
    packed, packed_scales = pack_ternary(weight)
    assert packed.dtype == np.uint8
    assert packed_scales.dtype == np.float32
    restored = unpack_ternary(packed, packed_scales, in_features=7)
    expected = quantized.numpy().astype(np.float32) * scales.numpy()[:, None]
    assert np.array_equal(restored, expected)
    assert np.all(restored[weight.numpy() == 0] == 0)


def test_groupwise_int4_quantization_round_trips_odd_width_and_zero_rows():
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0, -2.0, -1.0, 0.0, 0.5, 1.5, 2.5],
        [7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0],
    ])
    packed, scales = pack_int4_groupwise(weight, group_size=4)
    assert packed.dtype == np.uint8
    assert scales.shape == (3, 2)
    restored = unpack_int4_groupwise(
        packed, scales, in_features=7, group_size=4
    )
    assert restored.shape == tuple(weight.shape)
    assert np.array_equal(restored[0], np.zeros(7, dtype=np.float32))
    assert np.max(np.abs(restored - weight.numpy())) <= float(scales.max()) / 2 + 1e-6


def test_quantization_error_metrics_report_weight_error_not_quality():
    reference = torch.tensor([[1.0, -2.0, 0.0]])
    approximation = torch.tensor([[1.0, -1.5, 0.0]])
    report = quantization_error_metrics(reference, approximation)
    assert report["relative_l2_error"] == pytest.approx(0.5 / np.sqrt(5))
    assert 0.0 < report["cosine_similarity"] <= 1.0
    assert report["max_absolute_error"] == pytest.approx(0.5)


def test_experimental_quantizers_reject_invalid_shapes_and_nonfinite_values():
    from qmini.quantization import pack_ternary, pack_int4_groupwise

    with pytest.raises(ValueError, match="at least two dimensions"):
        quantize_ternary_rowwise(torch.ones(5))
    with pytest.raises(ValueError, match="group_size"):
        pack_int4_groupwise(torch.ones(2, 5), group_size=0)
    with pytest.raises(ValueError, match="non-finite"):
        pack_ternary(torch.tensor([[float("nan"), 1.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        pack_int4_groupwise(torch.tensor([[float("inf"), 1.0]]))


def test_quantization_ablation_script_runs_only_on_own_qmini_checkpoint(
    tiny_config, tmp_path
):
    from qmini.model import QMiniMoE
    from qmini.config import QMiniConfig
    from qmini.quantization_ablation import run_ablation

    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    config = QMiniConfig.from_dict(tiny_config.to_dict())
    config.save(model_dir / "config.json")
    model = QMiniMoE(config)
    torch.save(model.state_dict(), model_dir / "model.pt")

    tokens_path = tmp_path / "heldout.bin"
    rng = np.random.default_rng(603)
    np.asarray(
        rng.integers(0, config.vocab_size, size=256), dtype=np.uint16
    ).tofile(tokens_path)

    output = tmp_path / "reports" / "quantization.json"
    report = run_ablation(
        model_dir,
        output,
        group_size=8,
        eval_tokens=tokens_path,
        eval_batch_size=1,
        eval_seq_len=8,
        eval_batches=2,
        eval_seed=991,
    )
    assert output.is_file()
    assert report["source_parameter_count"] == model.parameter_report()["total"]
    assert set(report["formats"]) == {"fp16", "int8", "q2", "ternary", "int4"}
    assert set(report["heldout_language_evaluation"]) == {
        "fp16", "int8", "q2", "ternary", "int4"
    }
    for evaluation in report["heldout_language_evaluation"].values():
        assert np.isfinite(evaluation["cross_entropy"])
        assert evaluation["supervised_target_tokens"] > 0
        assert evaluation["seed"] == 991
    assert report["evaluation_dataset_sha256"] == hashlib.sha256(tokens_path.read_bytes()).hexdigest()
    for metrics in report["formats"].values():
        assert np.isfinite(metrics["relative_l2_error"])
        assert np.isfinite(metrics["cosine_similarity"])
        assert metrics["estimated_weight_payload_bytes"] > 0
    assert any("not semantic assistant-quality scores" in item for item in report["limitations"])


def test_quantization_ablation_accepts_qmini_owned_latest_checkpoint(
    tiny_config, tmp_path
):
    from qmini.model import QMiniMoE
    from qmini.config import QMiniConfig
    from qmini.quantization_ablation import run_ablation

    config = QMiniConfig.from_dict(tiny_config.to_dict())
    model = QMiniMoE(config)
    run_dir = tmp_path / "own-pretraining-run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "latest.pt"
    torch.save({
        "format": "qmini-checkpoint-v2",
        "model": model.state_dict(),
        "config": config.to_dict(),
        "step": 1,
    }, checkpoint_path)

    report = run_ablation(
        run_dir,
        tmp_path / "checkpoint-ablation.json",
        group_size=8,
    )
    assert report["source_kind"] == "qmini-training-checkpoint"
    assert report["source_weight_path"] == str(checkpoint_path.resolve())
    assert report["source_parameter_count"] == model.parameter_report()["total"]


def test_rowwise_int8_weight_quantization_round_trip_and_zero_rows():
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, -0.25, 0.5, 2.0],
        [0.01, -0.02, 0.03, -0.04],
    ])
    quantized, scales = quantize_int8_rowwise(weight)
    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert scales.shape == (3,)
    assert torch.all(quantized[0] == 0)
    restored = dequantize_int8_rowwise(quantized, scales)
    assert restored.shape == weight.shape
    assert torch.equal(restored[0], torch.zeros_like(restored[0]))
    assert (restored - weight).abs().max() <= scales.max() / 2 + 1e-6


def test_rowwise_int8_quantization_rejects_invalid_weights_and_scales():
    with pytest.raises(ValueError, match="non-finite"):
        quantize_int8_rowwise(torch.tensor([[float("nan"), 1.0]]))
    q = torch.zeros((2, 3), dtype=torch.int8)
    with pytest.raises(TypeError, match="float32 vector"):
        dequantize_int8_rowwise(q, torch.ones(2, dtype=torch.float16))
    with pytest.raises(ValueError, match="finite and positive"):
        dequantize_int8_rowwise(q, torch.tensor([1.0, float("nan")]))



def test_q2_v2_exact_normalized_codebook_row_scale_and_all_zero_rows():
    weight = torch.tensor([
        [-4.0, -1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    packed, scales = pack_quaternary(weight)
    assert scales.dtype == np.float32
    assert scales[0] == pytest.approx(2.5)
    assert scales[1] == 0.0

    codes = np.empty((2, 8), dtype=np.uint8)
    for offset in range(4):
        codes[:, offset::4] = (packed >> (2 * offset)) & 3
    assert codes[0].tolist() == [0, 1, 2, 2, 3, 2, 2, 2]
    assert codes[1].tolist() == [2] * 8

    restored = unpack_quaternary(packed, scales, in_features=8, format_version=2)
    expected = np.array([
        [-2.5, -1.25, 1.25, 1.25, 2.5, 1.25, 1.25, 1.25],
        [0.0] * 8,
    ], dtype=np.float32)
    assert np.array_equal(restored, expected)
    fake_quantized = quantize_quaternary(weight).detach().numpy()
    assert np.array_equal(fake_quantized, expected)


def test_q2_v1_legacy_decoder_preserves_historical_codebook_and_scales():
    packed = np.array([[0b11100100, 0b00011011]], dtype=np.uint8)
    scales = np.array([0.25], dtype=np.float32)
    actual = unpack_quaternary(
        packed, scales, in_features=8, format_version=1
    )
    expected = np.array([[-0.5, -0.25, 0.25, 0.5, 0.5, 0.25, -0.25, -0.5]])
    assert np.array_equal(actual, expected.astype(np.float32))


def test_q2_decoder_rejects_version_specific_invalid_scales():
    packed = np.zeros((2, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="v1 row scales must be positive"):
        unpack_quaternary(
            packed, np.array([0.0, 1.0], dtype=np.float32), 4, format_version=1
        )
    with pytest.raises(ValueError, match="v2 row scales must be non-negative"):
        unpack_quaternary(
            packed, np.array([-0.1, 1.0], dtype=np.float32), 4, format_version=2
        )



def test_q2_v2_pack_and_qat_agree_for_tiny_nonzero_rows():
    weight = torch.tensor([
        [1e-12, -1e-12, 0.0, 2e-12],
        [0.0, 0.0, 0.0, 0.0],
    ])
    packed, scales = pack_quaternary(weight)
    restored = unpack_quaternary(packed, scales, 4, format_version=2)
    fake = quantize_quaternary(weight).detach().numpy()
    assert scales[0] > 0.0
    assert scales[1] == 0.0
    assert np.array_equal(restored, fake)
