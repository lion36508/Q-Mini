import torch

from qmini.export import export_packed_state
from qmini.model import QMiniMoE
from qmini.packed_runtime import load_packed_model


def _packed_config(tiny_config, state_space=False, vocab_size=None):
    values = tiny_config.to_dict()
    values["use_state_space"] = state_space
    if vocab_size is not None:
        values["vocab_size"] = vocab_size
    return type(tiny_config).from_dict(values)


def test_packed_runtime_keeps_matrices_packed_and_matches_reference(tiny_config, tmp_path):
    torch.manual_seed(901)
    config = _packed_config(tiny_config)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "model.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)

    assert packed._qmini_packed_runtime is True
    assert packed.parameter_report()["total"] == config.estimated_parameter_count()
    assert packed.parameter_report()["trainable"] == 0
    assert all(parameter.device.type == "cpu" for parameter in packed.parameters())
    assert all(parameter.ndim < 2 for parameter in packed.parameters())

    ids = torch.randint(0, config.vocab_size, (2, 13))
    with torch.inference_mode():
        expected = reference(ids, thinking_steps=2)["logits"]
        actual = packed(ids, thinking_steps=2)["logits"]

        state = None
        streamed = []
        for position in range(ids.shape[1]):
            result, state = packed.forward_step(
                ids[:, position], state=state, thinking_steps=2
            )
            streamed.append(result["logits"])
        stream_logits = torch.stack(streamed, dim=1)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.allclose(stream_logits, actual, atol=3e-5, rtol=3e-5)


def test_packed_runtime_supports_adaptive_state_space(tiny_config, tmp_path):
    config = _packed_config(tiny_config, state_space=True)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "hybrid.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_packed_runtime_has_bounded_pytorch_fallback(
    tiny_config, tmp_path, monkeypatch
):
    import qmini.native as native_module

    config = _packed_config(tiny_config, vocab_size=300)
    reference = QMiniMoE(config).eval()
    path = export_packed_state(
        reference.state_dict(), config, tmp_path / "fallback.qmini.npz"
    )
    packed = load_packed_model(path, expected_config=config)
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: None)
    ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.inference_mode():
        expected = reference(ids)["logits"]
        actual = packed(ids)["logits"]

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
