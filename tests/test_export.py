import numpy as np
from qmini.export import export_packed_state, reconstruct_packed_state, verify_packed_export
from qmini.model import QMiniMoE
from qmini.quantization import quantize_quaternary


def test_packed_export_roundtrip_matches_quaternary_projection(tiny_config, tmp_path):
    model = QMiniMoE(tiny_config)
    target = export_packed_state(
        model.state_dict(), tiny_config, tmp_path / "model.qmini.npz"
    )
    config, restored = reconstruct_packed_state(target)
    assert config.to_dict() == tiny_config.to_dict()
    original = model.state_dict()
    assert set(restored) == set(original)

    for name, tensor in original.items():
        if tensor.ndim >= 2:
            expected = quantize_quaternary(tensor).detach().cpu().numpy()
            assert restored[name].shape == tuple(tensor.shape)
            assert np.allclose(restored[name], expected, atol=1e-6, rtol=1e-6), name
        else:
            assert np.allclose(
                restored[name], tensor.detach().cpu().numpy(), atol=1e-3, rtol=1e-3
            ), name

    report = verify_packed_export(target)
    assert report["tensor_count"] == len(original)
    assert report["parameter_count"] == sum(t.numel() for t in original.values())
    assert report["context"] == tiny_config.max_context
