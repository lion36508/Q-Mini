import numpy as np
from qmini.export import export_packed_state, reconstruct_packed_state, verify_packed_export
from qmini.model import QMiniMoE


def test_packed_export_roundtrip(tiny_config, tmp_path):
    model = QMiniMoE(tiny_config)
    target = export_packed_state(model.state_dict(), tiny_config, tmp_path / "model.qmini.npz")
    config, restored = reconstruct_packed_state(target)
    assert config.to_dict() == tiny_config.to_dict()
    original = model.state_dict()
    assert set(restored) == set(original)
    for name, tensor in original.items():
        if tensor.ndim >= 2:
            assert restored[name].shape == tuple(tensor.shape)
            assert np.isfinite(restored[name]).all()
        else:
            assert np.allclose(restored[name], tensor.detach().numpy(), atol=1e-3)
    report = verify_packed_export(target)
    assert report["tensor_count"] == len(original)
    assert report["parameter_count"] == sum(t.numel() for t in original.values())
