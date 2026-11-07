import torch
import torch.nn as nn
from qmini.model import QMiniMoE

def test_forward_shape_and_count(tiny_config):
    model = QMiniMoE(tiny_config); ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, tiny_config.vocab_size)
    assert out["aux_loss"].ndim == 0
    assert model.parameter_report()["total"] == sum(p.numel() for p in model.parameters())

def test_future_tokens_do_not_change_past_logits(tiny_config):
    torch.manual_seed(7); model = QMiniMoE(tiny_config).eval()
    a = torch.randint(0, tiny_config.vocab_size, (1, 20)); b = a.clone()
    b[:, 12:] = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.inference_mode(): x, y = model(a)["logits"], model(b)["logits"]
    assert torch.allclose(x[:, :12], y[:, :12], atol=1e-6, rtol=1e-6)

def test_router_gets_gradient(tiny_config):
    model = QMiniMoE(tiny_config); ids = torch.randint(0, tiny_config.vocab_size, (2, 12))
    out = model(ids); (out["logits"].float().square().mean() + out["aux_loss"]).backward()
    grad = model.blocks[0].moe.router.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

def test_no_attention_or_qkv_modules(tiny_config):
    model = QMiniMoE(tiny_config)
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(m, forbidden) for m in model.modules())
    names = [n.lower() for n, _ in model.named_modules()]
    assert not any(any(x in n for x in ("q_proj", "k_proj", "v_proj", "attention")) for n in names)

def test_refinement_step_bounds(tiny_config):
    model = QMiniMoE(tiny_config); ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    assert model(ids, thinking_steps=2)["logits"].shape[1] == 8
    try: model(ids, thinking_steps=3)
    except ValueError: return
    raise AssertionError("Out-of-range thinking_steps must raise ValueError.")
