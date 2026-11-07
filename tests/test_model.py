import pytest
import torch
import torch.nn as nn

from qmini.model import QMiniMoE


def test_forward_shape_and_count(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, tiny_config.vocab_size)
    assert out["aux_loss"].ndim == 0
    assert model.parameter_report()["total"] == sum(p.numel() for p in model.parameters())


def test_future_tokens_do_not_change_past_logits(tiny_config):
    torch.manual_seed(7)
    model = QMiniMoE(tiny_config).eval()
    first = torch.randint(0, tiny_config.vocab_size, (1, 20))
    changed = first.clone()
    changed[:, 12:] = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.inference_mode():
        logits_a = model(first)["logits"]
        logits_b = model(changed)["logits"]
    assert torch.allclose(logits_a[:, :12], logits_b[:, :12], atol=1e-6, rtol=1e-6)


def test_router_gets_gradient_from_language_model_loss(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 12))
    logits = model(ids)["logits"]
    logits.float().square().mean().backward()
    gradient = model.blocks[0].moe.router.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_router_balance_loss_has_gradient(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 12))
    result = model(ids)
    result["aux_loss"].backward()
    gradient = model.blocks[0].moe.router.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_no_attention_or_qkv_modules(tiny_config):
    model = QMiniMoE(tiny_config)
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    names = [name.lower() for name, _ in model.named_modules()]
    assert not any(any(token in name for token in ("q_proj", "k_proj", "v_proj", "attention")) for name in names)


def test_refinement_step_bounds(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    assert model(ids, thinking_steps=2)["logits"].shape[1] == 8
    with pytest.raises(ValueError):
        model(ids, thinking_steps=3)


def test_empty_sequence_and_batch_are_rejected(tiny_config):
    model = QMiniMoE(tiny_config)
    with pytest.raises(ValueError, match="Sequence length"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="Batch size"):
        model(torch.empty((0, 4), dtype=torch.long))
