import pytest
import torch
import torch.nn as nn

from qmini.config import QMiniConfig
from qmini.model import GatedStateSpaceMixer, QMiniMoE


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
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_router_balance_loss_has_gradient(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (2, 12))
    model(ids)["aux_loss"].backward()
    gradient = model.blocks[0].moe.router.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_no_attention_or_qkv_modules(tiny_config):
    model = QMiniMoE(tiny_config)
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    names = [name.lower() for name, _ in model.named_modules()]
    assert not any(
        any(token in name for token in ("q_proj", "k_proj", "v_proj", "attention"))
        for name in names
    )


def test_refinement_step_bounds(tiny_config):
    model = QMiniMoE(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
    assert model(ids, thinking_steps=2)["logits"].shape[1] == 8
    with pytest.raises(ValueError):
        model(ids, thinking_steps=3)


def test_empty_inputs_and_invalid_token_ids_are_rejected(tiny_config):
    model = QMiniMoE(tiny_config)
    with pytest.raises(ValueError, match="Sequence length"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="Batch size"):
        model(torch.empty((0, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="outside the configured vocabulary"):
        model(torch.tensor([[tiny_config.vocab_size]], dtype=torch.long))


def test_incremental_cache_matches_full_sequence_logits(tiny_config):
    torch.manual_seed(12)
    model = QMiniMoE(tiny_config).eval()
    ids = torch.randint(0, tiny_config.vocab_size, (2, 19))
    with torch.inference_mode():
        full = model(ids, thinking_steps=2)["logits"]
        state = None
        streamed = []
        for position in range(ids.shape[1]):
            result, state = model.forward_step(
                ids[:, position], state=state, thinking_steps=2
            )
            streamed.append(result["logits"])
        incremental = torch.stack(streamed, dim=1)

    assert state.tokens_processed == ids.shape[1]
    assert torch.allclose(incremental, full, atol=2e-5, rtol=2e-5)


def test_incremental_state_isolated_per_batch_item(tiny_config):
    torch.manual_seed(4)
    model = QMiniMoE(tiny_config).eval()
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))
    with torch.inference_mode():
        result_a, _ = model.forward_step(tokens[:, 0])
        result_b0, _ = model.forward_step(tokens[0:1, 0])
        result_b1, _ = model.forward_step(tokens[1:2, 0])
    assert torch.allclose(result_a["logits"][0], result_b0["logits"][0], atol=1e-6)
    assert torch.allclose(result_a["logits"][1], result_b1["logits"][0], atol=1e-6)


def test_incremental_step_count_cannot_change_mid_stream(tiny_config):
    model = QMiniMoE(tiny_config).eval()
    with torch.inference_mode():
        _, state = model.forward_step(torch.tensor([1]), thinking_steps=1)
        with pytest.raises(ValueError, match="cannot change"):
            model.forward_step(torch.tensor([2]), state=state, thinking_steps=2)

def test_non_integer_token_ids_are_rejected(tiny_config):
    model = QMiniMoE(tiny_config)
    with pytest.raises(TypeError, match="integer tensor dtype"):
        model(torch.tensor([[1.0, 2.0]]))

    model.eval()
    with torch.inference_mode(), pytest.raises(TypeError, match="integer tensor dtype"):
        model.forward_step(torch.tensor([1.0]))

def test_thinking_steps_arguments_are_strict_in_all_inference_paths(tiny_config):
    model = QMiniMoE(tiny_config).eval()
    with pytest.raises(TypeError, match="thinking_steps must be an integer"):
        model(torch.tensor([[1, 2]]), thinking_steps=1.5)
    with pytest.raises(TypeError, match="thinking_steps must be an integer"):
        model.new_stream_state(batch_size=1, thinking_steps="1")
    with torch.inference_mode(), pytest.raises(TypeError, match="thinking_steps must be an integer"):
        model.forward_step(torch.tensor([1]), thinking_steps=True)

def test_gated_state_space_parallel_scan_matches_sequential_recurrence():
    torch.manual_seed(101)
    decay = (torch.rand(2, 17, 5) * 0.8 + 0.1).requires_grad_()
    injection = torch.randn(2, 17, 5, requires_grad=True)
    actual = GatedStateSpaceMixer.parallel_scan(decay, injection)

    state = torch.zeros(2, 5)
    expected = []
    for position in range(decay.shape[1]):
        state = decay[:, position] * state + injection[:, position]
        expected.append(state)
    expected = torch.stack(expected, dim=1)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert decay.grad is not None and torch.isfinite(decay.grad).all()
    assert injection.grad is not None and torch.isfinite(injection.grad).all()


def test_state_space_model_is_causal_and_streaming_matches_full_forward(tiny_config):
    values = tiny_config.to_dict()
    values["use_state_space"] = True
    config = QMiniConfig.from_dict(values)
    torch.manual_seed(102)
    model = QMiniMoE(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 19))
    changed = ids.clone()
    changed[:, 12:] = torch.randint(0, config.vocab_size, (2, 7))

    with torch.inference_mode():
        full = model(ids, thinking_steps=2)["logits"]
        changed_full = model(changed, thinking_steps=2)["logits"]
        state = None
        streamed = []
        for position in range(ids.shape[1]):
            result, state = model.forward_step(
                ids[:, position], state=state, thinking_steps=2
            )
            streamed.append(result["logits"])
        incremental = torch.stack(streamed, dim=1)

    assert torch.allclose(full[:, :12], changed_full[:, :12], atol=2e-5, rtol=2e-5)
    assert torch.allclose(incremental, full, atol=3e-5, rtol=3e-5)
    assert state is not None and len(state.state_spaces) == config.n_layers + 2
    assert all(item is not None for item in state.state_spaces)


def test_gradient_checkpointed_state_space_path_backpropagates(tiny_config):
    values = tiny_config.to_dict()
    values.update({"use_state_space": True, "gradient_checkpointing": True})
    model = QMiniMoE(QMiniConfig.from_dict(values)).train()
    ids = torch.randint(0, tiny_config.vocab_size, (2, 9))
    result = model(ids)
    loss = result["logits"].float().square().mean() + result["aux_loss"]
    loss.backward()
    grad = model.blocks[0].state_mixer.state_input.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

def test_state_space_channels_start_with_logarithmic_timescales(tiny_config):
    from torch.nn.functional import softplus
    from qmini.model import GatedStateSpaceMixer

    mixer = GatedStateSpaceMixer(
        tiny_config.d_model, max_context=tiny_config.max_context
    )
    with torch.no_grad():
        rates = softplus(mixer.state_decay.bias)
        time_constants = rates.reciprocal()
    assert torch.all(time_constants[1:] > time_constants[:-1])
    assert torch.isclose(time_constants[0], torch.tensor(2.0), rtol=1e-4)
    assert torch.isclose(
        time_constants[-1], torch.tensor(float(tiny_config.max_context)), rtol=1e-4
    )


@pytest.mark.parametrize("use_state_space", [False, True])
def test_streaming_recurrent_state_stays_bounded_beyond_prefill_limit(
    tiny_config, use_state_space
):
    values = tiny_config.to_dict()
    values.update({"max_context": 8, "use_state_space": use_state_space})
    config = QMiniConfig.from_dict(values)
    torch.manual_seed(730)
    model = QMiniMoE(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 23))
    state = None
    reference_prefix = []

    with torch.inference_mode():
        for position in range(ids.shape[1]):
            result, state = model.forward_step(ids[:, position], state=state)
            if position < config.max_context:
                reference = model(ids[:, :position + 1])["logits"][:, -1]
                assert torch.allclose(
                    result["logits"], reference, atol=4e-5, rtol=4e-5
                )
            else:
                assert torch.isfinite(result["logits"]).all()
            if position == config.max_context - 1:
                cache_shapes_at_limit = [
                    tuple(item.shape) if item is not None else None
                    for item in state.caches
                ]
                ssm_shapes_at_limit = [
                    tuple(item.shape) if item is not None else None
                    for item in state.state_spaces
                ]

    assert state is not None
    assert state.tokens_processed == ids.shape[1]
    assert state.caches and len(state.caches) == config.n_layers + config.thinking_steps
    assert [
        tuple(item.shape) if item is not None else None
        for item in state.caches
    ] == cache_shapes_at_limit
    assert [
        tuple(item.shape) if item is not None else None
        for item in state.state_spaces
    ] == ssm_shapes_at_limit
    assert all(
        item is None or not item.requires_grad
        for item in [*state.caches, *state.state_spaces]
    )

    with torch.inference_mode(), pytest.raises(ValueError, match="Sequence length"):
        model(ids[:, :config.max_context + 1])


def test_forward_step_rejects_grad_enabled_streaming(tiny_config):
    model = QMiniMoE(tiny_config).eval()
    with pytest.raises(RuntimeError, match="inference-only"):
        model.forward_step(torch.tensor([1]))

    with torch.no_grad():
        result, state = model.forward_step(torch.tensor([1]))
    assert result["logits"].shape == (1, tiny_config.vocab_size)
    assert state.tokens_processed == 1
    assert all(
        item is None or not item.requires_grad
        for item in [*state.caches, *state.state_spaces]
    )


def test_single_expert_is_a_dense_ffn_without_degenerate_router():
    config = QMiniConfig(
        vocab_size=64, d_model=16, d_hidden=24, n_layers=1,
        n_experts=1, top_k=1, max_context=16, kernel_size=3,
        dilations=(1,), max_thinking_steps=1, thinking_steps=1,
    )
    from qmini.model import SparseTop1MoE

    moe = SparseTop1MoE(config)
    assert moe.router is None
    assert moe.router_norm is None
    assert not any(
        name.startswith(("router.", "router_norm."))
        for name, _ in moe.named_parameters()
    )

    values = torch.randn(2, 5, config.d_model)
    actual, auxiliary, counts = moe(values)
    expected = moe.experts[0](values.reshape(-1, config.d_model)).reshape_as(values)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert auxiliary.item() == 0.0
    assert torch.equal(counts, torch.tensor([10]))

    (actual.square().mean() + auxiliary).backward()
    gradients = [parameter.grad for parameter in moe.parameters()]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_single_expert_model_parameter_count_matches_dense_layout():
    config = QMiniConfig(
        vocab_size=64, d_model=16, d_hidden=24, n_layers=2,
        n_experts=1, top_k=1, max_context=16, kernel_size=3,
        dilations=(1, 2), max_thinking_steps=1, thinking_steps=1,
        use_state_space=True,
    )
    model = QMiniMoE(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert config.estimated_parameter_count() == actual
    assert config.estimated_unique_active_parameter_count() == actual
    assert not any(".moe.router." in name or ".moe.router_norm." in name for name, _ in model.named_parameters())
