from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QMiniConfig
from .quantization import (
    fake_quantize_int8,
    pack_quaternary,
    quantize_int8_per_token,
    quantize_quaternary,
)


def _cache_key(weight: torch.Tensor) -> tuple[int, int, tuple[int, ...]]:
    return (int(weight._version), int(weight.data_ptr()), tuple(weight.shape))


def _cached_quantized_weight(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Cache only immutable eval/no-grad weights; training always uses fresh STE ops."""
    if module.training or torch.is_grad_enabled():
        return quantize_quaternary(weight)
    key = _cache_key(weight)
    if getattr(module, "_qmini_quantized_key", None) != key:
        module._qmini_quantized_weight = quantize_quaternary(weight)
        module._qmini_quantized_key = key
    return module._qmini_quantized_weight


def _try_native_linear(
    module: nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    """Run the real packed INT8 C++ projection for CPU inference when installed."""
    if (
        module.training
        or torch.is_grad_enabled()
        or x.device.type != "cpu"
        or x.dtype != torch.float32
        or weight.device.type != "cpu"
        or weight.dtype != torch.float32
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
    ):
        return None

    from .native import get_native_kernel

    kernel = get_native_kernel()
    if kernel is None:
        return None

    key = _cache_key(weight)
    if getattr(module, "_qmini_native_packed_key", None) != key:
        packed, scales = pack_quaternary(weight)
        module._qmini_native_packed_weight = packed
        module._qmini_native_weight_scales = scales
        module._qmini_native_packed_key = key

    activations_i8, activation_scales = quantize_int8_per_token(x)
    output = kernel.linear(
        activations_i8.cpu().numpy(),
        module._qmini_native_packed_weight,
        module._qmini_native_weight_scales,
        activation_scales.cpu().numpy(),
        bias.detach().float().cpu().numpy() if bias is not None else None,
    )
    tensor = torch.from_numpy(output)
    return tensor.reshape(*x.shape[:-1], weight.shape[0])


class RMSNorm(nn.Module):
    """RMS normalization evaluated in FP32 for numerical stability."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale * self.weight.float()).to(x.dtype)


class QLinear(nn.Linear):
    """Quaternary weight-aware projection with per-token INT8 fake quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_qmini_packed_weight", None) is not None:
            from .packed_runtime import packed_linear_forward
            return packed_linear_forward(self, x, self.bias)
        native_output = _try_native_linear(self, x, self.weight, self.bias)
        if native_output is not None:
            return native_output
        return F.linear(fake_quantize_int8(x), _cached_quantized_weight(self, self.weight), self.bias)


class QEmbedding(nn.Module):
    """Tied token embedding and language-model output projection."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def quantized_weight(self) -> torch.Tensor:
        return _cached_quantized_weight(self, self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_qmini_packed_weight", None) is not None:
            from .packed_runtime import packed_embedding_lookup
            return packed_embedding_lookup(self, token_ids)
        return fake_quantize_int8(F.embedding(token_ids, self.quantized_weight()))

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_qmini_packed_weight", None) is not None:
            from .packed_runtime import packed_linear_forward
            return packed_linear_forward(self, hidden, None)
        native_output = _try_native_linear(self, hidden, self.weight, None)
        if native_output is not None:
            return native_output
        return F.linear(fake_quantize_int8(hidden), self.quantized_weight())


class CausalGatedDepthwiseConv(nn.Module):
    """Strictly causal depthwise convolution with a token-by-token streaming path."""

    def __init__(self, config: QMiniConfig, dilation: int):
        super().__init__()
        dim = config.d_model
        self.dim = dim
        self.kernel_size = config.kernel_size
        self.dilation = dilation
        self.history_length = dilation * (self.kernel_size - 1)
        self.content_gate = QLinear(dim, dim * 2)
        self.depthwise_weight = nn.Parameter(torch.empty(dim, 1, self.kernel_size))
        self.depthwise_bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.depthwise_weight, a=5 ** 0.5)
        self.output = QLinear(dim, dim)
        self.dropout = nn.Dropout(config.dropout)

    def _kernel(self) -> torch.Tensor:
        if getattr(self, "_qmini_packed_weight", None) is not None:
            cached = getattr(self, "_qmini_depthwise_dequantized", None)
            if cached is None:
                from .quantization import unpack_quaternary
                dense = unpack_quaternary(
                    self._qmini_packed_weight.detach().cpu().numpy(),
                    self._qmini_weight_scales.detach().cpu().numpy(),
                    self.kernel_size,
                )
                cached = torch.from_numpy(dense).reshape(
                    self.dim, 1, self.kernel_size
                )
                self._qmini_depthwise_dequantized = cached
            return cached
        quantized = _cached_quantized_weight(
            self, self.depthwise_weight.reshape(self.dim, self.kernel_size)
        )
        return quantized.reshape_as(self.depthwise_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("Convolution input must have shape [batch, sequence, d_model].")
        content, gate = self.content_gate(x).chunk(2, dim=-1)
        content = fake_quantize_int8(content).transpose(1, 2)
        content = F.pad(content, (self.history_length, 0))
        mixed = F.conv1d(
            content,
            self._kernel(),
            self.depthwise_bias.float(),
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.dim,
        ).transpose(1, 2)
        return self.dropout(self.output(F.silu(mixed) * torch.sigmoid(gate)))

    def forward_step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token; cache holds the previous projected content values."""
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError("Streaming convolution input must have shape [batch, d_model].")
        content, gate = self.content_gate(x).chunk(2, dim=-1)
        content = fake_quantize_int8(content)
        batch = x.shape[0]

        if self.history_length:
            if cache is None:
                cache = content.new_zeros((batch, self.history_length, self.dim))
            elif cache.shape != (batch, self.history_length, self.dim):
                raise ValueError("Streaming convolution cache has an incompatible shape.")
            elif cache.device != content.device or cache.dtype != content.dtype:
                raise ValueError("Streaming convolution cache device/dtype differs from the model.")
            window = torch.cat((cache, content.unsqueeze(1)), dim=1)
            next_cache = window[:, 1:, :].contiguous()
        else:
            window = content.unsqueeze(1)
            next_cache = window[:, :0, :].contiguous()

        mixed = F.conv1d(
            window.transpose(1, 2),
            self._kernel(),
            self.depthwise_bias.float(),
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.dim,
        )
        if mixed.shape[-1] != 1:
            raise RuntimeError("Streaming convolution did not produce exactly one output step.")
        mixed = mixed.squeeze(-1)
        output = self.dropout(self.output(F.silu(mixed) * torch.sigmoid(gate)))
        return output, next_cache


class GatedStateSpaceMixer(nn.Module):
    """Q-Mini gated diagonal SSM with a parallel training scan and recurrent inference.

    This is not an implementation of Mamba or RWKV. The causal update is:
        state[t] = decay[t] * state[t-1] + (1-decay[t]) * drive[t]
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        max_context: int = 2048,
    ):
        super().__init__()
        if max_context < 2:
            raise ValueError("State-space max_context must be at least 2.")
        self.dim = dim
        self.state_input = QLinear(dim, dim)
        self.state_decay = QLinear(dim, dim)
        self.state_gate = QLinear(dim, dim)
        self.output = QLinear(dim, dim)
        nn.init.zeros_(self.state_decay.weight)
        if self.state_decay.bias is not None:
            # Spread initial state time constants logarithmically from 2 to
            # max_context tokens, rather than giving every channel one timescale.
            with torch.no_grad():
                log_times = torch.linspace(
                    math.log(2.0), math.log(float(max_context)), dim
                )
                time_constants = torch.exp(log_times)
                rates = time_constants.reciprocal()
                self.state_decay.bias.copy_(torch.log(torch.expm1(rates)))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def parallel_scan(decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """Differentiable associative prefix scan of affine recurrent updates."""
        if decay.ndim != 3 or injection.shape != decay.shape:
            raise ValueError("decay and injection must share [batch, sequence, features] shape.")
        if decay.shape[1] < 1:
            raise ValueError("State-space scan requires a non-empty sequence.")
        a, b = decay.float(), injection.float()
        offset, length = 1, decay.shape[1]
        while offset < length:
            previous_a, previous_b = a, b
            earlier_a = torch.cat(
                (torch.ones_like(previous_a[:, :offset]), previous_a[:, :-offset]),
                dim=1,
            )
            earlier_b = torch.cat(
                (torch.zeros_like(previous_b[:, :offset]), previous_b[:, :-offset]),
                dim=1,
            )
            # Compose the current affine update after the earlier prefix.
            b = previous_b + previous_a * earlier_b
            a = previous_a * earlier_a
            offset *= 2
        return b

    def _drive_and_decay(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drive = torch.tanh(self.state_input(x)).float()
        decay = torch.exp(-F.softplus(self.state_decay(x).float())).clamp_min(1e-6)
        return drive, decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("State-space input must have shape [batch, sequence, d_model].")
        drive, decay = self._drive_and_decay(x)
        states = self.parallel_scan(decay, (1.0 - decay) * drive)
        gated = states.to(dtype=x.dtype) * torch.sigmoid(self.state_gate(x))
        return self.dropout(self.output(gated))

    def forward_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError("Streaming state-space input must have shape [batch, d_model].")
        drive = torch.tanh(self.state_input(x)).float()
        decay = torch.exp(-F.softplus(self.state_decay(x).float())).clamp_min(1e-6)
        if state is None:
            state = torch.zeros_like(drive)
        elif state.shape != drive.shape:
            raise ValueError("State-space cache has an incompatible shape.")
        elif state.device != drive.device:
            raise ValueError("State-space cache device differs from the model.")
        next_state = decay * state.float() + (1.0 - decay) * drive
        gated = next_state.to(dtype=x.dtype) * torch.sigmoid(self.state_gate(x))
        return self.dropout(self.output(gated)), next_state


class SparseExpert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = QLinear(dim, hidden)
        self.down = QLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class SparseTop1MoE(nn.Module):
    """Token-level Top-1 dispatch with a differentiable selected expert gate."""

    def __init__(self, config: QMiniConfig):
        super().__init__()
        self.router_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.router = QLinear(config.d_model, config.n_experts)
        self.experts = nn.ModuleList([
            SparseExpert(config.d_model, config.d_hidden)
            for _ in range(config.n_experts)
        ])
        self.n_experts = config.n_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probabilities = F.softmax(self.router(self.router_norm(x)).float(), dim=-1)
        flat_probs = probabilities.reshape(-1, self.n_experts)
        selected = flat_probs.argmax(dim=-1)
        selected_weights = flat_probs.gather(1, selected.unsqueeze(1))
        load = F.one_hot(selected, num_classes=self.n_experts).float().mean(dim=0)
        importance = flat_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(importance * load)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            indices = torch.nonzero(selected == expert_id, as_tuple=False).flatten()
            if indices.numel():
                expert_input = flat.index_select(0, indices)
                expert_output = expert(expert_input)
                gate = selected_weights.index_select(0, indices).to(expert_output.dtype)
                output = output.index_copy(0, indices, expert_output * gate)

        counts = torch.bincount(selected, minlength=self.n_experts)
        return output.reshape(shape), aux_loss, counts

    def forward_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same dispatch logic for a single [batch, d_model] token batch."""
        if x.ndim != 2:
            raise ValueError("MoE streaming input must have shape [batch, d_model].")
        output, aux_loss, counts = self.forward(x)
        return output, aux_loss, counts


class QMiniBlock(nn.Module):
    def __init__(self, config: QMiniConfig, dilation: int):
        super().__init__()
        self.mix_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mixer = CausalGatedDepthwiseConv(config, dilation)
        self.state_norm = (
            RMSNorm(config.d_model, config.rms_norm_eps)
            if config.use_state_space else None
        )
        self.state_mixer = (
            GatedStateSpaceMixer(config.d_model, config.dropout, config.max_context)
            if config.use_state_space else None
        )
        self.moe_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.moe = SparseTop1MoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.mixer(self.mix_norm(x))
        if self.state_mixer is not None:
            x = x + self.state_mixer(self.state_norm(x))
        expert_output, aux_loss, counts = self.moe(self.moe_norm(x))
        return x + expert_output, aux_loss, counts

    def forward_step(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        state_cache: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor | None,
    ]:
        mixed, next_cache = self.mixer.forward_step(self.mix_norm(x), cache)
        x = x + mixed
        next_state = state_cache
        if self.state_mixer is not None:
            state_output, next_state = self.state_mixer.forward_step(
                self.state_norm(x), state_cache
            )
            x = x + state_output
        expert_output, aux_loss, counts = self.moe.forward_step(self.moe_norm(x))
        return x + expert_output, aux_loss, counts, next_cache, next_state


@dataclass
class QMiniStreamState:
    """Bounded rolling context plus per-layer incremental inference caches."""

    caches: list[torch.Tensor | None]
    state_spaces: list[torch.Tensor | None]
    thinking_steps: int
    batch_size: int
    tokens_processed: int = 0
    # Token IDs stay on CPU so rolling-window resets do not pin GPU memory.
    token_history: torch.Tensor | None = None


class QMiniMoE(nn.Module):
    """From-scratch attention-free causal-convolution sparse-MoE language model."""

    def __init__(self, config: QMiniConfig | None = None):
        super().__init__()
        self.config = config or QMiniConfig()
        self.config.validate()
        self.embedding = QEmbedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList([
            QMiniBlock(self.config, dilation) for dilation in self.config.dilations
        ])
        self.refinement = QMiniBlock(self.config, dilation=1)
        self.final_norm = RMSNorm(self.config.d_model, self.config.rms_norm_eps)

    @staticmethod
    def _validate_token_id_dtype(input_ids: torch.Tensor) -> None:
        if (
            input_ids.dtype == torch.bool
            or input_ids.is_floating_point()
            or input_ids.is_complex()
        ):
            raise TypeError("input_ids must use an integer tensor dtype; implicit float-to-ID casts are refused.")

    def _resolve_thinking_steps(self, thinking_steps: int | None) -> int:
        if thinking_steps is None:
            steps = self.config.thinking_steps
        else:
            if isinstance(thinking_steps, bool) or not isinstance(thinking_steps, int):
                raise TypeError("thinking_steps must be an integer.")
            steps = thinking_steps
        if not 1 <= steps <= self.config.max_thinking_steps:
            raise ValueError(f"thinking_steps must be 1..{self.config.max_thinking_steps}.")
        return steps

    def forward(self, input_ids: torch.Tensor, thinking_steps: int | None = None) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.shape[1] < 1 or input_ids.shape[1] > self.config.max_context:
            raise ValueError(f"Sequence length must be in [1, {self.config.max_context}].")
        if input_ids.shape[0] < 1:
            raise ValueError("Batch size must be at least one.")
        self._validate_token_id_dtype(input_ids)
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside the configured vocabulary.")

        steps = self._resolve_thinking_steps(thinking_steps)

        x = self.embedding(input_ids.long())
        aux_losses: list[torch.Tensor] = []
        router_counts: list[torch.Tensor] = []
        if self.training and self.config.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint

            for block in self.blocks:
                x, aux, counts = checkpoint(block, x, use_reentrant=False)
                aux_losses.append(aux)
                router_counts.append(counts)
            for _ in range(steps):
                x, aux, counts = checkpoint(self.refinement, x, use_reentrant=False)
                aux_losses.append(aux)
                router_counts.append(counts)
        else:
            for block in self.blocks:
                x, aux, counts = block(x)
                aux_losses.append(aux)
                router_counts.append(counts)
            for _ in range(steps):
                x, aux, counts = self.refinement(x)
                aux_losses.append(aux)
                router_counts.append(counts)
        logits = self.embedding.project(self.final_norm(x))
        return {
            "logits": logits,
            "aux_loss": torch.stack(aux_losses).mean(),
            "router_counts": router_counts,
        }

    def new_stream_state(
        self,
        batch_size: int,
        thinking_steps: int | None = None,
    ) -> QMiniStreamState:
        if self.training:
            raise RuntimeError("Incremental inference requires model.eval().")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        steps = self._resolve_thinking_steps(thinking_steps)
        layer_count = self.config.n_layers + steps
        return QMiniStreamState(
            caches=[None] * layer_count,
            state_spaces=[None] * layer_count,
            thinking_steps=steps,
            batch_size=batch_size,
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: QMiniStreamState | None = None,
        thinking_steps: int | None = None,
    ) -> tuple[dict[str, Any], QMiniStreamState]:
        """Run one token with the same hard rolling context as the batch path.

        Streaming inference is inference-only: callers must use torch.no_grad()
        or torch.inference_mode(). Once the context is full, cached state is
        rebuilt from the last max_context - 1 token IDs before processing the
        next token. This drops old SSM state as well as convolution history.
        """
        if self.training:
            raise RuntimeError("Incremental inference requires model.eval().")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "forward_step is inference-only; wrap streaming calls in "
                "torch.no_grad() or torch.inference_mode()."
            )
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1 or input_ids.shape[0] < 1:
            raise ValueError("forward_step expects token IDs shaped [batch] or [batch, 1].")

        if state is None:
            state = self.new_stream_state(input_ids.shape[0], thinking_steps)
        else:
            if state.batch_size != input_ids.shape[0]:
                raise ValueError("Streaming state batch size does not match input.")
            configured_steps = self._resolve_thinking_steps(state.thinking_steps)
            if thinking_steps is not None and self._resolve_thinking_steps(thinking_steps) != configured_steps:
                raise ValueError("thinking_steps cannot change in the middle of a stream.")

        total_tokens_processed = state.tokens_processed
        history = state.token_history
        if history is not None:
            if (
                history.ndim != 2
                or history.shape[0] != state.batch_size
                or history.shape[1] > self.config.max_context
                or history.dtype != torch.long
                or history.device.type != "cpu"
            ):
                raise ValueError("Streaming token history is malformed or belongs to another model.")
            if history.shape[1] >= self.config.max_context:
                rolling_prefix = history[:, 1:]
                rebuilt = self.new_stream_state(
                    state.batch_size, thinking_steps=state.thinking_steps
                )
                for position in range(rolling_prefix.shape[1]):
                    _, rebuilt = self._forward_step_raw(
                        rolling_prefix[:, position].to(device=input_ids.device),
                        state=rebuilt,
                    )
                state = rebuilt

        result, next_state = self._forward_step_raw(
            input_ids, state=state, thinking_steps=thinking_steps
        )
        current_ids = input_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1, 1)
        if history is None:
            next_history = current_ids
        else:
            next_history = torch.cat((history, current_ids), dim=1)
        next_state.token_history = next_history[:, -self.config.max_context:].contiguous()
        next_state.tokens_processed = total_tokens_processed + 1
        return result, next_state

    def _forward_step_raw(
        self,
        input_ids: torch.Tensor,
        state: QMiniStreamState | None = None,
        thinking_steps: int | None = None,
    ) -> tuple[dict[str, Any], QMiniStreamState]:
        """Internal one-token update used to rebuild and advance a rolling cache."""
        if self.training:
            raise RuntimeError("Incremental inference requires model.eval().")
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            input_ids = input_ids[:, 0]
        if input_ids.ndim != 1 or input_ids.shape[0] < 1:
            raise ValueError("forward_step expects token IDs shaped [batch] or [batch, 1].")
        self._validate_token_id_dtype(input_ids)
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside the configured vocabulary.")

        if state is None:
            state = self.new_stream_state(input_ids.shape[0], thinking_steps)
        elif state.batch_size != input_ids.shape[0]:
            raise ValueError("Streaming state batch size does not match input.")
        else:
            configured_steps = self._resolve_thinking_steps(state.thinking_steps)
            if thinking_steps is not None:
                requested_steps = self._resolve_thinking_steps(thinking_steps)
                if requested_steps != configured_steps:
                    raise ValueError("thinking_steps cannot change in the middle of a stream.")
        expected_layers = self.config.n_layers + state.thinking_steps
        if len(state.caches) != expected_layers or len(state.state_spaces) != expected_layers:
            raise ValueError("Streaming state does not match this model configuration.")

        x = self.embedding(input_ids.long())
        new_caches: list[torch.Tensor | None] = []
        new_state_spaces: list[torch.Tensor | None] = []
        aux_losses: list[torch.Tensor] = []
        router_counts: list[torch.Tensor] = []

        for index, block in enumerate(self.blocks):
            x, aux, counts, cache, memory_state = block.forward_step(
                x, state.caches[index], state.state_spaces[index]
            )
            new_caches.append(cache)
            new_state_spaces.append(memory_state)
            aux_losses.append(aux)
            router_counts.append(counts)

        offset = self.config.n_layers
        for index in range(state.thinking_steps):
            state_index = offset + index
            x, aux, counts, cache, memory_state = self.refinement.forward_step(
                x, state.caches[state_index], state.state_spaces[state_index]
            )
            new_caches.append(cache)
            new_state_spaces.append(memory_state)
            aux_losses.append(aux)
            router_counts.append(counts)

        logits = self.embedding.project(self.final_norm(x))
        new_state = QMiniStreamState(
            caches=new_caches,
            state_spaces=new_state_spaces,
            thinking_steps=state.thinking_steps,
            batch_size=state.batch_size,
            tokens_processed=state.tokens_processed + 1,
        )
        return {
            "logits": logits,
            "aux_loss": torch.stack(aux_losses).mean(),
            "router_counts": router_counts,
        }, new_state

    def parameter_report(self) -> dict[str, int]:
        if getattr(self, "_qmini_packed_runtime", False):
            total = int(self._qmini_original_parameter_count)
            return {"total": total, "trainable": 0, "non_trainable": total}
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "non_trainable": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }
