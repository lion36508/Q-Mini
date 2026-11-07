from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QMiniConfig
from .quantization import fake_quantize_int8, quantize_quaternary


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale * self.weight.float()).to(x.dtype)


class QLinear(nn.Linear):
    """Linear projection using quaternary master weights and INT8 activations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(fake_quantize_int8(x), quantize_quaternary(self.weight), self.bias)


class QEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def quantized_weight(self) -> torch.Tensor:
        return quantize_quaternary(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return fake_quantize_int8(F.embedding(token_ids, self.quantized_weight()))

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(fake_quantize_int8(hidden), self.quantized_weight())


class CausalGatedDepthwiseConv(nn.Module):
    """Causal sequence mixer: left padding only, so future tokens cannot leak backward."""

    def __init__(self, config: QMiniConfig, dilation: int):
        super().__init__()
        dim = config.d_model
        self.kernel_size = config.kernel_size
        self.dilation = dilation
        self.content_gate = QLinear(dim, dim * 2)
        self.depthwise_weight = nn.Parameter(torch.empty(dim, 1, self.kernel_size))
        self.depthwise_bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.depthwise_weight, a=5 ** 0.5)
        self.output = QLinear(dim, dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.content_gate(x).chunk(2, dim=-1)
        content = fake_quantize_int8(content).transpose(1, 2)
        left_pad = self.dilation * (self.kernel_size - 1)
        content = F.pad(content, (left_pad, 0))
        kernel = quantize_quaternary(self.depthwise_weight.reshape(self.depthwise_weight.shape[0], -1))
        kernel = kernel.reshape_as(self.depthwise_weight)
        mixed = F.conv1d(
            content, kernel, self.depthwise_bias, stride=1,
            padding=0, dilation=self.dilation,
            groups=self.depthwise_weight.shape[0],
        ).transpose(1, 2)
        return self.dropout(self.output(F.silu(mixed) * torch.sigmoid(gate)))


class SparseExpert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = QLinear(dim, hidden)
        self.down = QLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class SparseTop1MoE(nn.Module):
    def __init__(self, config: QMiniConfig):
        super().__init__()
        self.router_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.router = QLinear(config.d_model, config.n_experts)
        self.experts = nn.ModuleList([
            SparseExpert(config.d_model, config.d_hidden) for _ in range(config.n_experts)
        ])
        self.n_experts = config.n_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probabilities = F.softmax(self.router(self.router_norm(x)).float(), dim=-1)
        flat_probs = probabilities.reshape(-1, self.n_experts)
        selected = flat_probs.argmax(dim=-1)
        load = F.one_hot(selected, num_classes=self.n_experts).float().mean(dim=0)
        importance = flat_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(importance * load)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            indices = torch.nonzero(selected == expert_id, as_tuple=False).flatten()
            if indices.numel():
                expert_output = expert(flat.index_select(0, indices))
                output = output.index_copy(0, indices, expert_output)
        counts = torch.bincount(selected, minlength=self.n_experts)
        return output.reshape(shape), aux_loss, counts


class QMiniBlock(nn.Module):
    def __init__(self, config: QMiniConfig, dilation: int):
        super().__init__()
        self.mix_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mixer = CausalGatedDepthwiseConv(config, dilation)
        self.moe_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.moe = SparseTop1MoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.mixer(self.mix_norm(x))
        expert_output, aux_loss, counts = self.moe(self.moe_norm(x))
        return x + expert_output, aux_loss, counts


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

    def forward(self, input_ids: torch.Tensor, thinking_steps: int | None = None) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        sequence_length = int(input_ids.shape[1])
        if not 1 <= sequence_length <= self.config.max_context:
            raise ValueError(
                f"Sequence length must be in [1, {self.config.max_context}]."
            )
        if input_ids.shape[0] < 1:
            raise ValueError("Batch size must be at least one.")
        steps = self.config.thinking_steps if thinking_steps is None else int(thinking_steps)
        if not 1 <= steps <= self.config.max_thinking_steps:
            raise ValueError(f"thinking_steps must be 1..{self.config.max_thinking_steps}.")
        x = self.embedding(input_ids.long())
        aux_losses, router_counts = [], []
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

    def parameter_report(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "non_trainable": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }
