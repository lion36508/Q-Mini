from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QMiniConfig:
    vocab_size: int = 4096
    d_model: int = 256
    d_hidden: int = 384
    n_layers: int = 6
    n_experts: int = 4
    top_k: int = 1
    max_context: int = 512
    kernel_size: int = 10
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    max_thinking_steps: int = 4
    thinking_steps: int = 2
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    router_aux_loss_coef: float = 0.01
    use_state_space: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42

    def validate(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "max_context": self.max_context,
            "kernel_size": self.kernel_size,
            "max_thinking_steps": self.max_thinking_steps,
            "thinking_steps": self.thinking_steps,
        }
        invalid_types = [
            name for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_types:
            raise TypeError("These Q-Mini settings must be integers: " + ", ".join(invalid_types))

        non_positive = [
            name for name, value in integer_fields.items() if value <= 0
        ]
        if non_positive:
            raise ValueError("Must be positive: " + ", ".join(non_positive))
        if self.max_context < 2:
            raise ValueError("max_context must be at least 2 tokens.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isinstance(self.use_state_space, bool):
            raise TypeError("use_state_space must be a boolean.")
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")

        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k != 1:
            raise ValueError("Q-Mini currently implements sparse Top-1 routing only.")
        if not isinstance(self.dilations, (tuple, list)):
            raise TypeError("dilations must be a tuple or list of integers.")
        if (
            len(self.dilations) != self.n_layers
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.dilations
            )
        ):
            raise ValueError("dilations must contain one positive integer per main block.")
        if self.thinking_steps > self.max_thinking_steps:
            raise ValueError("thinking_steps cannot exceed max_thinking_steps.")

        numeric_fields = {
            "dropout": self.dropout,
            "rms_norm_eps": self.rms_norm_eps,
            "router_aux_loss_coef": self.router_aux_loss_coef,
        }
        for name, value in numeric_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.rms_norm_eps <= 0 or self.router_aux_loss_coef < 0:
            raise ValueError("Invalid RMSNorm epsilon or router auxiliary loss coefficient.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "QMiniConfig":
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a dictionary.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown Q-Mini config keys: " + ", ".join(unknown))
        clean = dict(values)
        if "dilations" in clean:
            clean["dilations"] = tuple(clean["dilations"]) if isinstance(clean["dilations"], (list, tuple)) else clean["dilations"]
        config = cls(**clean)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "QMiniConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object.")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return target

    def _block_parameter_counts(self) -> tuple[int, int]:
        """Return (full block parameters, one expert's parameters)."""
        d, h, experts = self.d_model, self.d_hidden, self.n_experts
        shared = 3 * d * d + 7 * d + d * self.kernel_size + d * experts + experts
        if self.use_state_space:
            # Four QLinear projections plus the state-space RMSNorm.
            shared += 4 * d * d + 5 * d
        one_expert = 2 * d * h + h + d
        return shared + experts * one_expert, one_expert

    def estimated_parameter_count(self) -> int:
        """Exact model-layout count without allocating model tensors."""
        self.validate()
        block_total, _ = self._block_parameter_counts()
        return self.vocab_size * self.d_model + (self.n_layers + 1) * block_total + self.d_model

    def estimated_unique_active_parameter_count(self) -> int:
        """Unique parameters touched per token under the current Top-1 routing policy."""
        self.validate()
        block_total, one_expert = self._block_parameter_counts()
        shared = block_total - self.n_experts * one_expert
        return (
            self.vocab_size * self.d_model
            + self.d_model
            + (self.n_layers + 1) * (shared + one_expert)
        )

    def estimated_adamw_training_state_bytes(self) -> int:
        """FP32 parameters, gradients and two FP32 Adam moments; excludes activations."""
        return self.estimated_parameter_count() * 16

    def reference_tokens_at_20_per_parameter(self) -> int:
        """Dense-LM planning heuristic only; not a guarantee or sparse-MoE scaling law."""
        return self.estimated_parameter_count() * 20

    def main_receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)
