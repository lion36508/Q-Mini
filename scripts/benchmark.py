from __future__ import annotations

import argparse
import platform
import time

import torch

from qmini.config import QMiniConfig
from qmini.model import QMiniMoE
from qmini.native import native_kernel_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Q-Mini reference/hybrid CPU or CUDA inference."
    )
    parser.add_argument("--config", default="configs/qmini_base.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--thinking-steps", type=int, default=2)
    args = parser.parse_args()

    if min(args.batch_size, args.iterations) < 1 or args.warmup < 0:
        raise SystemExit("batch-size and iterations must be positive; warmup cannot be negative.")

    config = QMiniConfig.from_json(args.config)
    if not 1 <= args.seq_len <= config.max_context:
        raise SystemExit("seq-len must be within configured context.")
    if not 1 <= args.thinking_steps <= config.max_thinking_steps:
        raise SystemExit("thinking-steps exceeds configured limit.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QMiniMoE(config).to(device).eval()
    ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(ids, thinking_steps=args.thinking_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(args.iterations):
            model(ids, thinking_steps=args.thinking_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    tokens = args.batch_size * args.seq_len * args.iterations
    status = native_kernel_status()
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    print(f"cpu_projection_backend={status['backend']}")
    print(f"arm_neon={status['uses_neon']}")
    print(f"parameters={model.parameter_report()['total']:,}")
    print(f"context={args.seq_len} thinking_steps={args.thinking_steps}")
    print(f"warmup_iterations={args.warmup} measured_iterations={args.iterations}")
    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"input_tokens_per_second={tokens / elapsed:.2f}")
    if device.type == "cuda":
        print(f"peak_cuda_memory_bytes={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
