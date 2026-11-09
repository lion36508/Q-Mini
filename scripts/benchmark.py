from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import torch

from qmini.config import QMiniConfig
from qmini.model import QMiniMoE
from qmini.native import native_kernel_status


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_process_rss_bytes() -> int | None:
    """Return process peak RSS where the platform exposes getrusage."""
    try:
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # macOS reports bytes; Linux and the supported BSD CI targets report KiB.
    return value if platform.system() == "Darwin" else value * 1024


def cpu_model_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware", "processor")):
                    candidate = line.split(":", 1)[-1].strip()
                    if candidate:
                        return candidate
        except OSError:
            pass
    return platform.machine() or "unknown"


@torch.inference_mode()
def prefill(
    model: QMiniMoE,
    prompt_ids: torch.Tensor,
    thinking_steps: int,
):
    """Build the incremental convolution caches for a batched prompt."""
    state = None
    output = None
    for position in range(prompt_ids.shape[1]):
        output, state = model.forward_step(
            prompt_ids[:, position],
            state=state,
            thinking_steps=thinking_steps,
        )
    if state is None or output is None:
        raise RuntimeError("Benchmark prefill did not create a model state.")
    return output, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Q-Mini full forward, prompt prefill and cached decode throughput."
    )
    parser.add_argument("--config", default="configs/qmini_base.json")
    parser.add_argument(
        "--model-dir", default=None,
        help="load a trained model bundle; prefers the packed Q2 CPU runtime when available",
    )
    parser.add_argument(
        "--device", default=None,
        help="execution device; leave unset to auto-select (packed model bundles default to CPU)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128, help="Synthetic prompt length.")
    parser.add_argument("--iterations", type=int, default=20, help="Measured iterations or decode tokens.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--thinking-steps", type=int, default=2)
    parser.add_argument("--mode", choices=("forward", "decode", "both"), default="both")
    args = parser.parse_args()

    if args.batch_size < 1 or args.iterations < 1 or args.warmup < 0:
        raise SystemExit("batch-size and iterations must be positive; warmup cannot be negative.")

    config = QMiniConfig.from_json(args.config)
    if not 1 <= args.seq_len <= config.max_context:
        raise SystemExit("seq-len must be within configured context.")
    if not 1 <= args.thinking_steps <= config.max_thinking_steps:
        raise SystemExit("thinking-steps exceeds the configured limit.")

    if args.model_dir:
        from qmini.generate import load_model
        model, _tokenizer, device = load_model(args.model_dir, args.device)
        config = model.config
        if args.thinking_steps > config.max_thinking_steps:
            raise SystemExit(
                f"thinking-steps exceeds model limit {config.max_thinking_steps}."
            )
    else:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = QMiniMoE(config).to(device).eval()
    prompt_ids = torch.randint(
        0, config.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"platform={platform.platform()}")
    print(f"device={device}")
    print(f"cpu_model={cpu_model_name()}")
    print(f"runtime_model_bundle={args.model_dir or 'random-initialized-config'}")
    print(f"packed_q2_runtime={bool(getattr(model, '_qmini_packed_runtime', False))}")
    backend = native_kernel_status()
    print(f"cpu_projection_backend={backend['backend']}")
    print(f"arm_neon={backend['uses_neon']}")
    print(f"parameters={model.parameter_report()['total']:,}")
    print(f"prompt_tokens={args.batch_size * args.seq_len}")
    print(f"thinking_steps={args.thinking_steps}")
    print(f"warmup={args.warmup} measured_iterations={args.iterations}")
    print("inputs=synthetic token IDs; tokenizer and sampling overhead are excluded")

    with torch.inference_mode():
        if args.mode in {"forward", "both"}:
            for _ in range(args.warmup):
                model(prompt_ids, thinking_steps=args.thinking_steps)
            synchronize(device)

            start = time.perf_counter()
            for _ in range(args.iterations):
                model(prompt_ids, thinking_steps=args.thinking_steps)
            synchronize(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tokens = args.batch_size * args.seq_len * args.iterations
            print(f"forward_elapsed_seconds={elapsed:.4f}")
            print(f"forward_input_tokens_per_second={tokens / elapsed:.2f}")

        if args.mode in {"decode", "both"}:
            # Warm up the same streaming path used by actual autoregressive generation.
            _, warm_state = prefill(model, prompt_ids, args.thinking_steps)
            for _ in range(args.warmup):
                next_ids = torch.randint(
                    0, config.vocab_size, (args.batch_size,), device=device
                )
                _, warm_state = model.forward_step(next_ids, state=warm_state)
            synchronize(device)

            prefill_start = time.perf_counter()
            _, state = prefill(model, prompt_ids, args.thinking_steps)
            synchronize(device)
            prefill_elapsed = max(time.perf_counter() - prefill_start, 1e-9)

            decode_start = time.perf_counter()
            for _ in range(args.iterations):
                next_ids = torch.randint(
                    0, config.vocab_size, (args.batch_size,), device=device
                )
                _, state = model.forward_step(next_ids, state=state)
            synchronize(device)
            decode_elapsed = max(time.perf_counter() - decode_start, 1e-9)
            decoded_tokens = args.batch_size * args.iterations

            print(f"prefill_elapsed_seconds={prefill_elapsed:.4f}")
            print(
                f"prefill_input_tokens_per_second="
                f"{args.batch_size * args.seq_len / prefill_elapsed:.2f}"
            )
            print(f"decode_elapsed_seconds={decode_elapsed:.4f}")
            print(f"decode_tokens_per_second={decoded_tokens / decode_elapsed:.2f}")
            print(f"decode_tokens_per_request={args.iterations}")

        if device.type == "cuda":
            print(f"peak_cuda_memory_bytes={torch.cuda.max_memory_allocated(device)}")
        rss = peak_process_rss_bytes()
        print(f"peak_process_rss_bytes={rss if rss is not None else 'unavailable'}")


if __name__ == "__main__":
    main()
