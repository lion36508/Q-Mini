from __future__ import annotations
import argparse, time, torch
from qmini.config import QMiniConfig
from qmini.model import QMiniMoE

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/qmini_smoke.json")
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--iterations", type=int, default=10); p.add_argument("--thinking-steps", type=int, default=1)
    a = p.parse_args()
    if min(a.batch_size, a.iterations) < 1: raise SystemExit("Batch and iterations must be positive.")
    cfg = QMiniConfig.from_json(a.config)
    if not 1 <= a.seq_len <= cfg.max_context: raise SystemExit("seq-len exceeds configured context.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = QMiniMoE(cfg).to(device).eval()
    ids = torch.randint(0, cfg.vocab_size, (a.batch_size, a.seq_len), device=device)
    with torch.inference_mode():
        for _ in range(2): model(ids, thinking_steps=a.thinking_steps)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(a.iterations): model(ids, thinking_steps=a.thinking_steps)
        if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"device={device} parameters={model.parameter_report()['total']:,} elapsed={elapsed:.3f}s tokens_per_second={a.batch_size*a.seq_len*a.iterations/elapsed:.2f}")
if __name__ == "__main__": main()
