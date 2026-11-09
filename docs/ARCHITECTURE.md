# Q-Mini architecture lock

## Forward graph

```text
Token IDs
  -> quaternary token embedding
  -> [QMiniBlock x 6]
       -> RMSNorm
       -> quantized content/gate projection
       -> left-padded depthwise causal convolution
       -> SiLU(content) * sigmoid(gate)
       -> output projection + residual
       -> RMSNorm
       -> sparse Top-1 router
       -> selected expert MLP only
       -> residual
  -> one shared QMiniBlock repeated 1..4 times
  -> final RMSNorm
  -> tied embedding projection
  -> next-token logits
```

No Transformer attention, Q/K/V or dot-product attention modules are present. Legacy profiles use left-padded causal depthwise convolution only. New hybrid profiles combine that convolution with a gated diagonal state-space recurrence. Router and expert operations remain token-local.

## Causality and memory

Each convolution is padded only on the left by `dilation * (kernel_size - 1)`. No future token can affect an earlier output. The convolution path has a finite receptive field of `1 + (kernel_size - 1) * sum(dilations)`. Hybrid profiles additionally maintain a fixed-width learned recurrent state, updated as `s[t] = a[t] * s[t-1] + (1-a[t]) * u[t]`. Training uses a differentiable parallel associative scan; token-by-token generation uses the equivalent recurrent update. This state compresses prior information and cannot guarantee exact retrieval of arbitrary earlier text.

## Quantization

Trainable master weights remain FP32 and are fake-quantized row-wise to {-2,-1,+1,+2} with STE. Activations use per-token AbsMax scaling to signed INT8 [-127,127], also with STE. Packed export stores four 2-bit codes per byte and row scales. Optimizer state is not 2-bit.

During CPU evaluation/inference, QLinear projections and the tied output projection use the C++17 kernel when `libqmini_kernel.so` is available. Activations are quantized to INT8 per token; each output row's quaternary weights are packed to four 2-bit codes per byte; the native inner loop accumulates integer products before applying row/token scales and bias. Packed weights are cached until the underlying parameter version changes. Training and autograd always use the PyTorch STE reference path.

## Sparse experts and refinement

Each token is sent to one expert per block. Router auxiliary loss discourages imbalance but does not guarantee meaningful expert specialization. One additional block is shared and reused from one to four times. This is learned hidden-state refinement, not symbolic theorem proving or a guarantee of correct reasoning.

The authoritative parameter count is computed from `sum(p.numel() for p in model.parameters())` on the instantiated model.

## Native runtime boundary

The native kernel covers dense linear projections only. AArch64/NEON builds use vector widening multiply and pairwise integer accumulation for the INT8-by-quaternary dot product; other targets use the scalar integer fallback. Causal depthwise convolution, RMSNorm, token embedding lookup, sparse dispatch and state management remain PyTorch operations. This is intentionally a hybrid runtime; it must not be described as a full integer-only or fully native ARM64 model. CI cross-compiles the C++ primitive for AArch64, but only a real Android device benchmark can establish phone performance.


## Local memory boundary

The qmini/memory.py module provides an optional local SQLite store. FTS5 indexes note content and tags when available; otherwise search falls back to literal substring matching. Writes happen only through explicit user commands. This subsystem is not a neural memory layer, does not alter model weights and does not grant arbitrary tool execution.


## Higher-capacity profile

The compact baseline is preserved in configs/qmini_base.json. For GPU training, configs/qmini_research_134m.json supplies width 640, expert hidden width 1,024, ten causal blocks, eight Top-1 experts per block, an 8,192-token target vocabulary and a 2,048-token configured context. The dilation schedule is 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.

For the full 8,192-token vocabulary, the instantiated shape contains 134,426,200 parameters, including one shared refinement block. The CI audit computes this value from actual PyTorch parameter tensors. If tokenizer training produces fewer than 8,192 entries, the active config and model parameter count are adjusted to that actual vocabulary.


## Hybrid profile and resource limits

`configs/qmini_research_152m.json` contains 152,483,800 parameters at its full 8,192-token vocabulary. The state-space block is inserted in every main block and in the shared refinement block. Top-1 routing means only one expert per block processes a token, but all experts remain part of the stored model and optimizer state.

`configs/qmini_5m.json` contains 5,245,996 parameters. `configs/qmini_5b.json` contains 5,132,630,288 parameters. Use `python -m qmini.cli plan --config configs/qmini_5b.json` to inspect it without allocating a 5B model.

The 5B FP32 parameter, gradient and two AdamW moment arrays require about 82.1 GB before activations and runtime overhead. FSDP/ZeRO and expert/tensor parallel execution are not implemented, so the 5B config is not yet an executable H100 distributed-training setup.
