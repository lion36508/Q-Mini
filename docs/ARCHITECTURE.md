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

No Transformer attention, Q/K/V, dot-product sequence mixing or retrieval-attention modules are present. The only sequence mixer is causal left-padded depthwise convolution. Router and expert operations are token-local.

## Causality and receptive field

Each convolution is padded only on the left by `dilation * (kernel_size - 1)`. No future token can affect an earlier output. Main-block receptive field is `1 + (kernel_size - 1) * sum(dilations)`. With kernel size 10 and dilations 1, 2, 4, 8, 16 and 32, it is 568 tokens; default context is 512.

## Quantization

Trainable master weights remain FP32 and are fake-quantized row-wise to {-2,-1,+1,+2} with STE. Activations use per-token AbsMax scaling to signed INT8 [-127,127], also with STE. Packed export stores four 2-bit codes per byte and row scales. Optimizer state is not 2-bit.

During CPU evaluation/inference, QLinear projections and the tied output projection use the C++17 kernel when `libqmini_kernel.so` is available. Activations are quantized to INT8 per token; each output row's quaternary weights are packed to four 2-bit codes per byte; the native inner loop accumulates integer products before applying row/token scales and bias. Packed weights are cached until the underlying parameter version changes. Training and autograd always use the PyTorch STE reference path.

## Sparse experts and refinement

Each token is sent to one expert per block. Router auxiliary loss discourages imbalance but does not guarantee meaningful expert specialization. One additional block is shared and reused from one to four times. This is learned hidden-state refinement, not symbolic theorem proving or a guarantee of correct reasoning.

The authoritative parameter count is computed from `sum(p.numel() for p in model.parameters())` on the instantiated model.

## Native runtime boundary

The native kernel covers the dense linear projections only. Causal depthwise convolution, RMSNorm, token embedding lookup, sparse dispatch and state management remain PyTorch operations. This is intentionally a hybrid runtime; it must not be described as a full integer-only or fully native ARM64 model.
