# Q-Mini Q2 weight quantization specification

## Non-negotiable production weight format

The production matrix-weight format is exactly two bits per weight, with four normalized signed values:

| 2-bit code | Normalized value |
|---|---:|
| `00` | `-1.0` |
| `01` | `-0.5` |
| `10` | `+0.5` |
| `11` | `+1.0` |

No production matrix weight may be encoded using ternary, INT4, INT8, FP16 or another wider codebook. INT8 activations are a separate activation representation and do not change the two-bit weight format.

## Row scale and reconstruction

For each output row (w_r):

[
s_r = 2 \cdot \operatorname{mean}(|w_r|)
]

The reconstructed value is:

[
\hat{w}_{r,i} = c_{r,i} \cdot s_r
]

where (c_{r,i}) is one of the four normalized codebook values above. The larger-magnitude symbol is selected when (|w_{r,i}|/s_r \ge 0.75); otherwise the smaller-magnitude symbol is selected. Sign chooses positive or negative. Four symbols are packed per byte, from least-significant pair to most-significant pair.

### Zero behavior

The four-symbol alphabet has no zero code. In a nonzero row, an exact-zero input element maps deterministically to (+0.5s_r). This is a known approximation trade-off, not exact sparse storage.

For a completely zero row, (s_r=0) and every decoded weight is exactly zero. Export/runtime validation allows finite non-negative scales and rejects negative or non-finite scales.

## Training versus deployment

Training uses FP32 master parameters and optimizer state with a straight-through estimator; the forward path is fake-quantized to the Q2 reconstruction above. This is quantization-aware training, not a claim that Adam slots or gradients are two-bit.

The packed deployment archive stores:
- matrix-weight symbols at exactly two bits each;
- FP32 per-row scales;
- one-dimensional biases/norm vectors in FP16;
- configuration and tensor metadata.

The total archive therefore exceeds the raw (2N/8)-byte symbol payload. Scale/vector/metadata overhead must be itemized and measured.

Activations continue to use per-token AbsMax signed INT8 fake quantization. That does not widen the model-weight representation.

## Archive compatibility

- New archives use `qmini-packed-v2` and declare the normalized Q2-v2 codebook and scale convention.
- Legacy `qmini-packed-v1` archives retain their historical `{-2,-1,+1,+2}` interpretation and original scale convention.
- Runtime fallback, embedding/depthwise paths and C++ projection must follow the archive version.
- A legacy archive must never be silently reinterpreted as v2.

## Experimental helper scope

Ternary and INT4 helper functions are not production Q-Mini formats and must never be selected by default training, export, checkpoint loading or inference. They may remain isolated mathematical utilities, but do not replace or widen the production Q2 model.

## Required regression and quality gates

1. Tests prove normalized fake-quantized values belong only to the four-symbol codebook.
2. Pack/unpack tests cover odd widths, sign extremes, all-zero rows and exact-zero entries in nonzero rows.
3. Export metadata identifies the v2 codebook and scale convention.
4. The loader reads v1 and v2 without mixing their scale semantics.
5. PyTorch fallback, packed embedding/depthwise paths and native scalar/NEON projection agree within stated tolerances.
6. Save → load → inference parity is tested on a Q-Mini-owned checkpoint.
7. Held-out NLL/perplexity and task-level capability deltas are measured against the same higher-precision Q-Mini checkpoint before quality impact is called acceptable.
8. The final artifact report shows two bits per matrix weight plus separately itemized scales, vectors and metadata.

Passing serialization tests establishes format correctness only; it does not prove preservation of language-model or assistant quality.
