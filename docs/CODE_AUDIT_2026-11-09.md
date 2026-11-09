# Q-Mini source-code audit — 2026-11-09

## Scope and honesty

This is a **static source and test-coverage audit** of the `main` branch at commit `4aa23e8ed024045988a571eab555c375ac72c1c4`. The review covered the model, configuration, data ingestion, tokenizer, training loop, generation/chat, packed export/runtime, quantization, native projection kernel, evaluation, CLI, benchmark, project docs, and the corresponding test files.

The source was reviewed remotely through GitHub. The full test suite was **not executed in this audit session**, and no training run or hardware benchmark was performed. Findings below distinguish confirmed code defects from engineering gaps and research risks. A static review cannot prove that every possible defect has been found.

**Training ownership:** training remains the user's task. This audit does not start training, download a pretrained checkpoint, or change the project's random-initialization-only rule.

## Executive verdict

Q-Mini is not throwaway code. It already contains a real, non-trivial research implementation: an attention-free causal-convolution language model; an optional gated diagonal state-space path; sparse Top-1 expert routing; a shared iterative refinement block; a tokenizer and document-split pipeline; response-loss masks; a single-device training/checkpoint path; packed Q2 export; a CPU C++ projection primitive; streaming decode; evaluation tools; and meaningful unit tests.

However, it is still a **research prototype**, not yet a demonstrated high-quality general assistant or a proven 5B/4GB-phone model. Code execution, a low-bit archive, a large parameter count, and a falling training loss are not by themselves evidence of assistant capability.

## 1. Architecture review

Current hybrid architecture, in order:

1. Token embedding (tied to the language-model output projection).
2. Main blocks with RMSNorm, gated causal depthwise convolution, optional gated diagonal SSM, Top-1 router and expert MLP.
3. One shared Q-Mini block reused for the configured refinement/thinking passes.
4. Final RMSNorm and tied vocabulary projection.

There is no Transformer self-attention, Q/K/V projection or attention module in the implementation. The current hybrid profile is a custom causal-convolution + diagonal-recurrence + sparse-MoE design. It is **not** an implementation of Mamba-2 or RWKV-7.

### Strengths found in source/tests

- The convolution path is left-padded and has a token-by-token cache path.
- The SSM scan has tests against the sequential recurrence, gradient checks, causality checks and batch-versus-streaming parity tests.
- Streaming state is fixed-size and the tests exercise generation beyond the prefill context limit.
- Top-1 expert routing has a differentiable selected-gate path and a balance auxiliary loss; the one-expert case correctly bypasses the degenerate one-element router.
- Training includes masked response loss, gradient accumulation, clipping, mixed-precision selection, checkpoint/resume contracts and dataset/tokenizer fingerprints.
- Checkpoint loading uses restricted `weights_only=True` loading for model/checkpoint paths.
- The packed loader avoids materializing all dense FP32 matrices at inference and validates tensor names, shapes, dtypes and scales.
- The native C++ primitive has scalar and ARM NEON code paths, and CI compiles the primitive and cross-compiles an ARM64 shared library.

These are real engineering foundations, not evidence that the design is already optimal.

## 2. Confirmed defects reported as focused issues

### #25 — JSON arrays with legal leading whitespace are rejected
File: `qmini/data.py`, `_iter_json_file()` and `_iter_json_array()`.

Format detection skips leading whitespace, then seeks back to the start. The streaming array parser immediately reads one character and requires it to be `[`. Valid files beginning with spaces/newlines therefore fail. The report asks for bounded whitespace skipping and regression cases for spaces, tabs, blank lines and BOM.

https://github.com/lion36508/Q-Mini/issues/25

### #26 — long interactive-chat history can lose role/control markers
File: `qmini/generate.py`, `interactive_chat()`.

The trim branch slices an arbitrary token suffix and calls `tokenizer.decode()` without explicitly preserving special tokens. The Hugging Face Tokenizers API documents `skip_special_tokens=True` as the default for `tokenizers.Tokenizer.decode()`; Q-Mini's role and end markers are special tokens. The result can remove message boundaries, and the arbitrary slice can begin mid-turn. The issue requests turn-aware truncation and a regression test that continues chatting after truncation.

Reference for the Tokenizers decode default: https://huggingface.co/docs/tokenizers/python/latest/api/reference.html

https://github.com/lion36508/Q-Mini/issues/26

### #27 — tokenizer's zero validation-fraction contract is inconsistent
Files: `qmini/tokenizer.py` and `qmini/data.py`.

`train_tokenizer()` accepts `validation_fraction=0`, but delegates to `iter_training_text_records()`, which rejects `valid_fraction=0`. The CLI also accepts the float and fails later in the helper. The issue requests a single documented contract: either support tokenizer fitting with no holdout, or reject zero at the public API/CLI boundary.

https://github.com/lion36508/Q-Mini/issues/27

### #28 — sparse response masks can make random window sampling abort
File: `qmini/train.py`, `_sample_batch()`.

The loader checks that a mask has some supervised tokens somewhere in the full stream. The sampler then makes at most 128 uniformly random proposals per sample and raises if all those windows miss the supervised positions. With extremely sparse response-only labels, the dataset can be globally valid while a training/evaluation batch fails probabilistically. The issue requests a supervised-position-aware or deterministic fallback sampler and tests for sparse/boundary masks.

https://github.com/lion36508/Q-Mini/issues/28

These four reports describe code-level defects or robustness failures; they are not claims that normal default-path training always fails.

## 3. Major architecture, quality and deployment risks

The following are important, but should not be mislabeled as proven bugs:

### A. Long-range information and model quality
The convolution path has a finite receptive field. The diagonal SSM carries a learned compressed state, not exact access to arbitrary earlier tokens. The code correctly documents this boundary. Large context values and repeated refinement do not guarantee retrieval or reasoning. Long-context and real task results are required.

### B. Sparse MoE quality and efficiency
Top-1 routing reduces active expert compute, but all experts still contribute to total model storage and optimizer state. Python-level per-expert dispatch may become a bottleneck. Router balance loss is a useful signal but does not prove specialization or prevent starvation on real corpora. The current benchmark/test scale is not a 5B expert-dispatch validation.

### C. Quantization quality
The Q2 code alphabet is `{-2, -1, +1, +2}`, so it has no exact-zero code. Current fake quantization and packed export apply this representation to matrix weights. A consistent pack/unpack test only proves serialization correctness; it does not prove preserved perplexity, instruction-following, or reasoning. Compare against Q-Mini's own higher-precision checkpoint before treating Q2 as the quality-preserving default.

### D. Native runtime boundary
The C++ primitive accelerates linear projections. Embedding lookup, causal convolution, RMSNorm, router/expert dispatch, state management, tokenizer and generation still involve Python/PyTorch. This is a hybrid packed runtime, not a complete native or integer-only graph. Real target-device memory, latency, thermal and energy measurements are still needed.

### E. 5B training is not implemented by the current trainer
The current trainer selects one CUDA device when available, otherwise CPU. It does not implement distributed process setup, FSDP/ZeRO sharding, or expert parallelism. The 5B configuration's FP32 weights, gradients and two Adam moments alone are roughly 82.1 GB before activations/workspaces. A 5B config file is not a 5B training system.

### F. Assistant capability is unproven
The current task evaluator reports normalized exact match and lexical token F1; these are diagnostics, not semantic correctness. The docs already call out the need for verified math scoring, constrained code execution, multilingual evaluation, multi-turn consistency and other capability measures.

Existing tracked work for these broader gaps:
- #12 — 5B architecture and real 4GB inference proof.
- #13 — optional random-initialized Mamba-2/SSD reference experiment, with no pretrained weights.
- #18 — distributed training infrastructure.
- #19 — evaluation across every supported refinement count.
- #20 — long-sequence/mixed-precision SSM numerical stress tests.
- #21 — reproducible Q-Mini-owned pretraining to SFT workflow.
- #22 — Q2 versus ternary/INT4/FP16 ablations.
- #23 — assistant capability benchmarks.
- #8 — trained-model prefill/decode and target-runtime benchmark.

## 4. Existing areas that looked sound in this static pass

The following were checked and did not produce a new confirmed defect in this pass:

- Parameter-count formulas are reconciled against actual instantiated parameters in CI for the named profiles.
- The parallel SSM prefix composition matches the sequential recurrence in the unit test.
- The stream-state carries the refinement-step count forward; generation does not accidentally change the number of passes on subsequent tokens.
- Training validation aggregates masked NLL by supervised-token count.
- Prepared dataset manifests verify stream and tokenizer hashes; assistant-target masks are required for auto/assistant prepared datasets.
- The JSON top-level array parser is chunked and validates trailing data; issue #25 is a specific leading-whitespace regression, not a claim that the earlier bounded-array work is absent.
- The C++ linear kernel validates shapes/scales and uses 64-bit scalar accumulation; CI compiles the CPU implementation and cross-compiles ARM64.

This section is a record of reviewed paths, not a formal proof of correctness.

## 5. Priority order

1. Fix #25 and #26 first: they affect data ingestion and user-visible multi-turn conversation structure.
2. Resolve #27 and #28: make public API contracts and sparse response-mask sampling reliable.
3. Complete #19 and #20 before trusting results across refinement counts or mixed precision.
4. Complete #21 and #23 before calling a checkpoint a conversational assistant.
5. Complete #22 before claiming the Q2 format preserves model quality.
6. Treat #18 as a prerequisite to any serious 5B training attempt.
7. Treat #12 and #8 as release gates for the stated 5B/4GB target.

## 6. Scope explicitly not performed

- No model training or fine-tuning was started.
- No external pretrained weights were downloaded or loaded.
- No source-code bug fix was merged as part of this audit; the four findings were recorded as issues for controlled fixes and tests.
- The full pytest suite was not run in this session. Review of source and test files is not the same as executing the tests.
- No GPU, Colab, ARM phone, memory, speed, energy or thermal benchmark was performed.
- This report does not claim that every possible defect has been discovered.

## Audit conclusion

Q-Mini has a credible experimental code foundation and is not “useless code.” It needs targeted correctness fixes, numerical/quality ablations, stronger evaluation and a genuinely scalable training/runtime path before it can be described as a powerful general assistant. Keep the from-scratch random initialization requirement unchanged. Make every improvement test-backed, benchmarked and traceable to an issue.
