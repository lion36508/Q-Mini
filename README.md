# Q-Mini — Text-Only CausalConv-MoE Language Model

**Q-Mini is a from-scratch language-model research project aimed at becoming a compact, capable, offline-friendly text assistant.** It is not an Android app, a hosted-chatbot wrapper, or a fine-tune of downloaded pretrained weights. The model starts from random initialization. PyTorch supplies tensor operations, automatic differentiation and optimization; the tokenizer is trained locally from permitted corpus data.

> **Current status:** the repository contains an executable small-model baseline, training and data pipeline, reference text generation, packed-weight export utilities, a C++ projection primitive, tests and CI. It does **not** yet contain a trained, generally capable assistant, a complete packed-integer runtime, a finished Android app, persistent personal memory or unrestricted autonomous tool execution. This README separates implemented code from long-term goals.

[![Q-Mini CI](https://github.com/lion36508/Q-Mini/actions/workflows/tests.yml/badge.svg)](https://github.com/lion36508/Q-Mini/actions/workflows/tests.yml)

## 1. Project vision

The long-term goal is to train Q-Mini into a useful general-purpose **text assistant** that can understand user instructions, maintain coherent conversations, explain difficult topics, solve structured problems, assist with programming and help users work through technical tasks.

Intended capability areas:

- **Conversation and language:** multi-turn dialogue, instruction following, context-aware answers, explanations, summaries, rewriting and question answering.
- **Reasoning and problem solving:** structured problem decomposition, logic, comparisons, planning and checking intermediate results.
- **Mathematics and science:** arithmetic, algebra, geometry, basic statistics, physics, chemistry, scientific concepts and engineering explanations.
- **Programming and software engineering:** code explanation, small program generation, debugging, algorithm review, test generation and development workflows.
- **Technical knowledge:** computers, operating systems, electronics, hardware, networks, software tools and engineering concepts.
- **Text and document work:** summarization, classification, extraction, transformation and answers grounded in supplied text.
- **Multilingual communication:** language coverage developed from verified training data, with English and Indian-language coverage as important evaluation targets. Fluency must be demonstrated by held-out tests, not assumed from tokenizer support.
- **Efficient local use:** progressively reduce inference cost and memory footprint so a trained model can run without a paid hosted inference API.

These are **training and product objectives**, not claims that the current random-initialized baseline already performs these tasks well.

## 2. Non-negotiable design constraints

1. **Text-only model.** Q-Mini is for text and language. Image understanding, image generation, vision encoders, video understanding and multimodal branches are outside this model's scope.
2. **No Transformer self-attention.** No Q/K/V projections, dot-product attention, softmax attention or retrieval-attention modules.
3. **Causal sequence processing.** A token at a future position must never influence an earlier position. The sequence mixer is a left-padded causal 1D depthwise convolution. This is a text-sequence operation, not an image CNN.
4. **From-scratch training.** Do not silently load or fine-tune third-party pretrained model weights. The tokenizer is trained locally from the selected corpus.
5. **Sparse expert computation.** Use learned Top-1 routing across the configured feed-forward experts in the current baseline; monitor expert usage rather than assuming specialization.
6. **Low-bit research path.** Use quaternary weights and INT8 fake-quantized activations with an STE during model training. Clearly distinguish fake quantization, packed storage and true integer inference.
7. **Reproducible development.** Version architecture/configuration, test causal behavior, record data provenance, keep evaluation data held out and report actual parameter counts and runtime measurements.
8. **Free/open-source-first development.** Prefer open-source libraries and local execution. Do not make a paid hosted API or proprietary model service a required dependency.

## 3. Current architecture baseline

The current configuration is a compact research baseline, not a claimed 5B or production-scale model.

| Component | Current baseline |
|---|---:|
| Vocabulary target | 4,096; actual tokenizer size is written into active config |
| Model width | 256 |
| Expert hidden width | 384 |
| Main causal blocks | 6 |
| Experts per block | 4 |
| Active experts per token | Top-1 |
| Maximum context | 512 tokens |
| Convolution kernel width | 10 |
| Convolution dilations | 1, 2, 4, 8, 16, 32 |
| Shared refinement | One block reused for 1–4 passes |
| Weight alphabet | {-2, -1, +1, +2} |
| Activation quantization | Per-token AbsMax INT8 fake quantization |

### Forward path

~~~text
Token IDs
  -> Quaternary token embedding
  -> Six Q-Mini causal blocks
       -> RMSNorm
       -> Quantized content/gate projection
       -> Left-padded depthwise causal convolution
       -> SiLU(content) * sigmoid(gate)
       -> Output projection + residual
       -> RMSNorm
       -> Sparse Top-1 expert router
       -> Selected expert MLP + residual
  -> Shared refinement block (1–4 iterations)
  -> Final RMSNorm
  -> Tied embedding projection
  -> Next-token logits
~~~

With kernel width 10 and the configured dilations, the six main blocks have a theoretical receptive field of 568 tokens. The configured context is 512 tokens. Parameter count must always be read from the instantiated model using the sum of parameter tensor sizes; configuration changes alter the total.

### Quantization — what it does and does not mean

- Trainable master weights remain FP32 and are mapped to {-2, -1, +1, +2} during the fake-quantized forward pass using a straight-through estimator.
- Activations use per-token AbsMax INT8 fake quantization, also with an STE.
- Export packs four 2-bit weight codes per byte and stores row scales; selected metadata tensors are stored separately.
- FP32 master weights, gradients and optimizer states are still used during ordinary training. The small packed archive size is **not** the training RAM requirement.
- The C++17 kernel is now wired into `QLinear` and the tied language-model output projection for CPU evaluation/inference when `qmini/libqmini_kernel.so` is present. It consumes per-token INT8 activations and packed quaternary weights, using integer accumulation in the native inner loop. If the library is absent, Q-Mini transparently uses the PyTorch reference path.
- The causal depthwise convolution, RMSNorm, embedding lookup and sparse dispatch are still executed by PyTorch. This is a hybrid native/PyTorch CPU path, not a complete integer-only model runtime.
- The packed NPZ archive remains a storage/export format; direct execution from that archive is not yet implemented. Reference model loading still uses `model.pt`.

The shared refinement block is a learned hidden-state update that reuses parameters. It is not a symbolic theorem prover and does not guarantee correct reasoning. More refinement passes increase inference compute and must be evaluated.

## 4. Implemented now vs. intended later

| Area | Implemented in this repository | Longer-term objective |
|---|---|---|
| Language model | From-scratch causal-convolution MoE baseline | Improve language quality through measured training |
| Conversation | Reference text generation and terminal multi-turn chat | Reliable, context-aware assistant dialogue |
| Reasoning | Shared latent refinement mechanism and next-token training | Demonstrated math, logic and multi-step problem-solving accuracy |
| Coding | General text-generation path can generate code | Dedicated code data, debugging evaluations and executable test-based scoring |
| Data | Local BPE tokenizer and text/chat data readers | Larger curated, deduplicated, licensed and balanced corpora |
| Training | AdamW, gradient accumulation, available mixed precision, assistant-only loss masks for structured chat/Q&A, evaluation and fingerprinted resumable checkpoints | Longer curriculum training and measured instruction tuning |
| Experts | Sparse Top-1 MoE and router/load statistics | Verified useful specialization without expert collapse |
| Quantization | Quaternary/INT8 fake quantization and packed export | Calibrated, numerically verified low-bit execution |
| Memory | Opt-in local SQLite memory with FTS5 search, explicit save/recall/delete and JSON export | Improve retrieval and add user-controlled import/backup workflows |
| Tools/tasks | CLI for project operations; no model-controlled general tool executor | Permission-gated registered tools for carefully scoped tasks |
| Mobile | Standalone C++ projection primitive | Complete ARM64 runtime, integrated kernels and measured on-device inference |
| Multilingual | Tokenizer and text pipeline support Unicode text | Language-specific training and held-out language evaluations |

A feature in the right-hand column is a roadmap item, not an existing capability.

## 5. Training and learning plan

A chat-only corpus is insufficient for the complete target. Training data should be versioned by source, license, language, quality and task family.

### Suggested curriculum

1. **Language foundation:** clean, deduplicated, well-formed text with known provenance and permitted licenses.
2. **Technical and educational material:** mathematics, logic, science, programming, engineering, documentation and explanations.
3. **Instruction and conversation:** role-marked multi-turn examples, clear task instructions, context retention and grounded responses.
4. **Reasoning practice:** verified arithmetic, algebra, logic, structured problem solving and worked explanations with checked answers.
5. **Programming practice:** code explanation, algorithm design, debugging examples, test generation and code-review cases with verified expected outcomes.
6. **Safety and calibration:** uncertainty, correction, safe boundaries, refusal examples and instruction hierarchy.
7. **Mixed replay:** continue mixing core language examples with specialist skills to reduce catastrophic forgetting.

Keep validation documents separate from training documents. Do not treat unverified synthetic answers as ground truth. Respect dataset licenses and remove private or sensitive data.

### Training pipeline

- Discover supported local data files: TXT, Markdown, JSON, JSONL/NDJSON, CSV and optional Parquet.
- Split documents deterministically and de-duplicate exact documents using disk-backed indexes.
- Train the local byte-level BPE tokenizer on training documents only; validation documents are excluded from tokenizer vocabulary learning.
- Convert documents into role-marked text where appropriate and create uint16 token streams with aligned uint8 loss masks.
- In `auto` mode, plain text receives ordinary next-token loss while structured chat/Q&A applies supervised loss to assistant response spans and leaves user/system text as context.
- Run a tiny smoke training before the full run.
- Train next-token prediction with cross-entropy and router load-balancing loss.
- Use AdamW, gradient accumulation, warmup/cosine scheduling and available GPU mixed precision.
- Save model, optimizer, configuration and RNG state so a compatible run can resume.
- Evaluate on held-out data and retain JSONL training metrics.

## 6. Evaluation: capability must be measured

A bigger model, more experts or additional refinement steps do not automatically make a better assistant. Record:

- Validation cross-entropy and perplexity, tokens seen and training steps.
- Arithmetic/math exact-match and numeric-answer accuracy.
- Logic and multi-step task accuracy using held-out problems.
- Instruction adherence and multi-turn conversation consistency.
- Language-wise performance and invalid/repetitive-output rates.
- Code generation pass rate on sandboxed unit tests, not just subjective examples.
- Router token counts per expert and signs of starvation/collapse.
- Tokens per second, peak RAM/VRAM, cold-start time, model artifact size and thermal stability.
- Quantized-vs-reference numerical error and generation-quality change.

Use fixed datasets, prompts and scoring methods when comparing revisions. Report sample counts and limitations; do not claim benchmark results from training data or cherry-picked samples.

## 7. Memory and task execution

**Local memory is implemented as an optional subsystem outside the neural model.** It uses SQLite for persistence and FTS5 full-text search when the local SQLite build provides it; a literal-substring fallback is available when FTS5 is absent. Memory is never written automatically.

Start chat with `python -m qmini.cli chat --model-dir PATH --memory-db PATH_TO_LOCAL_SQLITE`. The chat supports `/remember TEXT`, `/recall QUERY`, `/memories` and `/forget ID`. Relevant saved notes may be surfaced as explicit memory context during a turn. The CLI also provides `python -m qmini.cli memory --db PATH add --text TEXT`, `search --query QUERY`, `forget --id ID` and `export --output BACKUP.json`. Records can be individually deleted or exported to JSON. SQLite database files are ignored by Git.

Model-directed task execution remains a future, permission-controlled extension. It should use typed tool schemas and a small allow-list; consequential actions must require confirmation. No unrestricted shell, filesystem, account or network actions should occur merely because the model generated a command.

## 8. Hardware and deployment direction

**Development and training:** GitHub is the source of truth for code, configuration, tests and documentation. Google Colab is the GPU training environment. Google Drive is the recommended location for datasets, tokenizer artifacts, token streams, checkpoints, logs and model exports.

**Local CPU inference:** the hybrid path uses the C++17 packed-quaternary × INT8 projection kernel when the shared library is available; AArch64/NEON builds use vector widening multiply and integer accumulation, while other architectures use the scalar integer fallback. The convolution, RMSNorm, embedding lookup and sparse dispatch remain PyTorch. The benchmark script reports the selected projection backend; performance must be measured on each target phone.

**Android ARM64 / Termux:** the current native projection source has an AArch64/NEON compile path and CI cross-compiles it. A complete offline runtime still requires native embedding lookup, causal depthwise convolution, RMSNorm, sparse expert dispatch, output handling and tokenizer integration, plus differential tests and real-device benchmarks. CI cross-compilation is not an Android phone performance benchmark. This repository does not currently build an APK or complete Android runtime.

## 9. Roadmap

- **Phase A — Core correctness:** architecture invariants, causal tests, parameter audit, tokenizer/data validation, smoke training and safe checkpoint resume.
- **Phase B — Training quality:** curated language/technical corpora, longer runs, held-out validation, router diagnostics and repeatable checkpoints.
- **Phase C — Assistant competence:** instruction and multi-turn data, mathematics/logic, science and programming curricula, capability-specific evaluations and calibration.
- **Phase D — Efficient inference:** packed-format validation, quantized/reference comparisons, optimized kernels and measured CPU performance.
- **Phase E — Local assistant extensions:** optional user-controlled memory and permission-gated tools, only after core model behavior is reliable.
- **Phase F — ARM64 deployment:** complete native graph, differential tests, performance and memory measurements on supported phones.

Each phase should be accepted using tests and measurements, not by adding a feature name to the README.

## 10. Quick start

### Local environment

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python QMINI_MASTER.py verify
python -m qmini.cli audit --config configs/qmini_smoke.json
~~~

### Google Colab

1. Put permitted training files in Google Drive/MyDrive/QMini/datasets/.
2. Open colab/QMini_Training.ipynb and select an available GPU runtime.
3. Run the notebook from top to bottom. It mounts Drive, obtains the source, installs dependencies, trains the tokenizer, prepares data and runs a small smoke training before the base run.
4. Keep artifacts in Google Drive/MyDrive/QMini/artifacts/ so they survive Colab runtime resets.

The notebook and CLI contain the tokenizer, data preparation, training, resume, generation, chat and held-out evaluation commands. The training step count is the target total optimizer step, not an extra number of steps after resume.

## 11. Repository map

- QMINI_MASTER.py — single project entry point.
- qmini/model.py — causal-convolution MoE model.
- qmini/quantization.py — STE fake quantization and quaternary packing.
- qmini/tokenizer.py, qmini/data.py — local tokenizer and corpus pipeline.
- qmini/train.py — training, validation and resumable checkpoints.
- qmini/generate.py — cached autoregressive generation and terminal chat.
- qmini/evaluate.py — held-out language-model and prompt/reference evaluation.
- qmini/export.py — packed archive export and verification.
- qmini/memory.py — opt-in local SQLite/FTS5 memory with explicit CRUD and JSON export.
- qmini/native_kernel.cpp, qmini/native.py — packed INT8/quaternary CPU projection, AArch64/NEON path, scalar fallback and backend detection.
- scripts/benchmark.py — repeatable model-forward throughput and resource benchmark.
- configs/ — base and smoke configurations.
- tests/ — model, causality, quantization, data, training, export and native tests.
- docs/ — architecture, training, evaluation and project scope.
- colab/ — Colab training notebook.
- .github/workflows/ — automated CI.

## 12. Important limitations and data security

- A random initialization is not a useful assistant. Real capability requires enough high-quality training data, compute and independent evaluation.
- The current baseline is a compact research model, not a large foundation model or a claim of state-of-the-art performance.
- FP32 master weights and optimizer states are used in training.
- The packed archive is not yet a complete integer-only runtime.
- The native kernel accelerates CPU linear projections only; convolution, normalization, embedding lookup and MoE dispatch remain on the PyTorch reference path.
- There are no bundled trained weights or private datasets.
- Local memory is stored in the path supplied by the user; protect that database as personal data and export/delete records deliberately.
- The GitHub repository is public. Never commit credentials, private datasets, personal information or private checkpoints. Keep such artifacts in private storage.

**Project principle:** build a truthful, testable, efficient text model first; add assistant capabilities only when the training data, implementation and evaluations support them.
