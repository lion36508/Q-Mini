# Q-Mini — Project Scope and Capability Specification

**Purpose:** record the intended product direction, non-negotiable architecture rules, capability goals, implementation status and acceptance criteria. This document is a scope contract, not a claim that every roadmap feature already exists.

## 1. Product objective

Build a from-scratch, text-only language model that can ultimately serve as a compact general-purpose assistant. The desired assistant should support natural conversation, instruction following, technical explanation, mathematics and logic, science/engineering questions, programming assistance and structured text tasks. It should progressively become efficient enough for local and offline inference.

The project is not an Android IDE, not an image model, not a multimodal model and not a wrapper around a third-party pretrained chatbot.

## 2. Hard architecture constraints

- No Transformer self-attention.
- No Q/K/V projections, dot-product attention, softmax attention or retrieval-attention.
- No image encoder, image CNN, vision adapter, VAE, GAN or video model branch.
- Text sequence mixing uses strictly causal left-padded 1D depthwise convolution.
- Decoder-only next-token language modeling, initialized from random parameters.
- Sparse Top-1 expert routing in the current baseline.
- Quaternary weight alphabet {-2, -1, +1, +2} with FP32 master parameters and STE.
- Per-token AbsMax INT8 activation fake quantization with STE.
- RMSNorm and residual paths.
- Shared iterative refinement reuses one block for a configurable number of passes; it is not a symbolic proof engine.
- Use local, open-source-first tooling. No required paid API, hosted model or pretrained model weights.

The convolution mentioned here is a text-sequence operator; it does not imply image or visual processing.

## 3. Current baseline

| Item | Baseline |
|---|---:|
| Vocabulary target | 4,096; actual tokenizer size is saved in active config |
| Width | 256 |
| Expert hidden width | 384 |
| Main blocks | 6 |
| Experts per block | 4 |
| Active experts | 1 (Top-1) |
| Context limit | 512 tokens |
| Kernel width | 10 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| Shared refinement | 1–4 repeated passes |
| Weight values | -2, -1, +1, +2 |
| Activation path | Per-token INT8 fake quantization |

Parameter accounting must be computed from the instantiated model. The six main causal convolution blocks have a theoretical receptive field of 568 tokens.

## 4. Capability targets

### 4.1 Language and conversation
- Understand ordinary written prompts and respond coherently.
- Follow explicit instructions and preserve relevant conversation context.
- Explain, summarize, rewrite, classify and transform text.
- Maintain consistent multi-turn dialogue.
- Support multilingual training and evaluation; language quality must be measured independently for each language.

### 4.2 Reasoning and problem solving
- Break structured problems into manageable steps.
- Compare options and explain the basis of a conclusion.
- Solve logic and multi-step tasks using verified training examples.
- Express uncertainty and correct mistakes when evidence requires it.
- Use evaluation tasks designed to detect memorized answers and data leakage.

### 4.3 Mathematics, science and engineering
- Arithmetic, algebra, geometry and introductory statistics.
- Physics, chemistry and general scientific explanations.
- Computing, electronics, hardware, operating systems, networks and engineering fundamentals.
- Checkable answers and transparent scoring rather than unverified generated explanations.

### 4.4 Programming and technical work
- Explain code and algorithms.
- Generate code from explicit specifications.
- Diagnose and explain common programming errors.
- Propose fixes, create tests and review code.
- Evaluate generated programs by running tests in a constrained sandbox; do not treat plausible-looking code as proof of correctness.

### 4.5 Text and document tasks
- Summarization, extraction, classification, question answering and transformation.
- Conversation-formatted datasets and structured Q/A records.
- Source-aware answers when source text is supplied.

### 4.6 Persistent memory and task tools — optional future extension

The present model core has no persistent user-memory database or autonomous tool executor. If added later, those systems must be separate, auditable and permission-controlled:

- Long-term memory is opt-in and user-visible.
- Users can inspect, correct, export and delete saved records.
- The model may propose a registered action, but a host policy validates the schema and permissions.
- Consequential actions require user confirmation.
- No unrestricted command execution, hidden file changes or silent network/account actions.

## 5. Training-data design

Maintain data categories with source provenance and licenses:

1. General language and reading comprehension.
2. Educational, scientific and technical explanation.
3. Mathematics and logic with checked answers.
4. Programming, algorithms, debugging and test-driven examples.
5. High-quality instruction following and multi-turn conversation.
6. Multilingual and Indic-language examples where data quality is sufficient.
7. Safety, uncertainty and factual correction examples.

The pipeline should:
- Deduplicate and normalize data.
- Remove personal/private information and respect licenses.
- Split by document or source before tokenization to avoid train/validation leakage.
- Keep test data separate from training and tuning.
- Track corpus versions, language mix and tokens seen.
- Reject malformed records and report skipped files/rows.
- Preserve original datasets and checkpoints; never overwrite without explicit intent.

## 6. Training and checkpoints

Current implementation includes next-token cross-entropy, router load-balancing auxiliary loss, AdamW, gradient accumulation, warmup/cosine scheduling, available GPU mixed precision, evaluation and checkpoint/resume support.

Checkpoint metadata should remain sufficient to reproduce a run: model state, optimizer state, configuration, step number and RNG state. Resumption must reject a mismatched architecture/configuration. Logs should include train loss, validation loss/perplexity, learning rate, tokens seen, step and expert-load statistics.

## 7. Inference and deployment

- PyTorch reference inference is the correctness baseline.
- Packed quaternary export is a storage format; it is not evidence of native integer execution.
- The current C++17 primitive accelerates/tests a single projection operation only and is not wired into the full model.
- A complete Android ARM64/Termux runtime is a later deliverable. It must implement embedding, quantized projection, causal convolution, normalization, expert dispatch, output projection and tokenizer compatibility.
- Native output must be compared to the reference implementation with stated numerical tolerances.
- Device tests must report peak RAM, model size, cold-start time, tokens/second, thermal behavior and long-session stability.
- No APK or Android UI is implied by this model repository.

## 8. Acceptance criteria by phase

### Phase A — Foundation correctness
- Architecture tests prove no attention/QKV components exist.
- Causality tests prove future tokens do not change earlier logits.
- Quaternary packing/unpacking and STE gradient tests pass.
- Router gradients and per-expert assignment counts are verified.
- Data, tokenizer, checkpoint/resume and export tests pass.
- CI is green on the current source revision.

### Phase B — Training reproducibility
- Smoke training completes end to end.
- A resumed run continues the same optimizer step and rejects config mismatches.
- Validation data is document-separated and never silently reused for training.
- Logs and checkpoints are persisted outside ephemeral training runtimes.

### Phase C — Capability improvement
- Instruction and conversation evaluations are held out.
- Math/logic scoring uses verifiable answers.
- Code-generation scoring uses executable sandboxed tests.
- Per-language performance is measured instead of inferred from tokenization support.
- Improvements are compared against a fixed baseline and data split.

### Phase D — Low-bit inference
- Packed archive round-trip is verified for every tensor.
- Quantized generation is compared with reference generation.
- Native projection and full-model paths have explicit parity tests.
- Memory and speed are measured on target hardware.

### Phase E — Assistant extensions
- Memory has user controls, provenance and data deletion.
- Tool schemas, allow-lists, error handling and confirmation gates are tested.
- Tool actions cannot run merely because generated text contains a command.

### Phase F — ARM64/Termux runtime
- Full graph runs without Python/PyTorch if that remains the chosen deployment goal.
- Correct tokenizer/config/model bundle loads.
- Differential tests pass on supported devices.
- Real-device resource and stability reports are published.

## 9. Non-claims

Q-Mini must not be described as:
- Already trained or already a capable general-purpose assistant.
- A 5B model or a model with an unverified parameter count.
- A Transformer, RWKV implementation or official Microsoft BitNet implementation.
- A model with currently working long-term memory or unrestricted agents.
- A complete integer-only or full ARM64 runtime.
- A system that has passed benchmarks unless results and test conditions are recorded.

The project is successful when measured capabilities, reproducible training and a reliable local runtime catch up with the stated goals—not when the feature list becomes longer.
