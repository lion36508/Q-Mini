# Evaluation protocol

Training loss is a learning signal, not proof that Q-Mini can answer real user questions. Keep a versioned, held-out evaluation set that is never mixed into training or tokenizer fitting.

## Language-model evaluation

The standalone evaluator reports token-weighted cross-entropy and perplexity, supervised target count, input throughput, peak CUDA allocation and per-block expert token loads.

~~~bash
python -m qmini.cli eval-lm \
  --model-dir /path/to/final_model \
  --tokens /path/to/held-out/valid.bin \
  --mask /path/to/held-out/valid.mask.bin \
  --batch-size 2 --seq-len 512 --batches 100 --seed 42 \
  --output /path/to/reports/language-eval.json
~~~

Use evaluation streams produced by the same tokenizer and configuration as the model. Keep the document split and tokenizer training split identical. Compare runs with the same data hashes, seed, batch/sequence settings and model configuration.

## Task evaluation

Prepare a UTF-8 JSONL file with one JSON object per prompt. Each row must contain a non-empty prompt and one of reference, expected or answer; category is optional.

~~~json
{"category":"math","prompt":"What is 6 times 7?","reference":"42"}
{"category":"science","prompt":"What gas do plants absorb?","reference":"Carbon dioxide"}
{"category":"coding","prompt":"What does a Python list store?","reference":"An ordered collection of items"}
~~~

Run deterministic greedy evaluation.

~~~bash
python -m qmini.cli eval-tasks \
  --model-dir /path/to/final_model \
  --dataset /path/to/evaluation-tasks.jsonl \
  --max-new-tokens 96 --output /path/to/reports/task-eval.json
~~~

The report contains exact-match and lexical token F1 overall and by category, plus every prompt/reference/prediction. These are transparent string-comparison diagnostics. They do not establish semantic correctness, factuality, instruction adherence or code correctness. Mathematics requires verified numeric scoring; code generation requires execution in a constrained test sandbox; reasoning requires independently checked multi-step tasks.

## Required capability reporting

Track:
- Validation cross-entropy and perplexity, tokens seen and training steps.
- Arithmetic/math exact-match and numeric-answer accuracy.
- Logic and multi-step task accuracy using held-out problems.
- Instruction adherence and multi-turn conversation consistency.
- Language-wise performance and invalid/repetitive-output rates.
- Code generation pass rate on sandboxed unit tests, not subjective examples.
- Router token counts per expert and signs of starvation/collapse.
- Tokens per second, peak RAM/VRAM, cold-start time, model artifact size and thermal stability.
- Quantized-vs-reference numerical error and generation-quality change.

Use fixed datasets, prompts and scoring methods when comparing revisions. Report sample counts and limitations; do not claim benchmark results from training data or cherry-picked samples.


## Hybrid architecture tests

The gated diagonal state-space path is accepted only when the parallel scan agrees with the sequential recurrence, gradients are finite, future-token changes cannot affect earlier logits, and incremental streaming logits match batch logits within the configured prefill length. Longer streaming tests must additionally demonstrate fixed-size convolution/SSM caches and no prompt replay after max_context. These tests establish implementation correctness, not conversational competence.

A capability report must additionally include fixed held-out tests for multi-turn continuity, instruction adherence, Assamese/Hindi/English performance, arithmetic and independently verified maths, science explanation, logic/state tracking, code-generation unit tests in a constrained runner, long-context recall and repetition/termination failures. Exact match and lexical token F1 are diagnostic metrics, not semantic reasoning scores.


## Quantized deployment and sparse/dense comparison

For the 5B design, compare the sparse profile against \`configs/qmini_dense_853m.json\` under a controlled compute budget. Keep tokenizer, train/validation documents, tokens seen, optimizer schedule and evaluation prompts identical. Report:
- Validation NLL/perplexity and response-mask target count.
- Multi-turn consistency, factuality checks, maths/logic accuracy, and sandboxed coding pass rate.
- Per-expert load distribution, starvation/collapse, active compute and total resident model size.
- Q2 packed versus dense/fake-quant output differences, including capability regression by task.
- Prefill throughput, autoregressive decode latency/tokens/sec, peak process RSS, cold start and thermal behavior.

The planner's 1.409 GB CPU working-set estimate for batch 1 is not a measured process RSS and does not establish a 4 GB-phone target. A release claim requires measurements on the actual device with the trained packed archive and tokenizer loaded.
