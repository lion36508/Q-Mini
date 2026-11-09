# Training and data protocol

Place permitted text data in `Google Drive/MyDrive/QMini/datasets/`. Supported formats are TXT, Markdown, JSON, JSONL/NDJSON, CSV and optional Parquet. Conversation rows with a `messages` list or common question/answer fields are converted to dedicated role control tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`) and an explicit `<|end|>` turn marker in newly trained tokenizers. The loader does not download pretrained models or remote corpora.

## Curriculum

1. Language foundation: broad, deduplicated, well-formed text with source provenance.
2. Structured competence: mathematics, explanations, logic, programming and technical documentation.
3. Instruction and dialogue: high-quality multi-turn conversations, task demonstrations and safe tool-use examples.
4. Mixed replay: retain general language alongside target skills to reduce catastrophic forgetting. Keep held-out validation documents isolated from training.

## Pipeline

Split documents deterministically first; train the local byte-level BPE tokenizer only on unique training documents (never validation documents); then store token IDs as uint16; run a tiny smoke training; train with AdamW, gradient accumulation and warmup/cosine scheduling; evaluate periodically; save model/optimizer/config/RNG checkpoints; export model and tokenizer together.

The trainer refuses to overwrite an existing checkpoint unless `--resume` is set. Token preparation refuses to overwrite existing train/valid streams. For version-3 prepared datasets, the trainer verifies tokenizer, train/validation stream and supplied mask SHA-256 fingerprints against dataset.json; auto/assistant loss modes require both response masks. Resume rejects a mismatched config, data, tokenizer or mask. Begin with batch size 1, sequence length 128 and gradient accumulation 1; scale only after measuring the actual accelerator memory.

Checkpoint trust: use `latest.pt` only from a training run you created or otherwise trust. Resume and model initialization use PyTorch's restricted `weights_only=True` loader; arbitrary pickle loading is not used. Do not accept untrusted checkpoints or model files as safe merely because restricted loading is enabled. Keep a copy of a known-good checkpoint before resuming or fine-tuning.

The compact convolution-only qmini_base.json and historical qmini_research_134m.json profiles remain available for compatibility. The updated Colab pipeline uses qmini_research_152m.json, a 152,483,800-parameter hybrid CausalConv + gated diagonal SSM + sparse Top-1 MoE profile. The separate qmini_5m.json and qmini_5b.json profiles provide capacity estimates. The 5B profile is not distributed-training-ready: its FP32 parameter/gradient/AdamW state alone is about 82.1 GB before activations. More parameters, steps or refinement iterations do not automatically improve quality.


## Training budget and staged fine-tuning

The Colab notebook uses a configurable initial training-token budget based on 20 tokens per unique active Top-1 parameter. This is only a compute-budget heuristic; sparse-MoE scaling has to be measured. It reports how many passes the requested token budget implies over the available training stream, making corpus repetition visible.

For a separate instruction/SFT stage, use `--init-from /path/to/pretrain/final_model/model.pt`. The trainer loads only the model state dictionary and creates a new optimizer. The source path/hash is kept in the run manifest and verified when resuming that fine-tuning run. Use response-only masks for structured chat examples.

## 5B resource boundary

The 5B profile's FP32 weights, gradients and two AdamW moments require approximately 82.1 GB before activation memory, workspaces or fragmentation. The current trainer is single-process and lacks FSDP/ZeRO and expert/tensor parallelism. Do not treat the 5B file as a ready-to-run H100 training job.


## Near-duplicate isolation

Exact SHA-256 deduplication is always active. Optionally group near-duplicates using a disk-backed 64-bit SimHash over word trigrams. The threshold is the maximum Hamming distance from 1 to 3; zero disables SimHash grouping and preserves exact-only behavior. The entire similarity group is assigned to one split, and the dataset manifest records the threshold, group count and number of documents absorbed into groups.

Use the **same** holdout fraction and near-duplicate distance in both commands:

\`\`\`bash
python -m qmini.cli tokenizer --data-dir DATA --output tokenizer.json \\
  --vocab-size 32768 --valid-fraction 0.02 \\
  --near-duplicate-hamming-distance 3 \\
  --config-output config.json

python -m qmini.cli prepare --data-dir DATA --tokenizer tokenizer.json \\
  --output-dir prepared --valid-fraction 0.02 \\
  --near-duplicate-hamming-distance 3 --loss-mask auto
\`\`\`

This is an approximate similarity filter, not semantic plagiarism detection. Inspect the grouped-document counts and keep the validation corpus independently curated.


## Validation metric definition

Periodic \`eval_loss\` is the global token-weighted cross-entropy over all sampled validation targets:

\`eval_loss = sum(masked per-token NLL) / sum(supervised target tokens)\`.

For assistant-only datasets, context tokens do not contribute to the numerator or denominator. The \`eval_supervised_tokens\` field records the denominator. \`eval_perplexity\` is \`exp(eval_loss)\` when representable. This matches standalone \`eval-lm\` and avoids bias from an unweighted arithmetic mean of batch means with unequal response lengths.



## Colab learning proof and optional own-checkpoint SFT

\`colab/QMini_Training.ipynb\` now defaults to a **50-step small-model smoke run**. It measures held-out token-weighted NLL on a fixed evaluation seed before and after training and records a \`learning_proof_*.json\` file in Drive. A long 152M research run is disabled until the user explicitly sets \`RUN_LONG_TRAINING = True\`; it is capped by \`LONG_MAX_OPTIMIZER_STEPS\`, reports the actual token budget/corpus passes and resumes from a hash-keyed Drive checkpoint. The 5B configuration is not a Colab training target.

The optional SFT stage is disabled by default (\`RUN_OWN_CHECKPOINT_SFT = False\`). Put a separate licensed chat/instruction corpus under \`MyDrive/QMini/artifacts/sft_datasets/\`, with rows containing \`messages\` or supported instruction/answer fields, then set the flag to \`True\`. The notebook:
1. Uses the matching tokenizer/config from a Q-Mini-produced checkpoint.
2. Prepares a separate SFT dataset with assistant-only loss masks and a held-out split.
3. Initializes only from the Q-Mini model's packed \`model.qmini.npz\` artifact via \`--init-from\`; no external model weights are accepted and the new optimizer is fresh.
4. Resumes to a new hash-keyed run directory on Drive.
5. Optionally compares the SFT checkpoint with its pre-SFT source on the separate \`sft-heldout-tasks.jsonl\` file, reporting task-pass-rate and per-language changes.

An NLL decrease or task-score increase on a small sample is evidence only for that run and dataset. Review sample counts and cases; do not generalize it to broad assistant capability.




## Reliable sampling with sparse response masks

When a prepared assistant-only mask contains very few positive target tokens, the trainer builds a disk-backed \`SupervisedWindowIndex\` once before model/optimizer allocation. It counts valid windows in bounded blocks and samples uniformly from all starts whose shifted label mask contains at least one supervised target. It preserves the exact alignment \`mask[start + 1:start + seq_len + 1]\`. It fails early only when the stream has no legal supervised window, rather than failing probabilistically after 128 random proposals.




Final model manifests now bind the dense model.pt snapshot and packed model.qmini.npz export with independent SHA-256 fingerprints and byte counts. Quantization ablations reject a dense reference whose manifest, config, parameter count, packed artifact size or hashes do not match. This is separate from the resumable latest.pt checkpoint contract.
