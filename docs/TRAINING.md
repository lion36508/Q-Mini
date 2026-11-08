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

The compact qmini_base.json profile is retained for smoke tests and low-resource development. The Colab pipeline uses the separate 134M-class qmini_research_134m.json profile by default. Its full-vocabulary shape is runtime-audited at 134,426,200 parameters. Neither profile is a state-of-the-art assistant by itself: quality requires substantial permitted data, compute and held-out evaluation. More parameters, steps or refinement iterations do not automatically improve quality.
