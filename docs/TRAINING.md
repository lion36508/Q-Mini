# Training and data protocol

Place permitted text data in `Google Drive/MyDrive/QMini/datasets/`. Supported formats are TXT, Markdown, JSON, JSONL/NDJSON, CSV and optional Parquet. Conversation rows with a `messages` list or common question/answer fields are converted to role-marked text. The loader does not download pretrained models or remote corpora.

## Curriculum

1. Language foundation: broad, deduplicated, well-formed text with source provenance.
2. Structured competence: mathematics, explanations, logic, programming and technical documentation.
3. Instruction and dialogue: high-quality multi-turn conversations, task demonstrations and safe tool-use examples.
4. Mixed replay: retain general language alongside target skills to reduce catastrophic forgetting. Keep held-out validation documents isolated from training.

## Pipeline

Train a local byte-level BPE tokenizer; split whole documents deterministically into train and validation streams; store token IDs as uint16; run a tiny smoke training; train with AdamW, gradient accumulation and warmup/cosine scheduling; evaluate periodically; save model/optimizer/config/RNG checkpoints; export model and tokenizer together.

The trainer refuses to overwrite an existing checkpoint unless `--resume` is set. Token preparation refuses to overwrite existing train/valid streams. Resume rejects a mismatched config. Begin with batch size 1, sequence length 128 and gradient accumulation 1; scale only after measuring the actual accelerator memory.

The baseline is a compact 8M-class research experiment, not a state-of-the-art assistant. Quality requires substantial permitted data, compute and held-out evaluation. More parameters, steps or refinement iterations do not automatically improve quality.
