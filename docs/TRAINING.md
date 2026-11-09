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

The compact convolution-only qmini_base.json and historical qmini_research_134m.json profiles remain available for compatibility. The updated Colab pipeline uses qmini_research_152m.json, a 152,483,800-parameter hybrid CausalConv + gated diagonal SSM + sparse Top-1 MoE profile. The separate qmini_5m.json and qmini_5b.json profiles provide capacity estimates. The 5B profile is not distributed-training-ready: its FP32 parameter/gradient/AdamW state alone is about 82.1 GB before activations. More parameters, steps or refinement iterations do not automatically improve quality.


## Training budget and staged fine-tuning

The Colab notebook uses a configurable initial training-token budget based on 20 tokens per unique active Top-1 parameter. This is only a compute-budget heuristic; sparse-MoE scaling has to be measured. It reports how many passes the requested token budget implies over the available training stream, making corpus repetition visible.

For a separate instruction/SFT stage, use `--init-from /path/to/pretrain/final_model/model.pt`. The trainer loads only the model state dictionary and creates a new optimizer. The source path/hash is kept in the run manifest and verified when resuming that fine-tuning run. Use response-only masks for structured chat examples.

## 5B resource boundary

The 5B profile's FP32 weights, gradients and two AdamW moments require approximately 82.1 GB before activation memory, workspaces or fragmentation. The current trainer is single-process and lacks FSDP/ZeRO and expert/tensor parallelism. Do not treat the 5B file as a ready-to-run H100 training job.
