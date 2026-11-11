# Q-Mini training-ready Colab build

## Recommended Colab path

Use [QMini Training-Ready Model Builder](https://colab.research.google.com/github/lion36508/Q-Mini/blob/main/colab/QMini_Training_Ready_Build.ipynb).

This notebook is for the repository's **152M-class hybrid research profile**. It builds a package that the existing single-device QAT trainer can consume later. It does not start neural-model training by default.

Before running, place reviewed corpus files in:

`MyDrive/QMini/datasets/`

The notebook fits the byte-level BPE tokenizer from those files, derives the active config from the actual tokenizer vocabulary, prepares train/validation token streams and masks, builds a random-initialized Q2-v2 model, and verifies all fingerprints before publishing the package in:

`MyDrive/QMini/training_ready/QMini-152M-Q2v2-<fingerprint>/`

The package contains:

- `initial_brain/model.qmini.npz`, `config.json`, and `model_manifest.json`
- `tokenizer/tokenizer.json` and its Q-Mini provenance sidecar
- `dataset/train.bin`, `valid.bin`, both loss masks, and `dataset.json`
- `BUILD_REPORT.json` with architecture, data, tokenizer, precision, and hash evidence

## Training later

In the notebook's settings cell, set `RUN_TRAINING_NOW = True` only when ready. Set a total optimizer-step target and review the GPU memory check. The first run uses `--init-from-initial` with the verified Q2-v2 weights. An existing run resumes from its Drive checkpoint; the trainer checks the previous initialization fingerprint and data contract.

The default 1024-token sequence length, batch size 1 and gradient accumulation 8 are conservative starting values, not a universal memory guarantee. The notebook requires a CUDA GPU with at least the configured free-memory floor before it starts training. The trainer performs its own additional memory preflight.

## Scope and hard limits

- No Transformer self-attention or Q/K/V attention is added.
- No third-party pretrained model weights are downloaded or loaded.
- Building the neural model performs zero optimizer steps and zero model-weight updates.
- Q2-v2 means the matrix symbols are stored at 2 bits each. Row scales are FP32, one-dimensional parameters FP16; QAT master weights and AdamW state are FP32 in the current trainer.
- The 5B profile is **not** a supported Colab training target. The repository does not yet implement FSDP/ZeRO sharded optimizer/checkpoint training. A planning estimate is not a distributed trainer.
- A complete package is an untrained model plus matching tokenizer/data assets. General assistant capability is not present until training and independent held-out evaluation are completed.
- Training data must be reviewed for rights, provenance, privacy and quality. Set `RELEASE_MODE=True` with a valid source metadata file to enable the stricter provenance/PII gate.
