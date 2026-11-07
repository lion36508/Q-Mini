# Q-Mini — Quaternary CausalConv-MoE

Q-Mini is a from-scratch decoder-only language model, not an Android app, hosted-chatbot wrapper or fine-tune of downloaded pretrained weights. PyTorch supplies tensor operations, autograd and optimization. A byte-level BPE tokenizer is trained locally from the user's permitted corpus; no pretrained model weights or paid APIs are required.

## Architecture lock

- Six causal blocks; width 256; expert hidden width 384; vocabulary target 4,096; context 512.
- Four feed-forward experts per block; sparse Top-1 token routing.
- Causal gated depthwise convolution with dilations 1, 2, 4, 8, 16, 32.
- One shared refinement block reused for 1–4 inference steps.
- Quaternary weights {-2, -1, +1, +2}; FP32 master weights with a straight-through estimator (STE).
- Per-token AbsMax INT8 activation fake quantization with STE.
- RMSNorm, residual paths, next-token cross-entropy and router load-balancing loss.
- No Transformer self-attention, no Q/K/V projections, no dot-product attention and no retrieval-attention.

The theoretical receptive field of the six main convolution blocks is 568 tokens with kernel width 10. Exact parameter count is computed from the instantiated model, not hard-coded to an experimental target.

## Repository

`qmini/` contains model, quantization, tokenizer, data, trainer, inference and CLI. `configs/` contains base and smoke profiles. `tests/` checks causal behavior, routing gradients, quantization, data parsing and C++ parity. `docs/` describes architecture, training and evaluation. `colab/` contains a Google Colab pipeline. `.github/workflows/` runs tests on pushes and pull requests.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m qmini.cli audit --config configs/qmini_smoke.json
```

## Google Colab

Place licensed dataset files in `Google Drive/MyDrive/QMini/datasets/`. Open `colab/QMini_Training.ipynb`, select a GPU runtime and run the cells in order. Tokenizer, token streams, checkpoints, metrics and exports are stored in `Google Drive/MyDrive/QMini/artifacts/`, not in this public repository.

Example commands after installation:

```bash
python -m qmini.cli tokenizer --data-dir /content/drive/MyDrive/QMini/datasets --output /content/drive/MyDrive/QMini/artifacts/tokenizer/tokenizer.json --vocab-size 4096 --config-output /content/drive/MyDrive/QMini/artifacts/qmini_active_config.json
python -m qmini.cli prepare --data-dir /content/drive/MyDrive/QMini/datasets --tokenizer /content/drive/MyDrive/QMini/artifacts/tokenizer/tokenizer.json --output-dir /content/drive/MyDrive/QMini/artifacts/tokens
python -m qmini.cli train --config /content/drive/MyDrive/QMini/artifacts/qmini_active_config.json --train-bin /content/drive/MyDrive/QMini/artifacts/tokens/train.bin --valid-bin /content/drive/MyDrive/QMini/artifacts/tokens/valid.bin --tokenizer /content/drive/MyDrive/QMini/artifacts/tokenizer/tokenizer.json --output-dir /content/drive/MyDrive/QMini/artifacts/runs/base --steps 10000 --batch-size 2 --seq-len 512 --grad-accum 8
```

Resume with `--resume`; the step value is the target total optimizer step, not an extra increment. Start with a smoke run and inspect actual GPU memory before scaling batch or sequence length.

## Native kernel and mobile boundary

`bash scripts/build_native_kernel.sh` builds an optional C++17 packed-quaternary INT8 linear projection primitive. It is not a complete native model runtime. Full ARM64 inference still needs kernels for embeddings, causal convolution, normalization, sparse expert dispatch and output projection, plus numerical parity tests on actual devices.

## Limits and honest status

A randomly initialized model is not a capable assistant. Quality depends on corpus rights and quality, number of tokens, optimization, training duration and independent held-out evaluation. The shared refinement loop is learned hidden-state refinement, not a symbolic theorem prover. Training uses FP32 master weights and optimizer state; a 2-bit export does not mean 2-bit training RAM. This repository includes source code and tests, not private datasets or trained checkpoints. The GitHub repository is public: never commit credentials, personal data, private corpora or checkpoints.
