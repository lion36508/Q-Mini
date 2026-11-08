from __future__ import annotations

import math
from pathlib import Path

import torch

from .config import QMiniConfig
from .model import QMiniMoE
from .tokenizer import load_tokenizer, special_token_id


def load_model(model_dir: str | Path, device: str | None = None):
    root = Path(model_dir)
    config = QMiniConfig.from_json(root / "config.json")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = QMiniMoE(config)
    weights_path = root / "model.pt"
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(target).eval()
    tokenizer = load_tokenizer(root / "tokenizer.json")
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer.get_vocab_size()}) does not match "
            f"model config ({config.vocab_size})."
        )
    return model, tokenizer, target


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    forbidden_ids: tuple[int, ...],
) -> int:
    logits = logits.float().clone()
    if forbidden_ids:
        valid = [token_id for token_id in forbidden_ids if 0 <= token_id < logits.shape[-1]]
        if valid:
            logits[:, valid] = float("-inf")

    if temperature == 0:
        return int(logits.argmax(dim=-1).item())

    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, indices, sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
        raise RuntimeError("Sampling produced invalid probabilities; check model logits and filters.")
    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.inference_mode()
def generate(
    model: QMiniMoE,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    thinking_steps: int | None = None,
) -> str:
    """Autoregressive generation using cached causal-convolution histories."""
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer.")
    if not math.isfinite(float(top_p)) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1.")
    if model.training:
        raise RuntimeError("Call model.eval() before generation.")
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model configuration.")

    device = next(model.parameters()).device
    bos_id = special_token_id(tokenizer, "<bos>")
    eos_id = special_token_id(tokenizer, "<eos>")
    forbidden_ids = tuple(
        token_id for token_id in (
            tokenizer.token_to_id("<pad>"),
            bos_id,
            tokenizer.token_to_id("<unk>"),
        )
        if token_id is not None
    )

    prompt_ids = tokenizer.encode(prompt).ids
    context_ids = [bos_id, *prompt_ids]
    if len(context_ids) > model.config.max_context:
        context_ids = [bos_id, *context_ids[-(model.config.max_context - 1):]]

    state = None
    next_logits = None
    for token_id in context_ids:
        result, state = model.forward_step(
            torch.tensor([token_id], dtype=torch.long, device=device),
            state=state,
            thinking_steps=thinking_steps,
        )
        next_logits = result["logits"]

    if next_logits is None or state is None:
        raise RuntimeError("Prompt prefill did not produce a model state.")

    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = _sample_next_token(
            next_logits, float(temperature), top_k, float(top_p), forbidden_ids
        )
        if next_id == eos_id:
            break
        generated.append(next_id)
        result, state = model.forward_step(
            torch.tensor([next_id], dtype=torch.long, device=device),
            state=state,
        )
        next_logits = result["logits"]

    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_chat(model_dir: str | Path, max_new_tokens: int = 160) -> None:
    model, tokenizer, _ = load_model(model_dir)
    print("Q-Mini reference chat. Type /exit to stop.")
    history = ""
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if user_text.lower() in {"/exit", "exit", "quit"}:
            return
        if not user_text:
            continue
        history += f"[USER]\n{user_text}\n[ASSISTANT]\n"
        answer = generate(model, tokenizer, history, max_new_tokens=max_new_tokens)
        print("Q-Mini: " + answer)
        history += answer + "\n"
        ids = tokenizer.encode(history).ids
        if len(ids) > model.config.max_context * 3 // 4:
            history = tokenizer.decode(ids[-model.config.max_context // 2:])
