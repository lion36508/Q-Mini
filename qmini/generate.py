from __future__ import annotations

import math
from pathlib import Path

import torch

from .config import QMiniConfig
from .model import QMiniMoE
from .tokenizer import load_tokenizer, special_token_id


def load_model(model_dir, device=None):
    root = Path(model_dir)
    config = QMiniConfig.from_json(root / "config.json")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = QMiniMoE(config)
    try:
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(root / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(target).eval()
    tokenizer = load_tokenizer(root / "tokenizer.json")
    return model, tokenizer, target


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    thinking_steps=None,
):
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if not math.isfinite(float(top_p)) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1.")

    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [special_token_id(tokenizer, "<bos>")]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    eos = tokenizer.token_to_id("<eos>")
    generated = []

    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.max_context:]
        logits = model(context, thinking_steps=thinking_steps)["logits"][:, -1, :].float()
        if temperature == 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            logits = logits / temperature
            if top_k > 0 and top_k < logits.shape[-1]:
                cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            if 0.0 < top_p < 1.0:
                sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
                probabilities = torch.softmax(sorted_logits, dim=-1)
                remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, indices, sorted_logits)
            next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

        generated.append(next_id)
        next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        tokens = torch.cat([tokens, next_tensor], dim=1)
        if eos is not None and next_id == eos:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_chat(model_dir, max_new_tokens=160):
    model, tokenizer, _ = load_model(model_dir)
    history = ""
    print("Q-Mini reference chat. Type /exit to stop.")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if user.lower() in {"/exit", "exit", "quit"}:
            return
        if not user:
            continue
        history += f"[USER]\n{user}\n[ASSISTANT]\n"
        answer = generate(model, tokenizer, history, max_new_tokens=max_new_tokens)
        print("Q-Mini: " + answer)
        history += answer + "\n"
        ids = tokenizer.encode(history).ids
        if len(ids) > model.config.max_context * 3 // 4:
            history = tokenizer.decode(ids[-model.config.max_context // 2:])
