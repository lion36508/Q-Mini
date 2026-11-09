from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import expand_data_paths, iter_training_text_records

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|other|>", "<|end|>",
]


def train_tokenizer(
    data_paths: Iterable[str | Path],
    output_path: str | Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    overwrite: bool = False,
    validation_fraction: float = 0.02,
    near_duplicate_distance: int = 0,
) -> Tokenizer:
    """Train byte-level BPE on unique training documents, excluding validation docs."""
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Tokenizer already exists: {target}; pass overwrite=True to replace it."
        )
    minimum_vocab = 256 + len(SPECIAL_TOKENS)
    if vocab_size < minimum_vocab:
        raise ValueError(
            f"Byte-level BPE vocab_size must be at least {minimum_vocab} "
            "to reserve the complete byte alphabet and Q-Mini control tokens."
        )
    if min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer.")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5).")
    paths = expand_data_paths(data_paths)
    if not paths:
        raise FileNotFoundError("No supported dataset files were found.")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_training_text_records(
            paths,
            valid_fraction=validation_fraction,
            near_duplicate_distance=near_duplicate_distance,
        ),
        trainer,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        tokenizer.save(str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {target}")
    return Tokenizer.from_file(str(target))


def special_token_id(tokenizer: Tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        raise ValueError(f"Tokenizer lacks required token {token!r}.")
    return int(value)
