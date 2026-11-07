from __future__ import annotations
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from .data import expand_data_paths, iter_text_records
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def train_tokenizer(data_paths, output_path, vocab_size: int = 4096, min_frequency: int = 2, overwrite: bool = False):
    target = Path(output_path)
    if target.exists() and not overwrite: raise FileExistsError(f"Tokenizer exists: {target}; use overwrite=True to replace it.")
    paths = expand_data_paths(data_paths)
    if not paths: raise FileNotFoundError("No supported dataset files found.")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tokenizer.train_from_iterator(iter_text_records(paths), trainer)
    target.parent.mkdir(parents=True, exist_ok=True); tokenizer.save(str(target))
    return tokenizer

def load_tokenizer(path):
    target = Path(path)
    if not target.is_file(): raise FileNotFoundError(f"Tokenizer not found: {target}")
    return Tokenizer.from_file(str(target))

def special_token_id(tokenizer, token: str) -> int:
    value = tokenizer.token_to_id(token)
    if value is None: raise ValueError(f"Tokenizer lacks required token {token!r}.")
    return int(value)
