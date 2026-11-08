from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np

SUPPORTED_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".json", ".jsonl", ".ndjson", ".csv", ".parquet"}

def expand_data_paths(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(path.resolve())
        elif path.is_dir():
            found.extend(p.resolve() for p in sorted(path.rglob("*"))
                         if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        elif not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return sorted(set(found))

def extract_text(record: object) -> str:
    if isinstance(record, str): return record
    if isinstance(record, list):
        return "\n".join(text for item in record if (text := extract_text(item)).strip())
    if not isinstance(record, dict): return ""
    messages = record.get("messages")
    if isinstance(messages, list):
        lines = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "unknown")).strip().upper()
                body = extract_text(message.get("content", message.get("text", "")))
                if body.strip(): lines.append(f"[{role}]\n{body}")
        if lines: return "\n".join(lines)
    keys = ("text", "content", "instruction", "input", "prompt", "question", "completion", "output", "answer", "response", "reasoning")
    parts = [extract_text(record[key]) for key in keys if record.get(key) is not None]
    parts = [part for part in parts if part.strip()]
    if parts: return "\n".join(parts)
    fallback = [extract_text(v) for v in record.values() if isinstance(v, (str, list, dict))]
    return "\n".join(v for v in fallback if v.strip())

def _iter_plain(path: Path, chunk_chars: int = 8192) -> Iterator[str]:
    buffer, size = [], 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                if buffer: yield "\n".join(buffer); buffer, size = [], 0
                continue
            buffer.append(line.rstrip()); size += len(line)
            if size >= chunk_chars: yield "\n".join(buffer); buffer, size = [], 0
    if buffer: yield "\n".join(buffer)

def iter_text_records(paths: Iterable[str | Path]) -> Iterator[str]:
    for path in expand_data_paths(paths):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".md", ".markdown"}: yield from _iter_plain(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as h: payload = json.load(h)
            for item in (payload if isinstance(payload, list) else [payload]):
                text = extract_text(item).strip()
                if text: yield text
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as h:
                for line_no, line in enumerate(h, 1):
                    if not line.strip(): continue
                    try: record = json.loads(line)
                    except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                    text = extract_text(record).strip()
                    if text: yield text
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as h:
                for row in csv.DictReader(h):
                    text = extract_text(row).strip()
                    if text: yield text
        elif suffix == ".parquet":
            try: import pyarrow.parquet as pq
            except ImportError as exc: raise RuntimeError("Parquet input requires pyarrow.") from exc
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                columns = batch.to_pydict()
                for i in range(batch.num_rows):
                    text = extract_text({k: v[i] for k, v in columns.items()}).strip()
                    if text: yield text

def prepare_token_streams(data_paths, tokenizer_path, output_dir, valid_fraction: float = 0.01):
    from .tokenizer import load_tokenizer, special_token_id
    if not 0.0 < valid_fraction < 0.5: raise ValueError("valid_fraction must be greater than 0 and below 0.5.")
    paths = expand_data_paths(data_paths)
    if not paths: raise FileNotFoundError("No supported dataset files found.")
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = target / "train.bin", target / "valid.bin"
    if train_path.exists() or valid_path.exists(): raise FileExistsError("Existing token streams are preserved; choose a new output folder.")
    tokenizer = load_tokenizer(tokenizer_path)
    bos, eos = special_token_id(tokenizer, "<bos>"), special_token_id(tokenizer, "<eos>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535: raise ValueError("uint16 token files require vocab_size <= 65535.")
    counts = {"train_tokens": 0, "valid_tokens": 0, "train_documents": 0, "valid_documents": 0}
    train_tmp, valid_tmp = target / "train.bin.tmp", target / "valid.bin.tmp"
    try:
        with train_tmp.open("wb") as train_file, valid_tmp.open("wb") as valid_file:
            for text in iter_text_records(paths):
                text = text.strip()
                if len(text) < 2: continue
                digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
                valid = int.from_bytes(digest[:8], "big") / (2**64) < valid_fraction
                ids = np.asarray([bos, *tokenizer.encode(text).ids, eos], dtype=np.uint16)
                destination = valid_file if valid else train_file
                ids.tofile(destination)
                counts["valid_tokens" if valid else "train_tokens"] += int(ids.size)
                counts["valid_documents" if valid else "train_documents"] += 1
        if not counts["train_tokens"] or not counts["valid_tokens"]:
            raise ValueError("Train or validation stream is empty; add documents or adjust valid_fraction.")
        train_tmp.replace(train_path); valid_tmp.replace(valid_path)
    except Exception:
        train_tmp.unlink(missing_ok=True); valid_tmp.unlink(missing_ok=True)
        train_path.unlink(missing_ok=True); valid_path.unlink(missing_ok=True)
        raise
    metadata = {**counts, "vocab_size": int(vocab_size), "valid_fraction": valid_fraction,
                "source_files": [str(p) for p in paths], "dtype": "uint16"}
    with (target / "dataset.json").open("w", encoding="utf-8") as h: json.dump(metadata, h, indent=2)
    return metadata
