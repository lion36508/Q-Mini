from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".md", ".markdown", ".json",
    ".jsonl", ".ndjson", ".csv", ".parquet",
}


def expand_data_paths(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            found.append(path.resolve())
        elif path.is_dir():
            found.extend(
                item.resolve()
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return sorted(set(found))


def extract_text(record: object) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, list):
        return "\n".join(
            text for item in record
            if (text := extract_text(item)).strip()
        )
    if not isinstance(record, dict):
        return ""

    messages = record.get("messages")
    if isinstance(messages, list):
        lines = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role", "unknown")).strip().upper()
                body = extract_text(message.get("content", message.get("text", "")))
                if body.strip():
                    lines.append(f"[{role}]\n{body}")
        if lines:
            return "\n".join(lines)

    keys = (
        "text", "content", "instruction", "input", "prompt", "question",
        "completion", "output", "answer", "response", "reasoning",
    )
    parts = [
        extract_text(record[key])
        for key in keys
        if record.get(key) is not None
    ]
    parts = [part for part in parts if part.strip()]
    if parts:
        return "\n".join(parts)

    fallback = [
        extract_text(value)
        for value in record.values()
        if isinstance(value, (str, list, dict))
    ]
    return "\n".join(part for part in fallback if part.strip())


def _iter_plain(path: Path, chunk_chars: int = 8192) -> Iterator[str]:
    buffer: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                if buffer:
                    yield "\n".join(buffer)
                    buffer, size = [], 0
                continue
            buffer.append(line.rstrip())
            size += len(line)
            if size >= chunk_chars:
                yield "\n".join(buffer)
                buffer, size = [], 0
    if buffer:
        yield "\n".join(buffer)


def iter_text_records(paths: Iterable[str | Path]) -> Iterator[str]:
    for path in expand_data_paths(paths):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".md", ".markdown"}:
            yield from _iter_plain(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            for item in payload if isinstance(payload, list) else [payload]:
                text = extract_text(item).strip()
                if text:
                    yield text
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {path} at line {line_number}: {exc}"
                        ) from exc
                    text = extract_text(record).strip()
                    if text:
                        yield text
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    text = extract_text(row).strip()
                    if text:
                        yield text
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet input requires pyarrow. Install with pip install pyarrow."
                ) from exc
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                columns = batch.to_pydict()
                for index in range(batch.num_rows):
                    row = {key: values[index] for key, values in columns.items()}
                    text = extract_text(row).strip()
                    if text:
                        yield text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(paths: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_token_streams(
    data_paths: Iterable[str | Path],
    tokenizer_path: str | Path,
    output_dir: str | Path,
    valid_fraction: float = 0.02,
) -> dict[str, object]:
    """Build deduplicated document-separated token streams with a reproducible split.

    Exact duplicate documents are removed using a temporary SQLite index, so the
    de-duplication index remains disk-backed instead of growing with corpus RAM.
    A deterministic hash split is used, with a guaranteed non-empty train/valid
    partition whenever at least two unique documents are available.
    """
    from .tokenizer import load_tokenizer, special_token_id

    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be greater than 0 and below 0.5.")

    paths = expand_data_paths(data_paths)
    if not paths:
        raise FileNotFoundError("No supported dataset files were found.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.bin"
    valid_path = target / "valid.bin"
    manifest_path = target / "dataset.json"
    if train_path.exists() or valid_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Existing dataset artifacts in {target} are preserved; choose a new output directory."
        )

    tokenizer = load_tokenizer(tokenizer_path)
    bos_id = special_token_id(tokenizer, "<bos>")
    eos_id = special_token_id(tokenizer, "<eos>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError("uint16 token streams support vocabulary sizes up to 65,535.")

    source_manifest_before = _file_manifest(paths)
    database_path = target / ".qmini_dedup.sqlite3"
    train_tmp = target / "train.bin.tmp"
    valid_tmp = target / "valid.bin.tmp"
    stats: dict[str, object] = {
        "unique_documents": 0,
        "duplicate_documents": 0,
        "skipped_short_documents": 0,
        "train_tokens": 0,
        "valid_tokens": 0,
        "train_documents": 0,
        "valid_documents": 0,
    }

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE documents (digest BLOB PRIMARY KEY, is_valid INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE written (digest BLOB PRIMARY KEY)"
        )

        # First pass: build a bounded-memory exact-document dedup index.
        with connection:
            for document in iter_text_records(paths):
                normalized = document.strip()
                if len(normalized) < 2:
                    stats["skipped_short_documents"] += 1
                    continue
                digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).digest()
                valid = int.from_bytes(digest[:8], "big") / (2**64) < valid_fraction
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO documents(digest, is_valid) VALUES (?, ?)",
                    (digest, int(valid)),
                )
                if cursor.rowcount == 1:
                    stats["unique_documents"] += 1
                else:
                    stats["duplicate_documents"] += 1

        unique_count = int(stats["unique_documents"])
        if unique_count < 2:
            raise ValueError(
                "At least two unique documents are required to create separate train and validation sets."
            )

        valid_count = int(connection.execute(
            "SELECT COUNT(*) FROM documents WHERE is_valid=1"
        ).fetchone()[0])
        if valid_count == 0:
            chosen = connection.execute(
                "SELECT digest FROM documents ORDER BY digest LIMIT 1"
            ).fetchone()[0]
            connection.execute("UPDATE documents SET is_valid=1 WHERE digest=?", (chosen,))
        elif valid_count == unique_count:
            chosen = connection.execute(
                "SELECT digest FROM documents ORDER BY digest DESC LIMIT 1"
            ).fetchone()[0]
            connection.execute("UPDATE documents SET is_valid=0 WHERE digest=?", (chosen,))
        connection.commit()

        # Second pass: stream each unique document into its deterministic split.
        with train_tmp.open("wb") as train_file, valid_tmp.open("wb") as valid_file:
            for document in iter_text_records(paths):
                normalized = document.strip()
                if len(normalized) < 2:
                    continue
                digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).digest()
                row = connection.execute(
                    "SELECT is_valid FROM documents WHERE digest=?", (digest,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Source data changed during token preparation.")
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO written(digest) VALUES (?)", (digest,)
                )
                if inserted.rowcount == 0:
                    continue

                token_ids = [bos_id, *tokenizer.encode(normalized).ids, eos_id]
                if token_ids and max(token_ids) >= vocab_size:
                    raise ValueError("Tokenizer emitted an ID outside its declared vocabulary.")
                values = np.asarray(token_ids, dtype=np.uint16)
                if int(row[0]) == 1:
                    values.tofile(valid_file)
                    stats["valid_tokens"] += int(values.size)
                    stats["valid_documents"] += 1
                else:
                    values.tofile(train_file)
                    stats["train_tokens"] += int(values.size)
                    stats["train_documents"] += 1

            train_file.flush()
            valid_file.flush()
            os.fsync(train_file.fileno())
            os.fsync(valid_file.fileno())

        source_manifest_after = _file_manifest(paths)
        if source_manifest_after != source_manifest_before:
            raise RuntimeError("Source files changed while token streams were being prepared.")

        if int(stats["train_tokens"]) == 0 or int(stats["valid_tokens"]) == 0:
            raise RuntimeError("Token preparation produced an empty train or validation stream.")

        os.replace(train_tmp, train_path)
        os.replace(valid_tmp, valid_path)

        corpus_descriptor = json.dumps(
            source_manifest_before, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        metadata: dict[str, object] = {
            "format": "qmini-token-stream-v2",
            **stats,
            "vocab_size": int(vocab_size),
            "valid_fraction_requested": float(valid_fraction),
            "dtype": "uint16",
            "deduplication": "exact-document-sha256",
            "source_files": source_manifest_before,
            "corpus_sha256": hashlib.sha256(corpus_descriptor).hexdigest(),
            "tokenizer_sha256": _sha256_file(Path(tokenizer_path)),
        }
        _write_json_atomic(manifest_path, metadata)
        return metadata

    except Exception:
        train_tmp.unlink(missing_ok=True)
        valid_tmp.unlink(missing_ok=True)
        train_path.unlink(missing_ok=True)
        valid_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        Path(str(database_path) + "-journal").unlink(missing_ok=True)
        Path(str(database_path) + "-wal").unlink(missing_ok=True)
        Path(str(database_path) + "-shm").unlink(missing_ok=True)
