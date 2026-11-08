from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".md", ".markdown", ".json",
    ".jsonl", ".ndjson", ".csv", ".parquet",
}
ASSISTANT_ROLES = {"assistant", "bot", "model"}


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


def iter_source_records(paths: Iterable[str | Path]) -> Iterator[object]:
    """Yield source records without flattening away chat roles."""
    for path in expand_data_paths(paths):
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text", ".md", ".markdown"}:
            yield from _iter_plain(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            yield from (payload if isinstance(payload, list) else [payload])
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {path} at line {line_number}: {exc}"
                        ) from exc
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield from csv.DictReader(handle)
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
                    yield {key: values[index] for key, values in columns.items()}


def iter_text_records(paths: Iterable[str | Path]) -> Iterator[str]:
    for record in iter_source_records(paths):
        text = extract_text(record).strip()
        if text:
            yield text


def iter_training_text_records(
    paths: Iterable[str | Path],
    valid_fraction: float = 0.02,
) -> Iterator[str]:
    """Yield only unique training documents using the exact preparation split.

    The split plan is built in a disk-backed SQLite index. If hashing assigns every
    document to one side, the same deterministic rebalancing rule as
    prepare_token_streams() moves one digest to the other side. This prevents
    tokenizer/validation leakage for small corpora.
    """
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be greater than 0 and below 0.5.")
    source_paths = expand_data_paths(paths)
    if not source_paths:
        raise FileNotFoundError("No supported dataset files were found.")

    with tempfile.TemporaryDirectory(prefix="qmini-tokenizer-") as temp_dir:
        database_path = Path(temp_dir) / "split.sqlite3"
        connection = sqlite3.connect(database_path)
        yielded = 0
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE documents (digest BLOB PRIMARY KEY, is_valid INTEGER NOT NULL)"
            )
            connection.execute("CREATE TABLE written (digest BLOB PRIMARY KEY)")

            # First pass: use the exact same deterministic hash split as preparation.
            with connection:
                for document in iter_text_records(source_paths):
                    normalized = document.strip()
                    if len(normalized) < 2:
                        continue
                    digest = hashlib.sha256(
                        normalized.encode("utf-8", errors="replace")
                    ).digest()
                    score = int.from_bytes(digest[:8], "big") / (2**64)
                    connection.execute(
                        "INSERT OR IGNORE INTO documents(digest, is_valid) VALUES (?, ?)",
                        (digest, int(score < valid_fraction)),
                    )

            unique_count = int(connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0])
            if unique_count < 2:
                raise ValueError(
                    "No unique training documents: at least two unique documents are required "
                    "for a separated training/validation tokenizer split."
                )

            valid_count = int(connection.execute(
                "SELECT COUNT(*) FROM documents WHERE is_valid=1"
            ).fetchone()[0])
            if valid_count == 0:
                chosen = connection.execute(
                    "SELECT digest FROM documents ORDER BY digest LIMIT 1"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE documents SET is_valid=1 WHERE digest=?", (chosen,)
                )
            elif valid_count == unique_count:
                chosen = connection.execute(
                    "SELECT digest FROM documents ORDER BY digest DESC LIMIT 1"
                ).fetchone()[0]
                connection.execute(
                    "UPDATE documents SET is_valid=0 WHERE digest=?", (chosen,)
                )
            connection.commit()

            # Second pass: only emit unique documents assigned to the train split.
            for document in iter_text_records(source_paths):
                normalized = document.strip()
                if len(normalized) < 2:
                    continue
                digest = hashlib.sha256(
                    normalized.encode("utf-8", errors="replace")
                ).digest()
                row = connection.execute(
                    "SELECT is_valid FROM documents WHERE digest=?", (digest,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("Source data changed during tokenizer training.")
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO written(digest) VALUES (?)", (digest,)
                )
                if inserted.rowcount == 0 or int(row[0]) == 1:
                    continue
                yielded += 1
                if yielded % 10000 == 0:
                    connection.commit()
                yield normalized

            if yielded == 0:
                raise ValueError("No unique training documents remain for tokenizer training.")
        finally:
            connection.close()


def _record_segments(record: object) -> tuple[list[tuple[str, bool]], bool, bool]:
    """Return (text, target-loss) segments, structured flag, assistant-target flag."""
    if isinstance(record, dict):
        messages = record.get("messages")
        if isinstance(messages, list):
            segments: list[tuple[str, bool]] = []
            has_assistant = False
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "unknown")).strip().lower()
                body = extract_text(message.get("content", message.get("text", ""))).strip()
                if not body:
                    continue
                is_assistant = role in ASSISTANT_ROLES
                has_assistant = has_assistant or is_assistant
                if role in ASSISTANT_ROLES:
                    control_role = "assistant"
                elif role in {"user"}:
                    control_role = "user"
                elif role in {"system", "developer"}:
                    control_role = "system"
                elif role in {"tool", "function"}:
                    control_role = "tool"
                else:
                    control_role = "other"
                segments.append((f"<|{control_role}|>", is_assistant))
                segments.append((body + "\n", is_assistant))
                segments.append(("<|end|>\n", is_assistant))
            if segments:
                return segments, True, has_assistant

        user_fields = ("instruction", "input", "prompt", "question")
        assistant_fields = ("completion", "output", "answer", "response")
        user_text = "\n".join(
            extract_text(record[key]).strip()
            for key in user_fields
            if record.get(key) is not None and extract_text(record[key]).strip()
        )
        assistant_text = "\n".join(
            extract_text(record[key]).strip()
            for key in assistant_fields
            if record.get(key) is not None and extract_text(record[key]).strip()
        )
        if user_text and assistant_text:
            return [
                ("<|user|>", False),
                (user_text + "\n", False),
                ("<|end|>\n", False),
                ("<|assistant|>", True),
                (assistant_text + "\n", True),
                ("<|end|>\n", True),
            ], True, True

    text = extract_text(record).strip()
    if not text:
        return [], False, False
    return [(text, True)], False, True


def _encode_record(
    record: object,
    tokenizer,
    bos_id: int,
    eos_id: int,
    loss_mask_mode: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    segments, structured, has_assistant = _record_segments(record)
    if not segments:
        return None

    if loss_mask_mode == "assistant" and (not structured or not has_assistant):
        return None

    ids = [bos_id]
    mask = [0]
    for segment, assistant_target in segments:
        encoded = tokenizer.encode(segment).ids
        if not encoded:
            continue
        ids.extend(encoded)
        if loss_mask_mode == "all" or not structured:
            mask.extend([1] * len(encoded))
        elif loss_mask_mode == "assistant":
            mask.extend([int(assistant_target)] * len(encoded))
        else:  # auto: train assistant turns only for structured chat/Q&A records.
            mask.extend([int(assistant_target)] * len(encoded))

    eos_is_target = (
        loss_mask_mode == "all"
        or not structured
        or (loss_mask_mode in {"auto", "assistant"} and has_assistant)
    )
    ids.append(eos_id)
    mask.append(int(eos_is_target))

    if not any(mask):
        return None
    return (
        np.asarray(ids, dtype=np.uint16),
        np.asarray(mask, dtype=np.uint8),
    )


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
    loss_mask_mode: str = "auto",
) -> dict[str, object]:
    """Prepare deduplicated train/validation token and assistant-loss-mask streams.

    loss_mask_mode:
      auto      - general text trains all tokens; chat/Q&A records train assistant outputs.
      all       - ordinary next-token pretraining over every token.
      assistant - only structured assistant responses contribute to loss.
    """
    from .tokenizer import load_tokenizer, special_token_id

    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be greater than 0 and below 0.5.")
    if loss_mask_mode not in {"auto", "all", "assistant"}:
        raise ValueError("loss_mask_mode must be one of: auto, all, assistant.")

    paths = expand_data_paths(data_paths)
    if not paths:
        raise FileNotFoundError("No supported dataset files were found.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train_path = target / "train.bin"
    valid_path = target / "valid.bin"
    train_mask_path = target / "train.mask.bin"
    valid_mask_path = target / "valid.mask.bin"
    manifest_path = target / "dataset.json"
    output_paths = (train_path, valid_path, train_mask_path, valid_mask_path, manifest_path)
    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"Existing dataset artifacts in {target} are preserved; choose a new output directory."
        )

    tokenizer_path_obj = Path(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path_obj)
    bos_id = special_token_id(tokenizer, "<bos>")
    eos_id = special_token_id(tokenizer, "<eos>")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError("uint16 token streams support vocabulary sizes up to 65,535.")

    source_manifest_before = _file_manifest(paths)
    fd, database_name = tempfile.mkstemp(
        prefix=".qmini_dedup_", suffix=".sqlite3", dir=target
    )
    os.close(fd)
    database_path = Path(database_name)
    temp_outputs = [
        target / "train.bin.tmp",
        target / "valid.bin.tmp",
        target / "train.mask.bin.tmp",
        target / "valid.mask.bin.tmp",
    ]
    stats: dict[str, object] = {
        "unique_documents": 0,
        "duplicate_documents": 0,
        "skipped_short_documents": 0,
        "skipped_without_targets": 0,
        "train_tokens": 0,
        "valid_tokens": 0,
        "train_target_tokens": 0,
        "valid_target_tokens": 0,
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
        connection.execute("CREATE TABLE written (digest BLOB PRIMARY KEY)")

        # First pass: exact-content de-dup index stored on disk, not in Python RAM.
        with connection:
            for record in iter_source_records(paths):
                normalized = extract_text(record).strip()
                if len(normalized) < 2:
                    stats["skipped_short_documents"] += 1
                    continue
                digest = hashlib.sha256(
                    normalized.encode("utf-8", errors="replace")
                ).digest()
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
                "At least two unique documents are required for non-empty train and validation sets."
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

        # Second pass: each unique source record is encoded once and split atomically.
        with (
            temp_outputs[0].open("wb") as train_file,
            temp_outputs[1].open("wb") as valid_file,
            temp_outputs[2].open("wb") as train_mask_file,
            temp_outputs[3].open("wb") as valid_mask_file,
        ):
            for record in iter_source_records(paths):
                normalized = extract_text(record).strip()
                if len(normalized) < 2:
                    continue
                digest = hashlib.sha256(
                    normalized.encode("utf-8", errors="replace")
                ).digest()
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

                encoded = _encode_record(
                    record, tokenizer, bos_id, eos_id, loss_mask_mode
                )
                if encoded is None:
                    stats["skipped_without_targets"] += 1
                    continue
                token_ids, loss_mask = encoded
                if token_ids.size and int(token_ids.max()) >= vocab_size:
                    raise ValueError("Tokenizer emitted an ID outside its declared vocabulary.")
                if token_ids.size != loss_mask.size:
                    raise RuntimeError("Token and loss-mask lengths do not match.")
                if not np.isin(loss_mask, (0, 1)).all():
                    raise RuntimeError("Loss mask must contain only binary values.")

                if int(row[0]) == 1:
                    token_ids.tofile(valid_file)
                    loss_mask.tofile(valid_mask_file)
                    stats["valid_tokens"] += int(token_ids.size)
                    stats["valid_target_tokens"] += int(loss_mask.sum())
                    stats["valid_documents"] += 1
                else:
                    token_ids.tofile(train_file)
                    loss_mask.tofile(train_mask_file)
                    stats["train_tokens"] += int(token_ids.size)
                    stats["train_target_tokens"] += int(loss_mask.sum())
                    stats["train_documents"] += 1

            for handle in (train_file, valid_file, train_mask_file, valid_mask_file):
                handle.flush()
                os.fsync(handle.fileno())

        source_manifest_after = _file_manifest(paths)
        if source_manifest_after != source_manifest_before:
            raise RuntimeError("Source files changed while token streams were being prepared.")
        if int(stats["train_tokens"]) == 0 or int(stats["valid_tokens"]) == 0:
            raise RuntimeError("Token preparation produced an empty train or validation stream.")
        if int(stats["train_target_tokens"]) == 0 or int(stats["valid_target_tokens"]) == 0:
            raise RuntimeError(
                "At least one split has no supervised target tokens. Add assistant answers "
                "or use --loss-mask all for ordinary next-token pretraining."
            )

        for temporary, final in zip(temp_outputs, (
            train_path, valid_path, train_mask_path, valid_mask_path
        )):
            os.replace(temporary, final)

        corpus_descriptor = json.dumps(
            source_manifest_before, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        metadata: dict[str, object] = {
            "format": "qmini-token-stream-v3",
            **stats,
            "vocab_size": int(vocab_size),
            "valid_fraction_requested": float(valid_fraction),
            "dtype": "uint16",
            "loss_mask_dtype": "uint8",
            "loss_mask_mode": loss_mask_mode,
            "deduplication": "exact-document-sha256",
            "source_files": source_manifest_before,
            "corpus_sha256": hashlib.sha256(corpus_descriptor).hexdigest(),
            "tokenizer_sha256": _sha256_file(tokenizer_path_obj),
            "train_tokens_sha256": _sha256_file(train_path),
            "valid_tokens_sha256": _sha256_file(valid_path),
            "train_mask_sha256": _sha256_file(train_mask_path),
            "valid_mask_sha256": _sha256_file(valid_mask_path),
        }
        _write_json_atomic(manifest_path, metadata)
        return metadata

    except Exception:
        for path in (*temp_outputs, *output_paths):
            path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        Path(str(database_path) + "-journal").unlink(missing_ok=True)
        Path(str(database_path) + "-wal").unlink(missing_ok=True)
        Path(str(database_path) + "-shm").unlink(missing_ok=True)
