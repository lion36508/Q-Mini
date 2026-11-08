import json

import numpy as np
import pytest

from qmini.data import extract_text, iter_text_records, prepare_token_streams
from qmini.tokenizer import train_tokenizer


def test_extract_chat_messages():
    text = extract_text({"messages": [
        {"role": "user", "content": "Explain gravity."},
        {"role": "assistant", "content": "Gravity attracts masses."},
    ]})
    assert "[USER]" in text and "Explain gravity." in text
    assert "Gravity attracts masses." in text


def test_jsonl_and_plain_text_readers(tmp_path):
    jsonl = tmp_path / "chat.jsonl"
    jsonl.write_text(
        json.dumps({"question": "Q?", "answer": "A!"}) + "\n",
        encoding="utf-8",
    )
    plain = tmp_path / "notes.txt"
    plain.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    records = list(iter_text_records([jsonl, plain]))
    assert any("Q?" in item and "A!" in item for item in records)
    assert any("First paragraph." in item for item in records)
    assert any("Second paragraph." in item for item in records)


def test_prepare_deduplicates_and_guarantees_nonempty_splits(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"text": "The first distinct document has enough text."},
        {"text": "The second distinct document also has enough text."},
        {"text": "The first distinct document has enough text."},
    ]
    corpus.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer([corpus], tokenizer_path, vocab_size=300, min_frequency=1)

    output = tmp_path / "tokens"
    report = prepare_token_streams(
        [corpus], tokenizer_path, output, valid_fraction=0.01
    )
    assert report["unique_documents"] == 2
    assert report["duplicate_documents"] == 1
    assert report["train_documents"] == 1
    assert report["valid_documents"] == 1
    assert report["train_tokens"] > 0
    assert report["valid_tokens"] > 0
    assert report["format"] == "qmini-token-stream-v3"
    assert (output / "dataset.json").is_file()

    for split in ("train", "valid"):
        tokens = np.memmap(output / f"{split}.bin", mode="r", dtype=np.uint16)
        mask = np.memmap(output / f"{split}.mask.bin", mode="r", dtype=np.uint8)
        assert len(tokens) == len(mask)
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) > 0

    with pytest.raises(FileExistsError):
        prepare_token_streams([corpus], tokenizer_path, output, valid_fraction=0.01)


def test_assistant_loss_mask_excludes_user_prompt_tokens(tmp_path):
    corpus = tmp_path / "conversations.jsonl"
    conversations = [
        {"messages": [
            {"role": "user", "content": "What is a star?"},
            {"role": "assistant", "content": "A star is a luminous sphere of plasma."},
        ]},
        {"messages": [
            {"role": "user", "content": "Explain gravity."},
            {"role": "assistant", "content": "Gravity is the attraction between masses."},
        ]},
    ]
    corpus.write_text(
        "\n".join(json.dumps(item) for item in conversations),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "conversation-tokenizer.json"
    train_tokenizer([corpus], tokenizer_path, vocab_size=400, min_frequency=1)
    output = tmp_path / "assistant-targets"

    report = prepare_token_streams(
        [corpus], tokenizer_path, output,
        valid_fraction=0.01, loss_mask_mode="assistant",
    )
    assert report["loss_mask_mode"] == "assistant"
    assert report["train_target_tokens"] > 0
    assert report["valid_target_tokens"] > 0

    for split in ("train", "valid"):
        tokens = np.memmap(output / f"{split}.bin", mode="r", dtype=np.uint16)
        mask = np.memmap(output / f"{split}.mask.bin", mode="r", dtype=np.uint8)
        assert len(tokens) == len(mask)
        assert 0 < int(mask.sum()) < len(mask)
