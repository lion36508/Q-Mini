import pytest

from qmini.tokenizer import load_tokenizer, train_tokenizer


def test_local_byte_bpe_round_trip_and_no_overwrite(tmp_path):
    corpus = tmp_path / "corpus.txt"
    text = "Q-Mini learns carefully. Namaste नमस्ते 🌌\nSmall models need clean data."
    corpus.write_text(text + "\n" + text, encoding="utf-8")
    output = tmp_path / "tokenizer.json"

    trained = train_tokenizer([corpus], output, vocab_size=300, min_frequency=1)
    loaded = load_tokenizer(output)
    encoded = loaded.encode(text)
    assert encoded.ids
    assert loaded.decode(encoded.ids) == text
    assert trained.get_vocab_size() <= 300

    with pytest.raises(FileExistsError):
        train_tokenizer([corpus], output, vocab_size=300, min_frequency=1)
