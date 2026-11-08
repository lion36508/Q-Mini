import torch

from qmini.generate import generate, load_model
from qmini.model import QMiniMoE
from qmini.tokenizer import train_tokenizer


def test_load_and_generate_end_to_end(tiny_config, tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("Q-Mini predicts the next token.\n\n"
         "A causal model must not use future tokens.\n\n"
         "Small models require careful evaluation.\n\n") * 4,
        encoding="utf-8",
    )
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    tokenizer_path = model_dir / "tokenizer.json"
    tokenizer = train_tokenizer([corpus], tokenizer_path, vocab_size=300, min_frequency=1)

    tiny_config.vocab_size = tokenizer.get_vocab_size()
    tiny_config.validate()
    model = QMiniMoE(tiny_config)
    tiny_config.save(model_dir / "config.json")
    torch.save(model.state_dict(), model_dir / "model.pt")

    loaded_model, loaded_tokenizer, device = load_model(model_dir, device="cpu")
    assert device.type == "cpu"
    output = generate(
        loaded_model, loaded_tokenizer, "Q-Mini",
        max_new_tokens=2, temperature=0, thinking_steps=1,
    )
    assert isinstance(output, str)
