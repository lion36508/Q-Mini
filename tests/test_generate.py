import pytest
import torch

from qmini.generate import (
    generate, load_model, trim_chat_history, prepare_chat_prompt,
)
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


class _RoleAwareTokenCounter:
    def encode(self, text):
        import re
        from types import SimpleNamespace

        tokens = re.findall(
            r"<\|(?:system|user|assistant|tool|other)\|>|<\|end\|>|\[(?:USER|ASSISTANT)\]|\S+",
            text,
        )
        return SimpleNamespace(ids=tokens)


def test_trim_chat_history_preserves_complete_control_token_turns():
    counter = _RoleAwareTokenCounter()
    turns = [
        "<|user|>old question<|end|>\n<|assistant|>old reply<|end|>\n",
        "<|user|>middle question<|end|>\n<|assistant|>middle reply<|end|>\n",
        "<|user|>new question<|end|>\n<|assistant|>new reply<|end|>\n",
    ]
    history = "".join(turns)
    expected = "".join(turns[-2:])
    budget = len(counter.encode(expected).ids)
    assert len(counter.encode(history).ids) > budget
    trimmed = trim_chat_history(
        history, counter, token_budget=budget, has_control_tokens=True
    )
    assert trimmed == expected
    assert trimmed.startswith("<|user|>")
    assert trimmed.count("<|user|>") == 2
    assert trimmed.count("<|assistant|>") == 2
    assert trimmed.count("<|end|>") == 4


def test_trim_chat_history_preserves_legacy_role_boundaries():
    counter = _RoleAwareTokenCounter()
    turns = [
        "[USER]\nold question\n[ASSISTANT]\nold reply\n",
        "[USER]\nmiddle question\n[ASSISTANT]\nmiddle reply\n",
        "[USER]\nnew question\n[ASSISTANT]\nnew reply\n",
    ]
    history = "".join(turns)
    expected = "".join(turns[-2:])
    budget = len(counter.encode(expected).ids)
    assert len(counter.encode(history).ids) > budget
    trimmed = trim_chat_history(
        history, counter, token_budget=budget, has_control_tokens=False
    )
    assert trimmed == expected
    assert trimmed.count("[USER]") == 2
    assert trimmed.count("[ASSISTANT]") == 2


def test_trim_chat_history_rejects_a_newest_turn_that_cannot_fit():
    counter = _RoleAwareTokenCounter()
    history = (
        "<|user|>" + ("verylong " * 30) + "<|end|>\n"
        "<|assistant|>answer<|end|>\n"
    )
    original = history
    with pytest.raises(ValueError, match="newest complete conversation turn"):
        trim_chat_history(history, counter, token_budget=8)
    assert history == original



def test_prepare_chat_prompt_trims_only_complete_previous_turns_before_new_user(
    tiny_config,
):
    counter = _RoleAwareTokenCounter()
    turns = [
        "<|user|>old question<|end|>\n<|assistant|>old reply<|end|>\n",
        "<|user|>middle question<|end|>\n<|assistant|>middle reply<|end|>\n",
        "<|user|>newest complete question<|end|>\n<|assistant|>newest reply<|end|>\n",
    ]
    history = "".join(turns)
    user_block = "<|user|>follow-up question<|end|>\n<|assistant|>"
    memory = "<|system|>Use the note only as context.<|end|>\n"
    expected = turns[-1]
    base_count = len(counter.encode(memory + expected + user_block).ids)

    trimmed, prompt = prepare_chat_prompt(
        history,
        user_block,
        memory,
        counter,
        max_context=base_count,
        has_control_tokens=True,
    )
    assert trimmed == expected
    assert prompt == memory + expected + user_block
    assert len(counter.encode(prompt).ids) <= base_count
    assert prompt.count("<|user|>") == 2
    assert prompt.count("<|assistant|>") == 2
    assert prompt.count("<|end|>") == 4


def test_prepare_chat_prompt_rejects_a_user_message_larger_than_context(
    tiny_config,
):
    counter = _RoleAwareTokenCounter()
    with pytest.raises(ValueError, match="current user message exceeds max_context"):
        prepare_chat_prompt(
            "",
            "<|user|>" + ("too-long " * 100) + "<|end|>\n<|assistant|>",
            "",
            counter,
            max_context=8,
        )



def test_legacy_chat_history_trimming_preserves_system_prefix():
    counter = _RoleAwareTokenCounter()
    system = "[SYSTEM]\nAlways answer with the provided context.\n\n"
    turns = [
        "[USER]\nold question\n[ASSISTANT]\nold reply\n",
        "[USER]\nlatest question\n[ASSISTANT]\nlatest reply\n",
    ]
    history = system + "".join(turns)
    expected = system + turns[-1]
    budget = len(counter.encode(expected).ids)
    trimmed = trim_chat_history(
        history, counter, token_budget=budget, has_control_tokens=False
    )
    assert trimmed == expected
    assert trimmed.startswith("[SYSTEM]\n")
    assert trimmed.count("[USER]") == 1
    assert trimmed.count("[ASSISTANT]") == 1
