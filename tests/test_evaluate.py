import pytest
import json

import numpy as np
import torch

import qmini.evaluate as evaluate
from qmini.model import QMiniMoE


def test_answer_normalization_and_lexical_f1():
    assert evaluate.normalize_answer("The Answer! 42.") == "the answer 42"
    assert evaluate.normalize_answer("Namaste, दुनिया!") == "namaste दुनिया"
    assert evaluate.normalize_answer("café CAFÉ") == "café café"
    assert evaluate.token_f1("red red blue", "red blue") == 0.8
    assert evaluate.token_f1("", "") == 1.0
    assert evaluate.token_f1("blue", "red") == 0.0


def test_language_evaluation_reports_masked_metrics_and_expert_load(tiny_config, tmp_path):
    model = QMiniMoE(tiny_config).eval()
    token_path = tmp_path / "valid.bin"
    mask_path = tmp_path / "valid.mask.bin"
    rng = np.random.default_rng(31)
    np.asarray(rng.integers(0, tiny_config.vocab_size, 256), dtype=np.uint16).tofile(token_path)
    (rng.random(256) > 0.25).astype(np.uint8).tofile(mask_path)

    report = evaluate.evaluate_language_model(
        model, token_path, mask_path,
        batch_size=2, seq_len=8, batches=3, seed=7,
    )
    assert report["format"] == "qmini-language-evaluation-v1"
    assert report["supervised_target_tokens"] > 0
    assert report["cross_entropy"] > 0
    assert report["perplexity"] >= 1
    assert len(report["router_load"]) == tiny_config.n_layers + tiny_config.thinking_steps
    assert sum(report["router_load"][0]["tokens_per_expert"]) == 2 * 8 * 3


def test_task_evaluation_aggregates_categories(tiny_config, tmp_path, monkeypatch):
    model = QMiniMoE(tiny_config)
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({"category": "math", "prompt": "1+1?", "reference": "2"}),
            json.dumps({"category": "science", "prompt": "sky?", "expected": "blue sky"}),
        ]),
        encoding="utf-8",
    )
    predictions = iter(["2", "The sky is blue."])
    monkeypatch.setattr(evaluate, "generate", lambda *args, **kwargs: next(predictions))

    report = evaluate.evaluate_task_file(
        model, tokenizer=object(), dataset_path=dataset, max_new_tokens=4
    )
    assert report["examples"] == 2
    assert report["exact_match"] == 0.5
    assert report["categories"]["math"]["exact_match"] == 1.0
    assert report["categories"]["science"]["mean_token_f1"] > 0.0
    assert len(report["cases"]) == 2



def test_task_evaluation_uses_numeric_and_json_scorers_with_language_reports(
    tiny_config, tmp_path, monkeypatch
):
    model = QMiniMoE(tiny_config)
    dataset = tmp_path / "structured-tasks.jsonl"
    dataset.write_text(
        "\n".join([
            json.dumps({
                "category": "math", "language": "en",
                "scorer": "numeric", "tolerance": 0.01,
                "prompt": "What is 6 times 7?", "reference": "42",
            }),
            json.dumps({
                "category": "format", "language": "hi",
                "scorer": "json",
                "prompt": "Return the requested JSON object.",
                "reference": '{"ok":true,"count":3}',
            }),
            json.dumps({
                "category": "format", "language": "as",
                "scorer": "json",
                "prompt": "Return invalid JSON for this deliberately failed example.",
                "reference": '{"ok":false}',
            }),
        ]),
        encoding="utf-8",
    )
    predictions = iter([
        "The answer is 42.0",
        '{"count": 3, "ok": true}',
        "not-json",
    ])
    monkeypatch.setattr(
        evaluate, "generate", lambda *args, **kwargs: next(predictions)
    )

    report = evaluate.evaluate_task_file(
        model, tokenizer=object(), dataset_path=dataset, max_new_tokens=8
    )
    assert report["format"] == "qmini-task-evaluation-v2"
    assert report["examples"] == 3
    assert report["task_pass_rate"] == pytest.approx(2 / 3)
    assert report["task_pass_rate_wilson_95"][0] < 2 / 3
    assert report["categories"]["math"]["task_pass_rate"] == 1.0
    assert report["categories"]["math"]["scorers"] == ["numeric"]
    assert report["categories"]["format"]["examples"] == 2
    assert set(report["languages"]) == {"en", "hi", "as"}
    assert report["languages"]["as"]["task_pass_rate"] == 0.0
    assert report["cases"][0]["passed"] is True
    assert report["cases"][2]["scoring_error"].startswith("Invalid JSON:")
