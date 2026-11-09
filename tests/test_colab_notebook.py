import json
from pathlib import Path


def test_colab_long_training_has_no_fixed_optimizer_step_cap():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    long_stage = "".join(notebook["cells"][9]["source"])

    assert "LONG_MAX_OPTIMIZER_STEPS" not in long_stage
    assert "LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "BASE_STEPS = LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert (
        "if LONG_OPTIMIZER_STEPS_OVERRIDE is None:\\n"
        "        if TARGET_TOKENS < TOKENS_PER_OPTIMIZER_STEP:"
    ) in long_stage
    assert "No arbitrary maximum optimizer-step cap is applied." in long_stage


def test_colab_training_targets_do_not_change_checkpoint_run_identity():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    long_stage = "".join(notebook["cells"][9]["source"])
    long_identity = long_stage.split("base_identity = json.dumps({", 1)[1].split("BASE_KEY =", 1)[0]
    assert "'steps': BASE_STEPS" not in long_identity
    assert "'target_tokens': ACTUAL_TOKENS" not in long_identity
    assert "args.append('--extend-schedule')" in long_stage

    five_m_stage = "".join(notebook["cells"][7]["source"])
    five_m_identity = five_m_stage.split("identity_5m = {", 1)[1].split("KEY_5M =", 1)[0]
    assert "'steps': INTERMEDIATE_5M_STEPS" not in five_m_identity
    assert "ARGS_5M.append('--extend-schedule')" in five_m_stage

    sft_stage = "".join(notebook["cells"][11]["source"])
    sft_identity = sft_stage.split("json.dumps({", 1)[1].split("SFT_KEY =", 1)[0]
    assert "'steps': SFT_STEPS" not in sft_identity
    assert "sft_args.append('--extend-schedule')" in sft_stage

def test_sft_never_silently_selects_the_tiny_smoke_checkpoint():
    notebook_path = Path(__file__).resolve().parents[1] / 'colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][11]['source'])
    assert 'SFT_SOURCE_OVERRIDE = None' in sft_stage
    assert 'SFT_MINIMUM_SOURCE_PARAMETERS = 50_000_000' in sft_stage
    assert 'base_*/final_model/model_manifest.json' in sft_stage
    assert 'qmini-final-model-v2' in sft_stage
    assert "SFT_SOURCE = SMOKE/'final_model'" not in sft_stage
    assert 'The tiny smoke model is never selected silently.' in sft_stage
    assert sft_stage.index('SFT_SOURCE = select_sft_source()') > sft_stage.index('if not sft_files:')


def test_sft_requires_independent_task_and_conversation_holdouts():
    notebook_path = Path(__file__).resolve().parents[1] / 'colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][11]['source'])
    gate = sft_stage.index('if not SFT_HELDOUT_TASKS.is_file() or not SFT_HELDOUT_CONVERSATIONS.is_file():')
    training_run = sft_stage.index("SFT_RUN = ART/'runs'/f'sft_{SFT_KEY}'")
    identity = sft_stage.index("sft_identity.update(SFT_HELDOUT_TASKS_SHA256.encode('utf-8'))")
    assert gate < training_run
    assert identity < sft_stage.index('SFT_KEY =')
    assert 'training_file_hashes.intersection(heldout_hashes)' in sft_stage
