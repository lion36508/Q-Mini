import json
from pathlib import Path



def test_colab_setup_preserves_preinstalled_cuda_pytorch():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])

    assert "'--no-deps','-e','.'" in setup
    assert "'pip','install','--upgrade'" not in setup
    assert "from packaging.version import Version" in setup
    assert "Version('2.10.0')" in setup
    assert "pyarrow>=15.0" in setup
    assert "pip','install','-e','.[dev,colab]'" not in setup


def test_colab_expensive_stages_fail_closed_without_gpu_and_preflight_memory():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    five_m_stage = "".join(notebook["cells"][8]["source"])
    long_stage = "".join(notebook["cells"][10]["source"])
    sft_stage = "".join(notebook["cells"][12]["source"])

    assert "if not torch.cuda.is_available():" in five_m_stage
    assert "if not torch.cuda.is_available():" in long_stage
    assert "torch.cuda.mem_get_info(0)" in long_stage
    assert "LONG_MIN_FREE_GPU_GIB = 8.0" in long_stage
    assert "if not torch.cuda.is_available():" in sft_stage
    assert "torch.cuda.mem_get_info(0)" in sft_stage
    assert "LONG_TARGET_TOKENS_OVERRIDE must be a positive integer or None." in long_stage

def test_colab_long_training_has_no_fixed_optimizer_step_cap():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    long_stage = "".join(notebook["cells"][10]["source"])

    assert "LONG_MAX_OPTIMIZER_STEPS" not in long_stage
    assert "LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "BASE_STEPS = LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "if LONG_OPTIMIZER_STEPS_OVERRIDE is None:" in long_stage
    assert "if TARGET_TOKENS < TOKENS_PER_OPTIMIZER_STEP:" in long_stage
    assert "No arbitrary maximum optimizer-step cap is applied." in long_stage


def test_colab_training_targets_do_not_change_checkpoint_run_identity():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    long_stage = "".join(notebook["cells"][10]["source"])
    long_identity = long_stage.split("base_identity = json.dumps({", 1)[1].split("BASE_KEY =", 1)[0]
    assert "'steps': BASE_STEPS" not in long_identity
    assert "'target_tokens': ACTUAL_TOKENS" not in long_identity
    assert "args.append('--extend-schedule')" in long_stage

    five_m_stage = "".join(notebook["cells"][8]["source"])
    five_m_identity = five_m_stage.split("identity_5m = {", 1)[1].split("KEY_5M =", 1)[0]
    assert "'steps': INTERMEDIATE_5M_STEPS" not in five_m_identity
    assert "ARGS_5M.append('--extend-schedule')" in five_m_stage

    sft_stage = "".join(notebook["cells"][12]["source"])
    sft_identity = sft_stage.split("json.dumps({", 1)[1].split("SFT_KEY =", 1)[0]
    assert "'steps': SFT_STEPS" not in sft_identity
    assert "sft_args.append('--extend-schedule')" in sft_stage

def test_sft_never_silently_selects_the_tiny_smoke_checkpoint():
    notebook_path = Path(__file__).resolve().parents[1] / 'colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][12]['source'])
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
    sft_stage = ''.join(notebook['cells'][12]['source'])
    gate = sft_stage.index('if not SFT_HELDOUT_TASKS.is_file() or not SFT_HELDOUT_CONVERSATIONS.is_file():')
    training_run = sft_stage.index("SFT_RUN = ART/'runs'/f'sft_{SFT_KEY}'")
    identity = sft_stage.index("sft_identity.update(SFT_HELDOUT_TASKS_SHA256.encode('utf-8'))")
    assert gate < training_run
    assert identity < sft_stage.index('SFT_KEY =')
    assert 'training_file_hashes.intersection(heldout_hashes)' in sft_stage



def test_colab_runs_random_brain_architecture_preflight_before_dataset_loading():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])
    preflight = "".join(notebook["cells"][3]["source"])
    tokenizer_data_stage = "".join(notebook["cells"][4]["source"])

    assert "qmini.cli','preflight'" in preflight
    assert "qmini_research_152m.json" in preflight
    assert "'--sequence-length','8'" in preflight
    assert "if torch.cuda.is_available():" in preflight
    assert "qmini_research_152m.json" in preflight
    assert "qmini_smoke.json" in preflight
    assert "PREFLIGHT_DEVICE = 'cpu'" in preflight
    assert "'--device',PREFLIGHT_DEVICE" in preflight
    assert "optimizer step" in preflight
    assert "weights updated" in preflight
    import ast
    ast.parse(preflight)
    assert "subprocess.run(ARCHITECTURE_PREFLIGHT,check=True)" in preflight
    assert "qmini.cli','tokenizer'" in tokenizer_data_stage
    assert setup.index("pip','install','--no-deps','-e','.'") < len(setup)

def test_colab_smoke_learning_proof_loads_training_checkpoint_not_dense_snapshot():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    smoke_stage = "".join(notebook["cells"][6]["source"])

    assert "checkpoint_payload = torch.load(checkpoint_path" in smoke_stage
    assert "checkpoint_payload.get('format') != 'qmini-checkpoint-v2'" in smoke_stage
    assert "trained_state = checkpoint_payload.get('model')" in smoke_stage
    assert "SMOKE/'final_model'/'model.pt'" not in smoke_stage
    assert "minimum_nll_improvement': 0.05" in smoke_stage
    assert "before_report['cross_entropy'] - after_report['cross_entropy'] >= 0.05" in smoke_stage
    assert "if not checkpoint_path.is_file():" in smoke_stage
    assert smoke_stage.index("if not checkpoint_path.is_file():") < smoke_stage.index(
        "checkpoint_payload = torch.load(checkpoint_path"
    )
