from __future__ import annotations
import argparse, json, platform, shutil
import torch
from .config import QMiniConfig
from .data import prepare_token_streams
from .generate import generate, interactive_chat, load_model
from .evaluate import emit_report, evaluate_language_model, evaluate_task_file
from .model import QMiniMoE

def main() -> int:
    parser = argparse.ArgumentParser(prog="qmini", description="Q-Mini model, dataset and training pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="inspect model configuration and runtime")
    audit.add_argument("--config", default="configs/qmini_base.json")
    def do_audit(args):
        from .model import QMiniMoE
        cfg = QMiniConfig.from_json(args.config); model = QMiniMoE(cfg)
        print("Q-MINI AUDIT"); print("=" * 60)
        print("Python:", platform.python_version(), "PyTorch:", torch.__version__)
        print("Platform:", platform.platform(), "CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GiB)")
        print("Parameters:", f"{model.parameter_report()['total']:,}")
        print("Main receptive field:", cfg.main_receptive_field(), "tokens; max context:", cfg.max_context)
        print("C++ compiler:", shutil.which("clang++") or shutil.which("g++") or "not found")
        from .native import native_kernel_status
        native = native_kernel_status()
        print("CPU projection backend:", native["backend"])
        print("ARM NEON:", native["uses_neon"])
        print("Native library:", native["path"])
        print("Attention/QKV: absent by design"); return 0
    audit.set_defaults(func=do_audit)

    plan = commands.add_parser("plan", help="estimate model size and resource requirements without allocating the model")
    plan.add_argument("--config", required=True)
    def do_plan(args):
        cfg = QMiniConfig.from_json(args.config)
        report = {
            "config": args.config,
            "total_parameters": cfg.estimated_parameter_count(),
            "unique_active_parameters_per_token_top1": cfg.estimated_unique_active_parameter_count(),
            "fp32_weight_bytes": cfg.estimated_parameter_count() * 4,
            "fp32_adamw_state_lower_bound_bytes": cfg.estimated_adamw_training_state_bytes(),
            "estimated_q2_fp16_export_bytes": cfg.estimated_packed_export_bytes(),
            "estimated_q2_fp16_export_gb_decimal": cfg.estimated_packed_export_bytes() / 1_000_000_000,
            "estimated_q2_fp16_export_gib": cfg.estimated_packed_export_bytes() / (1024 ** 3),
            "deployment_weight_budget_bytes": 1_500_000_000,
            "within_1_5gb_weight_artifact_budget": (
                cfg.estimated_packed_export_bytes() <= 1_500_000_000
            ),
            "packed_weight_format": "2-bit signed quaternary weights + FP32 row scales + FP16 vectors",
            "dense_reference_tokens_at_20_per_parameter": cfg.reference_tokens_at_20_per_parameter(),
            "active_top1_reference_tokens_at_20_per_active_parameter": cfg.reference_tokens_at_20_per_active_parameter(),
            "max_context": cfg.max_context,
            "convolution_receptive_field": cfg.main_receptive_field(),
            "state_space_enabled": cfg.use_state_space,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "experts": cfg.n_experts,
            "active_experts_per_token": cfg.top_k,
            "warning": (
                "The packed export estimate is a raw storage upper bound for model weights, not a quality "
                "guarantee; tokenizer and application files are separate. Two-bit quantization can reduce "
                "accuracy and must be checked against held-out benchmarks. 20 tokens per parameter is a "
                "dense-LM planning heuristic, not a validated sparse-MoE scaling law. Training memory "
                "excludes activations, workspaces, fragmentation and distributed communication."
            ),
        }
        print(json.dumps(report, indent=2))
        return 0
    plan.set_defaults(func=do_plan)
    tok = commands.add_parser("tokenizer", help="train a local BPE tokenizer")
    tok.add_argument("--data-dir", nargs="+", required=True); tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=4096); tok.add_argument("--min-frequency", type=int, default=2)
    tok.add_argument("--base-config", default="configs/qmini_base.json"); tok.add_argument("--config-output", required=True)
    tok.add_argument("--valid-fraction", type=float, default=0.02,
        help="use the same document holdout fraction as data preparation")
    tok.add_argument("--overwrite", action="store_true")
    def do_tokenizer(args):
        from .tokenizer import train_tokenizer
        t = train_tokenizer(
            args.data_dir, args.output, args.vocab_size, args.min_frequency,
            args.overwrite, validation_fraction=args.valid_fraction
        )
        cfg = QMiniConfig.from_json(args.base_config); cfg.vocab_size = t.get_vocab_size(); cfg.validate(); cfg.save(args.config_output)
        print("Tokenizer:", args.output, "vocab:", cfg.vocab_size, "config:", args.config_output); return 0
    tok.set_defaults(func=do_tokenizer)
    prep = commands.add_parser("prepare", help="prepare immutable train and validation token streams")
    prep.add_argument("--data-dir", nargs="+", required=True); prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output-dir", required=True); prep.add_argument("--valid-fraction", type=float, default=0.02)
    prep.add_argument("--loss-mask", choices=("auto", "all", "assistant"), default="auto",
        help="auto: assistant-only targets for chats/Q&A, full LM loss for plain text")
    prep.set_defaults(func=lambda a: (
        print(json.dumps(prepare_token_streams(
            a.data_dir, a.tokenizer, a.output_dir, a.valid_fraction, a.loss_mask
        ), indent=2)) or 0
    ))
    train = commands.add_parser("train", help="train from scratch or resume a checkpoint")
    for flag in ("config", "train-bin", "valid-bin", "tokenizer", "output-dir"): train.add_argument("--" + flag, required=True)
    train.add_argument("--steps", type=int, default=1000); train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seq-len", type=int, default=512); train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4); train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--eval-every", type=int, default=250); train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--eval-batches", type=int, default=20); train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--init-from",
        help=(
            "initialize a new fine-tuning run from final_model/model.pt or "
            "a packed final_model/model.qmini.npz; optimizer state is reset"
        ),
    )
    train.add_argument(
        "--train-mask-bin",
        help="aligned uint8 target mask; required for prepared auto/assistant datasets",
    )
    train.add_argument(
        "--valid-mask-bin",
        help="aligned uint8 validation target mask; required for prepared auto/assistant datasets",
    )
    train.add_argument("--extend-schedule", action="store_true",
        help="explicitly extend the saved learning-rate schedule")
    train.add_argument("--allow-legacy-resume", action="store_true",
        help="resume a checkpoint without data fingerprints after manual verification")
    def do_train(a):
        from .train import train_model
        train_model(QMiniConfig.from_json(a.config), a.train_bin, a.valid_bin, a.output_dir,
            a.steps, a.batch_size, a.seq_len, a.grad_accum, a.lr, a.weight_decay,
            a.eval_every, a.save_every, a.eval_batches, a.resume, a.tokenizer, None,
            a.extend_schedule, a.allow_legacy_resume, a.train_mask_bin, a.valid_mask_bin,
            init_from=a.init_from); return 0
    train.set_defaults(func=do_train)
    gen = commands.add_parser("generate", help="generate from a trained model")
    gen.add_argument("--model-dir", required=True); gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=128); gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-k", type=int, default=40); gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--thinking-steps", type=int, choices=range(1, 5)); gen.add_argument("--device", default=None)
    def do_generate(a):
        model, tokenizer, _ = load_model(a.model_dir, a.device)
        print(generate(model, tokenizer, a.prompt, a.max_new_tokens, a.temperature, a.top_k, a.top_p, a.thinking_steps)); return 0
    gen.set_defaults(func=do_generate)
    verify_export = commands.add_parser("verify-export", help="validate a packed Q-Mini NPZ export")
    verify_export.add_argument("--path", required=True)
    def do_verify_export(a):
        from .export import verify_packed_export
        print(json.dumps(verify_packed_export(a.path), indent=2)); return 0
    verify_export.set_defaults(func=do_verify_export)
    memory = commands.add_parser("memory", help="manage explicit local SQLite memories")
    memory.add_argument("--db", required=True, help="local SQLite database path")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)

    memory_add = memory_commands.add_parser("add", help="save a note explicitly")
    memory_add.add_argument("--text", required=True)
    memory_add.add_argument("--tag", action="append", default=[])
    def do_memory_add(a):
        from .memory import LocalMemoryStore
        store = LocalMemoryStore(a.db)
        print(json.dumps(store.remember(a.text, tags=a.tag), ensure_ascii=False, indent=2))
        return 0
    memory_add.set_defaults(func=do_memory_add)

    memory_search = memory_commands.add_parser("search", help="search saved notes")
    memory_search.add_argument("--query", default="")
    memory_search.add_argument("--limit", type=int, default=8)
    def do_memory_search(a):
        from .memory import LocalMemoryStore
        print(json.dumps(
            LocalMemoryStore(a.db).search(a.query, a.limit),
            ensure_ascii=False, indent=2
        ))
        return 0
    memory_search.set_defaults(func=do_memory_search)

    memory_forget = memory_commands.add_parser("forget", help="delete one saved note by ID")
    memory_forget.add_argument("--id", type=int, required=True)
    def do_memory_forget(a):
        from .memory import LocalMemoryStore
        removed = LocalMemoryStore(a.db).forget(a.id)
        print("deleted" if removed else "not-found")
        return 0
    memory_forget.set_defaults(func=do_memory_forget)

    memory_export = memory_commands.add_parser("export", help="export memories to JSON")
    memory_export.add_argument("--output", required=True)
    def do_memory_export(a):
        from .memory import LocalMemoryStore
        target = LocalMemoryStore(a.db).export_json(a.output)
        print(target)
        return 0
    memory_export.set_defaults(func=do_memory_export)

    chat = commands.add_parser("chat", help="interactive terminal chat")
    chat.add_argument("--model-dir", required=True)
    chat.add_argument("--max-new-tokens", type=int, default=160)
    chat.add_argument("--memory-db", default=None,
        help="opt-in local SQLite memory database; writes require /remember")
    chat.set_defaults(func=lambda a: (
        interactive_chat(a.model_dir, a.max_new_tokens, a.memory_db) or 0
    ))

    eval_lm = commands.add_parser("eval-lm", help="evaluate held-out language-model loss and expert usage")
    eval_lm.add_argument("--model-dir", required=True)
    eval_lm.add_argument("--tokens", required=True)
    eval_lm.add_argument("--mask", default=None)
    eval_lm.add_argument("--batch-size", type=int, default=2)
    eval_lm.add_argument("--seq-len", type=int, default=512)
    eval_lm.add_argument("--batches", type=int, default=100)
    eval_lm.add_argument("--seed", type=int, default=42)
    eval_lm.add_argument("--device", default=None)
    eval_lm.add_argument("--output", default=None)
    def do_eval_lm(a):
        model, _tokenizer, _device = load_model(a.model_dir, a.device)
        report = evaluate_language_model(
            model, a.tokens, a.mask, a.batch_size, a.seq_len, a.batches, a.seed
        )
        emit_report(report, a.output)
        return 0
    eval_lm.set_defaults(func=do_eval_lm)

    eval_tasks = commands.add_parser("eval-tasks", help="evaluate a JSONL prompt/reference set")
    eval_tasks.add_argument("--model-dir", required=True)
    eval_tasks.add_argument("--dataset", required=True)
    eval_tasks.add_argument("--max-new-tokens", type=int, default=128)
    eval_tasks.add_argument("--limit", type=int, default=None)
    eval_tasks.add_argument("--thinking-steps", type=int, choices=range(1, 5), default=None)
    eval_tasks.add_argument("--device", default=None)
    eval_tasks.add_argument("--output", default=None)
    def do_eval_tasks(a):
        model, tokenizer, _device = load_model(a.model_dir, a.device)
        report = evaluate_task_file(
            model, tokenizer, a.dataset, a.max_new_tokens, a.limit, a.thinking_steps
        )
        emit_report(report, a.output)
        return 0
    eval_tasks.set_defaults(func=do_eval_tasks)

    args = parser.parse_args()
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
