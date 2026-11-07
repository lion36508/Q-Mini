from __future__ import annotations
import argparse, json, platform, shutil
import torch
from .config import QMiniConfig
from .data import prepare_token_streams
from .generate import generate, interactive_chat, load_model
from .model import QMiniMoE

def main():
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
        print("Attention/QKV: absent by design"); return 0
    audit.set_defaults(func=do_audit)
    tok = commands.add_parser("tokenizer", help="train a local BPE tokenizer")
    tok.add_argument("--data-dir", nargs="+", required=True); tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=4096); tok.add_argument("--min-frequency", type=int, default=2)
    tok.add_argument("--base-config", default="configs/qmini_base.json"); tok.add_argument("--config-output", required=True)
    tok.add_argument("--overwrite", action="store_true")
    def do_tokenizer(args):
        from .tokenizer import train_tokenizer
        t = train_tokenizer(args.data_dir, args.output, args.vocab_size, args.min_frequency, args.overwrite)
        cfg = QMiniConfig.from_json(args.base_config); cfg.vocab_size = t.get_vocab_size(); cfg.validate(); cfg.save(args.config_output)
        print("Tokenizer:", args.output, "vocab:", cfg.vocab_size, "config:", args.config_output); return 0
    tok.set_defaults(func=do_tokenizer)
    prep = commands.add_parser("prepare", help="prepare immutable train and validation token streams")
    prep.add_argument("--data-dir", nargs="+", required=True); prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output-dir", required=True); prep.add_argument("--valid-fraction", type=float, default=0.01)
    prep.set_defaults(func=lambda a: (print(json.dumps(prepare_token_streams(a.data_dir, a.tokenizer, a.output_dir, a.valid_fraction), indent=2)) or 0))
    train = commands.add_parser("train", help="train from scratch or resume a checkpoint")
    for flag in ("config", "train-bin", "valid-bin", "tokenizer", "output-dir"): train.add_argument("--" + flag, required=True)
    train.add_argument("--steps", type=int, default=1000); train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--seq-len", type=int, default=512); train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4); train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--eval-every", type=int, default=250); train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--eval-batches", type=int, default=20); train.add_argument("--resume", action="store_true")
    def do_train(a):
        from .train import train_model
        train_model(QMiniConfig.from_json(a.config), a.train_bin, a.valid_bin, a.output_dir,
            a.steps, a.batch_size, a.seq_len, a.grad_accum, a.lr, a.weight_decay,
            a.eval_every, a.save_every, a.eval_batches, a.resume, a.tokenizer); return 0
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
    chat = commands.add_parser("chat", help="interactive terminal chat")
    chat.add_argument("--model-dir", required=True); chat.add_argument("--max-new-tokens", type=int, default=160)
    chat.set_defaults(func=lambda a: (interactive_chat(a.model_dir, a.max_new_tokens) or 0))
    args = parser.parse_args(); result = args.func(args); raise SystemExit(result if isinstance(result, int) else 0)

if __name__ == "__main__": main()
