"""Phase 2d — DPO on PKU-SafeRLHF pairs, with or without the interpost Mode-2
probe loss term.

    # vanilla
    python -m examples.probe_guided_refusal_dpo.run_dpo --out .../dpo_vanilla --no-probe-term
    # + probe term (interpost OfflineDPOTrainer)
    python -m examples.probe_guided_refusal_dpo.run_dpo --out .../dpo_probe --signal-weight 0.5

Run from the repo root. Needs HF_TOKEN in .env. Instruct model, LoRA.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from examples._shared.data import load_pku_preference_pairs
from examples._shared.probe import load_probe
from interpost.trainers import OfflineDPOTrainer

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--probe", default=str(HERE / "artifacts" / "refusal_probe.npz"))
    p.add_argument(
        "--probe-term", action=argparse.BooleanOptionalAction, default=True,
        help="add the Mode-2 probe loss term (--no-probe-term = vanilla DPO)",
    )
    p.add_argument("--signal-weight", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=20000, help="PKU rows to scan for pairs")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft-weight", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    from peft import LoraConfig

    use_bf16 = torch.cuda.is_available()
    print(f"cuda={use_bf16}  model={args.model}  probe_term={args.probe_term}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if use_bf16 else torch.float32
    )

    ds = load_pku_preference_pairs("train", limit=args.limit, seed=args.seed, conversational=True)
    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
    print(f"pairs: {len(split['train'])} train / {len(split['test'])} eval")

    signal_kwargs: dict = {}
    if args.probe_term:
        probe = load_probe(args.probe)
        mean = torch.tensor(probe.mean_, dtype=torch.float32)
        scale = torch.tensor(probe.scale_, dtype=torch.float32)
        coef = torch.tensor(probe.coef_, dtype=torch.float32)
        b = float(probe.intercept_)

        def term_fn(chosen_pooled, rejected_pooled):
            dev = rejected_pooled.device
            x = (rejected_pooled - mean.to(dev)) / scale.to(dev)
            return torch.relu(x @ coef.to(dev) + b)  # penalize "harmful" reps on rejected

        signal_kwargs = dict(
            signal_layer=probe.layer, signal_term_fn=term_fn, signal_weight=args.signal_weight
        )
        print(f"probe term: layer {probe.layer}, weight {args.signal_weight}")

    if args.sft_weight > 0:
        loss_type, loss_weights = ["sigmoid", "sft"], [1.0, args.sft_weight]
    else:
        loss_type, loss_weights = ["sigmoid"], None

    config = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        loss_type=loss_type,
        loss_weights=loss_weights,
        max_length=args.max_length,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        bf16=use_bf16,
        fp16=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
        seed=args.seed,
    )

    trainer = OfflineDPOTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        ),
        **signal_kwargs,
    )
    trainer.train()
    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
