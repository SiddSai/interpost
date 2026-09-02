"""Phase 1d — run stock TRL DPO on a preference dataset from build_pairs.py.

No interpost trainer subclass here: this is the Wehner & Fritz "probe-based DPO"
setup, which is *ordinary* DPO on probe-selected pairs. Run it once on
``pairs_probe`` and once on ``pairs_classifier``; the comparison is Phase 1e.

    python -m examples.probe_based_toxicity_dpo.run_dpo \
        --pairs examples/probe_based_toxicity_dpo/artifacts/pairs_probe \
        --out   examples/probe_based_toxicity_dpo/artifacts/dpo_probe

Run from the repo root. Full fine-tune (not LoRA). Checkpoints every --save-steps
for the later probe-transfer analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

KEEP_COLUMNS = ("prompt", "chosen", "rejected")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--pairs", required=True, help="dir written by build_pairs.py")
    p.add_argument("--out", required=True, help="output dir for checkpoints + final model")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--per-device-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8, help="effective batch = per-device * this")
    p.add_argument("--lr", type=float, default=5e-5, help="LoRA runs hot; use ~5e-6 for --no-lora")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LoRA (stable for DPO on a raw base model); --no-lora for full fine-tune",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--sft-weight",
        type=float,
        default=0.1,
        help="weight of an auxiliary NLL(chosen) term; guards against DPO collapse. 0 = pure DPO",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=-1, help="override for quick runs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    use_bf16 = torch.cuda.is_available()
    print(f"cuda={torch.cuda.is_available()}  bf16={use_bf16}  model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPOTrainer requires left padding

    # LoRA: base is frozen, so bf16 base weights are fine. Full FT: keep fp32 master
    # weights so small DPO updates aren't rounded away.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if (args.lora and use_bf16) else torch.float32
    )

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    print(f"lora={args.lora}  effective_batch={args.per_device_batch_size * args.grad_accum}")

    ds = load_from_disk(args.pairs)
    ds = ds.remove_columns([c for c in ds.column_names if c not in KEEP_COLUMNS])
    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
    print(f"pairs: {len(split['train'])} train / {len(split['test'])} eval  from {args.pairs}")

    if args.sft_weight > 0:
        loss_type = ["sigmoid", "sft"]
        loss_weights = [1.0, args.sft_weight]
    else:
        loss_type = ["sigmoid"]
        loss_weights = None
    print(f"loss_type={loss_type}  loss_weights={loss_weights}")

    config = DPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
        gradient_checkpointing=False,  # keep activations clean for the 1f subclass
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

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    # merge LoRA so downstream (eval, probe extraction) loads a plain model
    if args.lora:
        trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
