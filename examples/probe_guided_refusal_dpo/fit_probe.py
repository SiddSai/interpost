"""Phase 2b — fit the refusal / harmful-intent probe on PKU-SafeRLHF.

Classes: label 1 = harmful compliance (an unsafe assistant response), label 0 =
refusal / benign (a safe response). Activations are pooled over the assistant
response tokens of an instruct model (chat-templated), swept across layers.

    python -m examples.probe_guided_refusal_dpo.fit_probe --n-examples 12000

Run from the repo root. Needs HF_TOKEN in .env.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._shared.data import load_pku_probe_examples
from examples._shared.extract import extract_pooled
from examples._shared.probe import fit_probe, probe_summary, save_probe_verified

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--split", default="train")
    p.add_argument("--n-examples", type=int, default=20000, help="PKU rows to scan (x2 responses)")
    p.add_argument("--per-class", type=int, default=3000)
    p.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "artifacts" / "refusal_probe.npz"))
    return p.parse_args()


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    load_dotenv()
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={device}  model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = (
        list(range(n_layers))
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )

    prompts, responses, labels = load_pku_probe_examples(args.split, limit=args.n_examples)
    n_harm = int(labels.sum())
    print(f"PKU examples: {len(labels)}  ({n_harm} harmful, {len(labels) - n_harm} safe)")

    # chat-templated prefix per prompt; response is the assistant content
    prefixes = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    # class-balance
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n = min(args.per_class, len(idx_pos), len(idx_neg))
    if n < 100:
        raise SystemExit(f"only {len(idx_pos)} harmful / {len(idx_neg)} safe; raise --n-examples")
    rng = np.random.default_rng(args.seed)
    keep = np.concatenate(
        [rng.choice(idx_pos, n, replace=False), rng.choice(idx_neg, n, replace=False)]
    )
    rng.shuffle(keep)
    prefixes = [prefixes[i] for i in keep]
    responses = [responses[i] for i in keep]
    labels = labels[keep]
    print(f"balanced: {len(labels)} ({int(labels.sum())} harmful)")

    acts = extract_pooled(
        model, tokenizer, prefixes, responses, layers,
        pooling="mean", batch_size=args.batch_size, max_length=args.max_length,
        prompt_add_special_tokens=False, device=device,
    )
    probe = fit_probe(acts, labels, pooling="mean", seed=args.seed)
    print("layer sweep (val AUC):")
    for li in sorted(probe.layer_aucs):
        mark = "  <- selected" if li == probe.layer else ""
        print(f"  L{li:2d}: {probe.layer_aucs[li]:.4f}{mark}")
    print(probe_summary(probe))
    save_probe_verified(probe, args.out)


if __name__ == "__main__":
    main()
