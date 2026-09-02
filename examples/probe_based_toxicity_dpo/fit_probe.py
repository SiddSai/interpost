"""Phase 1b — fit the toxicity probe (Wehner & Fritz reproduction).

Pipeline: take comments from Civil Comments as prompts -> greedy-decode a continuation
from a non-aligned base model -> label the continuation with a RoBERTa toxicity
classifier (>= 0.5 = toxic) -> extract response-pooled activations across a layer
sweep -> fit an L2 logistic-regression probe, keep the best-AUC layer.

    python -m examples.probe_based_toxicity_dpo.fit_probe --n-prompts 1200

Run from the repo root. Needs HF_TOKEN in .env (gated base model).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._shared.data import stream_civil_comments_prompts
from examples._shared.extract import extract_pooled, generate_responses
from examples._shared.labelers import RobertaToxicity
from examples._shared.probe import fit_probe, probe_summary

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="non-aligned base model")
    p.add_argument("--n-prompts", type=int, default=1500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-prompt-chars", type=int, default=40)
    p.add_argument("--max-prompt-chars", type=int, default=300)
    p.add_argument(
        "--min-prompt-toxicity",
        type=float,
        default=0.5,
        help="keep only Civil Comments whose own toxicity >= this (higher toxic yield)",
    )
    p.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    p.add_argument("--per-class", type=int, default=600, help="target samples per class")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "artifacts" / "toxicity_probe.npz"))
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
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers
    layers = (
        list(range(n_layers))
        if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )

    # --- prompts: Civil Comments text, length- and toxicity-filtered ---
    # build_pairs.py skips the first --n-prompts of these for a clean held-out split,
    # so keep this filtering in sync (both call the shared helper).
    prompts = stream_civil_comments_prompts(
        min_toxicity=args.min_prompt_toxicity,
        min_chars=args.min_prompt_chars,
        max_chars=args.max_prompt_chars,
        limit=args.n_prompts,
    )
    print(f"prompts: {len(prompts)} (own toxicity >= {args.min_prompt_toxicity})")

    # --- generate one greedy continuation per prompt ---
    responses = [
        c[0]
        for c in generate_responses(
            model,
            tokenizer,
            prompts,
            k=1,
            do_sample=False,  # greedy, per Wehner & Fritz
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    ]

    # --- label with RoBERTa, balance classes ---
    roberta = RobertaToxicity(device=device)
    tox_p = roberta(responses)
    labels = (tox_p >= 0.5).astype(int)
    print(f"toxic rate in generations: {labels.mean():.3f}")

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    n = min(args.per_class, len(idx_pos), len(idx_neg))
    if n < 50:
        raise SystemExit(
            f"only {len(idx_pos)} toxic / {len(idx_neg)} non-toxic generations; "
            "raise --n-prompts or use a more toxic prompt source"
        )
    rng = np.random.default_rng(args.seed)
    keep = np.concatenate(
        [rng.choice(idx_pos, n, replace=False), rng.choice(idx_neg, n, replace=False)]
    )
    rng.shuffle(keep)
    prompts = [prompts[i] for i in keep]
    responses = [responses[i] for i in keep]
    labels = labels[keep]
    print(f"balanced dataset: {len(labels)} ({int(labels.sum())} toxic)")

    # --- extract + fit ---
    acts = extract_pooled(
        model, tokenizer, prompts, responses, layers,
        pooling="mean", batch_size=args.batch_size, device=device,
    )
    probe = fit_probe(acts, labels, pooling="mean", seed=args.seed)
    print("layer sweep (val AUC):")
    for li in sorted(probe.layer_aucs):
        mark = "  <- selected" if li == probe.layer else ""
        print(f"  L{li:2d}: {probe.layer_aucs[li]:.4f}{mark}")
    print(probe_summary(probe))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    probe.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
