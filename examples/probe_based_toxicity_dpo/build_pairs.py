"""Phase 1c — build probe-selected and classifier-selected preference pairs.

Wehner & Fritz "probe-based DPO": for each held-out toxic prompt, sample k
continuations, then form a preference pair by taking the least-toxic candidate as
``chosen`` and the most-toxic as ``rejected`` — once using the linear probe as the
toxicity scorer, once using the RoBERTa classifier (the baseline).

    python -m examples.probe_based_toxicity_dpo.build_pairs --n-prompts 3000

Run from the repo root. Needs HF_TOKEN in .env. Assumes fit_probe.py has produced
artifacts/toxicity_probe.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples._shared.data import stream_civil_comments_prompts
from examples._shared.extract import extract_pooled, generate_responses
from examples._shared.labelers import RobertaToxicity
from examples._shared.probe import load_probe
from examples._shared.select import select_pairs

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--probe", default=str(HERE / "artifacts" / "toxicity_probe.npz"))
    p.add_argument("--n-prompts", type=int, default=3000)
    p.add_argument(
        "--skip",
        type=int,
        default=30000,
        help="qualifying prompts to skip (>= fit_probe's --n-prompts, for a held-out split)",
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-prompt-toxicity", type=float, default=0.5)
    p.add_argument("--min-gap", type=float, default=0.05, help="skip near-tie prompts")
    p.add_argument(
        "--match-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only prompts both scorers produced a pair for (controlled comparison)",
    )
    p.add_argument("--max-pairs", type=int, default=0, help="cap each dataset (0 = no cap)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(HERE / "artifacts"))
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

    probe = load_probe(args.probe)
    print(f"probe: layer {probe.layer}, pooling {probe.pooling}, val_auc {probe.val_auc:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device)
    model.eval()

    prompts = stream_civil_comments_prompts(
        min_toxicity=args.min_prompt_toxicity,
        skip=args.skip,
        limit=args.n_prompts,
    )
    print(f"held-out prompts: {len(prompts)} (skipped first {args.skip})")

    candidates = generate_responses(
        model,
        tokenizer,
        prompts,
        k=args.k,
        temperature=args.temperature,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    roberta = RobertaToxicity(device=device)

    def probe_scorer(flat_prompts: list[str], flat_responses: list[str]) -> np.ndarray:
        acts = extract_pooled(
            model,
            tokenizer,
            flat_prompts,
            flat_responses,
            [probe.layer],
            pooling=probe.pooling,
            batch_size=args.batch_size,
            device=device,
        )
        return probe.score(acts[probe.layer])

    def classifier_scorer(flat_prompts: list[str], flat_responses: list[str]) -> np.ndarray:
        return roberta(flat_responses)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, object] = {
        "n_prompts": len(prompts),
        "k": args.k,
        "temperature": args.temperature,
        "probe_layer": probe.layer,
        "skip": args.skip,
    }

    built = {
        name: select_pairs(prompts, candidates, scorer, min_gap=args.min_gap)
        for name, scorer in [("probe", probe_scorer), ("classifier", classifier_scorer)]
    }
    for name, ds in built.items():
        print(f"{name:10s}: {len(ds):5d} pairs before matching")

    if args.match_prompts:
        common = set.intersection(*(set(ds["prompt"]) for ds in built.values()))
        built = {n: ds.filter(lambda r: r["prompt"] in common) for n, ds in built.items()}
        print(f"matched prompt set: {len(common)}")

    if args.max_pairs:
        built = {
            n: ds.shuffle(seed=args.seed).select(range(min(args.max_pairs, len(ds))))
            for n, ds in built.items()
        }

    meta["match_prompts"] = bool(args.match_prompts)
    for name, ds in built.items():
        gap = float(np.mean(np.array(ds["rejected_score"]) - np.array(ds["chosen_score"])))
        dest = out_dir / f"pairs_{name}"
        ds.save_to_disk(str(dest))
        meta[f"pairs_{name}"] = {"n": len(ds), "mean_score_gap": gap}
        print(f"{name:10s}: {len(ds):5d} pairs kept, mean gap {gap:.3f} -> {dest}")

    (out_dir / "pairs_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta -> {out_dir / 'pairs_meta.json'}")


if __name__ == "__main__":
    main()
