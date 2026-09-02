# interpost — status & handoff

_Snapshot for an agent picking this up. Companion docs: [`prd.md`](./prd.md) (spec),
[`build-plan.md`](./build-plan.md) (phases), [`research-direction.md`](./research-direction.md)
(the research program that will consume this library — kept out of `interpost/`)._

---

## 1. What this is

An open-source library that wires mechanistic-interpretability signals (linear
probes, SAE features, activation directions) into post-training (DPO, later
PPO/GRPO) as **reusable primitives**, instead of a one-off script per paper. Three
intervention "modes":

1. **Data shaping** — score/filter/reweight a preference dataset with a signal.
2. **Loss/reward shaping** — add a signal-derived term to the objective.
3. **Monitoring-preservation** — regularize so the signal stays predictive of the
   behavior it tracks (don't let training launder behavior into an untracked
   representation).

Core is domain-agnostic. Safety (toxicity, refusal) lives only in `examples/`.

---

## 2. Where the build is

| Phase | What | Status |
|---|---|---|
| 0 | scaffold, CI, smoke tests | done (CI blocked by a GitHub billing lock on the account, not code) |
| 1 | **toxicity plumbing check** — reproduce Wehner & Fritz's probe-selected-DPO on a base model, LoRA | **done**, result below |
| 2 | **Mode-2 loss-term DPO on refusal** — the `OfflineDPOTrainer` primitive + a real before/after | **done**, result below |
| 3 | Mode-3 (preservation) | not started; design now well-constrained by Phase 1/2 |
| 4 | generalize `Signal` / `Intervention` ABCs | not started |
| 5 | Mode-1 data shaping (generalized) | `select_pairs` primitive exists; not generalized |
| 6 | online GRPO trainer + reward-model surface | not started |
| 7 | package, docs, examples | not started |

**44 tests pass** (`conda run -n interpost pytest -q`). Env: conda env `interpost`,
Python 3.11, `trl==1.12.0` (pinned — trainer subclasses depend on its internals),
`transformers` 5.16, torch 2.13. Local dev on Mac/MPS; real runs on a remote A100-40GB
box (`ubuntu@129.146.2.52:~/interpost`), synced with `rsync` (`push-vm` alias) or git.

---

## 3. What's built and validated (the offline half of the library)

All under `src/interpost/` unless noted. "Validated" = has real tests and ran in the
two experiments.

**`activations/hooks.py` — `HookManager`**
Registers forward hooks on a causal-LM's decoder blocks to capture (later: inject)
residual-stream activations *during a live training forward pass*, gradients intact.
BFS wrapper-attr resolver handles PEFT/accelerate nesting. `capture(layers, first=True)`
keeps only the first forward per layer — needed because a DPO step forwards twice
(policy, then a no-grad reference) through the same modules.

**`activations/pooling.py` — `pool(hidden, mask, mode)`**
Mask-aware `last | mean | max | per_token` pooling over completion tokens.

**`trainers/offline_dpo.py` — `OfflineDPOTrainer(trl.DPOTrainer)`**
The Mode-2 offline primitive. Extra kwargs: `signal_layer`, `signal_term_fn`
(`(chosen_pooled, rejected_pooled) -> Tensor`, differentiable), `signal_weight`,
`signal_pooling`. Overrides `compute_loss`: wraps `super().compute_loss` in
`HookManager.capture([layer], first=True)`, pools completion activations from the
concatenated `[chosen; rejected]` batch (`inputs["completion_mask"]`, chunk on dim 0),
adds `signal_weight * signal_term_fn(...).mean()` to the DPO loss, logs `signal/term`.
With no signal kwargs it's a plain `DPOTrainer`.

**`examples/_shared/`** (example-local; generalizes into `signals/` in Phase 4)
- `probe.py` — `Probe` dataclass (`score` / `logit` / `direction`), `fit_probe`
  (per-layer L2 logistic regression on standardized pooled acts, picks best val-AUC
  layer, records the full sweep), `save_probe_verified` (writes via an explicit file
  handle — `np.savez(path)` double-suffixes; then reloads and asserts).
- `extract.py` — `extract_pooled` (batched, teacher-forced forward over prompt+response,
  pool over the response span, `prompt_add_special_tokens` for chat-templated prefixes),
  `generate_responses` (k continuations, self-left-pads, greedy or temp-1.0, tqdm).
- `select.py` — `select_pairs(prompts, candidates, scorer)` → preference `Dataset`,
  `chosen` = min badness, `rejected` = max. The Mode-1 selection primitive.
- `labelers.py` — `RobertaToxicity` (P(toxic) on response text), `LlamaGuard`
  (1=unsafe on (prompt,response); builds the moderation prompt as a raw string —
  see gotchas), `llama_guard_prompt`.
- `data.py` — Civil Comments streamer (`stream_civil_comments_prompts`, held-out via
  `skip=`), PKU-SafeRLHF loaders (`load_pku_probe_examples` → flattened (prompt,
  response, unsafe-label); `load_pku_preference_pairs` → safe-vs-unsafe pairs,
  conversational format; `load_pku_prompts`), `load_xstest_safe_prompts`.

**Example drivers** (run with `python -u -m examples.<dir>.<script>`):
- `examples/probe_based_toxicity_dpo/`: `fit_probe`, `build_pairs`, `run_dpo`, `eval`
- `examples/probe_guided_refusal_dpo/`: `fit_probe`, `run_dpo`, `eval`, `check_guard`

**`eval` in both examples** computes, per model, on a **fixed shared labeled text
set** (built once from the base model so it works even when a detoxed model emits
nothing to score):
- `frozen_auc` — the fitted probe, weights unchanged, on this model's activations
- `dir_auc` — frozen direction, this model's *own* standardizer (isolates scale
  shift from direction rotation)
- `refit_auc` — a brand-new probe fit from scratch on this model's activations
- plus behavioral metrics (toxicity rate / Llama-Guard ASR / XSTest over-refusal)

**Not yet built:** the `Signal` / `Intervention` ABCs (`signals/`, `interventions/`
are empty namespaces), `HookManager.inject`, `Preservation` (Mode 3), online
trainers, `eval/signal_report.py` / `probe_transfer.py`, packaging.

---

## 4. Experiments run & findings

### Phase 1 — toxicity, `Llama-3.2-1B` (base), LoRA DPO

Probe: L9 (chosen from a flat 0.92–0.94 sweep), mean-pooled over response tokens,
Civil Comments, RoBERTa-labeled, ~4.4k **matched** preference pairs (same prompts,
probe- vs RoBERTa-selected chosen/rejected).

| | tox_rate | frozen AUC | dir + re-std | refit AUC |
|---|---|---|---|---|
| base | 0.35 | 0.80 | 0.80 | 0.82 |
| probe-selected DPO | 0.005 | 0.56 | 0.56 | 0.82 |
| RoBERTa-selected DPO | 0.005 | 0.58 | 0.58 | 0.82 |

- Both detoxify equally; generations stay coherent (LoRA + `--sft-weight 0.1` NLL
  anchor + low LR — full-FT DPO on a raw base model **collapses**, see gotchas).
- **Frozen probe dies (0.80→0.57). Re-standardizing recovers nothing** (`dir` ≈
  `frozen`) → the toxicity **direction rotates**; it is not a scale shift.
- **`refit` holds at 0.82** → the concept fully survives, relocated not destroyed.
- **No probe-vs-classifier gap** → Wehner & Fritz's preservation effect does *not*
  reproduce on toxicity. Consistent with Lee et al. (arXiv 2401.01967): DPO reduces
  toxicity by a *distributed bypass offset*, leaving the representation intact — so
  the pair-selector doesn't matter and there's little "drift" to differentiate.

### Phase 2 — refusal, `Llama-3.2-1B-Instruct`, LoRA DPO on PKU-SafeRLHF pairs

Probe: L7 (real mid-network peak, 0.88→0.96→0.95 sweep — a *computed* concept, unlike
flat toxicity), harmful-compliance vs safe responses. Runs: `dpo_vanilla`
(`--no-probe-term`) vs `dpo_probe` (`OfflineDPOTrainer`, `signal_weight=0.5`, term =
`ReLU(probe_logit)` on rejected acts).

| | ASR | over-refusal | frozen | dir | refit |
|---|---|---|---|---|---|
| base | 0.027 | 0.056 | 0.86 | 0.86 | 0.97 |
| dpo_vanilla | 0.013 | 0.100 | 0.86 | 0.85 | 0.96 |
| dpo_probe | 0.020 | 0.064 | 0.83 | 0.82 | 0.96 |

**Alignment ceiling:** base Instruct already refuses ~97% of PKU prompts, so the
behavioral deltas (ASR 2–4 prompts / 300) are noise. **But the training log is the
result** — `dpo_probe`'s `signal/term`:

```
step 10: signal/term 3.22   rewards/accuracies 0.48   mean_token_accuracy 0.43
step 20: signal/term 0.007  rewards/accuracies 0.68   0.44
step 30: signal/term 0.0    rewards/accuracies 0.76   0.43
step 40: signal/term 0.0    rewards/accuracies 0.81   0.44
```

- The Mode-2 hook **fired** (term = 3.22 at init, L7 probe strongly firing on harmful
  responses).
- **The model Goodharted the probe in ~15 steps** — `signal/term` → hard 0.0. The
  term only watched `ReLU(w·z + b)`; the model shifted harmful-response L7 activations
  across that one boundary (`logit ≤ 0` for all of them) without collapsing the
  classes together.
- **DPO preference learning proceeded normally alongside** (accuracies 0.48→0.81),
  **no capability cost** (token accuracy flat).
- Result in eval: `dpo_probe` frozen AUC lowest (0.83), refit intact (0.96) —
  **relocation to evade the monitor**, deliberate this time.

---

## 5. Headline findings

1. **"Optimize against a linear probe → the probe goes stale via direction
   rotation; the concept relocates; a refit probe still finds it; behavior improves
   anyway."** Seen twice — as a *side effect* of ordinary DPO (Phase 1, gradual) and
   *caused directly* by a Mode-2 term (Phase 2, complete in 15 steps).

2. **The frozen / dir / refit triple is the right measurement** and it works —
   cleanly separates *monitor stale* from *concept gone* from *scale shift*.

3. **Naive Mode-2 (`ReLU(probe_logit)` on policy activations) is trivially
   gameable** — the model needs `logit ≤ 0`, nothing more. This is a *finding*, not
   a bug. It mandates the `source="policy" | "reference"` axis (score the frozen
   reference model's activations, which the policy can't move) as a real feature,
   not a "seam".

4. **Mode 3's target is now concrete:** the probe's *specific direction* rotates.
   Preservation must constrain that rotation (direction-cosine / CKA / subspace
   penalty) or reframe as "a linear direction stays aligned with an *independent*
   behavioral label" (refit-probe agreement) — never "keep the frozen probe firing"
   (that's Mode 2 again) and never "re-standardize" (doesn't touch rotation).

5. **The 1B regime is hard.** Toxicity/base → DPO collapses or bypasses.
   Refusal/instruct → alignment ceiling. Neither gave the textbook "behavior
   improves + representation drifts + preservation recovers it". The *library*
   works; the *phenomena* at 1B are muddy. A clean positive result for preservation
   likely needs 7–8B.

---

## 6. Immediate next steps

1. **Fix the refusal headroom.** Recommended: swap to an **abliterated** instruct
   model (`huihui-ai/Llama-3.2-1B-Instruct-abliterated`, refusal-direction ablation
   per Arditi et al.) so base ASR is high; re-fit the refusal probe on it; re-run
   vanilla vs probe DPO. This makes the *behavioral* before/after real (does probe
   term make it safer even as the probe is gamed?). Alternative (official weights):
   jailbreak-wrapped prompts (WildJailbreak).
2. **Phase 3 — Mode 3.** With the abliterated setup, build `Preservation`:
   periodically refit a probe at a pinned layer, score against an *independent*
   Llama-Guard label, penalize drops in that refit AUC (not the frozen probe's
   score). Compare vanilla / +M2 / +M2+M3.
3. **Also worth doing early:** implement the `source="reference"` variant of the
   Mode-2 term — score `HookManager.capture` on a frozen ref model — and show it is
   *not* Goodharted the way the policy-activation term was. That's a clean
   standalone result and it's small.
4. **Phase 4** (generalize `Signal` / `Intervention`) only after Phase 3 — extract
   the ABCs from what Phases 1–3 actually needed.

---

## 7. How to run

```bash
conda activate interpost
pytest -q                                   # 44 pass (2 slow smoke sub-suites)

# toxicity example (base model)
python -u -m examples.probe_based_toxicity_dpo.fit_probe   --n-prompts 15000 --per-class 2000 --layers 9 --batch-size 64
python -u -m examples.probe_based_toxicity_dpo.build_pairs --probe <ART>/toxicity_probe.npz --n-prompts 8000 --k 5 --skip 30000 --batch-size 64
python -u -m examples.probe_based_toxicity_dpo.run_dpo     --pairs <ART>/pairs_probe --out <ART>/dpo_probe
python -u -m examples.probe_based_toxicity_dpo.eval        --runs <ART>/dpo_probe <ART>/dpo_classifier --auc-prompts 3000 --per-class 800

# refusal example (instruct model)
python -u -m examples.probe_guided_refusal_dpo.fit_probe   --n-examples 20000 --per-class 3000
python -u -m examples.probe_guided_refusal_dpo.run_dpo     --out <RF>/dpo_vanilla --no-probe-term
python -u -m examples.probe_guided_refusal_dpo.run_dpo     --out <RF>/dpo_probe   --signal-weight 0.5
python -u -m examples.probe_guided_refusal_dpo.eval        --runs <RF>/dpo_vanilla <RF>/dpo_probe
python -u -m examples.probe_guided_refusal_dpo.check_guard   # sanity-check Llama-Guard
```

Artifacts (probes, datasets, trained models) live under each example's `artifacts/`
dir on the A100 box — **gitignored**, not synced by `rsync`. `run_dpo` LoRA-merges
before saving so `eval` loads a plain model. Gated models used: `meta-llama/Llama-3.2-1B`,
`-1B-Instruct`, `meta-llama/Llama-Guard-3-8B`, `PKU-Alignment/PKU-SafeRLHF` — HF token
in `.env` (gitignored) covers all.

---

## 8. Gotchas / hard-won lessons

- **`trl==1.12.0` is pinned.** `OfflineDPOTrainer` overrides `compute_loss` and reads
  `inputs["completion_mask"]` + the `[chosen;rejected]` dim-0 concat. Re-verify on
  any TRL bump. `warmup_ratio` is not a valid `DPOConfig` field in this version —
  use `warmup_steps`.
- **Full-FT DPO on a raw base model collapses** (`rewards/chosen` → −2.5,
  `mean_token_accuracy` craters, generations degenerate). Effective batch was also
  ~16× too small. Fix that stuck: LoRA + effective batch ≥ 64 + `loss_type=["sigmoid","sft"]`
  with `loss_weights=[1.0, 0.1]` (NLL-on-chosen anchor) + 1 epoch + `--lr` ~5e-5 (LoRA).
- **DPO reduces likelihood of chosen too** (asymmetric dynamics); the SFT anchor is
  the guard. Watch `mean_token_accuracy` — if it drops >10% from base, LR too high.
- **`np.savez(path)` appends `.npz` even when present** → silent `name.npz.npz`.
  `save_probe` now writes via an open file handle; `fit_probe` reloads and asserts.
- **Llama-Guard-3-1B and -3-8B have incompatible chat templates** (1B wants list-form
  message `content`, 8B wants a string, and both silently misbehave with the wrong
  one). `LlamaGuard` builds the moderation prompt as a raw string instead — works for
  both. Also: **Llama-Guard-3-1B mislabels clear-cut unsafe content as safe** — use
  the 8B (default).
- **`eval` uses a fixed shared text set built from the base model**, not each model's
  own generations — a well-detoxed model emits nothing scoreable otherwise (Phase 1
  first eval hit "0 toxic / 6000").
- **`generate_responses` self-sets `padding_side="left"`** per call and pins
  `top_p=1.0, top_k=0` so a model's `generation_config` can't inject sampling params.
- Toxicity probe curves are **flat across layers** (surface/lexical feature, carried
  through the residual stream — documented, not a bug). Refusal curves have real
  mid-network structure. `fit_probe` reports the full sweep so you can tell.

---

## 9. File map

```
src/interpost/
  activations/hooks.py      HookManager (capture, first=; inject = NotImplemented)
  activations/pooling.py    pool()
  trainers/offline_dpo.py   OfflineDPOTrainer  ← the Phase-2 primitive
  signals/ interventions/ eval/ config.py   ← empty namespaces (Phase 4+)
examples/_shared/
  probe.py extract.py select.py labelers.py data.py
examples/probe_based_toxicity_dpo/   fit_probe build_pairs run_dpo eval
examples/probe_guided_refusal_dpo/   fit_probe run_dpo eval check_guard
tests/                     44 tests; tests/smoke/ = TRL-integration tripwires
docs/                      prd.md build-plan.md research-direction.md status.md(this)
research/README.md         firewall marker: research/ imports interpost, never reverse
```
