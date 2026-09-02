# interpost — Build Plan

Companion to [`prd.md`](./prd.md). This is the execution plan: phases, tasks, and
the definition of done for each. The architecture and literature rationale live in
the [structure doc](https://claude.ai/code/artifact/9bc48659-ac0c-417d-9584-4174e86b8181).

Day estimates are rough relative sizing for one person, not commitments.

---

## Purpose & guardrails

Build a library that wires interpretability signals (probes, SAE features, activation
directions) into DPO and online RL post-training as reusable primitives, covering
modes 1–3 across offline and online training.

Non-negotiables, carried from the PRD:

- **No general abstraction before Phase 4.** Phases 1–3 are hardcoded vertical
  slice on one model, one probe, one dataset. The `Signal` / `Intervention` ABCs are
  *extracted* from working code, not designed up front.
- **Build on TRL.** `DPOTrainer` and `GRPOTrainer` are subclassed, never reimplemented.
  The value is the interpretability↔training bridge.
- **Nothing safety-specific in `interpost/`.** Refusal, toxicity, PKU-SafeRLHF loading,
  Llama-Guard, HarmBench configs — all of that lives in `examples/`. Core has no
  `refusal` / `toxicity` / `safety` symbol.
- **Modes 1–3 only.** Mode 4 (circuit-redirection) is a docstring stub.
- **Correctness before extension.** Phase 1 runs the Wehner & Fritz
  ([arXiv 2510.21531](https://arxiv.org/abs/2510.21531)) toxicity pipeline end-to-end
  as a plumbing check before the real work starts. It is not expected to reproduce
  their preservation effect at full strength — toxicity is mechanistically a poor
  drift testbed ([Lee et al. 2401.01967](https://arxiv.org/abs/2401.01967)).

---

## Prerequisites

**Runtime**

| | |
|---|---|
| Python | 3.11 |
| Core deps | `torch`, `transformers`, `trl` (pinned), `peft`, `accelerate`, `datasets`, `scikit-learn`, `safetensors` |
| SAE deps (Phase 4+) | `sae-lens`, `sparsify` |
| Eval deps (extra) | `lm-eval` (optional), `nnsight` / `transformer-lens` (eval-only) |
| Model — toxicity run | `meta-llama/Llama-3.2-1B` (base; non-aligned) |
| Model — refusal run + Phase 2+ | `meta-llama/Llama-3.2-1B-Instruct` |
| Data (Phase 1) | `google/civil_comments` + a RoBERTa toxicity classifier (`s-nlp/roberta_toxicity_classifier` or `unitary/toxic-bert`) for labels; `PKU-Alignment/PKU-SafeRLHF` (`-10K` subset) |
| Behavioral labeler (Phase 2+) | `meta-llama/Llama-Guard-3-1B` |
| Pretrained SAE (Phase 4+) | `EleutherAI/sae-Llama-3.2-1B-131k` |
| Fallback base model | `google/gemma-3-1b-pt` (the paper's exact model) |
| Hardware target | single GPU (1× A100-40GB is ample); **full fine-tune by default** |
| Fine-tuning mode | **Full FT at 1B scale** — VRAM is not a constraint and it keeps the residual-stream activations the probe reads identical to what training actually moves (LoRA freezes the base and shifts only adapters, which muddies Mode 3's "did the representation relocate" question). LoRA stays as a config flag for when someone brings a 7B+ model. |

**Pin the TRL version in `pyproject.toml`.** The trainer subclasses depend on
`DPOTrainer.get_batch_loss_metrics` / `concatenated_forward` and
`GRPOTrainer.compute_loss` / rollout internals. Record the exact version and
re-verify the hook points on every bump.

---

## Phase 0 — Scaffolding

**Effort:** ~0.5 day

**Tasks**

- [ ] `pyproject.toml` — package `interpost`, optional extras `[sae]`, `[eval]`, `[dev]`.
- [ ] Package skeleton (empty modules with docstrings):
      `signals/`, `activations/`, `interventions/`, `trainers/`, `eval/`, `config.py`.
- [ ] `examples/`, `tests/`, `docs/` (this folder).
- [ ] `tests/smoke/` — a 5-step DPO run on `gpt2` + 50 synthetic preference rows,
      no intervention. Wired into CI. This is the "did we break TRL integration" tripwire.
- [ ] `ruff` + `pytest` config; a trivial GitHub Actions workflow.
- [ ] `README.md` placeholder pointing at `docs/`.

**Done when:** `pip install -e ".[dev]"` works and `pytest tests/smoke` passes.

---

## Phase 1 — Toxicity plumbing check (Wehner & Fritz, lightweight)

**Effort:** ~1–2 days (mostly done). **Status: near-complete — see below.**

> **Why this is now a checkbox, not the deliverable.** Wehner & Fritz's "probe-based
> DPO" = probe picks `chosen`/`rejected` from k=5 candidates, then stock DPO; their
> finding is that probe-selection *preserves* probe AUC where RoBERTa-selection
> degrades it. But the mechanistic literature says this effect has little room to
> show up on **toxicity specifically**: DPO reduces toxicity by *bypassing* toxic
> regions with a distributed offset, not by rewriting the representation, which
> **persists** ([Lee et al. 2401.01967](https://arxiv.org/abs/2401.01967)); and the
> toxicity signal is linear at *every* layer (our L0–L15 sweep: 0.92–0.94 flat),
> because it's a surface feature carried unchanged through the residual stream. So a
> faithful toxicity reproduction mostly validates plumbing.

**Setup:** `meta-llama/Llama-3.2-1B` (base — an Instruct model won't emit toxic
candidates), Civil Comments prompts, RoBERTa-labeled, probe mean-pooled over response
tokens. `build_pairs` → probe-selected and RoBERTa-selected preference sets on the
**same** prompts. **LoRA DPO** (matches the paper; far more stable than full-FT DPO
on a raw base model), effective batch ≥ 64, β=0.1, 1 epoch, `--sft-weight 0.1`.
`eval.py` scores a **fixed shared text set** through each model.

- [x] 1a activation capture (`HookManager.capture`, `pool`) + tests
- [x] 1b `fit_probe` / `Probe` / layer sweep; toxicity probe fitted (val AUC ~0.94)
- [x] 1c `select_pairs` + `build_pairs.py` (matched prompt sets)
- [x] 1d `run_dpo.py` (stock `DPOTrainer`)
- [x] 1e `eval.py` (fixed shared AUC text set — works even when detox zeroes toxic gens)
- [ ] add LoRA + batch-size defaults to `run_dpo.py`
- [ ] one clean LoRA run: probe-selected vs RoBERTa-selected, then `eval.py`

**Done when:** the pipeline runs end-to-end; probe-selected DPO measurably lowers
`tox_rate` vs. base; `eval.py` produces sane frozen/refit AUC for all three models.
The size of the probe-vs-classifier preservation gap is a **reported finding**
("small on toxicity, as Lee et al. predicts"), not a gate.

---

## Phase 2 — Mode 2 loss-term DPO on refusal (the first real result)

**Effort:** ~5–7 days. **This builds interpost's core offline primitive.**

Refusal, not toxicity: it's a *computed* post-alignment behavior with real
representation structure (room to drift), DPO on an **instruct** model is stable (it
*is* the post-SFT starting point), and **PKU-SafeRLHF ships preference pairs** — no
candidate generation, no RoBERTa/Llama-Guard-for-selection. Method is interpost
**Mode 2** (probe term in the loss), which is what the PRD's build-order step 1 asked
for: "one probe, offline DPO, mode-2 loss shaping, before/after vs. vanilla DPO."

**2a — `OfflineDPOTrainer(trl.DPOTrainer)`** (`trainers/offline_dpo.py`)
- [ ] wrap the policy forward in `HookManager.capture` inside `get_batch_loss_metrics`
- [ ] add one hardcoded term `λ · f(probe_score(pooled_rejected_acts))` (e.g. ReLU of
      the probe logit on the rejected/harmful side)
- [ ] log the term + probe score alongside DPO metrics; `dataloader_pin_memory=False`

**2b — Refusal probe** (`examples/probe_guided_refusal_dpo/fit_probe.py`)
- [ ] `Llama-3.2-1B-Instruct`, chat template. PKU-SafeRLHF: harmful-compliance
      responses vs. refusals as the two classes. Layer sweep; pin the winner.
- [ ] keep the 19 harm-category labels for Phase 5's `ProbeBank`.

**2c — Behavioral labeler** (`examples/_shared/labelers.py`)
- [ ] `Llama-Guard-3-1B` wrapper `completion -> {0,1}` — **eval only** here.

**2d — Experiment + eval**
- [ ] vanilla DPO vs. `OfflineDPOTrainer` + probe term, on PKU-SafeRLHF pairs, instruct
      model, LoRA, β=0.1.
- [ ] `eval.py` (refusal variant): frozen vs. refit probe AUC on a fixed text set;
      HarmBench-subset ASR ↓; XSTest over-refusal (guard "always refuse"); capability slice.

**Done when:** a vanilla-vs-M2 table shows the probe term shifts refusal behavior
(ASR ↓ beyond vanilla, or a cleaner refusal representation) at acceptable capability
cost — and the `HookManager`-in-`get_batch_loss_metrics` machinery is proven.

---

## Phase 3 — Mode 3: monitoring-preservation

**Effort:** ~4–6 days (the metric design is the real work).
**Goal:** on Phase 2's refusal + `OfflineDPOTrainer` setup, show M2-alone erodes a
re-fit probe's accuracy and M2+M3 holds it near target at small capability cost.

**3a — `Preservation` intervention** (example-local; generalized in Phase 4)
- [ ] `fit_probe(layers=[L])` to pin the layer for mid-training re-fits.
- [ ] rolling buffer of `(pooled_activation, behavioral_label)` from recent batches.
- [ ] `refit_every` schedule: refit a fresh probe at pinned layer L on the buffer,
      score against an **independent** Llama-Guard label → `legibility(t)`, logged.
- [ ] per-step penalty `weight · ReLU(legibility_target − legibility(t))`.
- [ ] **never** feed the training-time probe's own score back as a target.

**3b — Experiment + eval**
- [ ] three runs: vanilla DPO / +M2 / +M2+M3.
- [ ] `eval/signal_report.py` v0: frozen-probe AUC, refit-probe AUC, the gap between
      them (obfuscation vs. relocation), behavioral agreement.
- [ ] HarmBench-subset ASR, XSTest, capability slice.

**Done when:** the three-way table shows M3 preserves refit-probe AUC that M2-alone
loses, without collapsing to "always refuse".

---

## Phase 4 — Generalize the `Signal` / `Intervention` interface

**Effort:** ~4–6 days
**Goal:** turn Phase 2–3's hardcoded code into the real ABCs, proven by adding
second and third `Signal` implementations that the interface was *not* shaped around.

**3a — `Signal` ABC** (`signals/base.py`)
- [ ] `fit`, `score`, `direction`; attrs `requires_fit`, `layers`, `pooling`.
- [ ] Extract `LinearProbeSignal` from the Phase 1/2 probe code. Support multi-output
      (a `(B, K)` score) for the category bank.

**3b — `Intervention` ABC** (`interventions/base.py`)
- [ ] `on_dataset`, `on_offline_batch`, `on_rollout`, `on_reward_model` — all optional,
      default no-op/identity; `weight`, `contributes`.
- [ ] Refactor Phase 2's term into `LossShaping` (`mode="term"`).
- [ ] Refactor Phase 3's preservation into `Preservation`.
- [ ] `OfflineDPOTrainer` now iterates `self.interventions` generically.

**3c — `SAEFeatureSignal` + adapters** (`signals/sae_feature.py`, `signals/sae_adapters.py`)
- [ ] `SAEAdapter` protocol: `encode(acts) -> feature_acts`, `decode_direction(k) -> vec`.
- [ ] `SAELensAdapter` (wraps `sae_lens.SAE`, incl. `load_from_disk`).
- [ ] `SparsifyAdapter` (wraps `sparsify.Sae.load_from_hub`, for the EleutherAI Llama SAE).
- [ ] `RawSAEAdapter` — the escape hatch: `safetensors` of `W_enc/b_enc/W_dec/b_dec`
      + a 5-field config (`d_in`, `d_sae`, `activation`, `k_or_threshold`, `hook_layer`).
- [ ] `SAEFeatureSignal(adapter, feature_idx | feature_set, pooling)`.

**3d — `ActivationDirectionSignal`** (`signals/direction.py`)
- [ ] `fit` computes contrastive mean-difference; also expose probe-weight and CCS
      direction options. `score` = projection.

**3e — `AggregateSignal` / `ProbeBank`** (`signals/aggregate.py`)
- [ ] `list[Signal] -> (B, K)`, plus reductions `mean | min | vote`.

**3f — Regression gate**
- [ ] Re-run Phase 2 end-to-end through the generalized interface, with the probe
      signal swapped for an `ActivationDirectionSignal`. Result must still hold.
- [ ] `tests/conformance/` — every `Signal` round-trips `fit`→`score`→`direction`;
      every `Intervention` is a safe no-op through hooks it doesn't implement.

**Done when:** the generalized interface reproduces Phase 2 with a different signal,
and conformance tests pass.

---

## Phase 5 — Mode 1 (data-level shaping)

**Effort:** ~2–3 days

- [ ] `BatchScorer(model, signal)` (`activations/scoring.py`) — batched forward +
      extract + `signal.score` over a `Dataset`, returns per-row scores for
      `(prompt, chosen)` and `(prompt, rejected)`.
- [ ] `DataShaping` intervention (`interventions/data_shaping.py`), `mode ∈
      {filter, reweight, relabel}`, a `predicate` / `weight_fn`.
- [ ] `OfflineDPOTrainer.__init__` runs `iv.on_dataset(self.train_dataset, scorer)`
      for each intervention **before** training. `DataShaping`'s batch/rollout hooks
      stay no-ops.
- [ ] Confirm a `DataShaping` handed to the (Phase 6) online trainer is inert.
- [ ] Experiment: probe-filtered PKU subset vs. full set, vanilla DPO both, compare
      safety + capability.

**Done when:** `data_shaping.py` produces a before/after and does nothing when
attached to an online trainer.

---

## Phase 6 — Online trainer + reward-model surface

**Effort:** ~6–9 days. **This is research, not porting** — there is no published
result for online M2/M3, and the "DPO preserves representations" mechanism may not
carry to on-policy RL.

**Full fine-tune** for the policy here (not LoRA) — the whole point of online M3 is
whether the policy relocates a representation to escape the monitor under RL pressure,
and it can only do that if all its weights are free to move.

- [ ] `OnlineTrainer` mixin (`trainers/online_base.py`):
  - [ ] `_rollout(prompts)` — generate, then one hooked scoring forward pass to
        capture policy activations for the completions.
  - [ ] `_apply_interventions(rewards)` — `contributes="reward"` adds a per-rollout
        delta pre-advantage; `contributes="loss"` accumulates an aux loss term.
- [ ] `OnlineGRPOTrainer(trl.GRPOTrainer, OnlineTrainer)` — custom `rollout_func`
      or `compute_loss` override (whichever the pinned TRL version supports cleanly).
- [ ] `RewardModelWrapper` (`trainers/reward_model.py`) — RM forward also captures RM
      activations; exposes `on_reward_model` so an intervention can compute
      `R_final = R_rm − λ · probe(rm_acts)`. **Online only.**
- [ ] Port `LossShaping.on_rollout` (probe score → reward delta) and
      `Preservation.on_rollout` (per-rollout legibility penalty).
- [ ] Experiments on a harmful-prompt set (AdvBench / HarmBench / WildJailbreak):
  - [ ] Llama-Guard reward + KL-to-ref baseline.
  - [ ] + probe-as-reward term (M2 online).
  - [ ] + RM-side probe penalty (RM surface).
  - [ ] + per-rollout preservation (M3 online).
  - [ ] Guard against degenerate "always refuse" (XSTest / capability reward).
- [ ] `OnlinePPOTrainer(trl.PPOTrainer, OnlineTrainer)` — stub + a TODO; full
      implementation deferred but the mixin must not assume GRPO specifics.

**Done when:** a GRPO run trains with an activation-derived reward term, the RM
surface is exercised, and `signal_report` shows what happened to probe legibility
under on-policy optimization.

---

## Phase 7 — Package, docs, examples

**Effort:** ~3–4 days

- [ ] Finalize `pyproject.toml`, `pip install interpost` metadata, version `0.1.0`.
- [ ] `examples/probe_based_toxicity_dpo/` — clean rewrite of Phase 1's toxicity run
      on the general interface. One script, one config, one results table.
- [ ] `examples/probe_guided_refusal_dpo/` — Phases 2 + 3 + 5 +
      the online piece from 5. Category-probe bank demo. All safety-specific code
      lives here.
- [ ] `docs/`:
  - [ ] `quickstart.md` — install, run an example, read the table.
  - [ ] `bring-your-own-signal.md` — implement a `Signal` subclass without reading core.
  - [ ] `modes.md` — what M1/M2/M3 do, offline vs. online, with code.
  - [ ] `api.md` — `Signal`, `Intervention`, both trainers, `signal_report`.
  - [ ] `roadmap.md` — M4 (circuit-redirection), PPO, multi-node (all out of scope now).
- [ ] `interventions/circuit_redirection.py` — docstring stub, raises `NotImplementedError`.

**Done when:** someone unfamiliar with the project can add their own probe and run
an offline + online experiment using only `docs/`.

---

## Testing strategy

| Layer | Test |
|---|---|
| Integration tripwire | `tests/smoke` — 5-step DPO on gpt2, runs in CI on every push |
| Interface conformance | every `Signal` round-trips `fit`/`score`/`direction` with correct shapes; every `Intervention` is a no-op through unimplemented hooks |
| Hook correctness | captured activations match a manual `output_hidden_states=True` forward; grad flows through a captured tensor |
| Data shaping | `on_dataset` filter/reweight/relabel produce expected row counts/weights on a fixture |
| Numerics | fixed seed → identical loss for the first N steps with and without a zero-weight intervention |
| **Not tested** | exact RL reward curves, exact eval scores — these are experiment outputs, asserted loosely (monotone direction) at most |

---

## Cross-cutting conventions

- **Config:** `OfflineConfig` / `OnlineConfig` dataclasses wrapping `DPOConfig` /
  `GRPOConfig` + `signal` + `interventions`. No bespoke arg parsing in core.
- **Metrics:** every intervention prefixes its logged metrics with its own name
  (`m2/term`, `m3/legibility`). Logger-agnostic (works with whatever TRL's
  `report_to` is set to).
- **Devices/dtype:** signals and probes follow the policy model's device/dtype;
  `RawSAEAdapter` casts on load.
- **Full fine-tune by default** for the policy (offline and online); LoRA behind a
  config flag, for 7B+ models only. Rationale: the interpretability signal reads the
  policy's residual stream *during* training — with LoRA the base weights never move,
  so the activations the probe sees drift differently than they would under real
  fine-tuning, and Mode 3's laundering/obfuscation claims lose their teeth (the model
  can't fully relocate a representation if most of it is frozen).
- **Determinism:** single `seed` in config, threaded to torch / numpy / `datasets`.

---

## Milestones

| Phase | Artifact | Gate to advance |
|---|---|---|
| 0 | installable skeleton, green smoke test | CI passes |
| 1 | toxicity pipeline runs end-to-end (LoRA) | probe-selected DPO lowers tox_rate; eval gives sane frozen/refit AUC |
| 2 | vanilla vs. M2 refusal table (`OfflineDPOTrainer`) | probe term shifts refusal behavior at acceptable capability cost; hook machinery proven |
| 3 | vanilla / M2 / M2+M3 refusal table | M3 preserves refit-probe AUC that M2-alone loses |
| 4 | generalized ABCs + 3 signal types | Phase 2 reproduces through the interface with a swapped signal |
| 5 | `data_shaping` before/after | inert when attached to an online trainer |
| 6 | GRPO run with activation reward + RM surface | `signal_report` characterizes online legibility |
| 7 | pip package + examples + docs | third party runs an experiment from docs alone |

---

## Research seams

`docs/research-direction.md` describes a research program — *monitor stability under
post-training* — that will **consume** interpost, never live inside it. The invariant:
`research/` imports `interpost`, never the reverse. That program is out of scope for
this build, but a few of its needs are cheap to wire now and a refactor to retrofit.
Wire the seams; don't build the experiments.

| Seam | Land it in | Why now |
|---|---|---|
| **`source: "policy" \| "reference"`** on every signal-consuming intervention (`LossShaping`, `FeatureReward`, `Preservation`). Phase 1 implements `"policy"` only; the param exists from day one. | param in Phase 2; formal in the Phase 4 ABC | Policy-vs-frozen-reference is *the* axis the research compares (Goodfire's key choice). Threading a second model + a mode flag through both trainers later is a refactor; a default-valued param is not. Offline DPO already has `ref_model`; GRPO loads one when `beta>0`. |
| **`SignalTrackingCallback`** — at each checkpoint, log frozen-probe + refit-probe AUROC and stash pooled activations for a fixed held-out eval set. | Phase 2 (offline), Phase 6 (online) | `signal_report` on a schedule. Without it the probe-transfer matrix can't be built retroactively — the activations/checkpoints are gone. Additive `TrainerCallback`, ~50 lines. |
| **`eval/probe_transfer.py`** (`P[i,j]` = probe fit at checkpoint *i*, scored at *j*) and **`eval/representation_drift.py`** (probe-direction cosine, CKA/subspace overlap, retained causal-steering effect). | Phase 4, alongside generalizing `signal_report` | General "how did this representation move" primitives, useful beyond the research program. Make the matrix a loop over `signal_report`, not a parallel impl. |
| **`research/` dir** + `README.md` with the import invariant. Empty otherwise. | now | Marks the boundary so nothing paper-specific leaks into `interpost/`. |

Mode 3's preservation objective stays deliberately unspecified (frozen-probe
performance / class geometry / monitored subspace / direction alignment / old-vs-new
monitor agreement / causal effectiveness) — which one works is the empirical question
the research program answers, not a thing to pick now.

---

## Open decisions

Tracked from the design discussion; resolve in-phase, don't pre-decide.

1. **Probe layer & pooling** — Phase 1b sweep (Llama-3.2-1B, 8k balanced samples,
   toxicity) came back **flat**: every layer 0.92–0.94 val AUC, argmax = L2 within
   noise. A flat curve ⇒ toxicity is a pervasive surface feature carried unchanged
   through the residual stream. Leaning: **pin L9** (~mid-depth, matching the paper's
   L20/34) rather than the lexical L2 — ~0.005 AUC cost, more honest test of the
   representation-level claim. Strict argmax (paper's literal method) = L2. Pooling
   stays `mean`.
2. **Probe backend** — `sklearn` LR (simple, Phase 1) vs. a torch linear head
   (needed if we ever want the probe itself trainable in-loop). Default sklearn;
   revisit only if a use case demands it.
3. **Online headline** — probe-as-reward (M2 online, cleanest thesis demo) vs.
   `R_rm − λ·probe` (RM surface). Do both in Phase 6; pick the headline from results.
4. **PPO timing** — after Phase 6 lands GRPO, unless a user needs a value model sooner.
5. **PRD citation** — whether to name Wehner & Fritz explicitly in `prd.md`'s
   Validation section (currently generic "probe-guided-DPO literature").
