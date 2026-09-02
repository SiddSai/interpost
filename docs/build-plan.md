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

- **No general abstraction before Phase 3.** Phases 1–2 are a hardcoded vertical
  slice on one model, one probe, one dataset. The `Signal` / `Intervention` ABCs are
  *extracted* from working code, not designed up front.
- **Build on TRL.** `DPOTrainer` and `GRPOTrainer` are subclassed, never reimplemented.
  The value is the interpretability↔training bridge.
- **Nothing safety-specific in `interpost/`.** Refusal, toxicity, PKU-SafeRLHF loading,
  Llama-Guard, HarmBench configs — all of that lives in `examples/`. Core has no
  `refusal` / `toxicity` / `safety` symbol.
- **Modes 1–3 only.** Mode 4 (circuit-redirection) is a docstring stub.
- **Correctness before extension.** Phase 1 reproduces a published result
  (Wehner & Fritz, [arXiv 2510.21531](https://arxiv.org/abs/2510.21531)) before
  anything new is attempted.

---

## Prerequisites

**Runtime**

| | |
|---|---|
| Python | 3.11 |
| Core deps | `torch`, `transformers`, `trl` (pinned), `peft`, `accelerate`, `datasets`, `scikit-learn`, `safetensors` |
| SAE deps (Phase 3+) | `sae-lens`, `sparsify` |
| Eval deps (extra) | `lm-eval` (optional), `nnsight` / `transformer-lens` (eval-only) |
| Model — toxicity run | `meta-llama/Llama-3.2-1B` (base; non-aligned) |
| Model — refusal run + Phase 2+ | `meta-llama/Llama-3.2-1B-Instruct` |
| Data (Phase 1) | `google/civil_comments` + a RoBERTa toxicity classifier (`s-nlp/roberta_toxicity_classifier` or `unitary/toxic-bert`) for labels; `PKU-Alignment/PKU-SafeRLHF` (`-10K` subset) |
| Behavioral labeler (Phase 2+) | `meta-llama/Llama-Guard-3-1B` |
| Pretrained SAE (Phase 3+) | `EleutherAI/sae-Llama-3.2-1B-131k` |
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

## Phase 1 — Offline vertical slice: probe-selected preference DPO

**Effort:** ~6–8 days

> **What the paper actually does** (Wehner & Fritz, 2510.21531). Their "probe-based
> DPO" uses the probe as a **preference-pair selector**, not a loss term: generate
> k=5 candidates per prompt (temp 1.0), take the lowest-probe-score response as
> `chosen` and the highest as `rejected`, then run **stock DPO** (β=0.1). The
> classifier baseline is identical with RoBERTa picking the pairs. Their finding:
> **probe-selected DPO preserves held-out + retrained probe AUC; classifier-selected
> DPO degrades it.** Their loss-term method (NLL + λ·probe-penalty, λ=1) is the *SFT*
> variant and is their *negative* result — SFT does not preserve. Base model:
> Gemma-3-1B (non-aligned); probe at layer 20/34, mean-pooled over response tokens,
> logistic regression w/ L2 on standardized activations; ≥500 train / 50 val samples;
> LoRA; DPO dataset = 10k prompts, 70/20/10 split.

**Goal:** reproduce that result, then run the same pipeline on a refusal probe.

1. **Toxicity** — the faithful reproduction. Model: `meta-llama/Llama-3.2-1B` (base;
   an Instruct model won't emit enough toxic candidates). Probe: Civil Comments,
   response-token mean-pool, RoBERTa ≥ 0.5 label.
2. **Refusal / harmful-intent** — same pipeline, `Llama-3.2-1B-Instruct` +
   PKU-SafeRLHF. Tests whether the finding generalizes and stands up the probe
   Phases 2–3 build on.

Only the model, probe data, and eval metric differ between the two runs.

**1a — Activation capture** (`activations/hooks.py`, `activations/pooling.py`) — **DONE**
- [x] `HookManager.capture`, mask-aware `pool` (`last|mean|max|per_token`), tests.
- [ ] `HookManager.inject` still deferred to Phase 3.

**1b — Probe training** (`examples/_shared/`, local for now; generalized in Phase 3)
- [ ] `extract.py`: batched pooled-activation extraction over a layer sweep
      (teacher-forced forward on prompt+response, pool over the response span);
      `generate_responses` helper for candidate sampling.
- [ ] `probe.py`: `fit_probe(acts_by_layer, labels)` — standardize, `LogisticRegression`
      (L2), pick the layer by val AUC; `Probe` dataclass with `score` / `logit` /
      `direction`, `save_probe` / `load_probe` (joblib).
- [ ] **Toxicity probe** — base Llama, Civil Comments, RoBERTa-labeled.
- [ ] **Refusal probe** — Instruct Llama, PKU-SafeRLHF (harmful-compliance vs
      refusal/benign); keep the 19 category labels for Phase 3.
- [ ] Record chosen layers.

**1c — Signal-selected preference pairs** (`examples/_shared/select.py`) — the Mode-1
primitive, arriving early
- [ ] `select_pairs(model, prompts, scorer, k=5, temperature=1.0) -> preference Dataset`:
      sample k candidates, `chosen` = min score, `rejected` = max score.
- [ ] `scorer` is any `texts -> scores` callable — a `Probe`, or an output classifier.

**1d — Run stock DPO** (no subclass yet)
- [ ] `trl.DPOTrainer` on the probe-selected pairs and, separately, the
      classifier-selected pairs. β=0.1, **full fine-tune**, `dataloader_pin_memory=False`.
      (Paper used LoRA; at 1B full FT is cheap and cleaner for the activation work.)
- [ ] Toxicity: RoBERTa-selected baseline. Refusal: Llama-Guard-selected baseline.

**1e — Eval** (`examples/probe_based_toxicity_dpo/eval.py`, `examples/probe_guided_refusal_dpo/eval.py`)
- [ ] **Toxicity run:** toxicity rate = fraction of generations with RoBERTa ≥ 0.5 on
      100 held-out toxic prompts.
- [ ] **Refusal run:** HarmBench-subset ASR ↓ and XSTest over-refusal.
- [ ] **Both — the key metric:** held-out probe AUC (2 fresh probes on separate data,
      never in training) **and** retrained-probe AUC (2 probes refit on
      post-training activations). Plus a capability check (perplexity / short MMLU).

**1f — Loss-term variant + the negative result** (`trainers/offline_dpo.py`)
- [ ] `OfflineDPOTrainer(trl.DPOTrainer)` — capture acts in `get_batch_loss_metrics`,
      add one hardcoded `λ · f(probe_score(...))` term. (This is interpost's real
      Mode-2 machinery; Phase 2 builds on it.)
- [ ] Reproduce the paper's negative result: the analogous **probe-penalty SFT**
      (NLL + λ·penalty, λ=1) does **not** preserve probe AUC the way probe-selected
      DPO does — i.e. the effect is method-specific.

**Done when:**
- Toxicity: probe-selected DPO ↓ toxicity with retained held-out + retrained probe
  AUC; classifier-selected DPO ↓ toxicity with **degraded** AUC — the paper's result
  reproduces (direction + rough magnitude). Probe-penalty SFT does not preserve.
- Refusal: the same comparison is run and characterized — same pattern (good) or a
  documented divergence (input to Phase 2).

**Risks:**
- 1B may not separate probe- vs classifier-selected cleanly → larger model or sharper
  eval, decided from the table.
- Base Llama-3.2-1B may generate too little toxicity for good k=5 pairs → fall back to
  `google/gemma-3-1b-pt` (the paper's exact model) or a more toxic prompt set.
- The two domains may diverge. That is a result, not a bug.

---

## Phase 2 — Mode 3 as a designed intervention

**Effort:** ~4–6 days (the metric design is the real work)
**Goal:** on the refusal setup from Phase 1, show M2-alone erodes a re-fit probe's
accuracy and M2+M3 holds it near target at a small capability cost.

Reuses Phase 1's refusal probe, `PKU-SafeRLHF` loading, and `OfflineDPOTrainer` —
Phase 2 only adds the `Preservation` intervention on top.

**2a — `Preservation` intervention** (still example-local; generalized in Phase 3)
- [ ] Add a `layers=[L]` arg to `fit_probe` (pins the layer instead of sweeping).
- [ ] Rolling buffer of `(pooled_activation, behavioral_label)` from recent batches.
- [ ] `refit_every` schedule: refit a fresh probe on the buffer **at the pinned
      layer L** (so `legibility(t)` compares like-for-like across steps), compute
      held-out AUROC → `legibility(t)`, logged every step.
- [ ] Per-step penalty `weight · ReLU(legibility_target − legibility(t))`.
- [ ] **Do not** feed the training-time probe's own score back as a target.

**2b — Behavioral labeler** (`examples/.../labelers.py`)
- [ ] Wrap `Llama-Guard-3-1B` (or PKU's own safety label) as `completion -> {0,1}`
      for buffer labels and eval ground truth.

**2c — Experiment + eval**
- [ ] Three runs: vanilla DPO / +M2 / +M2+M3 (M2 comes straight from Phase 1).
- [ ] `eval/signal_report.py` **v0**: frozen-probe AUROC, refit-probe AUROC, the gap
      between them (obfuscation vs. relocation), behavioral agreement.
- [ ] Behavior eval: HarmBench-subset ASR, XSTest over-refusal, capability slice.
- [ ] Optional: run the same M2/M2+M3 comparison on the toxicity probe too, if the
      Phase 1 domains diverged and it's worth understanding why.

**Done when:** the three-way table shows M3 preserves refit-probe AUROC that
M2-alone loses, without collapsing to "always refuse" (XSTest holds).

---

## Phase 3 — Generalize the `Signal` / `Intervention` interface

**Effort:** ~4–6 days
**Goal:** turn Phase 1–2's hardcoded code into the real ABCs, proven by adding
second and third `Signal` implementations that the interface was *not* shaped around.

**3a — `Signal` ABC** (`signals/base.py`)
- [ ] `fit`, `score`, `direction`; attrs `requires_fit`, `layers`, `pooling`.
- [ ] Extract `LinearProbeSignal` from Phase 1's probe code. Support multi-output
      (a `(B, K)` score) for the category bank.

**3b — `Intervention` ABC** (`interventions/base.py`)
- [ ] `on_dataset`, `on_offline_batch`, `on_rollout`, `on_reward_model` — all optional,
      default no-op/identity; `weight`, `contributes`.
- [ ] Refactor Phase 1's term into `LossShaping` (`mode="term"`).
- [ ] Refactor Phase 2's preservation into `Preservation`.
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
- [ ] Re-run Phase 1 end-to-end through the generalized interface, with the probe
      signal swapped for an `ActivationDirectionSignal`. Result must still hold.
- [ ] `tests/conformance/` — every `Signal` round-trips `fit`→`score`→`direction`;
      every `Intervention` is a safe no-op through hooks it doesn't implement.

**Done when:** the generalized interface reproduces Phase 1 with a different signal,
and conformance tests pass.

---

## Phase 4 — Mode 1 (data-level shaping)

**Effort:** ~2–3 days

- [ ] `BatchScorer(model, signal)` (`activations/scoring.py`) — batched forward +
      extract + `signal.score` over a `Dataset`, returns per-row scores for
      `(prompt, chosen)` and `(prompt, rejected)`.
- [ ] `DataShaping` intervention (`interventions/data_shaping.py`), `mode ∈
      {filter, reweight, relabel}`, a `predicate` / `weight_fn`.
- [ ] `OfflineDPOTrainer.__init__` runs `iv.on_dataset(self.train_dataset, scorer)`
      for each intervention **before** training. `DataShaping`'s batch/rollout hooks
      stay no-ops.
- [ ] Confirm a `DataShaping` handed to the (Phase 5) online trainer is inert.
- [ ] Experiment: probe-filtered PKU subset vs. full set, vanilla DPO both, compare
      safety + capability.

**Done when:** `data_shaping.py` produces a before/after and does nothing when
attached to an online trainer.

---

## Phase 5 — Online trainer + reward-model surface

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

## Phase 6 — Package, docs, examples

**Effort:** ~3–4 days

- [ ] Finalize `pyproject.toml`, `pip install interpost` metadata, version `0.1.0`.
- [ ] `examples/probe_based_toxicity_dpo/` — clean rewrite of Phase 1's toxicity run
      on the general interface. One script, one config, one results table.
- [ ] `examples/probe_guided_refusal_dpo/` — Phase 1's refusal run + Phases 2 + 4 +
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
| 1 | toxicity + refusal probe-vs-classifier DPO tables | toxicity reproduces Wehner & Fritz; refusal run characterized |
| 2 | vanilla / M2 / M2+M3 refusal table | M3 preserves refit-probe AUROC M2 loses |
| 3 | generalized ABCs + 3 signal types | Phase 1 reproduces through the interface with a swapped signal |
| 4 | `data_shaping` before/after | inert when attached to an online trainer |
| 5 | GRPO run with activation reward + RM surface | `signal_report` characterizes online legibility |
| 6 | pip package + two examples + docs | third party runs an experiment from docs alone |

---

## Open decisions

Tracked from the design discussion; resolve in-phase, don't pre-decide.

1. **Layer & pooling** — sweep once in Phase 1, fix, expose as config.
2. **Probe backend** — `sklearn` LR (simple, Phase 1) vs. a torch linear head
   (needed if we ever want the probe itself trainable in-loop). Default sklearn;
   revisit only if a use case demands it.
3. **Online headline** — probe-as-reward (M2 online, cleanest thesis demo) vs.
   `R_rm − λ·probe` (RM surface). Do both in Phase 5; pick the headline from results.
4. **PPO timing** — after Phase 5 lands GRPO, unless a user needs a value model sooner.
5. **PRD citation** — whether to name Wehner & Fritz explicitly in `prd.md`'s
   Validation section (currently generic "probe-guided-DPO literature").
