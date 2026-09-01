# interpost — Project Requirements

## What this is

`interpost` is an open source library that lets mechanistic interpretability signals (probes, SAE features, activation directions) be wired into post-training (DPO and online RL) as first-class, reusable primitives. Today, every attempt to connect interpretability to post-training is a one-off research script tied to a single paper's specific method. There's no general interface that lets an arbitrary interpretability signal plug into an arbitrary post-training run. `interpost` is that connective layer.

The library is general-purpose by design — the core has no notion of "safety" or "refusal" baked in. The first worked example happens to be a safety application (probe-guided refusal training), but that lives in `examples/`, not in core.

## Problem framing

Post-training treats the model as a black box: optimize a scalar reward or preference signal, never inspect internals. Interpretability tools exist to inspect internals but aren't connected to the training loop in any reusable way. We're building the missing connective infrastructure, organized around a small number of distinct "modes" of intervention:

1. **Data-level shaping** — score/filter/reweight a preference dataset using an interpretability signal, before any training starts.
2. **Loss/reward shaping** — add a signal-derived term directly into the training objective (DPO loss, or RL reward), so gradients are informed by internals, not just outputs.
3. **Monitoring-preservation** — regularize training so a signal stays predictive of the behavior it tracks, preventing the model from "laundering" behavior into an untraceable representation while optimizing for it. This is a different goal from mode 2: not "hit a target," but "don't destroy legibility."
4. **Circuit-redirection** — reward the model for solving a task via one internal mechanism over another, not just for solving it. Out of scope for this build; document as roadmap only, do not implement.

## Scope for this build

Modes 1-3, across **both** offline (DPO) and online (PPO/GRPO) training. Mode 4 is explicitly not implemented — stub it out in docs/roadmap only.

Modes do not generalize uniformly across offline and online training, and that's intentional, not a bug to fix:
- Mode 2 and mode 3 apply in both settings, but the mechanism differs — a loss term on a static batch (offline) vs. a per-rollout reward/regularization term on live generations (online).
- Mode 1 is inherently a static-dataset operation. It doesn't have a real analog inside an online RL loop (there's no fixed dataset once the policy is generating its own rollouts). It still applies to any upstream data stage (e.g., SFT data, prompt curation) feeding into an online run — just not to the online loop itself.

Online RL also introduces a reward-model-level intervention surface that offline DPO doesn't have (DPO has no reward model). This should be its own extension point, not folded into the policy-side interventions.

## Architecture requirements

**Signal layer** — a common interface wrapping any interpretability tool:
- Takes model activations, returns a scalar or vector score.
- Needs concrete implementations for: a linear probe, an SAE feature, and a raw activation-direction projection.
- Must work identically whether the activations come from a static offline batch or a live online rollout — no offline/online-specific logic belongs here.

**Intervention layer** — implements the three in-scope modes, shared across offline and online training via a small set of hook points (something like: a pre-training dataset hook for mode 1, a batch-level hook for offline mode 2/3, a rollout-level hook for online mode 2/3). A mode-1 intervention should simply have nothing to attach to when used with an online trainer, rather than requiring special-case handling elsewhere.

**Trainer layer** — offline and online training are genuinely different underneath (different libraries, different math) and should be separate implementations that both call into the same Signal/Intervention interface. Don't force them into one shared trainer class. Offline should build on top of an existing, well-tested DPO implementation rather than reimplementing DPO math from scratch. Online should build on top of an existing PPO/GRPO implementation for the same reason — the value of this project is the interpretability-training bridge, not reimplementing RL algorithms.

**Reward-model intervention surface** — a distinct extension point for online training, for interventions that operate on the reward model itself rather than the policy.

**Eval tooling** — the library should make it easy to evaluate a trained model against a signal after the fact (e.g., checking whether a probe's predictiveness held up, or whether an intervention's behavior generalizes), not just apply signals during training.

## Build order

1. Get a single, real, end-to-end case working before generalizing anything: one small open model, one probe, offline DPO, mode-2 (loss shaping) intervention, one clear before/after comparison against vanilla DPO on some measurable property.
2. Add mode 3 (preservation) to the same setup, comparable against the mode-2 and vanilla baselines.
3. Only after both modes work end-to-end on a real case, generalize the Signal/Intervention interface so it isn't hardcoded to this one experiment.
4. Add mode 1 (data-level filtering) using the now-generalized interface.
5. Build the online (PPO/GRPO) trainer against the same interface, including the reward-model intervention surface.
6. Worked example(s), docs, README sufficient for a third party to plug in their own signal without reading source.

Do not design the general abstraction before step 1 produces a working, real result — the risk of building the abstraction first is guessing wrong about what it needs to support.

## Validation

Before trusting the implementation, reproduce a known result from existing probe-guided-DPO literature as a correctness check, then extend past it (existing work is offline/DPO-only and doesn't cover mode 1, mode 3, or online training).

## Explicit non-goals for this build

- Mode 4 (circuit-redirection) — document as future work, don't implement.
- Any hosted/platform component, UI, or dashboard — this is a library, not a service.
- Multi-node / distributed training support.
- Baking any safety-specific or refusal-specific logic into the core library — that belongs in examples/, not core.

## What "done" looks like

A pip-installable package with a general Signal/Intervention interface, working offline (DPO) and online (PPO/GRPO) trainers implementing modes 1-3 as applicable to each, a reward-model intervention surface for online training, at least one complete worked example with a real comparative result, and documentation sufficient for someone unfamiliar with the project to use it with their own interpretability signal.
