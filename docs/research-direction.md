# interpost — Project Context + Research Direction

## 1. What interpost is

`interpost` is an open-source library for connecting mechanistic interpretability signals to post-training.

The core motivation is that post-training methods such as DPO, PPO, and GRPO usually optimize behavioral signals: preference labels, reward-model scores, task rewards, verifier outputs, etc. Separately, mechanistic interpretability gives us signals derived from model internals: probes, SAE features, activation directions, representation subspaces, and related tools.

Today, attempts to combine these two areas are mostly implemented as paper-specific research code. The goal of `interpost` is to provide a reusable layer where an arbitrary internal signal can be connected to different stages of the post-training process.

The library itself should remain **general-purpose**.

Safety is the first major experimental application, but concepts such as refusal, harmfulness, or safety should not be baked into the core abstractions. A researcher should also be able to use `interpost` for factuality, reasoning, uncertainty, deception, reward hacking, style, sycophancy, planning, or any other property for which an internal signal can be constructed.

---

# 2. Core abstraction

At a high level:

```text
                         interpost

            ┌────────────────────────────┐
            │          SIGNALS           │
            │                            │
activations ─► LinearProbeSignal         │
            │ SAEFeatureSignal           │
            │ ActivationDirectionSignal  │
            │ CustomSignal               │
            └──────────────┬─────────────┘
                           │
                           ▼
            ┌────────────────────────────┐
            │       INTERVENTIONS        │
            │                            │
            │ Data shaping               │
            │ Loss shaping               │
            │ Reward shaping             │
            │ Monitor preservation       │
            └──────────────┬─────────────┘
                           │
                           ▼
            ┌────────────────────────────┐
            │          TRAINING          │
            │                            │
            │ SFT / DPO / GRPO / PPO     │
            └────────────────────────────┘
```

Signals should expose model-internal information in a common form.

The initial signal implementations are:

* linear probes
* SAE features
* raw activation-direction projections

A signal should work regardless of whether activations came from a static offline dataset or live online rollouts. Offline/online-specific logic belongs in the trainer/intervention layer, not in the signal itself.

---

# 3. The intervention surfaces

There are several distinct ways an internal signal can affect post-training.

These should not be treated as interchangeable.

## Mode 1 — Data shaping

Use internal signals before training to:

* score examples
* filter examples
* reweight examples
* construct curricula
* identify difficult examples
* cluster or select training samples

Conceptually:

```text
dataset
   ↓
model activations
   ↓
interpretability signal
   ↓
score/filter/reweight
   ↓
training dataset
```

This includes SAERL-like workflows.

Mode 1 is naturally a static-data intervention. It does not directly operate inside an online RL rollout loop, although it can curate prompts or upstream SFT data used by an online trainer.

Example desired API:

```python
signal = SAEFeatureSignal(
    sae=sae,
    feature_ids=[...],
    layer=...
)

scored = interpost.score_dataset(
    model=model,
    dataset=dataset,
    signal=signal
)

filtered = interpost.filter_dataset(
    scored,
    threshold=...
)
```

---

## Mode 2A — Reward shaping

Use the internal signal as part of an online RL reward.

For example:

```text
policy rollout
      ↓
internal feature score
      ↓
R_internal
      +
task / safety / verifier reward
      ↓
GRPO or PPO
```

Conceptually:

$$
R = R_{\text{task}} + \lambda R_{\text{internal}}
$$

This includes Features-as-Rewards-like methods.

Example API:

```python
feature_reward = FeatureReward(
    signal=signal,
    weight=0.25,
)

trainer = InterpostGRPOTrainer(
    model=model,
    reward_funcs=[
        task_reward,
        feature_reward,
    ],
)
```

There should eventually be an explicit distinction between evaluating the internal signal on:

1. the current policy model
2. a frozen reference model

because these have very different optimization properties.

For example:

```python
FeatureReward(
    signal=probe,
    source="reference_model",
)
```

versus:

```python
FeatureReward(
    signal=probe,
    source="policy_model",
)
```

---

## Mode 2B — Loss shaping

Offline methods such as DPO do not have the same reward structure as online RL.

Instead an internal signal can contribute an auxiliary differentiable objective:

$$
L =
L_{\text{DPO}}
+
\lambda L_{\text{internal}}
$$

Example:

```python
intervention = AuxiliarySignalLoss(
    signal=probe,
    weight=0.1,
)
```

This might encourage:

* a probe score
* activation alignment
* activation separation
* SAE feature activation/suppression
* preservation of some representation property

The important architectural principle is that reward shaping and loss shaping can belong to the same broad conceptual family while remaining separate concrete intervention types.

Offline and online trainers should therefore remain separate implementations while consuming a common Signal/Intervention interface. `interpost` should build on existing tested DPO/PPO/GRPO implementations rather than reimplementing the algorithms themselves.

---

## Mode 3 — Monitoring preservation

This is simultaneously:

1. a useful general library primitive
2. the beginning of a potentially important research direction

The basic problem is that an interpretability signal is usually constructed on some model state:

$$
p_0(h_{\theta_0})
$$

For example, a linear probe might reliably predict whether a representation corresponds to harmful versus safe behavior.

Post-training changes the model:

$$
\theta_0
\rightarrow
\theta_1
\rightarrow
...
\rightarrow
\theta_T
$$

There is no guarantee that:

$$
p_0(h_{\theta_T})
$$

still measures the same thing.

The representation itself can change underneath the monitor.

`MonitoringPreservation` therefore means introducing an intervention whose goal is not necessarily:

> maximize this internal feature

but instead:

> preserve the relationship between a monitor and the behavioral property it was originally capable of tracking.

This is already part of the intended `interpost` design: Mode 3 exists to prevent training from destroying the legibility of the representation being monitored.

Importantly, we should NOT prematurely assume what preservation means mathematically.

Candidate preservation objectives may include:

* retaining frozen-probe predictive performance
* preserving positive/negative class geometry
* preserving a monitored representational subspace
* preserving activation-direction alignment
* maintaining agreement between old and freshly trained monitors
* preserving causal effectiveness of an intervention direction
* constraining representation drift in a task-relevant subspace

Which version actually works is an empirical research question.

---

# 4. Relationship to existing work

Several recent research directions independently validate the motivation for `interpost`.

## SAERL-like work

SAE-derived features can be used for:

* filtering
* curriculum construction
* data selection
* training-example organization

This corresponds closely to Mode 1.

`interpost` should make this a reusable operation rather than something tied to a specific SAE-RL paper.

A researcher should be able to replace an SAE feature with a probe or activation direction without rebuilding the entire data pipeline.

---

## Features-as-Rewards / RLFR-like work

Goodfire demonstrated that internal features can be used as reward signals during RL.

An especially important design choice in their setup is that the feature monitor can operate on a frozen reference model rather than simply differentiating through the current policy's activations.

That provides one mechanism for reducing direct pressure on the representation being monitored.

`interpost` should be able to express this approach cleanly.

But `interpost` should also expose other choices.

That is important scientifically because different optimization paths may produce different failure modes.

---

# 5. The main research question enabled by interpost

The broad scientific question is:

# When we use internal representations as supervision signals during post-training, do those signals remain valid as the model changes?

More specifically:

> When does an internal monitor survive post-training, when does it become stale or Goodharted, and can monitor-preservation objectives prevent this degradation?

This question becomes particularly important as interpretability signals move from being passive diagnostic tools to active optimization targets.

Normally:

```text
MODEL
   ↓
activations
   ↓
PROBE
   ↓
measurement
```

The model is not optimizing against the probe.

Once the probe becomes training supervision:

```text
MODEL
   ↓
activations ─────────────→ PROBE
   ↓                         ↓
output                    reward/loss
   ↓                         ↓
behavioral objective ─────→ optimizer
                              ↓
                         update MODEL
```

there is now optimization pressure involving the monitor.

The desired semantic property might be:

```text
"be genuinely safer"
```

while the optimizer may only observe:

```text
"produce states for which this particular probe gives a favorable score"
```

Those objectives are not guaranteed to be equivalent.

This creates the possibility of **monitor Goodharting**.

---

# 6. Important distinction from Goodfire

The research question is NOT:

> Can we use features as rewards?

Goodfire and related work already demonstrate that this is possible.

It is also NOT:

> Feature rewards necessarily break probes.

Goodfire presents an important positive case where feature-based reward optimization works without obviously destroying the usefulness of the relevant representation.

The better question is:

> Under what optimization conditions does monitor transfer remain stable, and under what conditions does it fail?

One candidate hypothesis is that the optimization path matters.

For example, compare:

## Condition A — Frozen-reference feature reward

```text
student generates output
         ↓
frozen reference model
         ↓
probe / feature
         ↓
reward
         ↓
RL update student
```

There is no direct optimization through the student's monitored activation.

---

## Condition B — Policy-activation feature reward

```text
student
   ↓
student activations
   ↓
feature/probe
   ↓
reward
   ↓
RL update student
```

The reward depends on the policy's own current representation.

---

## Condition C — Differentiable internal loss

```text
student activations
      ↓
probe / feature objective
      ↓
loss
      ↓
gradient directly through representation
```

This creates even more direct optimization pressure on the monitored representation.

---

## Condition D — Internal objective + monitor preservation

Same as Condition B/C, but add an objective intended to preserve monitor validity.

These conditions give us a natural experimental continuum:

```text
external monitoring
      ↓
reference-model reward
      ↓
policy-model reward
      ↓
direct differentiable representation objective
```

The working hypothesis is:

> More direct optimization pressure on the monitored representation may increase the risk that the model changes the representation in ways that satisfy the monitor without preserving the semantic property the monitor originally tracked.

This is a hypothesis, not something the library should assume is true.

---

# 7. Initial research experiments

The first research goal should be observational.

Do NOT immediately invent a complicated monitor-preservation algorithm.

First determine whether monitor degradation actually occurs.

## Experiment 1 — Track monitor transfer through normal post-training

Take a small open model.

Train a safety-relevant probe at checkpoint:

$$
\theta_0
$$

Freeze that probe.

Perform ordinary post-training:

```text
base/instruct model
      ↓
DPO or GRPO
      ↓
θ1
      ↓
θ2
      ↓
...
      ↓
θT
```

At every checkpoint evaluate:

### Behavioral metric

For example:

* safety/refusal accuracy
* HarmBench
* jailbreak robustness
* helpfulness
* whatever task is selected

### Frozen-probe performance

Evaluate:

$$
p_0(h_{\theta_t})
$$

Does the original monitor continue to predict the behavior?

### Fresh-probe performance

At each checkpoint, train a new probe:

$$
p_t(h_{\theta_t})
$$

Then compare:

$$
p_0(h_{\theta_t})
$$

against:

$$
p_t(h_{\theta_t})
$$

This distinction is extremely important.

If:

```text
frozen probe performance ↓
fresh probe performance ≈ constant
```

then the concept may still be internally represented, but the representation has moved.

That would be **monitor staleness / representation drift**, not disappearance of the concept.

---

# 8. Probe-transfer matrix

The evaluation tooling should eventually make it easy to construct:

$$
P_{i,j}
=
\text{performance of probe trained at checkpoint i on checkpoint j}
$$

For example:

```text
probe trained at →

model evaluated ↓      t0      t1      t2      t3

t0                    .94     .91     .84     .72
t1                    .90     .94     .88     .79
t2                    .77     .87     .95     .90
t3                    .63     .75     .89     .96
```

This lets us visualize how the representation changes over training.

This functionality belongs naturally in `interpost.eval`, because it is broadly useful beyond this specific paper.

---

# 9. Experiment 2 — Compare optimization pathways

Once the baseline instrumentation works, compare:

```text
1. Vanilla post-training

2. Frozen-reference feature reward

3. Current-policy feature reward

4. Differentiable internal auxiliary loss
```

Keep behavioral objectives as comparable as possible.

Measure:

* behavioral performance
* OOD generalization
* original probe performance
* fresh probe performance
* probe-transfer matrix
* representation similarity
* activation-direction similarity

The goal is to ask:

> Does the way we inject an interpretability signal into optimization determine whether that signal remains trustworthy?

---

# 10. Experiment 3 — Explicit monitor Goodharting

If Conditions 2–4 produce interesting differences, investigate them directly.

An illustrative phenomenon would be:

```text
training step →

optimized probe score        ↑ ↑ ↑ ↑ ↑ ↑ ↑
ID behavioral safety         ↑ ↑ ↑ ─ ─ ─ ─
OOD behavioral safety        ↑ ↑ ─ ─ ─ ─ ─
fresh probe validity         ↑ ─ ↓ ↓ ↓ ↓ ↓
```

That would suggest the optimized monitor continues improving even though independent measurements of the semantic property do not.

The key point is that we should NOT call something Goodharting merely because representations drift.

To support that claim we need divergence between:

1. the optimized signal
2. independent behavioral measurements
3. independent representation-based measurements

Potential independent checks:

* new probes trained after optimization
* probes trained on held-out concepts/datasets
* OOD behavioral evaluation
* causal steering tests
* representation similarity
* alternative SAE features
* cross-model or cross-checkpoint transfer

---

# 11. Experiment 4 — Monitor preservation

Only after establishing a real failure mode should we implement and test preservation methods.

Compare:

```text
Vanilla DPO/GRPO

Internal-signal DPO/GRPO

Internal-signal DPO/GRPO
+
MonitoringPreservation
```

Ideal result:

```text
                            Task       OOD       Frozen      Fresh
                            score      score     monitor     monitor

Vanilla RL                  good       good      moderate    good

Feature-guided RL           best       worse     excellent   poor

Feature + preservation      ~best      best      good        good
```

If something like this occurs consistently, then we potentially have:

1. a failure mode
2. a mechanistic explanation
3. a mitigation

That is the structure of a strong paper.

---

# 12. Possible mechanistic analysis

Do not rely only on probe AUROC.

We should eventually compare:

### Linear decodability

Can a linear probe read out the concept?

### Representation geometry

Does the relevant direction/subspace rotate during training?

Possible measurements:

* cosine similarity between directions
* CKA / representation similarity
* subspace overlap
* probe-weight alignment
* activation-distribution changes

### Causal effect

Does steering along the original direction still affect behavior?

This is especially useful because:

```text
decodable ≠ causally used
```

A representation can remain linearly detectable without necessarily participating in the computation the same way.

Conversely, probe transfer can fail because the representation changed coordinates even though the underlying computation remains similar.

---

# 13. Secondary research direction: DPO vs GRPO

Once both offline and online trainers exist, `interpost` enables another potentially interesting project:

> When DPO and GRPO achieve similar external behavior, do they produce the same internal solution?

We could behavior-match models trained using:

```text
DPO
vs
GRPO
```

then compare:

* probe transfer
* representation drift
* internal direction similarity
* steering effects
* OOD safety/generalization
* monitor stability

This should be treated as a secondary direction rather than mixing it into the first paper immediately.

If a strong distinction appears, it can become its own project.

---

# 14. How the research fits the current library

The paper should be a **consumer of interpost**, not something baked into interpost.

The desired repository separation is approximately:

```text
interpost/
    signals/
        probe.py
        sae.py
        direction.py

    interventions/
        data.py
        reward.py
        loss.py
        preservation.py

    trainers/
        dpo.py
        grpo.py
        ppo.py

    eval/
        probe_transfer.py
        representation_drift.py
        signal_eval.py

    hooks/
    utils/

examples/
    sae_data_filtering/
    feature_reward_grpo/
    probe_guided_dpo/
    monitor_preservation/

research/
    monitor_stability/
        configs/
        experiments/
        analysis/
        README.md
```

Important invariant:

```text
research imports interpost
```

but:

```text
interpost NEVER imports research
```

`interpost` remains a reusable library.

---

# 15. Development philosophy

Do not overengineer the abstractions before real experiments force them into existence.

The current PRD intentionally says:

1. get one real probe-guided DPO experiment working
2. obtain a measurable result
3. add monitoring preservation
4. only then generalize the Signal/Intervention API
5. add data shaping
6. add online PPO/GRPO

That principle should remain.

The danger is building a theoretically elegant API around research workflows we have not actually implemented.

Instead:

```text
real experiment
      ↓
discover needed abstraction
      ↓
generalize abstraction
      ↓
second experiment
      ↓
stress-test abstraction
      ↓
improve API
```

The library and experiments should co-evolve.

---

# 16. Near-term implementation priorities

The immediate goal is NOT the full research paper.

The immediate goal is:

## Milestone 1

One complete end-to-end experiment:

```text
small open model
+
linear probe
+
DPO
+
internal auxiliary objective
+
vanilla DPO baseline
+
behavioral evaluation
+
probe evaluation
```

This serves as both:

* the first usable interpost training workflow
* an implementation correctness test

The PRD already requires reproducing a known probe-guided DPO-style result before trusting the implementation.

---

## Milestone 2

Add evaluation infrastructure early:

```text
frozen probe eval
fresh probe eval
checkpoint comparison
signal logging
```

Do this before implementing every signal type.

If monitor drift exists, we want to notice it from the beginning rather than discovering six months later that we never logged the necessary activations/checkpoints.

---

## Milestone 3

Generalize the signal interface.

Support:

```text
LinearProbeSignal
ActivationDirectionSignal
SAEFeatureSignal
```

with consistent semantics.

---

## Milestone 4

Add Mode 1 data tooling.

Support workflows such as:

```text
signal
  ↓
dataset scoring
  ↓
filter / weight / rank / curriculum
```

This is important independently of the research paper because we want `interpost` to be genuinely useful as an open-source library.

---

## Milestone 5

Add GRPO.

Then reproduce a feature-reward-style setup.

At this point we have the infrastructure necessary to begin serious experiments comparing optimization paths.

---

# 17. What success looks like

There are two independent outputs.

## Output A — Open-source infrastructure

`interpost` should eventually be a pip-installable library where another researcher can:

```python
signal = SAEFeatureSignal(...)

trainer = InterpostGRPOTrainer(
    model=model,
    interventions=[
        FeatureReward(signal)
    ]
)

trainer.train()
```

or switch to:

```python
signal = LinearProbeSignal(...)
```

without rewriting the training stack.

Likewise, they should be able to use the same signal for:

```text
data filtering
reward shaping
loss shaping
monitor preservation
evaluation
```

The PRD's existing definition of done remains the long-term engineering target: a general Signal/Intervention interface, offline and online trainers, multiple intervention modes, reward-model hooks, worked examples, and documentation sufficient for a third-party researcher to bring their own interpretability signal.

---

## Output B — Research

Use `interpost` to investigate:

> How stable are internal monitors under post-training, how does stability depend on how the signal enters optimization, and can monitor-preservation interventions maintain monitorability?

Possible paper structure if the phenomenon is real:

### Finding 1

Post-training causes measurable representation/monitor drift.

### Finding 2

Different uses of internal supervision cause dramatically different amounts of drift.

### Finding 3

Direct optimization of policy representations can cause the optimized monitor to diverge from independent behavioral and representation-level measurements.

### Finding 4

A monitor-preservation objective mitigates this failure while retaining most of the downstream performance gain.

The eventual contribution would therefore not be:

> We invented using probes during RL.

It would instead be:

> We systematically characterize when representation-based supervision remains trustworthy during post-training, identify conditions under which the monitor itself becomes unreliable, and introduce interventions for maintaining monitorability.

---

# 18. Important rule for future development

Whenever implementing a new `interpost` capability, ask two questions:

### Engineering question

> Would another researcher reasonably want to use this primitive with their own signal and task?

If yes, it belongs in `interpost`.

### Scientific question

> Does this capability enable a controlled comparison that teaches us something about how post-training changes model representations?

If yes, add an experiment under `research/`.

Do not contaminate the generic API with paper-specific assumptions.

The ideal feedback loop is:

```text
build reusable primitive
        ↓
run experiment
        ↓
observe phenomenon
        ↓
investigate phenomenon
        ↓
learn what abstraction is missing
        ↓
improve library
        ↓
run stronger experiment
```

That is the intended relationship between the open-source project and the research program.
