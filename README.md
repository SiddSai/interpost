# interpost

Wire mechanistic-interpretability signals — linear probes, SAE features, activation
directions — into post-training (DPO and online RL) as reusable primitives, instead
of a one-off script per paper.

**Status:** pre-alpha scaffold (Phase 0 of [`docs/build-plan.md`](docs/build-plan.md)).
No public API yet — `interpost.signals` / `interpost.interventions` are intentionally
empty until Phase 3.

## Docs

| | |
|---|---|
| [`docs/prd.md`](docs/prd.md) | What this is, scope, definition of done |
| [`docs/build-plan.md`](docs/build-plan.md) | Phase-by-phase execution plan |
| [`docs/README.md`](docs/README.md) | Doc index |

## Dev setup

```bash
conda env create -f environment.yml   # creates env "interpost" (py3.11) + pip install -e .[dev]
conda activate interpost
```

<details><summary>venv instead of conda</summary>

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

```bash
pytest -q -m "not smoke"   # fast: import + package-layout checks
pytest -q -m smoke         # slow: 5-step DPO run on gpt2 (TRL integration tripwire)
ruff check .
```
