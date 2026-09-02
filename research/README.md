# research/

Experiments that **consume** `interpost`. Charter: [`../docs/research-direction.md`](../docs/research-direction.md).

## Invariant

```
research/  imports  interpost      ✅
interpost  imports  research       ❌ never
```

`interpost/` stays a general-purpose library. Nothing paper-specific — no
monitor-Goodharting logic, no experiment configs, no dataset assumptions — leaks
into it. If a capability is generally useful ("would another researcher want this
primitive with their own signal?"), it goes in `interpost/`; if it only serves a
controlled comparison for the paper, it lives here.

## Status

Empty. The first research experiment starts only after the library reaches the point
where it can run it — earliest after Phase 2 (offline monitor-transfer tracking),
fully after Phase 5 (optimization-pathway comparison). See the build plan's
"Research seams" section for what the library is wiring in early to make this
possible without a later refactor.
