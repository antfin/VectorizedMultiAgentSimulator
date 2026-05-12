# Determinism test

> **Contract**: same config + same seed → byte-equal metrics across runs.
> Source: `tests/reproducibility/test_same_seed_byte_equal.py`.

## What it asserts

Two fresh local runs of an ER1 smoke YAML at `seed=0` must produce
identical M2, M3, M4 within a 1e-5 tolerance (absorbs float-precision
noise from non-deterministic CUDA reductions / threaded BLAS we can't
fully control).

```python
metrics_a = _run_once(cfg_dict)
metrics_b = _run_once(cfg_dict)
for k in ("M2_avg_return", "M3_steps", "M4_collisions"):
    assert abs(metrics_a[k] - metrics_b[k]) < 1e-5
```

Wall: ~5s per run × 2 = ~10s total.

## Why this matters

If a future change introduces non-determinism (forgets to seed an
RNG, switches to a non-deterministic op, etc.), this test fails before
downstream reproducibility experiments do — saving an OVH spend that
would have produced "not-quite-the-same numbers" silently.

## Distinct from `test_compare_to_reference.py`

That test pins F8.2's compare-to-reference *threshold logic* (within
±10% of rendezvous_comm; ≥1.5σ across seeds). Different contract —
neither subsumes the other.
