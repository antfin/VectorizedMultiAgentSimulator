# CSV aggregation

`multi-scenario consolidate <experiments_root>` walks the tree, reads
each run's `output/metrics.json`, and emits a `runs.csv` with one row
per run.

```bash
multi-scenario consolidate experiments/
# Writes experiments/runs.csv
```

Columns: `run_id`, `exp_id`, `scenario`, `algorithm`, `seed`,
`M1_success_rate`, `M2_avg_return`, …, `M9_spatial_spread`,
`run_timestamp`, `n_envs`, `n_eval_episodes`, plus a config snapshot
section.

Used by:

- `scripts/compare_to_reference.py` — applies the F8.2 threshold logic
  (within ±10% of rendezvous_comm; ≥1.5σ across seeds).
- Streamlit Comparison page — loads `runs.csv` for cross-run plots.

> Full schema + worked example land at F10.2 review.
