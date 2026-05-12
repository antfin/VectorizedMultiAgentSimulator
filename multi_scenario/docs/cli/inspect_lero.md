# `multi-scenario inspect-lero`

Pretty-print a LERO run's `final_summary.json` + per-candidate history.

```bash
multi-scenario inspect-lero <run_dir>
```

Output sections:

- Headline: iterations completed, candidates total, total cost (USD), best verdict.
- Inner-loop metrics (1M frames screening).
- **Post-full-train metrics (10M frames — the science result)**.
- Fallback chain — every full-training attempt with outcome.
- History records — per-iter candidate count.

The "post-full-train" block was added by Phase 1 of the LERO redesign;
older `final_summary.json`s (pre-fix) report `None` here and need
re-evaluation via [`multi-scenario eval`](eval.md).
