# `multi-scenario sweep`

Multi-seed sweep over one YAML; on OVH, the canonical "fire + wait-for-DONE + pull back" command.

```bash
multi-scenario sweep <yaml_path> --seeds 0,1,2 [--runner local|ovh] [--follow]
```

## `--follow` (recommended for OVH)

When set, the CLI polls each submitted OVH job to DONE, then pulls
results back into the local run-dir. Same lifecycle the Streamlit
Submit page's auto-poll runs.

## `--seeds`

Comma-separated list of integer seeds. The CLI clones the YAML's
`experiment.seed` for each value and fires one submission per seed.

## See also

- [`multi-scenario run`](run.md) — single-seed dispatch
- [Submitting experiments](../getting_started/submitting_experiments.md)
