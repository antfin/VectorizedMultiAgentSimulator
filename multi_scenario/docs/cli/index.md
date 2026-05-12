# CLI reference

`multi-scenario` is a Typer-based CLI; all subcommands live under
`src/multi_scenario/cli/`.

```bash
multi-scenario --help
```

| Subcommand | Purpose |
|---|---|
| [`run`](run.md) | Execute one experiment from a YAML — local or OVH dispatch |
| [`sweep`](sweep.md) | Multi-seed sweep over a YAML; `--follow` waits + pulls back |
| [`eval`](eval.md) | Re-evaluate a saved checkpoint (LERO-aware) |
| [`inspect-lero`](inspect_lero.md) | Pretty-print a LERO run's final_summary + history |
| [`regenerate-videos`](regenerate_videos.md) | Replay saved policy, write before/after mp4 |
| [`upload-code`](upload_code.md) | Push local source tree to OVH code bucket |
| [`validate`](validate.md) | Validate a YAML against the schema |
| `resume` | Continue an interrupted run from disk — see `cli/resume.py` |
| `consolidate` | Aggregate per-run results into a CSV — see `cli/consolidate.py` |
