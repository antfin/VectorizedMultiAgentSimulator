# `multi-scenario run`

Execute one experiment from a YAML config file.

```bash
multi-scenario run <yaml_path> [--runner local|ovh] [--json]
```

## Dispatch routing

- YAML's `runtime.runner.type` chooses local vs OVH by default.
- `--runner local|ovh` overrides the YAML (useful for debugging an OVH-targeted YAML locally).
- Both targets route through `application/submission.py`, the same code path the Streamlit Submit page uses.

## `--json` (chat-trigger / scripted use)

Emits a single JSON record on the last stdout line:

```bash
# Local
multi-scenario run --json baseline.yaml | jq
# {"runner": "local", "run_id": "...", "run_dir": "..."}

# OVH
multi-scenario run --json --runner ovh lero_s3b_local.yaml | jq
# {"runner": "ovh", "run_id": "...", "job_id": "...",
#  "run_dir": "...", "s3_prefix": "...", "dashboard_url": "..."}
```

Use to drive scripted pipelines or chat-based submission workflows.

## OVH auto-upload

Before submitting an OVH job, the CLI checks the bucket's `.code_hash`
against the local repo hash. On drift, it auto-runs `upload-code` so
the new YAML / source changes ship before the job starts.

## Local CUDA preflight

If `cfg.training.device == "cuda"` and `torch.cuda.is_available() == False`,
the command fails early with a clear message instead of mkdir-ing a
container mount path that doesn't exist locally.

## LERO YAMLs

When `cfg.lero is not None`, the local path delegates to
`experiment_service._run_lero` which primes `OPENAI_API_KEY` from
`MS_ENCRYPTED_SECRETS` (OVH) or `.env` (local) before constructing the
LiteLlmClient. See [Operations → Secrets and env](../operations/secrets_and_env.md).

## See also

- [`multi-scenario sweep`](sweep.md) — multi-seed + auto-pullback
- [`multi-scenario inspect-lero`](inspect_lero.md) — view LERO summary
- [Submitting experiments](../getting_started/submitting_experiments.md) — Streamlit + CLI workflow
