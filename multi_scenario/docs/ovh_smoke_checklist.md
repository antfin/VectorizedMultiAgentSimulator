# F6.5 — End-to-end OVH smoke checklist (manual)

A guided procedure for the **first** OVH submission. Costs roughly one
ai1-1-gpu credit (~5–10 minutes wall time, ~0.20 EUR). Re-run the checklist
whenever you change OVH project, region, buckets, or framework architecture.

> **Pre-requisite:** complete `docs/ovh_setup.md` first (one-time).

## 0. Sanity checks

```bash
ovhai --version                             # e.g. ovhai 3.35.0
ovhai user list                             # confirms login still valid
ls configs/ovh.yaml configs/s3.yaml         # both present + populated
echo "$AWS_ACCESS_KEY_ID" | head -c 4       # boto3 creds reachable
```

If any check fails, return to `docs/ovh_setup.md`.

## 1. Validate the smoke YAML

Pick one yaml. Default suggestion: a smoke we already trust on local.

```bash
multi-scenario validate experiments/discovery/baseline/configs/mappo_smoke.yaml
```

Should exit 0. (Catches typos before spending credit.)

## 2. Upload code to `bucket_code`

```bash
multi-scenario upload-code configs/s3.yaml --dry-run
multi-scenario upload-code configs/s3.yaml
```

Verify on OVH:

```bash
ovhai bucket object list ms-code@GRA | head -20
```

Expect `src/multi_scenario/...`, `experiments/...`, `pyproject.toml` keys.

## 3. Submit one smoke run

No CLI shortcut for this yet — F6.5 is "manual smoke", a one-off Python
script. Save as `scripts/ovh_smoke.py`:

```python
"""F6.5 — submit one smoke run to OVH; block until DONE; verify result."""
from pathlib import Path
from datetime import datetime, timezone

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.ovh import OvhRunner
from multi_scenario.adapters.runners.ovh_cli import OvhClient
from multi_scenario.adapters.secrets.fernet import FernetSecretsAdapter
from multi_scenario.adapters.storage.s3 import S3StorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig, OvhJobConfig, RunId, S3StorageConfig,
)

# Local artefacts land here so sync_to_local from S3 can repopulate them.
yaml_path = Path("experiments/discovery/baseline/configs/mappo_smoke.yaml")
cfg = ExperimentConfig.from_yaml(yaml_path)
ovh_cfg = OvhJobConfig.from_yaml(Path("configs/ovh.yaml"))
s3_cfg = S3StorageConfig.from_yaml(Path("configs/s3.yaml"))

run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
run_dir = Path("experiments/discovery/baseline") / run_id.folder_name(ts)
run_dir.mkdir(parents=True, exist_ok=True)

runner = OvhRunner(
    ovh_config=ovh_cfg,
    client=OvhClient(),
    secrets=FernetSecretsAdapter(),
    logger=FileLogger(run_dir / "logs" / "run.log"),
    s3_storage=S3StorageAdapter(s3_cfg),
    yaml_path_in_repo=str(yaml_path),
)
result = runner.run(cfg, run_dir=run_dir)
print(f"DONE: {result.run_id} → {run_dir}")
```

Run:

```bash
python scripts/ovh_smoke.py
```

Wall time ≈ 5–10 min. The script prints submit / poll / sync / DONE lines.

## 4. Verify the run folder

After the script returns:

```bash
ls experiments/discovery/baseline/mappo_smoke_s0__*
```

Should contain (per §3.5.2 layout):

```text
input/config.json
input/provenance.json
output/metrics.json
output/eval_episodes.json
output/report.json
output/benchmarl/<bm_run>/...
run_state.json
logs/run.log
```

Spot-check `output/metrics.json`:

```bash
jq . experiments/discovery/baseline/mappo_smoke_s0__*/output/metrics.json
```

Should have non-`null` `M1_success_rate`, `M2_avg_return`, etc. (or `null`
for not-applicable metrics).

## 5. Verify state machine

```bash
jq .state experiments/discovery/baseline/mappo_smoke_s0__*/run_state.json
```

Expect `"DONE"`. Transitions log includes `INITIALIZING → RUNNING → DONE`.

## 6. Optional — consolidate to runs.csv

```bash
multi-scenario consolidate experiments/discovery/baseline
```

`runs.csv` and `runs.json` should appear; one row each from the new run.

## 7. Cleanup

The OVH job auto-stops at completion (no idle cost). Result data costs are
~0.007 EUR/GB/month — negligible for individual runs. Periodically:

```bash
ovhai bucket object list ms-results@GRA | wc -l    # what's there
# Delete a specific run:
ovhai bucket object delete ms-results@GRA --prefix mappo_smoke_s0/ --yes
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Package 'multi-scenario' requires a different Python: 3.10.x not in '>=3.11'` | The `pytorch/pytorch:*-runtime` images ship Python 3.10. `pyproject.toml` is now `">=3.10"`; if it ever drifts back to 3.11, swap to a newer image. |
| Job exits 0 silently with empty results bucket | `cli.py` missing `if __name__ == "__main__": main()`. The guard exists in current code; if you fork / change cli.py, keep it. |
| Storage path errors with `Read-only file system` | The smoke yaml's `runtime.storage.path` points inside `/workspace/code` (mounted `:ro`). Use `mappo_ovh_smoke.yaml`-style yaml with `runtime.storage.path: /workspace/results` (which is `:rwd`). |
| `add_prefix should ends with a '/'` from `ovhai bucket object download` | `--output` arg without trailing `/`; pass `./` (or `dir/`). |
| Job state lands in `FAILED` immediately | wrong image / flavor; check `ovhai job logs <id> --tail=100` |
| `pip install` fails inside container | missing `HOME=/tmp`; verify `default_runner` in `configs/ovh.yaml` includes it |
| Polling times out | bump `timeout_sec` in `configs/ovh.yaml` |
| `boto3` `AccessDenied` on result sync | re-check `~/.aws/credentials` profile / `endpoint_url` matches bucket region |
| Job runs but sync_to_local pulls nothing | results bucket missing the per-run prefix — verify `--volume "<bucket>@<region>/<run_id>:<mount>:rwd"` in submitted args |

## What was confirmed in the live smoke (2026-05-07)

- `ovhai bucket object upload <local-dir> --remove-prefix "<repo-root>/" --add-prefix "multi_scenario/"` is the correct invocation for staging code at a sub-prefix without local-path leakage.
- `ovhai bucket object download <bucket>@<region> --prefix <key>/ --output ./ --workers N` pulls a synced run-folder back, recreating the directory tree.
- `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` works for CPU smoke — no GPU pre-installed but pip still wires `torch+cu121`. For real GPU runs, request `--flavor ai1-1-gpu --gpu 1`.
- The runner template in `configs/ovh.yaml` `default_runner: "export HOME=/tmp && pip install -e {mount_code} && cd {mount_code} && python -m multi_scenario.cli run {yaml_path_in_container}"` is what actually works inside the container.

## Where to look next

- `docs/ovh_setup.md` — one-time prereqs.
- `configs/ovh.yaml.example` — full config template.
- `implementation_plan.md` §F6.x — design notes.
- Project memory entries on OVH gotchas (trailing slash, parallel-finalize collisions).
