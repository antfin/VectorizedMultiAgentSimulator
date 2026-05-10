# OVH end-to-end smoke checklist

Guided procedure for the **first** OVH submission after a code change that
touches the runner / dispatch / scenario stack. Costs ~one ai1-1-gpu credit
(~5 min wall time, ~€0.02). Re-run whenever you change OVH project, region,
buckets, image, or framework architecture.

> **Pre-requisite:** complete `docs/ovh_setup.md` first (one-time setup —
> `ovhai login`, S3 keys, `configs/ovh.yaml` populated).

> **Architecture (F7.7.A4):** the smoke YAML at
> `experiments/discovery/baseline/configs/_ovh_smoke.yaml` has
> `runtime.runner.type: ovh` + `training.device: cuda`.
> `multi-scenario run` reads `runner.type` and dispatches to OvhRunner;
> the OVH container internally runs LocalRunner against the same YAML.

## 0. Sanity checks

```bash
ovhai --version                                     # e.g. ovhai 3.35.0
ovhai bucket list GRA                               # confirms login still valid + lists ms-code, ms-results
ls configs/ovh.yaml                                 # populated (region, buckets, flavor, n_gpu)
multi-scenario validate experiments/discovery/baseline/configs/_ovh_smoke.yaml
                                                    # exits 0 ("OK")
```

If any check fails, return to `docs/ovh_setup.md`.

## 1. Upload current source to `bucket_code`

```bash
multi-scenario upload-code configs/s3.yaml --dry-run    # preview; no S3 writes
multi-scenario upload-code configs/s3.yaml              # actual upload
```

Verify on OVH:

```bash
ovhai bucket object list ms-code@GRA --output json | head -20
```

Expect `src/multi_scenario/...`, `experiments/...`, `pyproject.toml` keys.
Plus a `.code_hash` blob — that's the F7.7.A2 preflight check anchor.

## 2. Submit one smoke run via the new CLI dispatch

```bash
multi-scenario run experiments/discovery/baseline/configs/_ovh_smoke.yaml
```

The CLI reads `runtime.runner.type: ovh` from the YAML and dispatches to
OvhRunner. Output should look like:

```text
submitting OVH job for ovh_smoke
OVH job submitted: id=abc-12345
SUBMITTED: ovh_smoke_s0 -> job_id=abc-12345
  results: ms-results@GRA/ovh_smoke_s0
  pull back: multi-scenario sweep --follow --runner ovh experiments/.../_ovh_smoke.yaml
```

To force-run locally instead (debugging without burning OVH credit):

```bash
multi-scenario run --runner local experiments/discovery/baseline/configs/_ovh_smoke.yaml
```

On a CUDA-less Mac this fails fast (F7.7.A4) with a clear message:
> `RuntimeError: training.device=cuda but torch.cuda.is_available()=False...`

## 3. Watch the job

```bash
ovhai job get <job_id>                # current state
ovhai job logs <job_id> --tail 50     # live stdout
```

States: `QUEUED` → `PENDING` → `RUNNING` → `DONE` (or `FAILED` / `KILLED`).

For the smoke YAML, expect ~2 min from QUEUED to DONE.

## 4. Pull results back (two equivalent paths)

Both paths invoke the same `application/ovh_pullback.py::pullback_run_dir`
(no AWS S3 keys; uses `ovhai bucket object` CLI), so the local layout is
identical. Pick whichever fits your flow.

### 4a. CLI: `sweep --follow` auto-pullback

```bash
multi-scenario sweep --follow --runner ovh experiments/discovery/baseline/configs/_ovh_smoke.yaml
```

Submits + polls + on each DONE job auto-pulls
`ms-results@GRA/ovh_smoke_s0__<ts>/` →
`experiments/discovery/baseline/ovh_smoke_s0__<ts>/`. Pullback failures
print to stderr but don't abort the sweep.

### 4b. Streamlit: 🔄 Refresh button (Stage 3, 2026-05-10)

After Submit, the panel shows `📤 SUBMITTED` with the job_id, S3 prefix,
OVH dashboard link, and a 🔄 Refresh button. Click Refresh whenever the
OVH dashboard shows the job near DONE:

- **Job still RUNNING** → panel updates the live state (`— state RUNNING`),
  status stays `submitted`. Click Refresh again later.
- **Job DONE** → pullback fires, status flips to `✅ DONE` with
  `Pullback: N files downloaded, M skipped`. Best-effort regenerate-videos
  also runs; failures show as a `regen_warning` panel (results are already
  on disk — open Run Detail to retry).
- **Job FAILED/KILLED** → status flips to `❌ CRASHED` with the OVH logs
  tail in an expander.

### Spot-check the layout (either path)

```bash
ls experiments/discovery/baseline/ovh_smoke_s0__*
```

Should contain (per §3.5.2) — note **flat** layout, no nested
`ovh_smoke_s0__*/ovh_smoke_s0__*` (Stage-1 lesson, fixed via the
`MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR` env-var short-circuit):

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

```bash
jq . experiments/discovery/baseline/ovh_smoke_s0__*/output/metrics.json
jq .state experiments/discovery/baseline/ovh_smoke_s0__*/run_state.json
                                                    # → "DONE"
```

`metrics.json` should have non-`null` `M1_success_rate`, `M2_avg_return`,
`M3_steps`, `M4_collisions` for discovery.

## 5. Browse in Streamlit

```bash
streamlit run src/multi_scenario/frontend/streamlit_app.py
```

The Run Detail page picks up `ovh_smoke_s0__*` automatically:

- Header with state badge + duration + timestamp
- M1–M9 metrics tiles
- Config viewer (collapsible JSON)
- BenchMARL training curves (CSV picker → live line plots)
- **Videos** section: when the OVH container was headless and Pyglet
  failed, the "No videos found" alert is paired with a **🎬 Regenerate
  videos** button (Stage 3, 2026-05-10). Click it to re-render from the
  trained policy. Local runs normally produce 2 videos (begin + end)
  during training, so this button rarely appears for them.

The Submit page (with `_ovh_smoke.yaml` picked) shows preflight cascade:

- 🟢 Configuration (Config schema valid + OVH config valid)
- 🟢 System (OVH CLI installed + Runner provisioning consistent + …)
- 🟢 Storage (Results bucket reachable + Code matches OVH bucket +
  Submitted YAML present + **Per-run prefix not occupied** — Stage 1's
  hard-block on accidental re-runs)

If any LED is red, the failed sub-check shows the root cause inline. The
prefix-collision row even tells you the exact `ovhai bucket object delete`
command to run for cleanup.

## 6. Cleanup (optional)

Smoke results are tiny (~few KB). Periodically:

```bash
ovhai bucket object list ms-results@GRA --output json | wc -l   # count what's there
ovhai bucket object delete ms-results@GRA --prefix ovh_smoke_s0/ --yes
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: training.device=cuda but torch.cuda.is_available()=False` | F7.7.A4 fail-fast: you're trying to run a `device: cuda` YAML on a Mac. Use `--runner ovh` (default for this YAML) or set `device: cpu` for local debug. |
| `flavor=ai1-1-cpu but device=cuda` from preflight | OvhJobConfig has CPU-only flavor; switch to `ai1-1-gpu` in `configs/ovh.yaml`. |
| `⚠ device=cpu on GPU flavor ai1-1-gpu` | Warning, not an error: you're paying for GPU but training on CPU. Either switch device or pick a CPU flavor. |
| `Package 'multi-scenario' requires ... 3.10.x not in '>=3.11'` | OVH `pytorch/pytorch:*-runtime` images ship Python 3.10. `pyproject.toml` is `>=3.10`; if it drifts back to 3.11, use a newer image. |
| Job exits 0 silently with empty results bucket | `cli.py` missing `if __name__ == "__main__": main()`. Guard exists in current code; if you fork, keep it. |
| `Read-only file system` errors in container | smoke YAML's `runtime.storage.path` points to `/workspace/code` (mounted `:ro`). Should be `/workspace/results` (mounted `:rwd`). The shipped `_ovh_smoke.yaml` is correct; check if you edited it. |
| `add_prefix should ends with a '/'` from `ovhai bucket object download` | `--output` arg without trailing `/`; pass `dir/`. |
| Job state lands in `FAILED` immediately | Wrong image / flavor; check `ovhai job logs <id> --tail=100`. |
| `pip install` fails inside container | Missing `HOME=/tmp`; verify `default_runner` in `configs/ovh.yaml` includes `export HOME=/tmp &&`. |
| Polling times out | Bump `timeout_sec` in `configs/ovh.yaml`. |
| Job runs but pullback finds nothing | Results bucket missing the per-run prefix — verify `--volume "<bucket>@<region>/<run_id>:<mount>:rwd"` in the submitted args (visible in `ovhai job get <id> --output json`). |
| Pullback lands at `<run_dir>/<container_run_dir>/input/...` (nested) | OvhRunner missed injecting `MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR=1` — without it the container's LocalRunner adds a SECOND timestamped subdir inside the host's per-run S3 prefix. Smoke 3, 2026-05-10. |
| Streamlit Refresh: `Read-only file system: '/workspace'` | Submit page tried to pull back to the OVH-container path. Fixed by `_run_ovh_submission` computing a `pullback_dir` from the YAML's grandparent (e.g. `experiments/discovery/baseline/<run_dir.name>/`). Smoke 4, 2026-05-10. |
| Streamlit Refresh: `FileNotFoundError: 'multi-scenario'` | The console script isn't on Streamlit's PATH. Fixed: regenerate-videos is now invoked via `sys.executable -m multi_scenario.cli` (in submit.py + run_detail.py). Smoke 4, 2026-05-10. |
| Streamlit "Code matches OVH bucket" FAIL after edits | Local code hash drifted from the bucket's `.code_hash`. Run `multi-scenario upload-code` and re-click "Run preflight". Stage 1 hard-block in action. |

## Where to look next

- `docs/ovh_setup.md` — one-time prereqs.
- `configs/ovh.yaml` — OVH deployment config (region/buckets/flavor/n_gpu).
- `implementation_plan.md` §F7.7.A4 — runner-device architecture rationale.
- `docs/_drafts/F8_F11_plan_draft.md` — full F8/F9/F10/F11 plan.
- Project memory (`~/.claude/.../memory/project_coopvmas_decisions.md`) — locked decisions.

## What's been confirmed live (history)

- **2026-05-07** — first OVH end-to-end smoke (`mappo_ovh_smoke.yaml`):
  - `ovhai bucket object upload <local-dir> --remove-prefix "<repo-root>/" --add-prefix "multi_scenario/"` is the correct invocation for staging code.
  - `ovhai bucket object download <bucket>@<region> --prefix <key>/ --output ./` pulls a synced run-folder back.
  - `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` works for CPU smoke.
- **2026-05-09** — first F7.7.A4 OVH+CUDA smoke (`_ovh_smoke.yaml`). Job `49c01e83…` ran in ~1m30s (most was pip install). Found and fixed:
  - **Recursive-submit bomb:** YAML's `runner.type: ovh` made the in-container CLI try to re-submit. Fixed by baking `--runner local` into `OvhJobConfig.default_runner`.
  - **`_parse_job_id` bug:** ovhai's `job run` output starts with "Created\n<id>"; the parser took "Created" as the id. Job runs fine (the real id is recorded by OvhRunner) but local echoes show `job_id=Created`. **TODO** before ER1.
  - **`upload-code` needs AWS creds:** bypass today is `ovhai bucket object upload` directly. **TODO** refactor to use OvhClient (mirror F7.7.A2).
  - Layout, metrics.json, run_state.json all landed cleanly. End-to-end pullback worked.
- **2026-05-10** — Stages 1+2+3 pre-ER1 smokes (4× passes after fixes). Findings & fixes:
  - **Stage 1 (data safety):** second-resolution timestamps in run-dir name + per-run S3 prefix `<bucket>@<region>/<run_dir.name>` so re-runs of the same exp_id+seed can't clobber. Hard-block in preflight when `<run_id>__*` already exists in bucket — error message includes the exact `ovhai bucket object delete --prefix <run_id>__ --yes` command.
  - **Stage 2 (pullback):** new `application/ovh_pullback.py::pullback_run_dir` uses `ovhai` (no AWS keys), idempotent (size-match skip), no-op-safe on empty prefix. Wired into `cli/sweep.py --follow` so each DONE job auto-pulls; failures don't abort the sweep.
  - **Stage 3 (FE UX):** Submit-page 🔄 Refresh button while job in flight; on DONE auto-pulls + best-effort regenerates videos. Run Detail "🎬 Regenerate videos" button replaces the previous text-only "no videos" hint.
  - **Smoke-3 nested-folder bug:** post-Stage-1 the host's S3 prefix has a timestamp AND the container's LocalRunner added another timestamped subdir → `<host_run_dir>/<container_run_dir>/input/...`. Fixed by `MULTI_SCENARIO_USE_STORAGE_ROOT_AS_RUN_DIR=1` env-var short-circuit in `build_run_dir`; OvhRunner injects it.
  - **Smoke-4 container-path bug:** Streamlit Submit's stored `run_dir` was `/workspace/results/<run_dir.name>/` (container path); pullback tried to write there on the macOS host → ReadOnly. Fixed by computing a separate `pullback_dir` from the YAML grandparent (`experiments/discovery/baseline/<run_dir.name>/`).
  - **Smoke-4 subprocess bug:** Streamlit's regenerate-videos invoked the `multi-scenario` console script, which isn't on the streamlit server's PATH outside an active venv. Fixed by `sys.executable -m multi_scenario.cli` in both submit.py and run_detail.py; OSError caught so a regen failure doesn't strand the panel on RUNNING.
  - All 4 smokes (CLI local, Streamlit local, CLI OVH, Streamlit OVH) pass cleanly. Final OVH cost ≈ €0.06 across 3 jobs (`d0237edc`, `2d38ef17`, `01e28129`). 511 + 1 added test = 512 tests pass; lint clean.
