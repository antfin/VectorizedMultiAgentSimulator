# Submit page

5-step workflow for triggering a new run. The **canonical entry point**
for experiments — the CLI exists as a developer escape hatch; both
fronts route through `application/submission.py`.

## Step 1 — Pick

Cascading picker over `experiments/<scenario>/<folder>/configs/*.yaml`:

- Scenario dropdown (discovery / navigation / transport / flocking).
- Folder dropdown (baseline / lero / etc.).
- Config dropdown (every YAML in that folder).
- "Use this config" button loads the YAML into the form snapshot.

## Step 2 — Inspect & edit

Auto-generated widgets for every section of the YAML:

| Section | Widget set |
|---|---|
| Experiment | `id`, `seed`, `name`, `description` |
| Scenario params | Data-driven from `Scenario.default_params()` — schema + YAML override union |
| Algorithm params | Same pattern, from `Algorithm.default_params()` |
| Training | `max_iters`, `num_envs`, `device`, `frames_per_batch`, `minibatch_size`, `n_minibatch_iters` |
| Evaluation | `interval_iters`, `episodes` |
| Runtime | `runner.type`, `storage.path`, runner params |
| **LERO** (when present) | All `LeroSection` fields — `n_iterations`, `n_candidates`, `evolve_reward`, `evolve_observation`, `prompt_version`, `reward_clip`, `eval_frames_per_candidate`, `meta_prompting`, `whitelist_strict` |
| **LLM** (when present) | `LlmSection`'s concrete-default fields — `model`, `temperature`, `max_tokens`, `cost_cap_per_day_eur`, `cost_cap_per_month_eur`, `usd_to_eur_rate`, `cache_enabled` |

The LERO + LLM sections only render when the loaded YAML carries them
(non-LERO submissions see the simpler form). `None`-default fields like
`llm.api_base` / `llm.seed` are NOT rendered as widgets when absent —
they'd otherwise force-add empty values that fail Pydantic validation.

### Dirty detection

The form's `current_form` is compared to a `snapshot_form` set on YAML
load. Both get the same Pydantic-default schema fill (via
`submit_workflow._fill_schema_defaults` + `_fill_pydantic_defaults`),
so a freshly-picked YAML with no edits stays clean. Pinned by
`tests/integration/dispatch_matrix/test_dispatch_matrix.py::test_lero_local_streamlit_cfg_matches_yaml`.

## Step 3 — Save

Auto-skipped when the form is clean. On edit → "Save as new" — the
original YAML stays untouched; the new save lives under the same
configs folder.

## Step 4 — Preflight

Three LED-rolled-up cards: Configuration / System / Storage. Each
contains rows for active checks. Click "Run preflight" to fire real
probes:

| Row | Local | OVH | Notes |
|---|---|---|---|
| Config schema valid | ✅ | ✅ | Pydantic validation |
| Required deps importable | ✅ | — | torch / vmas / benchmarl |
| OVH CLI installed | — | ✅ | `ovhai --version` |
| Runner provisioning consistent with device | ✅ | ✅ | cuda available locally; GPU flavor on OVH |
| **LLM API key present for cfg.lero** | ✅ | ✅ | Only shown when YAML has `lero:`; checks `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` etc. by model prefix |
| Storage path writable | ✅ | — | tempfile under runtime.storage.path |
| Run dir does not collide | ✅ | — | timestamped-folder freshness |
| Results bucket reachable | — | ✅ | `ovhai bucket list <region>` |
| Code matches OVH bucket | — | ✅ | Local hash vs bucket's `.code_hash` |
| Submitted YAML present in bucket | — | ✅ | Catches "saved YAML, forgot upload-code" |
| Per-run prefix not occupied | — | ✅ | No parallel-job clobber |
| No active OVH job with this run_id | — | ✅ | Defensive guard |
| Cost cap not exceeded | — | ✅ | `OvhJobConfig.estimate_cost_eur(...) < cost_cap_eur` |

## Step 5 — Submit

Gated on preflight green. On click:

- **Local**: synchronous; `submit_to_local(cfg, ...)` runs the full
  training in this process; results land under `<storage.path>/<run_id>__<ts>/`.
- **OVH**: async; submits via `OvhRunner`, returns a job_id; the panel
  switches to "submitted" with status, dashboard link, and an
  **auto-poll** checkbox (default ON).

## Auto-poll + auto-pullback (OVH)

When auto-poll is on, every page rerun calls `_refresh_ovh_status()`:

- Polls `ovhai job get <job_id>` for state.
- On `RUNNING` → no-op (keep status).
- On `DONE` → `pullback_run_dir(...)` syncs S3 → local; if local
  `videos/` is missing and a checkpoint exists, fires
  `subprocess.run([... "regenerate-videos", run_dir])`.
- On `FAILED` / `KILLED` → marks status crashed, fetches last 200 log
  lines for inspection.

User can uncheck auto-poll to navigate away without polling; the OVH
job runs regardless. Manual "Refresh status" button still works.

## See also

- [Submitting experiments](../getting_started/submitting_experiments.md) — workflow walkthrough
- [Operations → Secrets and env](../operations/secrets_and_env.md) — Fernet-shipping the API key to OVH
- [Operations → Testing](../operations/testing.md) — dispatch matrix + Playwright
