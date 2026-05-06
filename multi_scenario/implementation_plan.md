# Multi-Scenario Cooperative MARL — Implementation Plan

> **Status:** draft v2 — to review before any code is written.
> **Companion docs:** [`plan.md`](plan.md) (scenario rationale & descriptions).
> **Folder name `multi_scenario/` is a placeholder** — will be renamed when extracted to its own git repo.
> **Changes from v1:** added run-level conventions (§3.5), expanded Phase 1 with cross-cutting infra (logging, run-id, run-state, provenance, determinism), expanded Phase 2/5/6/7, added Phase 10 (polish/CI/extraction), added explicit "out of scope" list (§6.5) and additional gotchas list (§7.5).

---

## 1. Goals & guiding principles

### 1.1 Goals

1. A small, clean framework that can train and evaluate any of the 4 chosen VMAS cooperative scenarios (discovery, navigation, flocking, transport) with the same algorithms (MAPPO, IPPO, MADDPG, IDDPG, ISAC, MASAC).
2. Same code path runs **locally** or on **OVH**.
3. Results are emitted as **CSV** (per-run summary, per-iter training, per-eval — three distinct CSVs ported from rendezvous_comm; per-step long-format kept on the table for F5.4 decision).
4. A **Streamlit FE** for browsing/comparing results, then later submitting jobs.
5. Eventually portable to a **standalone repo** with its own dependencies.

### 1.2 Principles

- **Hexagonal architecture (ports & adapters)** — domain logic depends only on interfaces; everything VMAS/BenchMARL/OVH/Streamlit-specific lives in adapters.
- **Dependency injection** via constructor parameters + small factory functions (no DI framework — Python's `Protocol` types are enough).
- **TDD** with very small steps; every feature starts with a failing test.
- **Port from `rendezvous_comm/src` piece by piece**, not in one shot — each ported piece reviewed before next.
- **Validation gate after every feature** — checklist + demo command, user signs off before next feature starts.
- **Don't design LERO yet.** Phase 9 is a placeholder; we'll plan it when we're done with baselines. (LERO complexity flagged in §6 as v5–v9 + v4 meta-prompt + 23 prompt variants — design when we get there.)
- **No premature abstraction.** If only one adapter exists for a port, that's fine — the port still buys testability.

---

## 2. Architecture overview

```text
┌──────────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                                │
│   CLI (typer)        Streamlit FE        pytest                  │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                  APPLICATION LAYER (use-cases)                   │
│   ExperimentService — load config, build deps, run, persist     │
│   Factories         — name → adapter (DI registry)               │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DOMAIN LAYER (ports)                          │
│   Scenario | Algorithm | MetricSet | Storage | Runner | Logger   │
│   Models: ExperimentConfig, ExperimentResult, RunId, RunState,   │
│           Provenance, MetricRecord                               │
│   (no imports from VMAS / BenchMARL / Streamlit / boto3 / torch) │
└──────────────────────────────────────────────────────────────────┘
                          ▲
                          │ implemented by
┌──────────────────────────────────────────────────────────────────┐
│                    ADAPTERS LAYER                                │
│   scenarios/   — vmas_discovery, vmas_navigation, ...            │
│   algorithms/  — benchmarl_mappo, benchmarl_ippo, ...            │
│   metrics/     — common (M2/M3/M4...), discovery_m1, ...         │
│   storage/     — local_yaml_csv, s3 (later)                      │
│   runners/     — local, ovh (later)                              │
│   logging/     — file_logger, console_logger                     │
│   secrets/     — fernet_secrets (for LLM keys, OVH only)         │
└──────────────────────────────────────────────────────────────────┘
```

**Key design rule:** `domain/` files must not `import vmas`, `import benchmarl`, `import streamlit`, `import boto3`, or `import torch`. A unit test enforces this (F1.8).

---

## 3. Folder layout (final target)

```text
multi_scenario/
├── pyproject.toml                # uv-managed
├── README.md
├── plan.md                       # scenarios & rationale  (exists)
├── implementation_plan.md        # this doc
├── src/
│   └── multi_scenario/
│       ├── __init__.py
│       ├── domain/
│       │   ├── ports.py          # Protocols
│       │   └── models.py         # ExperimentConfig, RunId, RunState, Provenance, ...
│       ├── application/
│       │   ├── experiment_service.py
│       │   └── factories.py
│       ├── adapters/
│       │   ├── scenarios/
│       │   ├── algorithms/
│       │   ├── metrics/
│       │   ├── storage/
│       │   ├── runners/
│       │   ├── logging/
│       │   └── secrets/
│       ├── cli.py
│       └── frontend/
│           └── pages/
├── experiments/
│   ├── discovery/
│   │   ├── baseline/{configs,results}/
│   │   └── lero/{configs,results}/
│   ├── navigation/{baseline,lero}/{configs,results}/
│   ├── flocking/{baseline,lero}/{configs,results}/
│   └── transport/{baseline,lero}/{configs,results}/
├── docs/                         # architecture, scenarios, run_layout, decision notes (filled across phases)
└── tests/
    ├── unit/
    ├── integration/
    └── reproducibility/
```

---

## 3.5 Run-level conventions (ported from rendezvous_comm)

These conventions exist in `rendezvous_comm/` and we want to keep them — not silently re-invent.

### 3.5.1 Run ID (parametric, deterministic)

Format: `<exp_id>_s<seed>`. The folder name appends a timestamp:
`<run_id>__<timestamp>` = `<exp_id>_s<seed>__<YYYYMMDD_HHMM>`.

Example: run_id `disc_baseline_smoke_mappo_s0`,
folder `disc_baseline_smoke_mappo_s0__20260506_1423`.

Rules:

- Parametric, no timestamp in `run_id` itself → reproducible run identity. Timestamp lives only in the folder name and disambiguates re-runs of the same config.
- exp_id is the discriminator across algo / scenario / variant combos. To compare mappo vs ippo on the same scenario, use distinct exp_ids (e.g. `disc_baseline_smoke_mappo` and `disc_baseline_smoke_ippo`); seeds vary independently.
- Tested: identical config produces identical run_id.

### 3.5.2 Run folder layout

```text
experiments/<scenario>/<exp_type>/<run_id>__<timestamp>/
├── input/
│   ├── config.json               # resolved, fully-merged config (machine-read; YAML source stays in configs/)
│   └── provenance.json           # hashes, git_sha, hashed_source_files, library_versions, timestamps
├── logs/
│   └── run.log
├── output/
│   ├── metrics.json              # M1..M9 + config_snapshot + run metadata
│   ├── eval_episodes.json        # per-episode raw eval data (re-aggregatable)
│   ├── report.json               # manifest: status, summary, file links (relative paths)
│   ├── videos/                   # opt-in (default ON for non-smoke; record_video flag)
│   │   ├── before_training.mp4   # random-init policy, 1 eval episode
│   │   └── after_training.mp4    # final policy, 1 eval episode
│   └── benchmarl/                # untouched BenchMARL output (policy in checkpoints/, scalars/*.csv)
│       └── <bm_run>/...
└── run_state.json                # lifecycle (INITIALIZING|RUNNING|DONE|CRASHED|RESUMED)
```

Notes:

- No `results/` parent folder — run folders are direct children of `<exp_type>/`, sibling to `configs/`.
- All run-level files we own are JSON. BenchMARL's native CSVs (`scalars/`) remain in `benchmarl/` untouched.
- No standalone `policy.pt` at run-folder root: the policy lives in `output/benchmarl/<bm_run>/checkpoints/`. `report.json` records the exact path.
- `report.json` is a manifest with relative paths to `config`, `provenance`, `log`, `metrics`, `eval_episodes`, `policy`, `videos.{before,after}_training`, `benchmarl_dir`, `benchmarl_scalars`, plus a headline summary (status, duration, M1/M2 highlights). Streamlit run-detail page reads this — no globbing.

### 3.5.3 Cross-run aggregations

Cross-run files live at `experiments/<scenario>/<exp_type>/`, sibling to run folders. **Single canonical pair**, no timestamps — the per-run JSONs are the source of truth, the cross-run files are a derived view, fully reconstructable. Re-running consolidate replaces them in place (atomic write-rename). Each consolidate also copies the previous version to `runs.previous.csv` / `runs.previous.json` for one-step rollback.

- `runs.csv` — **long-format, one CSV** with a `record_type` column:
  - `record_type == "final"` → one row per run: final M1–M9 + flattened `config_snapshot` + run metadata. Equivalent of rendezvous_comm's `sweep_results.csv`.
  - `record_type == "eval"` → one row per (run, eval step): subset of M1–M9 sampled mid-training (M7/M8/M9 may be `N/A` when not yet computable).
  - Algorithm-agnostic columns. Stable schema across all scenarios. JSON nulls render as `N/A` via pandas `na_rep`.
  - **Out of scope at cross-run level**: per-iter algorithm-internal scalars (loss / entropy / clip_fraction / grad norms). Those live in each run's `output/benchmarl/.../scalars/*.csv` and Streamlit reads them at view time when showing a single run's training internals.
- `runs.json` — slim cross-run manifest:
  - `scope`: scenario, exp_type, n_runs, exp_ids, seeds, algorithms.
  - `csv`: link to `runs.csv`.
  - `rankings`: per-metric leaderboards as `[{run_id, value, report}]` arrays.
  - `runs`: flat list of `{run_id, report}` linking to each per-run `output/report.json`.
  - **No file duplication**: every per-run path (config, metrics, policy, videos, benchmarl_scalars) lives only in the per-run report; the cross-run manifest dereferences via `report` links.
- `runs.previous.csv`, `runs.previous.json` — one-step backups, overwritten on each consolidate.

### 3.5.4 Run state lifecycle

`run_state.json` enum: `INITIALIZING → RUNNING → DONE` (happy path), `→ CRASHED` (on exception), `→ RESUMED` (if resumed). Records timestamped transitions. Drives Streamlit status icons and resume detection.

### 3.5.5 Provenance

`input/provenance.json` per run, with:

- `config_hash` (sha256 of resolved config dict)
- `code_hash` (sha256 of the explicit `hashed_source_files` list — see §7.5/#8 for the staleness-tracking caveat)
- `hashed_source_files` (the curated list of files contributing to `code_hash`)
- `git_sha`, `git_dirty`
- `created_at`, `finished_at`
- `library_versions`: `python`, `torch`, `vmas`, `benchmarl`, `multi_scenario`

Used to flag stale results in Streamlit (current code hash differs → result is from older code).

---

## 4. Validation-gate template

Every feature ends with this exact checklist before we move to the next:

```text
[ ] All new tests pass            (pytest tests/<scope> -x)
[ ] Existing tests still pass     (pytest tests -x)
[ ] Lint clean                    (ruff check . && ruff format --check .)
[ ] Demo command produces stated output
[ ] User code-reviews the diff
[ ] User explicitly says "next"
```

Each feature lists its **demo command** + **expected output**.

---

## 5. Phased feature roadmap

> XS ≤ 30 min · S ≤ 2h · M ≤ ½ day. If a feature smells M+, we split it. **Backend first; Streamlit FE only after the BE is usable end-to-end.**

### Phase 0 — Scaffolding

#### F0.1 — Empty package + `pyproject.toml` (uv) — XS

- Goal: `import multi_scenario` works after `uv pip install -e .`.
- TDD: `tests/unit/test_smoke.py::test_import` asserts the import succeeds.
- Files: `pyproject.toml`, `src/multi_scenario/__init__.py` (with `__version__`), empty subpackages.
- Deps declared (not yet used): `torch`, `vmas`, `benchmarl`, `pyyaml`, `pydantic`, `pandas`, `typer`, `pytest`, `ruff`. (`streamlit`, `boto3`, `cryptography`, `litellm` deferred until needed.)
- **Demo:** `uv pip install -e . && python -c "import multi_scenario; print(multi_scenario.__version__)"` → prints `0.0.1`.

#### F0.2 — Dev tooling — XS

- Ruff config (line-length 100, target py3.11), pre-commit hooks, pytest config in `pyproject.toml`, `markdownlint` config.
- **Demo:** `pre-commit run --all-files` → all green; `pytest -q` → 1 passed.

#### F0.3 — Empty `experiments/` tree — XS

- Just create the 4×2 directory tree with `.gitkeep` files. No content yet.
- **Demo:** `find experiments -type d | sort` shows the expected 16 directories.

#### F0.4 — `tests/` skeleton + conftest — XS

- `tests/{unit,integration,reproducibility}/__init__.py`, shared fixtures in `conftest.py` (tmp results dir, fake config builder).
- **Demo:** `pytest --collect-only` lists all test directories.

#### F0.5 — Empty `docs/` tree — XS

- Create `docs/` with a `.gitkeep`. Placeholder for files added across later phases: `docs/example_config.yaml` (F1.1), `docs/csv_format_decision.md` (F5.5), `docs/architecture.md` / `docs/scenarios.md` / `docs/run_layout.md` (F10.3).
- **Demo:** `ls docs/` shows the directory exists.

#### F0.6 — Repo-prep files (`.gitignore` + stub `README.md`) — XS

- Drop the files a standalone repo will need so F10.4 extraction is mostly mechanical:
  - `.gitignore` — Python caches (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`), build artifacts (`*.egg-info/`, `dist/`, `build/`), venv (`.venv/`, `venv/`), IDE (`.vscode/`, `.idea/`), OS (`.DS_Store`), and experiment outputs (`experiments/**/results/*` excluding `.gitkeep`).
  - `README.md` — short stub (title, one-paragraph description, pointers to `plan.md` and `implementation_plan.md`). Already referenced in the §3 folder layout.
- Skipped on purpose: `.gitattributes` (no cross-platform line-ending pain in sight), `.editorconfig` (ruff covers formatting), `LICENSE` (deliberate choice — see F10.4 cleanup).
- **Demo:** `cat .gitignore` and `cat README.md` look reasonable; `pre-commit run --all-files` stays green.

---

### Phase 1 — Domain core (zero deps on VMAS / BenchMARL / torch)

#### F1.1 — `ExperimentConfig` model — S

- Pydantic model with sections: `experiment`, `scenario`, `algorithm`, `training`, `runner`, `metrics`, `storage`. Strict validation (`extra="forbid"`).
- TDD: round-trip `dict ↔ ExperimentConfig`; reject missing/unknown fields.
- **Demo:** `python -c "from multi_scenario.domain.models import ExperimentConfig; ExperimentConfig.from_yaml('docs/example_config.yaml')"` parses cleanly.

#### F1.2 — `ExperimentResult` & `MetricRecord` models — XS

- Dataclasses: per-run summary record + list of per-metric values; CSV-friendly flattening.

#### F1.3 — `RunId` + parametric naming — S

- Pure value object — frozen, hashable. Constructed from `(exp_id, seed)` per the §3.5.1 simplification (algo / scenario disambiguation is encoded in `exp_id` by the user; not synthesised). Stable string repr `<exp_id>_s<seed>`. Folder name helper `folder_name(timestamp) -> "<run_id>__<timestamp>"`.
- Reverse parsers: `RunId.from_string("..._sN")` and `RunId.from_folder_name("..._sN__YYYYMMDD_HHMM")` for reading existing run folders (Streamlit, consolidator).
- Validation: non-empty exp_id, no `__` in exp_id (collides with the timestamp separator), alphanumerics + `_-` only, non-negative seed.
- Tests: identical inputs produce identical id and identical hash; different seed → different id; folder-name round-trip; greedy-regex parser handles exp_ids that themselves end with `_sN`-shaped substrings; invalid inputs raise.

#### F1.4 — `RunState` enum + persisted shape — XS

- States: `INITIALIZING | RUNNING | DONE | CRASHED | RESUMED`. JSON-serialisable with timestamps.
- Tests: invalid transitions rejected (e.g., DONE → RUNNING).

#### F1.5 — `Provenance` model — S

- Fields per §3.5.5. Includes `config_hash` (sha256 of dict), `code_hash` (sha256 of given file list), `git_sha`, `git_dirty` flag.
- TDD: same config dict → same hash; modified file → different code_hash.

#### F1.6 — `Scenario` port — XS

- Runtime-checkable `Protocol` exposing: `name`, `make_env(cfg, num_envs, seed)`, `default_params()`, plus the four DI primitives that feed the always-on metric bundle (§3.5.3) — `has_comm()` (M5 applicability), `success_predicate(rollout)` (M1), `coverage_progress(rollout)` returning None when not applicable (M6), `utilization_predicate(state)` (M8).
- Domain stays torch/vmas-agnostic: tensor-shaped values are typed `Any` on the Protocol; concrete adapters in `adapters/scenarios/` know the real types.
- Single-file `domain/ports.py` for now. Refactor to a `ports/` package mirroring `models/` after the third Protocol lands (rule of three — F1.7 + F1.8 trigger the split).
- TDD: a fake scenario covering every member passes `isinstance(impl, Scenario)`; an incomplete fake fails the check.

#### F1.7 — `Algorithm` port — XS

- `Protocol`: `name`, `train(env, cfg) -> TrainArtifact`, `evaluate(artifact, env, cfg) -> Rollout`.

#### F1.8 — `MetricsBundle` port + `ports/` package refactor — S

- Renamed from "MetricSet" — we retired metric *sets* during the §3.5.3 redesign in favour of a single always-on bundle producing M1–M9 with `null` for non-applicable metrics.
- Runtime-checkable `Protocol` with one method: `compute(rollout, scenario) -> dict[str, float | None]`. Receives the rollout from `Algorithm.evaluate` plus the `Scenario` adapter (used for the four DI primitives feeding scenario-specific calculations). Returns the M1–M9 dict directly.
- **Rule-of-three refactor:** with three Protocols (Scenario, Algorithm, MetricsBundle) the single-file `ports.py` becomes a package: `ports/{__init__.py, scenario.py, algorithm.py, metrics.py}` mirroring `models/`. The `__init__.py` re-exports public names; existing imports unchanged.
- TDD: a fake bundle implementing `compute` passes `isinstance(_, MetricsBundle)`; one without fails.

#### F1.9 — `Storage` port — XS

- Runtime-checkable `Protocol` with 4 save + 4 load methods covering the per-run JSON files: `save_config` / `load_config`, `save_provenance` / `load_provenance`, `save_result` / `load_result`, `save_run_state` / `load_run_state`. Each takes a `run_dir: Path`.
- **Run-level only.** Cross-run aggregations (`runs.csv` / `runs.json`) live with the consolidator at F5.2/F5.3 — different concern, different lifecycle.
- **Optional artefacts** (`eval_episodes`, `report`, `videos`, `log`) are added on the concrete adapter when each writer feature lands (F2.5 / F2.10 / F2.11 / F2.7); they're not in the Protocol surface to keep it minimal.
- Adapters: `LocalStorageAdapter` (fs) at F2.5, `S3StorageAdapter` at F6.3. When S3 lands we generalise `Path` → `str | Path` if needed.
- TDD: a fake storage implementing all 8 methods passes `isinstance(_, Storage)`; an incomplete fake (missing one) fails.

#### F1.10 — `Logger` port + `RngState` model — S (domain part only)

- **Domain (this PR):**
  - `Logger` Protocol with `info`, `debug`, `warning`, `error`. Lives in `domain/ports/logger.py`.
  - `RngState` model with `seed: int` and `captures: dict[str, str]` (opaque encoded states keyed by RNG name — `"python.random"`, `"numpy"`, `"torch.cpu"`, `"torch.cuda"`). Encoding format is the adapter's choice; the model stays format-agnostic. Lives in `domain/models/rng_state.py`.
  - TDD: protocol fakes (full + incomplete) for Logger; round-trip for RngState.
- **Adapters (Phase 2+):**
  - `FileLogger` (writes `logs/run.log`) and `ConsoleLogger` — adapter implementations of `Logger`. Land at F2.7.
  - `seed_all(seed) -> RngState`, `save_rng_state() -> RngState`, `load_rng_state(state)` — pure functions wrapping `torch` / `numpy` / `random`. Land in `adapters/runtime/determinism.py` when ExperimentService first wires up determinism.
  - TDD for adapter functions: same seed → same first 100 numbers from each RNG.

#### F1.11 — `Runner` port + `ExperimentService` skeleton — S

- `Runner.run(cfg) -> ExperimentResult`. `ExperimentService` orchestrates: build deps → seed → write provenance → write run_state(RUNNING) → train → evaluate → compute metrics → save → write run_state(DONE).
- TDD: full use-case test with fakes only — no VMAS, no BenchMARL. <1s.
- **Demo:** `pytest tests/unit/application/test_experiment_service.py -v` → green, full pipeline executed against fakes.

#### F1.12 — Domain isolation enforcement — XS

- TDD: a test scans `src/multi_scenario/domain/**/*.py` and asserts none of them import `vmas`, `benchmarl`, `streamlit`, `boto3`, `torch`. Architecture lint.

---

### Phase 2 — First real adapter slice: discovery + MAPPO

> The smallest end-to-end vertical slice. We resist generalising before this slice works.

#### F2.1 — `VmasDiscoveryAdapter` — S

- Implements `Scenario`. Wraps `vmas.make_env(scenario="discovery", ...)`.
- **Port from rendezvous_comm:** discovery params from `src/config.py`; the `targets_respawn=False` invariant.
- TDD: env builds; observation/action spaces match for `n_agents=2, n_targets=2`.

#### F2.2 — Common metric adapters (M2, M3, M4) — S

- Pure functions: `m2_avg_return`, `m3_steps`, `m4_collisions`. Bundled into `CommonMetricSet`.
- **Port from rendezvous_comm:** logic from `src/metrics.py`.

#### F2.3 — `DiscoveryMetricSet` (M1, M6) — S

- M1 from `targets_covered` cumsum (NOT `terminated` — that bug is documented).

#### F2.4 — `BenchmarlBaseAdapter` + `MappoAdapter` — M

- Shared scaffolding (`benchmarl_base.py`) + MAPPO-specific subclass.
- Wraps a BenchMARL `Experiment`; exposes `train`/`evaluate` returning a serialisable artifact.
- **Port from rendezvous_comm:** BenchMARL wiring from `src/runner.py` (the working bits, not the LERO callbacks).
- TDD: 1-env, 2-iteration smoke training (slow, marked `@pytest.mark.slow`).

#### F2.5 — `LocalStorageAdapter` — S

- All run-level files we own are JSON: `input/config.json`, `output/metrics.json`, `output/eval_episodes.json`, `run_state.json`. BenchMARL's native CSVs in `benchmarl/` are preserved untouched. Per-run-summary CSV append is at the cross-run level (§3.5.3), not per-run. Writes to the §3.5.2 layout.
- TDD: round-trip; concurrent appends to the cross-run CSV serialise correctly.

#### F2.6 — `LocalRunner` + factories — S

- In-process runner. `factories.py` registers names → adapters.
- TDD: registry round-trip; unknown name raises clean error.

#### F2.7 — `FileLogger` + provenance + run_state writers — S

- `FileLogger` writes to `logs/run.log`. `ProvenanceWriter` computes hashes & writes `input/provenance.json` (JSON, with `hashed_source_files` and `library_versions` per §3.5.5). `RunStateWriter` updates `run_state.json` at lifecycle transitions.
- Wire all three into `ExperimentService`.
- **Port from rendezvous_comm:** `logging_setup.py`, `provenance.py`.

#### F2.8 — CLI `multi-scenario run <yaml>` — S

- Typer command. Provide `experiments/discovery/baseline/configs/mappo_smoke.yaml` (1 env, 2 iters).
- **Demo:** `multi-scenario run experiments/discovery/baseline/configs/mappo_smoke.yaml` → returns 0; writes the §3.5.2 run-folder layout (`<run_id>__<ts>/{input/config.json, input/provenance.json, output/metrics.json, output/eval_episodes.json, output/report.json, output/benchmarl/..., logs/run.log, run_state.json}`) and a row in `sweep_results_<ts>.csv`.

#### F2.9 — Smoke integration test — XS

- `tests/integration/smoke/test_discovery_mappo.py` runs the same YAML through the runner and asserts: CSV row exists, run_state=DONE, provenance fields populated.
- **Validation gate (Phase 2 milestone):** discovery + MAPPO produces a real CSV row locally with full lifecycle artifacts.

#### F2.10 — `report.json` writer — XS

- At run end (after metrics aggregation, before run_state→DONE), emit `output/report.json` per §3.5.2: a manifest with status, started/finished timestamps, duration, headline summary (a few key metrics), and relative-path links to every relevant artefact (`config`, `provenance`, `log`, `metrics`, `eval_episodes`, `policy` inside `benchmarl/`, `videos.before_training`, `videos.after_training`, `benchmarl_dir`, `benchmarl_scalars`).
- The exact `<bm_run>` directory name (BenchMARL-assigned) is captured in the manifest so consumers don't glob.
- TDD: given a fully-populated run folder, the writer produces a manifest whose every linked path resolves to an existing file (or is `null` if opt-in artefact wasn't generated).

#### F2.11 — Before/after training videos — S

- Generate `output/videos/before_training.mp4` (random-init policy, 1 eval episode, rendered) and `output/videos/after_training.mp4` (final policy, 1 eval episode, rendered) when `runtime.runner.params.record_video: true`. Default ON for non-smoke runs (any `experiment.id` that doesn't end in `_smoke`); default OFF for smoke (rendering is slow).
- Reuses the scenario adapter's render hook (`scenario.make_env(..., render=True)` + VMAS `Environment.render(mode="rgb_array")` per step).
- TDD: smoke test with `record_video: true` produces both files, both > 0 bytes; smoke test with the flag default-off produces neither.

---

### Phase 3 — Add remaining baseline algorithms (still discovery only)

> One algorithm per feature. Same TDD pattern: smoke test that 2-iter run completes + writes CSV.

- **F3.1 — IPPO adapter** (S)
- **F3.2 — MADDPG adapter** (S)
- **F3.3 — IDDPG adapter** (S)
- **F3.4 — ISAC adapter** (S)
- **F3.5 — MASAC adapter** (S)
- **F3.6 — Algorithm registry refactor** (XS) — extract common BenchMARL boilerplate into `benchmarl_base.py`. Refactor only after the 6th adapter, not before (rule of three).

**Phase 3 milestone demo:** loop over 6 yaml configs, all produce CSV rows in the same `results/`.

---

### Phase 4 — Extend to other scenarios

> One scenario per feature; each adds scenario adapter + scenario metric set + at least one MAPPO smoke run.

- **F4.1 — Navigation adapter + metrics** (S). Proposed success metric: fraction of agents within `goal_radius` at episode end. **User confirms before merge.**
- **F4.2 — Flocking adapter + metrics** (S). Proposed: fraction of timesteps where inter-agent distance ∈ [desired ± tol] AND |v − v*| < tol.
- **F4.3 — Transport adapter + metrics** (S). Proposed: package-at-goal flag + final distance.
- **F4.4 — Scenario registry refactor** (XS).

**Phase 4 milestone demo:** for each scenario, `multi-scenario run <scenario>_mappo_smoke.yaml` succeeds.

---

### Phase 5 — Configs, sweeps, three CSVs, eval-only, resume

#### F5.1 — YAML schema polish + `multi-scenario validate` — S

- Hardened error messages. Optional `multi-scenario schema` command emits JSON Schema from the Pydantic model.

#### F5.2 — `runs.csv` writer (long-format, single file) — S

- One CSV with `record_type` column, two row types per run: `final` (one row, full M1–M9 + config_snapshot + metadata) and `eval` (one per eval step, M1–M9 subset). Schema is algorithm-agnostic; JSON nulls → `N/A` via pandas `na_rep`. Atomic write-rename; on overwrite, copy current to `runs.previous.csv` for one-step rollback.
- **Port from rendezvous_comm:** structure of `consolidate.py`. Eval-step rows ported from the per-eval consolidation logic; final rows from the per-run aggregation.
- **Gotcha to handle (port the workaround):** custom eval scalars fire one step after native eval scalars; consolidator must shift custom keys back by 1. See §7.5/#1.

#### F5.3 — `runs.json` writer (slim cross-run manifest) — XS

- Cross-run manifest per §3.5.3: scope, link to `runs.csv`, rankings (`{run_id, value, report}` per metric), and a flat list of per-run `report` links. No duplication of per-run file paths — consumers dereference via `report` to each run's `output/report.json`. Atomic write-rename + `runs.previous.json` backup.
- TDD: given N populated run folders, the writer produces a manifest whose `runs[].report` paths all resolve and whose `rankings` agree with `final` rows in `runs.csv`.

#### F5.4 — Per-step long-format CSV (experimental) — S

- Behind a `storage.long_format: true` flag. Row count = `num_envs × max_steps × n_agents`.

#### F5.5 — **DECISION POINT: long vs summary** — S (analysis, not code)

- Mini-experiment: 6-algo × discovery × 3 seeds. Generate both formats; measure (a) disk size, (b) load time in pandas, (c) which downstream questions each can answer.
- Output: `docs/csv_format_decision.md` + recommendation. **User signs off** before defaults change.

#### F5.6 — Sweep config + combinatorial validator — S

- `SweepConfig` (lists for `seeds`, `algorithms`, scenario params); `iter_runs()` materialises combinations.
- Validator refuses sweeps over a configurable size cap (default 100); prints estimated wall-time before launching.
- **Port from rendezvous_comm:** `iter_runs()` semantics, but **return a list, not a generator** (see gotcha §7.5).

#### F5.7 — Resume from crash — M

- Checkpoint writer (sparse: every N eval intervals + last + best). Resume detection: on launch, if `run_state.json` exists with `RUNNING` and process is dead, resume from latest checkpoint; mark state `RESUMED`.
- **Not in rendezvous_comm yet** (in their cleanup plan, R10–R12). Port the *plan*, write the implementation.
- TDD: kill mid-run, relaunch, verify metrics consistent with uninterrupted run within tolerance.

#### F5.8 — Eval-only mode — S

- `multi-scenario eval <run_dir>` loads policy, runs N episodes, writes a separate `eval_run.json`. Useful for re-evaluating old policies under different conditions.

---

### Phase 6 — OVH runner & secrets

#### F6.1 — `SecretsAdapter` (Fernet) — S

- Encrypt/decrypt LLM API keys via Fernet with passphrase-derived key. Used to ship LLM keys to OVH jobs.
- **Port from rendezvous_comm:** `secrets_util.py`.

#### F6.2 — Port `ovh.py` (cleaned) — M

- Adapter implementing `Runner`. Builds the job spec, uploads code via S3, submits via `ovhai`, polls for completion. **Trailing-slash fix and per-experiment S3 prefix already baked in** (known gotchas, retain).

#### F6.3 — `S3StorageAdapter` — S

- Mirror local layout under `s3://<bucket>/<prefix>/experiments/...`. TDD with `moto` (mocked S3).

#### F6.4 — Code uploader — S

- Rsync-style upload of `src/` + `experiments/<scenario>/<exp_type>/configs/` to S3 before job submit.

#### F6.5 — End-to-end OVH smoke — S (manual)

- Submit `discovery_mappo_smoke.yaml`; verify results land at the right prefix; pull back via `S3StorageAdapter`.
- **Manual demo** — gated on user confirmation that they want to spend an OVH credit.

---

### Phase 7 — Streamlit FE (one page at a time)

> Inspired by `rendezvous_comm/Dashboard.py` + `pages/` (4 existing pages). No page is added until the previous one is approved.

#### F7.1 — Page 0 (Dashboard.py landing) — S

- Theme, navigation, summary tiles (total runs, success per scenario).
- **Port from rendezvous_comm:** `Dashboard.py` + `theme.py`.

#### F7.2 — Page 1: experiments browser — S

- Table of runs (filterable by scenario / algo / exp_type / state). Status badge driven by `run_state.json`.
- TDD: a unit test on the data-loading helper + a smoke test that the page imports cleanly.
- **Demo:** `streamlit run src/multi_scenario/frontend/streamlit_app.py` → Page 1 shows local runs.
- **Port:** `pages/1_*.py` from rendezvous_comm. **Gotcha:** Streamlit caches imports — restart on src changes (already known).

#### F7.3 — Page 2: per-run detail — S

- Click a run → config, metrics, learning curves from per-iter/per-eval CSVs, training videos if present.
- **Port:** `pages/3_*.py` (run detail page).

#### F7.4 — Page 3: cross-experiment comparison — M

- Bar / box plots comparing algorithms across scenarios. Selectable metric, scenario, algos.
- **Port:** `pages/4_*.py` (cross-exp comparison).

#### F7.5 — Page 4: local job submission — M

- Form → builds YAML → calls `LocalRunner` in a background thread → live tail of `logs/run.log`.

#### F7.6 — Page 5: OVH job submission — M

- Same form but routes to `OvhRunner`. Shows the per-job S3 prefix. **Port:** `pages/2_*.py` (OVH jobs page).

---

### Phase 8 — First cross-scenario baseline ablation (the "ER1 across 4 scenarios")

#### F8.1 — Heuristic baseline policies — S

- Per-scenario simple heuristics (e.g. discovery → greedy nearest-target; navigation → straight-line; transport → push-toward-goal). Used as a sanity floor.
- Implements `Algorithm` port (training is a no-op; only `evaluate` is real).
- **Not present in rendezvous_comm beyond a stub** — write fresh.

#### F8.2 — Ablation matrix definition — S

- 4 scenarios × 6 algorithms × N seeds + heuristic baseline. YAML matrix file + a script that fans it out into individual YAMLs.

#### F8.3 — Run the matrix — M (compute-bound)

- Locally for tiny smoke; OVH for real. Collect to one master CSV.

#### F8.4 — Comparison report — S

- Streamlit page or notebook → per-scenario leaderboard, best baseline per scenario.
- **Output:** identifies the best baseline candidate per scenario → input to Phase 9 LERO.

---

### Phase 9 — LERO (placeholder, design later)

Out of scope right now. **Note for future planning** (from deep analysis): rendezvous_comm has 7 LERO versions (v5–v9) plus a meta-prompt outer loop (v4) plus a 23-template prompt registry plus a disk-based LLM cache. Treat this as its own multi-phase mini-plan when we reach it. Best baseline candidates from Phase 8 inform which scenarios get LERO first.

---

### Phase 10 — Polish, CI, extraction

#### F10.1 — Reproducibility test — S

- Run the same config with the same seed twice; assert all metrics agree within tolerance. Lives in `tests/reproducibility/`.

#### F10.2 — CI pipeline — S

- GitHub Actions: lint + unit tests on push; smoke integration tests nightly. Coverage gate (start at 70%).

#### F10.3 — Documentation pass — S

- `README.md` (quick-start), `docs/architecture.md`, `docs/scenarios.md`, `docs/run_layout.md` (the §3.5 conventions formalised).

#### F10.4 — Repo extraction — M

- Rename package (`multi_scenario` → final name), pin VMAS to a released version (or commit hash), set up the new git repo, copy-with-history (`git filter-repo`).
- **Cleanup checklist on extraction:**
  - Remove the top-level `files: '^multi_scenario/'` line from `.pre-commit-config.yaml` — added in F0.2 to scope hooks while nested inside the VMAS repo; once `multi_scenario/` becomes the repo root, files no longer carry that prefix and the filter would silently make every hook no-op.
  - In the markdownlint hook, change `args: ["--config", "multi_scenario/.markdownlint.json"]` back to `args: ["--config", ".markdownlint.json"]` — the prefix is needed only while pre-commit runs from the VMAS toplevel.
  - Add a `LICENSE` file. Deliberately deferred from F0.6 because the choice (GPLv3 like parent VMAS, MIT, Apache-2.0, …) should be made on extraction, not pre-emptively.
- **Manual demo** — gated on user readiness to extract.

#### F10.5 — Comment cleanup pass — XS

- Sweep all source files (everything *outside* the planning markdowns) for comments that reference phase or feature IDs (`F0.1`, `F2.4`, `Phase 9`, etc.) or section anchors from this plan (`§3.5.2`, etc.). These references are scaffolding from the build process — useful during development to trace which feature added what, useless and confusing once the project is extracted to its own repo (where this plan no longer lives).
- For each comment found: if there is a substantive WHY behind the reference, keep that prose and drop the phase pointer. If the comment was *only* a phase pointer, delete the comment entirely. Per project style (CLAUDE.md) — comments justify *why*, not *which-feature-added-this*.
- Files in scope: `src/**`, `tests/**`, `docs/**` (non-markdown), `pyproject.toml`, `.pre-commit-config.yaml`, `.markdownlint.json`, `.gitignore`, YAML configs under `experiments/**`. Out of scope: `plan.md`, `implementation_plan.md`, and any other markdown documents in `docs/` whose purpose is planning/architectural narrative — those are allowed to keep phase references.
- **Demo:** `grep -rnE 'F[0-9]+\.[0-9]+|Phase [0-9]+|§[0-9]' src tests pyproject.toml .pre-commit-config.yaml .markdownlint.json .gitignore docs experiments --include='*.py' --include='*.toml' --include='*.yaml' --include='*.yml' --include='*.json' --include='.gitignore' --include='.pre-commit-config.yaml'` returns no matches.

---

## 6. Open questions / decisions deferred

| Topic | Latest moment |
| --- | --- |
| Default CSV format (long vs summary) | F5.5 |
| Per-scenario success metric (nav, flocking, transport) | F4.1–F4.3 |
| Final package name | before F10.4 |
| Algorithm hyperparameters per scenario | Phase 8 |
| Streamlit FE: keep in same repo or split? | post-Phase 7 |
| LERO architecture (v5–v9 + meta + registry) | start of Phase 9 |
| OVH cost vs local smoke threshold | F6.5 |
| Resume tolerance threshold (acceptable metric drift) | F5.7 |
| Heuristic baseline complexity per scenario | F8.1 |

---

## 6.5 Explicitly out of scope (for now)

To keep the project small and focused, the following are **not planned** and will only be revisited if the user asks:

- **Distributed / multi-GPU training** — single-GPU per run is enough; BenchMARL doesn't abstract DDP.
- **Population-Based Training, Bayesian HPO, gradient-based HPO** — grid sweep is enough for now.
- **W&B / MLflow / Neptune / TensorBoard** — CSV + Streamlit covers our reporting.
- **Multi-LLM provider abstraction** — LiteLLM (port from rendezvous_comm) when we get to LERO; no premature interface.
- **PDF / LaTeX report generation** — markdown reports are enough.
- **Async video pipeline** — videos generated synchronously per run when explicitly requested.
- **Slurm / cluster integration beyond OVH** — single OVH job per run is enough.
- **Cross-seed policy ensembling** — out of scope.

These are listed so we don't accidentally add complexity later under the guise of "porting". If the project grows and needs any of them, we'll plan them as their own phase.

---

## 7. Porting checklist from `rendezvous_comm/src`

Tracks what's ported and where it landed. Updated as we go.

| Source | Lands at | Phase | Status |
|---|---|---|---|
| `config.py` (discovery params) | `adapters/scenarios/discovery.py` | F2.1 | ⬜ |
| `config.py` (sweep iter_runs) | `application/sweep.py` | F5.6 | ⬜ |
| `metrics.py` (M2/M3/M4) | `adapters/metrics/common.py` | F2.2 | ⬜ |
| `metrics.py` (M1/M6 discovery) | `adapters/metrics/discovery.py` | F2.3 | ⬜ |
| `metrics.py` (M7 sample efficiency) | `adapters/metrics/common.py` | F5.2 | ⬜ |
| `metrics.py` (M8/M9) | `adapters/metrics/common.py` (with M8 bug fix) | F5.2 | ⬜ |
| `runner.py` (BenchMARL wiring) | `adapters/algorithms/benchmarl_base.py` | F2.4 | ⬜ |
| `runner.py` (eval callbacks) | `adapters/algorithms/benchmarl_base.py` | F2.4 | ⬜ |
| `runner.py` (video gen) | `adapters/runners/local.py` (sync) | F8.x | ⬜ |
| `storage.py` | `adapters/storage/local.py` | F2.5 | ⬜ |
| `consolidate.py` | `adapters/storage/csv_consolidator.py` | F5.2/F5.3 | ⬜ |
| `provenance.py` | `adapters/provenance/hashing.py` | F2.7 | ⬜ |
| `logging_setup.py` | `adapters/logging/file_logger.py` | F2.7 | ⬜ |
| `secrets_util.py` | `adapters/secrets/fernet.py` | F6.1 | ⬜ |
| `ovh.py` | `adapters/runners/ovh.py` | F6.2 | ⬜ |
| `Dashboard.py` + `theme.py` | `frontend/streamlit_app.py` | F7.1 | ⬜ |
| `pages/1_*.py` (browser) | `frontend/pages/` | F7.2 | ⬜ |
| `pages/3_*.py` (run detail) | `frontend/pages/` | F7.3 | ⬜ |
| `pages/4_*.py` (cross-exp) | `frontend/pages/` | F7.4 | ⬜ |
| `pages/2_*.py` (OVH jobs) | `frontend/pages/` | F7.6 | ⬜ |
| `plotting.py`, `display.py`, `report.py` | utility modules as Streamlit needs them | F7.x | ⬜ |
| `lero/` (v5–v9, meta, prompts, llm_cache) | `adapters/algorithms/lero/` | Phase 9 | ⬜ |

> Rule: **don't port code until the feature that needs it.** This forces every port to be reviewed in context.

### 7.1 What to NOT port

- `rendezvous_comm/tests/_archive_v3/` — obsolete LERO-MP iterations.
- Top-level run scripts `run_lero_v5.py … v9.py`, `run_lero_mp_v4.py`, `run_v7_inner_only_with_s3b_prompt.py` — one-off harnesses; reusable logic lives in `src/lero/v9/` already.
- Old prompt variants under `src/lero/prompts/v1*`, `v2_evolved_*` (except `v2_evolved_3x2M_best`), `v2_fewshot*`, `v2_twofn` — keep only the active set.
- `results/` directories (≈430 MB experiment data) — separate concern, not code.
- `configs/er1/archive/` — deprecated ablation YAMLs.
- `configs/ovh.yaml` — deployment-specific; ours will be its own deployment file.

---

## 7.5 Additional gotchas to watch for during porting

> Beyond the user's already-known list (targets_respawn, max_steps pop bug, OVH trailing slash, scenario_patch closure, M8 shared_reward, Streamlit import caching, pip vmas missing kwargs, PPO NaN crashes).

1. **Eval callback step alignment off-by-one.** Custom eval metrics fire in `on_evaluation_end()` AFTER the iter counter ticks; native BenchMARL eval scalars fire before. Consolidator in rendezvous_comm shifts custom keys back by 1. Port the workaround AND document the rule: new eval metrics must go in `on_evaluation_end()`. (F5.3)
2. **`iter_runs()` is a generator**: consuming it twice yields nothing the second time. Return a list. (F5.6)
3. **LLM cache key doesn't track model-version semantics.** When we change models we must invalidate or use `cache_mode="write_only"`. (Phase 9)
4. **`scenario_patch.exec()` has no shape/type checks** — LLM-generated reward functions can return wrong shapes and only fail at eval. Pydantic-validate the return shape if possible. (Phase 9)
5. **Whitelist applied at eval but not at LLM gen-time.** Mismatch makes LLM-generated code reference forbidden state keys then crash later. (Phase 9)
6. **BenchMARL pickles callbacks for an experiment-name hash** — undocumented contract. Our callbacks must implement `__getstate__/__setstate__` returning a stable dummy. (F2.4)
7. **M7 sample-efficiency picks the first crossing of 80%**, not the stable crossing — non-monotonic curves get a misleading value. Consider rolling average. (F5.2)
8. **Provenance freshness only tracks a hardcoded subset of files** (config/runner/metrics in rendezvous_comm). Plotting/display changes don't trigger staleness. Decide: include adapters in code_hash? (F1.5 / F2.7)
9. **Run ID and config seed can desync** if folders are renamed. Add an assertion that config.yaml's seed matches the run_id's `s<N>` token. (F2.5)
10. **Reward clipping is applied post-aggregation across agents** in scenario_patch; document this for any new scenario (Phase 9 / when we add scenario reward shaping).
11. **Task overrides silently shadow base task fields**, including typos. Validator should reject unknown override keys. (F1.1, F5.1)
12. **`generate_run_videos()` requires a VMAS eval loop** — coupled to the scenario adapter. Don't promise async video gen yet. (F8.x)

These will be revisited in the relevant feature; cross-referenced from the porting checklist.

---

## 8. Glossary

- **Port**: an interface (Python `Protocol`) the domain depends on.
- **Adapter**: concrete implementation of a port that talks to an external system (VMAS, BenchMARL, S3, Streamlit).
- **Factory**: a small function/dict that maps a name (`"mappo"`) to the constructor of the adapter that implements it.
- **Run**: one execution of one config → one CSV row.
- **Experiment**: a logical grouping of runs (e.g. discovery/baseline). Lives in `experiments/<scenario>/<type>/`.
- **Sweep**: a matrix of runs varying one or more axes (algorithm, seed, scenario param).
- **RunId**: parametric, deterministic identifier of a run (see §3.5.1).
- **RunState**: lifecycle status of a run on disk (see §3.5.4).
- **Provenance**: hashes + git SHA + library versions for a run (§3.5.5).

---

## 9. How we work each feature

1. I post: feature ID + goal + tests-to-write + files-to-touch.
2. You say go.
3. I write failing tests → red → implement → green → refactor.
4. I post the diff summary + the demo command + checklist.
5. You run the demo, review the diff, say "next" or send corrections.
6. I update the porting checklist in §7 and mark the feature done.

No feature begins until the previous one's gate is signed off.
