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
- `report.json` is a manifest with relative paths to `config`, `provenance`, `log`, `metrics`, `eval_episodes`, `policy`, `videos.{before,after}_training`, plus a `benchmarl: {dir, scalars: [...]}` block (where `dir` points at the *inner* BenchMARL run root and `scalars[i]` are paths relative to `dir`, typically `scalars/<name>.csv`), plus a headline summary (status, duration, M1/M2 highlights). Streamlit run-detail page reads this — no globbing. Resolve a scalar via `run_dir / benchmarl.dir / benchmarl.scalars[i]`.

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
  - **No file duplication**: every per-run path (config, metrics, policy, videos, `benchmarl.scalars`) lives only in the per-run report; the cross-run manifest dereferences via `report` links.
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

- **`Runner` Protocol** (`domain/ports/runner.py`) — `run(cfg, run_dir) -> ExperimentResult`. Concrete adapters: `LocalRunner` at F2.6 (wraps `ExperimentService`), `OvhRunner` at F6.2 (submits to cloud).
- **`ExperimentService`** (`application/experiment_service.py`) — in-process orchestrator. Constructor takes the five domain ports (`scenario`, `algorithm`, `metrics`, `storage`, `logger`). `run(cfg, run_dir, provenance) -> ExperimentResult` runs the full lifecycle: `INITIALIZING` → save config + provenance → `make_env` → `RUNNING` → train → evaluate → metric bundle → `ExperimentResult` → save result → `DONE`. Returns the result.
- **Provenance is injected**, not built inside the service — keeps the orchestrator free of git / package-version I/O. F2.7 supplies a real `ProvenanceWriter` to callers.
- **Crash handling deferred to F5.7.** F1.11 is happy-path only.
- TDD: full use-case test with port fakes only — no VMAS, no BenchMARL. <1s.
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

#### F2.2 — `CommonMetricsBundle` — S

- Renamed from "CommonMetricSet" — the always-on bundle replaced metric sets at the §3.5.3 redesign.
- `CommonMetricsBundle` (in `adapters/metrics/common.py`) implements `MetricsBundle.compute(rollout, scenario) -> dict[str, float | None]`. Routing:
  - **Universal**, computed here: M2 (return), M3 (steps), M4 (collisions) — means over per-episode tensors in the rollout.
  - **Scenario-DI**: M1 / M6 / M8 — delegated to `scenario.success_predicate` / `coverage_progress` / `utilization_predicate`. Return None when the scenario returns None (F2.1 stub case).
  - **Comm-gated**: M5 — None when `scenario.has_comm()` is False.
  - **Stubbed (None)**: M7 (sample-efficiency, end-of-run from eval-curve data) and M9 (spatial spread, needs position field in rollout). Filled in later features.
- **Rollout shape contract** (documented in the bundle's docstring): `{"episode_returns": Tensor[n], "episode_lengths": Tensor[n], "episode_collisions": Tensor[n]}`. BenchMARL adapter (F2.4) aggregates its TensorDict into this dict.
- **Port from rendezvous_comm:** logic from `src/metrics.py`.

#### F2.3 — Discovery DI primitives (M1, M6) — S

- Renamed from "DiscoveryMetricSet" — sets are gone. F2.3 implements the discovery-specific DI primitives on `VmasDiscoveryAdapter` (M1 / M6); the always-on `CommonMetricsBundle` from F2.2 picks them up automatically.
- `success_predicate(rollout)` — M1 from `targets_covered` cumsum max **(NOT from the `terminated` signal — documented bug from rendezvous_comm)**. Returns Tensor[n_episodes] of bools, or None when the rollout lacks `targets_covered` / `n_targets`.
- `coverage_progress(rollout)` — M6 = `max-over-T(targets_covered) / n_targets` per episode. Returns Tensor[n_episodes] of floats, or None when data missing.
- `utilization_predicate` — M8 stays stubbed at this stage; lands when needed.
- **Rollout-shape extension (documented in adapter docstring):** discovery rollouts carry `rollout["targets_covered"]: Tensor[n_episodes, T]` and `rollout["n_targets"]: int` on top of F2.2's universal contract. F2.4 (BenchMARL) populates these from VMAS info dicts.

#### F2.4 — `BenchmarlBaseAdapter` + `MappoAdapter` — M

- Shared scaffolding (`benchmarl_base.py`) + MAPPO-specific subclass.
- Wraps a BenchMARL `Experiment`; exposes `train`/`evaluate` returning a serialisable artifact.
- **Port from rendezvous_comm:** BenchMARL wiring from `src/runner.py` (the working bits, not the LERO callbacks).
- TDD: 1-env, 2-iteration smoke training (slow, marked `@pytest.mark.slow`).

#### F2.4.1 — Propagate BenchMARL training knobs from config — S

Background: F2.4 hard-coded several BenchMARL `ExperimentConfig` fields (`on_policy_collected_frames_per_batch=100`, `on_policy_minibatch_size=50`) and didn't propagate `lr` / `gamma` / `share_policy_params` / `n_minibatch_iters` at all. Worse, `lr` / `gamma` live on BenchMARL's `ExperimentConfig` but the `docs/example_config.yaml` (v5) routed them through `cfg.algorithm.params` where the strict-validating `MappoConfig` setattr loop would reject them. The F2.4 smoke test only passed because `algorithm.params` was empty.

Schema change — extend `TrainingSection` with the universal training-loop knobs BenchMARL puts on its `ExperimentConfig`:

- `lr: float = 3e-4`
- `gamma: float = 0.99`
- `frames_per_batch: int = 6000`
- `minibatch_size: int = 400`
- `n_minibatch_iters: int = 45`
- `share_policy_params: bool = True`

All have sensible defaults so the F1.1 round-trip and `fake_config_builder` tests stay green without edits.

Wiring change — `BenchmarlBaseAdapter._experiment_config(cfg)` reads from `cfg.training` instead of hard-coding. `evaluation_interval` set to `cfg.evaluation.interval_iters * cfg.training.frames_per_batch` (cadence in iters from the user's POV; frames internally). `render=False` set explicitly to avoid the BenchMARL default `True` (causes pyglet crashes in headless / OVH).

Algorithm-specific knobs (`lmbda`, `entropy_coef`, `clip_epsilon`) stay in `cfg.algorithm.params` — those genuinely live on `MappoConfig` / `IppoConfig` / etc.

`docs/example_config.yaml` updated: move `lr` / `gamma` from `algorithm.params` to `training`; keep `lmbda` in `algorithm.params`.

TDD:

- New unit test: building `_experiment_config(cfg)` produces a BenchMARL `ExperimentConfig` with our cfg's `lr` / `gamma` / batch sizes propagated.
- F2.4 smoke test still passes — config sets explicit small batch sizes for fast smoke runs.

#### F2.4.2 — Model arch + critic config (deferred placeholder)

**Trigger:** lift this into a real feature **before F8.3 (Run the matrix)** — that's when ER1-comparable training is queued and needs non-default MLP architecture / a separate critic. Or earlier if any production config explicitly requires custom `num_cells` / `activation_class` / critic shape.

Scope when activated:

- `num_cells` (MLP hidden layers) and `activation_class` plumbed from cfg.
- Independent `critic_model_config` (rendezvous_comm passes a separate `MlpConfig` for the critic).
- Model-type override (`"gnn"` topology) — Phase 9 LERO scope, not Phase 2.

Schema sketch (when implemented): add `algorithm.params.hidden_layers: list[int] | None` and `algorithm.params.activation: str | None` (these are model knobs that vary by algorithm config in practice, so live in algorithm.params not training).

#### F2.4.3 — Real rollout aggregation in `evaluate()` — S

**Background:** F2.4 wired up `train()` but `evaluate()` returns zero-filled tensors. F2.4.3 populates them with real per-episode data so `CommonMetricsBundle.compute(rollout, scenario)` produces meaningful M1–M6 numbers after a training run.

**Approach:** port the proven pattern from `rendezvous_comm/src/runner.py::evaluate_trained`. Inside `BenchmarlBaseAdapter.evaluate(artifact, env, cfg)`:

1. Pull `experiment.test_env`, `experiment.policy`, `experiment.max_steps`, `experiment.group_map` from the artifact.
2. Wrap in `torch.no_grad()` + `set_exploration_type(ExplorationType.DETERMINISTIC)`.
3. Run `test_env.rollout(max_steps, policy, auto_cast_to_device=True, break_when_any_done=False)` — enough times to gather `cfg.evaluation.episodes` total.
4. Extract per-env from the resulting TensorDict using path tuples:
   - `episode_returns` — sum of `("next", group, "reward")` over T per env.
   - `episode_lengths` — T (rollout length; refined later when episodes terminate naturally).
   - `episode_collisions` — count of `("next", group, "info", "collision_rew") < 0` per env.
5. **Discovery-specific:** `("next", group, "info", "targets_covered")` per-step → `cumsum(dim=1)` → `targets_covered: Tensor[n_episodes, T]`. Project memory invariant: cumsum of newly-covered counts (NOT terminated signal) — same logic as `VmasDiscoveryAdapter.success_predicate`.

**Tests:** the F2.4 smoke test now asserts that `evaluate()` returns non-zero `episode_returns` and `episode_lengths > 0`; for discovery, asserts `targets_covered` is a Tensor of the right shape and `n_targets` is set.

**Out of scope (later):**

- Per-step length detection (currently uses constant T — fine for `break_when_any_done=False` but loses info when episodes end early).
- Episode-truncation handling at iteration boundaries.
- Token-extraction for comm scenarios (M5 — when comm scenarios land).

#### F2.5 — `LocalStorageAdapter` — S

- All run-level files we own are JSON: `input/config.json`, `output/metrics.json`, `output/eval_episodes.json`, `run_state.json`. BenchMARL's native CSVs in `benchmarl/` are preserved untouched. Per-run-summary CSV append is at the cross-run level (§3.5.3), not per-run. Writes to the §3.5.2 layout.
- TDD: round-trip; concurrent appends to the cross-run CSV serialise correctly.

#### F2.6 — `LocalRunner` + factories — S

- In-process runner. `factories.py` registers names → adapters.
- TDD: registry round-trip; unknown name raises clean error.

#### F2.7 — `FileLogger` + `ConsoleLogger` + `ProvenanceWriter` — S

- `FileLogger(log_path)` — appends `<UTC ISO ts> <LEVEL> <msg>\n` to `logs/run.log`; auto-creates parent dirs.
- `ConsoleLogger(debug=False)` — info/debug → stdout, warning/error → stderr; debug suppressed by default.
- `ProvenanceWriter(hashed_source_files=(), git_root=None)` — callable building a `Provenance` for one run: `config_hash` from F1.5's `compute_config_hash`, `code_hash` from `compute_code_hash` (or `"sha256:empty"` when no files supplied), `git_sha` / `git_dirty` via `git rev-parse HEAD` and `git diff-index --quiet HEAD` with safe fallbacks (`"unknown"` / `False`), `library_versions` via `importlib.metadata` + `multi_scenario.__version__`.
- **Wiring:** `LocalRunner.__init__(provenance_factory=...)` becomes optional and defaults to `ProvenanceWriter()` — callers can omit it for the common case. The logger is still mandatory because it's run-dir-scoped (FileLogger needs the run_dir, only known at run time); a `make_local_runner(run_dir)` helper can land later if useful.
- **`RunStateWriter` is not a separate class.** Its role — writing `run_state.json` at every lifecycle transition — is fulfilled by `Storage.save_run_state` + `RunStateRecord.transition_to` + the explicit transitions inside `ExperimentService.run`. No new code needed for that part.
- **Port from rendezvous_comm:** `logging_setup.py`, `provenance.py`.

#### F2.8 — CLI `multi-scenario run <yaml>` — S

- Typer multi-command app (`version` + `run`). Provide `experiments/discovery/baseline/configs/mappo_smoke.yaml` (1 env, 1 iter, `max_steps=10`).
- **Actual deliverables (implemented):** `run_dir/<run_id>__<ts>/` containing:
  `input/config.json`, `input/provenance.json`, `output/metrics.json`, `output/benchmarl/...`, `logs/run.log`, `run_state.json`.
- **Not yet produced at this stage** (deferred to later features):
  - `output/eval_episodes.json` → F2.10 (report.json writer)
  - `output/report.json` → F2.10
  - `runs.csv` / `runs.json` → F5.2 / F5.3
- **Also fixed (F2.8 sub-fix):** Algorithm Protocol updated to accept `run_dir: Path | None = None` on both `train` and `evaluate`; `BenchmarlBaseAdapter.train` uses it to place BenchMARL output at `run_dir/output/benchmarl/` (creates dir before passing to BenchMARL).
- **Demo:** `multi-scenario run experiments/discovery/baseline/configs/mappo_smoke.yaml` → exit 0; run folder produced with all §3.5.2 files listed above.
- **Test:** `tests/integration/cli/test_run.py::test_run_command_succeeds` (slow) — verifies folder layout end-to-end.

#### F2.9 — Smoke integration test — XS

- `tests/integration/smoke/test_discovery_mappo.py` loads the real `experiments/discovery/baseline/configs/mappo_smoke.yaml` from disk, redirects `runtime.storage.path` to `tmp_path`, runs through `LocalRunner` (default `ProvenanceWriter`), and asserts the §3.5.2 milestone:
  - `run_state.json` → `state == "DONE"`.
  - `input/config.json` round-trips back to `ExperimentConfig`.
  - `input/provenance.json` has non-empty `git_sha` and populated `library_versions` (real, not stub).
  - `output/metrics.json` → `M1_success_rate`, `M2_avg_return`, `M3_steps` are real `float`s — proves F2.3 (discovery DI primitives) + F2.4.3 (rollout aggregation) are wired through.
  - `output/benchmarl/` directory exists and is non-empty.
  - `logs/run.log` exists and non-empty.
- **CSV row assertion deferred to F5.2** (was in the original wording; `runs.csv` writer doesn't exist yet at this stage).
- **Validation gate (Phase 2 milestone):** discovery + MAPPO produces a full §3.5.2 run-folder layout locally with non-stub metrics, provenance, and BenchMARL native output. ✅

#### F2.10 — `report.json` writer — XS

- At run end, emit `output/report.json` per §3.5.2: a manifest with status, started/finished timestamps, duration, headline summary (M1–M4), and relative-path links to every relevant artefact (`config`, `provenance`, `log`, `metrics`, `eval_episodes`, `policy` inside `benchmarl/`, `videos.before_training`, `videos.after_training`, plus a `benchmarl: {dir, scalars: [...]}` block enumerating every native scalar CSV).
- **Wiring:** built and saved by `LocalRunner` *after* `ExperimentService.run()` returns (not inside the service) so the report's `status` reflects the on-disk run state. Keeps the `Storage` Protocol surface minimal — `save_report` lives on the concrete `LocalStorageAdapter` only (per F1.9 design note).
- The exact `<bm_run>` directory name (BenchMARL-assigned) is captured in the manifest so consumers don't glob.
- **Out of scope:** `eval_episodes.json` writer — link is `null` until F2.10.1 lands. Video paths — `null` until F2.11 lands.
- TDD: given a fully-populated run folder, the writer produces a manifest whose every linked path resolves to an existing file (or is `null` if opt-in artefact wasn't generated).

#### F2.10.1 — `eval_episodes.json` writer — XS ✅

- `LocalStorageAdapter.save_eval_episodes(run_dir, rollout)` — serialises the rollout dict from `Algorithm.evaluate()` to `output/eval_episodes.json`. Tensors → lists via `.tolist()`. Schema: `{"episode_returns": [...], "episode_lengths": [...], "episode_collisions": [...], "targets_covered"?: [[...]], "n_targets"?: int}`. Discovery-specific fields included only when present; unknown keys silently dropped to keep schema stable.
- **Wiring:** `ExperimentService` accepts an optional `eval_episodes_writer: Callable[[Path, Any], None] | None = None` constructor arg; when set, called after `evaluate()` and before `metrics.compute()`. `LocalRunner` injects `LocalStorageAdapter().save_eval_episodes` (isinstance-narrowed to the concrete adapter, mirroring the F2.10 pattern). Off the `Storage` Protocol surface (F1.9 minimalism).
- `RunReport.links.eval_episodes` now resolves automatically via `ReportBuilder._optional_rel`.
- Tests: storage round-trip (universal + discovery + unknown-key drop), service wiring (writer called with correct args), F2.9 smoke extended with eval_episodes.json + report link assertions.

**Out of scope:** mid-training intermediate eval samples (only the final eval is captured here; the per-iter eval rows in F5.2 will need a different hook).

#### F2.11 — Before/after training videos — S ✅

- `VideoRecorder` (`adapters/video/recorder.py`) — rolls out one episode through `experiment.test_env` + `experiment.policy`, calls the underlying VMAS env's `render(mode="rgb_array", env_index=0)` per step, encodes via `imageio[ffmpeg]` (`mimsave(..., fps=15, codec="libx264")`). No state-dict reconstruction — reuses the BenchMARL TensorDictModule directly (cleaner than the rendezvous_comm port).
- `BenchmarlBaseAdapter.train()` split into `build_experiment(cfg, run_dir) -> Experiment` (no run) + `train()` (build + run). Exposing the random-init policy is what makes the "before" video possible without policy reconstruction.
- Recording gated by `_should_record_video(cfg, run_dir)`: requires `run_dir`; reads `cfg.runtime.runner.params.record_video`; default = `not cfg.experiment.id.endswith("_smoke")`.
- `RunReport.links.videos.{before,after}_training` populated automatically via `ReportBuilder._videos`.
- Tests: `tests/integration/video/test_recorder.py` (recorder MP4 round-trip), `tests/integration/smoke/test_discovery_mappo_videos.py` (videos enabled, both MP4s + report links resolve), `test_discovery_mappo.py` extended to assert default-off → no `videos/` dir.

**Out of scope:** multi-episode videos, FPS/codec configurability, headless/OVH-specific render testing.

**OVH/headless deferred to F6.6:** VMAS rendering needs Pyglet + OpenGL/X11, which OVH AI Training containers don't ship. Today, a non-smoke run on OVH would crash inside `VideoRecorder.record()` with `Error occurred while running 'from pyglet.gl import *'` (confirmed pattern from `rendezvous_comm/results/.../run.log`). F6.6 (Phase 6) bundles the three changes needed: fail-soft try/except around the recorder calls, a `multi-scenario regenerate-videos <run_dir>` CLI for post-import local regeneration, and flipping `bm.checkpoint_at_end = True` for non-smoke runs so the "after" video can be reproduced from the saved checkpoint.

---

### Phase 3 — Add remaining baseline algorithms (still discovery only)

> One algorithm per feature. Same TDD pattern: smoke test that 2-iter run completes + writes CSV.

- **F3.1 — IPPO adapter** (S) ✅ — `IppoAdapter` mirrors `MappoAdapter` (BenchMARL `IppoConfig`); `experiments/discovery/baseline/configs/ippo_smoke.yaml`; tests in `tests/integration/algorithms/test_ippo.py` (Protocol + 2-iter smoke). End-to-end via CLI confirmed.
- **F3.2 — MADDPG adapter** (S) ✅ — first **off-policy** adapter; extended `BenchmarlBaseAdapter._experiment_config` to wire the off_policy_* mirrors (`off_policy_collected_frames_per_batch`, `off_policy_train_batch_size`, `off_policy_n_optimizer_steps`) so PPO and DDPG/SAC families share the same `cfg.training` knobs. `MaddpgAdapter` + `experiments/discovery/baseline/configs/maddpg_smoke.yaml` + tests in `tests/integration/algorithms/test_maddpg.py`. End-to-end via CLI confirmed.
- **F3.3 — IDDPG adapter** (S) ✅ — `IddpgAdapter` (off-policy, same fields as MADDPG); smoke yaml + tests; no base changes (off_policy_* knobs already wired in F3.2).
- **F3.4 — ISAC adapter** (S) ✅ — `IsacAdapter` (off-policy SAC with alpha temperature / num_qvalue_nets / etc); smoke yaml + tests; same template as IDDPG.
- **F3.5 — MASAC adapter** (S) ✅ — `MasacAdapter` (off-policy multi-agent SAC, centralised critics); smoke yaml + tests; same template as ISAC.
- **F3.6 — Algorithm registry refactor** (XS) ✅ — `BenchmarlBaseAdapter` gained a `_config_class` class attribute + a default `_algorithm_config` that instantiates from YAML and applies `cfg.algorithm.params` overrides with strict field validation. Each of the 6 adapters shrank to ~12 lines of declarative metadata (name + _config_class). Subclasses can still override `_algorithm_config` directly if a custom build path is needed. Behavior-preserving — all 12 algorithm tests + 109 others still green.

**Phase 3 milestone demo:** loop over 6 yaml configs, all produce CSV rows in the same `results/`. (Per-algorithm smoke configs ready: `experiments/discovery/baseline/configs/{mappo,ippo,maddpg,iddpg,isac,masac}_smoke.yaml`. Cross-run CSV writer lands in F5.2.)

---

### Phase 4 — Extend to other scenarios

> One scenario per feature; each adds scenario adapter + scenario metric set + at least one MAPPO smoke run.

- **F4.1 — Navigation adapter + metrics** (S) ✅ — `VmasNavigationAdapter`. **M1 revised after user confirmation:** binary "all agents reached their goals during the episode" via the universal `episode_terminated` rollout key (mirrors discovery's all-or-nothing semantics; "fraction at goal" is semantically a coverage metric, not a success rate). Bundled changes:
  - `BenchmarlBaseAdapter._extract_terminated` — universal extraction of `("next", "terminated")` per episode → `episode_terminated: Tensor[n_eps, bool]`. Available to any scenario.
  - `_extract_collisions` — falls back to `("next", group, "info", "agent_collisions")` (navigation's key) if `collision_rew` not present (discovery's key still tried first).
  - `_EVAL_EPISODES_SCHEMA` extended with `episode_terminated` so the new universal key surfaces in `eval_episodes.json`.
  - `experiments/navigation/baseline/configs/mappo_smoke.yaml` + tests in `tests/integration/scenarios/test_navigation.py`. End-to-end via `LocalRunner` confirmed.
  - **Out of scope (deferred):** sharper M6 (per-agent on-goal fraction at episode end) — needs per-agent position extraction into the rollout dict; stub `None` for now.
- **F4.2 — Flocking adapter + metrics** (S) ✅ — `VmasFlockingAdapter`. **M1 revised after user confirmation:** `None` (no natural binary success metric — flocking is continuous-control optimisation; M2 / M4 carry the evaluation weight). Mirrors the `null` semantics already used for M5/M6/M7/M8/M9 when not applicable. Bundled changes:
  - `_extract_collisions` extended with a third info-key fallback: `agent_collision_rew` (flocking's key). Now tries `collision_rew` → `agent_collisions` → `agent_collision_rew` → zeros.
  - `experiments/flocking/baseline/configs/mappo_smoke.yaml` + tests in `tests/integration/scenarios/test_flocking.py`. End-to-end via `LocalRunner` confirmed (`M1=None`, `M2=-0.137`, `M4=0.0`).
  - **Out of scope (deferred):** sharper M1 like "fraction of timesteps in flocking-acceptable state" — needs per-step pos/vel extraction. Add later if you want it.
- **F4.3 — Transport adapter + metrics** (S) ✅ — `VmasTransportAdapter`. M1 uses universal `episode_terminated` (= "all packages delivered to goals" — same template as navigation). No base changes — VMAS transport has no `info()` so no new collision key needed. Defaults: heavy package (`package_mass=50`) requiring cooperative push. `experiments/transport/baseline/configs/mappo_smoke.yaml` + tests in `tests/integration/scenarios/test_transport.py`. End-to-end via `LocalRunner` confirmed. M6 stub `None` (deferred — needs per-package position extraction).
- **F4.4 — Scenario registry refactor** (XS) ✅ — `VmasScenarioBase` (`adapters/scenarios/base.py`) provides shared `make_env` (uses `self.name` as the VMAS scenario name), default `has_comm`/predicates returning `None`/`False`, plus a `_terminated_based_success` helper for navigation/transport. Each adapter shrunk:
  - **discovery** — kept its bespoke cumsum-based M1 + fraction-based M6.
  - **navigation, transport** — declarative metadata + 1-line `success_predicate` via the helper.
  - **flocking** — just `name + default_params` (everything else inherits `None` defaults).
  - Behavior-preserving — all 144 tests still green.

**Phase 4 milestone reached.** All 4 scenarios + 6 algorithms = **24 possible scenario × algorithm combos**, each one runnable end-to-end via the same `LocalRunner` pipeline producing the full §3.5.2 layout.

**Phase 4 milestone demo:** for each scenario, `multi-scenario run <scenario>_mappo_smoke.yaml` succeeds.

---

### Phase 5 — Configs, sweeps, three CSVs, eval-only, resume

#### F5.1 — YAML schema polish + `multi-scenario validate` — S ✅

- New `multi-scenario validate <yaml>` typer command. Parses the YAML through `ExperimentConfig.from_yaml`, exits 0 with `OK <path>` on success, exits 1 on any validation error with one readable line per issue: `<dotted.field.path>: <message>`. Uses Pydantic v2's multi-error reporting so all field issues surface in a single run (typos, missing fields, wrong types).
- **Pre-flight goal:** catch config errors before submitting OVH jobs (which would waste credits) or kicking off long local sweeps.
- Tests: valid YAML → exit 0; missing required field / unknown field / wrong type → exit 1 with the offending field path in the error output; missing file → typer's standard non-zero exit.
- **Scope drop after user review:** `multi-scenario schema` (JSON Schema export for IDE autocomplete) skipped — YAMLs are mostly template copies and validation alone covers the realistic use cases. Re-add later if hand-editing volume justifies it.

#### F5.2 — `runs.csv` writer (final rows) — S

- `RunsCsvWriter.consolidate(exp_type_dir)` walks `<exp_type_dir>/<run_folder>/`, filters runs with `run_state.state == "DONE"`, builds one `record_type=final` row per run from `output/metrics.json`. Schema is algorithm-agnostic; JSON nulls → `N/A` via pandas `to_csv(na_rep="N/A")`.
- Atomic write-rename: write to `runs.csv.tmp` → `os.replace` → `runs.csv`. If `runs.csv` exists at write time, copy to `runs.previous.csv` first (one-step rollback per §3.5.3).
- Schema: `record_type, run_id, exp_id, scenario, algorithm, seed, run_timestamp, M1_success_rate, ..., M9_spatial_spread, n_envs, n_eval_episodes, convergence_frame, duration_seconds, <flattened config_snapshot keys>`.
- New CLI: `multi-scenario consolidate <exp_type_dir>` invokes the writer.
- **Scope drop after user review:** `record_type=eval` rows deferred to F5.2.1 — needs BenchMARL eval callback or scalar-CSV aggregation mapping to M1-M9. Per-run leaderboard (final rows) is the load-bearing view; eval-time evolution can be read from per-run `output/benchmarl/.../scalars/eval_*.csv` at view time (Streamlit, Phase 7).
- **Port from rendezvous_comm:** structure of `_build_sweep_row` in `consolidate.py` (final rows). Eval-step rows + custom-key shift gotcha (§7.5/#1) deferred to F5.2.1.

#### F5.2.1 — `runs.csv` eval rows (deferred placeholder)

**Trigger:** lift this into a real feature **before F8.4 (Comparison report)** if the report needs cross-run training-curve aggregation (e.g. "compare MAPPO vs IPPO M2 at iter 50 across seeds in one table/plot"). Or earlier if Streamlit (Phase 7) grows a cross-run plotting page that wants this. Until then, single-run training curves are readable directly from each run's BenchMARL `eval_*.csv` scalars.

**Scope when activated:**

- Add `record_type=eval` rows to `runs.csv`. One row per (run, eval step).
- Source options:
  - **Option A (preferred):** custom BenchMARL eval callback that fires our `MetricsBundle.compute(rollout, scenario)` mid-training, persisting per-step M1-M9 to a new `output/eval_steps.json` file. F5.2.1 then aggregates these into runs.csv.
  - **Option B (lighter):** read BenchMARL native `output/benchmarl/.../scalars/eval_*.csv` and map onto a subset of M1-M9 (M2 from `eval_reward_episode_reward_mean`, M3 from `eval_reward_episode_len_mean`; M1/M4/M6/M8 as `N/A`).
- **Gotcha to port (rendezvous_comm §7.5/#1):** custom eval scalars fire one step after native eval scalars; consolidator must shift custom keys back by 1.

#### F5.3 — `runs.json` writer (slim cross-run manifest) — XS ✅

- `RunsManifest` domain model (`scope` / `csv` / `rankings` / `runs`) + `RunsJsonWriter.consolidate(exp_type_dir)` walking the same DONE-run folders as `RunsCsvWriter`. Atomic write-rename + `runs.previous.json` backup, mirroring the F5.2 pattern.
- Rankings: per-metric `[{run_id, value, report}]` arrays sorted descending by raw value (consumers know minimize-vs-maximize semantics). None-valued entries dropped per metric; metrics that are None across all runs absent from the rankings dict entirely.
- Pointer-only `runs[]`: `{run_id, report}` where `report` resolves to `<run_folder>/output/report.json` (relative to `exp_type_dir`) or is `None` if the run hasn't produced one yet. Zero path duplication — Streamlit dereferences via the `report` link to read each run's per-run manifest.
- CLI: `multi-scenario consolidate` now writes **both** `runs.csv` (F5.2) and `runs.json` (F5.3) in one shot from the same run scan.
- Tests: 9 unit tests covering scope aggregation, descending rankings, None-skipping, runs[] linking + missing-report → null, atomic backup, empty-dir → empty manifest. Plus end-to-end CLI confirmation.

#### F5.4 — Per-step long-format CSV (experimental) — S ✅

- `LocalStorageAdapter.save_eval_steps_long(run_dir, rollout_td, group_map)` writes `output/eval_steps.csv` with one row per `(env_idx, step, group:agent)` tuple. Row count = `num_envs × T × Σ|group|`. Universal schema: `env_idx, step, agent, reward, done, terminated, action_d{i}` (action dim discovered at runtime). Position / observation / per-info-key columns deliberately excluded — F5.5 measures whether they're worth adding.
- **Opt-in:** gated by `cfg.runtime.storage.params['long_format']: bool` (default `False`). Off the `Storage` Protocol per F1.9 minimalism.
- **Wiring:** `BenchmarlBaseAdapter.evaluate()` saves the LAST rollout's per-step data when the flag is on AND `run_dir` is set. Multi-rollout aggregation deferred — most evals fit in one rollout.
- Tests: 6 unit tests against fake rollout TensorDicts (row count, schema, value placement, group-map agent naming, done/terminated broadcast, output-dir creation) + 2 integration tests through `MappoAdapter.evaluate` (flag on → CSV produced; flag off → no file).

#### F5.5 — **DECISION POINT: long vs summary** — S (analysis) ✅

- `scripts/f5_5_format_decision.py` is the reproducer — runs 3 seeds (mappo, discovery, max_steps=100, n_agents=4, num_envs=1, long_format=true), then walks every per-run artefact + BenchMARL native scalars, captures sizes / columns / sample rows / load times, and emits `docs/csv_format_decision.md` directly. Re-runnable; no hand-typed numbers.
- **Scope cut from 18 → 3 runs** (documented in §1 of the doc): format-size analysis depends on `max_steps × n_agents × num_envs × episodes`, not on algorithm or seed. Three runs are enough to confirm consistency.
- Doc covers: per-file inventory with full content for `config.json` / `provenance.json` / `metrics.json` / `eval_episodes.json` / `report.json`, head + columns for `eval_steps.csv` (long format), all 39 BenchMARL `*.csv` scalars grouped by prefix (train_/eval_/collection_/timers_/counters_), side-by-side column comparison matrix, empirical sizes + load times, production-scale projection (1000 × 5 × 10 → 50k rows / ~3.7 MB per run), question matrix, recommendation, sign-off line.
- **Recommendation:** `long_format` stays opt-in (default off — F5.4 status quo). Cross-run leaderboard questions go through `runs.csv`; single-run drill-down via Streamlit + `eval_steps.csv` opt-in; training internals via BenchMARL native scalars (always-on, BenchMARL writes them anyway).
- **User signs off** before any defaults change. Sign-off boxes embedded at the bottom of the doc.
- **Post-review schema rev** (after first user pass): collapsed report's separate `benchmarl_dir` + `benchmarl_scalars` (single string) into a single `benchmarl: {dir, scalars: [...]}` block, with `dir` pointing at the *inner* BenchMARL run root and `scalars[i]` relative to `dir` (typically `scalars/<name>.csv`). Cleaner enumeration of every native CSV in one place; one resolve idiom (`run_dir / dir / scalars[i]`); no duplicated `<bm_run>` segment in scalar entries. See `domain/models/report.py::BenchmarlLinks`.

#### F5.6 — `multi-scenario sweep` (CLI-level expansion over per-experiment YAMLs) — S

**Design rev (after user review):** dropped the originally-planned `SweepConfig` Pydantic schema. The "1 YAML = 1 experiment" invariant stays; sweeps are pure CLI orchestration over filesystem selection.

- New CLI command: `multi-scenario sweep <input> [--seeds N1,N2,...] [--dry-run] [--max-runs N] [--seconds-per-run S]`.
- `<input>` resolution rules (in order):
  1. Existing regular file → single yaml.
  2. Existing directory → `<dir>/*.yaml`.
  3. Otherwise → glob pattern (Python's `glob.glob`, supports `*`, `**`, `?`, character classes).
  4. Filter to `*.yaml` extensions; error if zero matches.
- Cell semantics:
  - Without `--seeds`: each yaml's own `experiment.seed` is used; one run per yaml.
  - With `--seeds 0,1,2`: cartesian — each yaml × each seed. The yaml's own `experiment.seed` is **replaced** (not augmented) per cell.
  - `experiment.id` from the yaml is kept verbatim. Run-folder differentiation comes from `<exp_id>_s<seed>__<timestamp>` (F1.3).
- `--max-runs N` (default 100) — refuse to launch if expansion exceeds the cap. Exits 2 with cap + actual count.
- `--seconds-per-run S` — print wall-time estimate (`N cells × S sec ≈ total`).
- `--dry-run` — print the expansion (yaml × seed → resulting `<exp_id>_s<seed>` and target run_dir) and exit 0; no runs.
- Without `--dry-run`: runs each cell sequentially via `LocalRunner` with progress lines (`[3/12] running mappo_smoke_s2 → <run_dir>`).
- **No new domain models** — pure CLI orchestration. Glob via `pathlib`/`glob`; load via `ExperimentConfig.from_yaml`; override `experiment.seed`; run via `LocalRunner`.
- **Out of scope (deferred):**
  - **Heterogeneous overrides** (e.g. "mappo with seeds [0..4], ippo with seeds [0..1]") — workaround: two `sweep` invocations with different glob patterns. Lift only when a real use case needs it.
  - **Non-seed overrides** (`scenario.params.<field>` cartesian) — workaround: one variant yaml per cell (which is the "1 yaml = 1 experiment" principle).
  - **Parallel execution** — see F6.7.
- Tests:
  - Glob expansion: 4 yamls in dir → 4 cells; same dir × 3 seeds → 12 cells.
  - Single yaml + seeds; wildcard pattern.
  - Size cap raises non-zero exit with cap message.
  - `--dry-run` prints expansion without running.
  - Real sweep (slow): 2 yaml × 2 seeds → 4 run folders produced under storage path.

#### F5.7 — Resume from crash (local only) — M ✅

**Scope decision:** local-only. OVH and other distributed runners explicitly do **not** support resume — see "Capability flag" below. Rationale: most local crashes are environmental (laptop sleep, terminal close, OOM kill) and recovery saves real iteration time; OVH AI Training jobs are short and reliable, so the (resume infra effort) ÷ (compute saved) ratio doesn't justify the OVH path. If a real OVH crash pattern emerges (long jobs hitting transient failures), F6.8 would add it.

- **BenchMARL checkpoint enabling.** `BenchmarlBaseAdapter._experiment_config` sets `bm.checkpoint_interval = checkpoint_interval_iters × frames_per_batch` and `bm.checkpoint_at_end = True` for non-smoke runs (`*_smoke` exp_ids stay off — no point checkpointing 1-iter runs). New `TrainingSection.checkpoint_interval_iters: int = 10`.
- **`Algorithm` Protocol extension:** `train(env, cfg, run_dir=..., resume_from=...)` — optional `resume_from: Path | None = None`. `BenchmarlBaseAdapter.train` honours it via `experiment.load_state(resume_from)` between construction and `experiment.run()`. Other algorithm impls can ignore the kwarg.
- **`Runner` Protocol extension — capability flag:** `supports_resume: bool` class attribute. `LocalRunner.supports_resume = True`. Future runners (`OvhRunner` at F6.2, any SLURM / Modal / k8s adapter) set their own; the resume CLI checks the flag and refuses with a helpful message if `False`. Generalises cleanly per runner.
- **`LocalRunner.run` extension:** optional `resume_from: Path | None = None` threaded through `ExperimentService.run` to `Algorithm.train`.
- **`multi-scenario resume <run_dir>` CLI command:**
  - Loads `<run_dir>/input/config.json` → `ExperimentConfig`.
  - Builds runner via factory; refuses (exit 2) if `runner.supports_resume is False`.
  - Loads `run_state.json`; refuses (exit 2) if state is `DONE` (nothing to resume).
  - Locates latest BenchMARL checkpoint via `output/benchmarl/<bm_run>/.../checkpoints/*.pt` (mtime-newest).
  - Records state transitions: existing → `CRASHED` (if not already) → `RESUMED`. Both transitions persisted for the audit log.
  - Calls `runner.run(cfg, run_dir, resume_from=checkpoint_path)`. Service continues `RESUMED → RUNNING → DONE`.
- Tests:
  - Unit: capability flag exposed on `LocalRunner`; resume CLI refuses non-local cfg + DONE state.
  - Unit: state-machine `RUNNING → CRASHED → RESUMED` valid (already covered by F1.4).
  - Slow: train N iters, kill mid-run, relaunch via `multi-scenario resume`; verify final metrics agree with uninterrupted N-iter baseline within tolerance.
- **Out of scope:**
  - OVH resume (capability flag returns False for `OvhRunner`).
  - Resuming after `state == DONE` (refuse).
  - Cross-yaml resume (different config than original — that's a fresh run, not resume).

#### F5.8 — Eval-only mode — S ✅

- `multi-scenario eval <run_dir> [--episodes N] [--name TAG]` re-evaluates a trained policy without retraining.
- **Flow:** loads `<run_dir>/input/config.json`, optionally overrides `cfg.evaluation.episodes`, locates the latest BenchMARL checkpoint, reconstructs `Experiment` via `reload_from_file`, runs `BenchmarlBaseAdapter.evaluate` (reuses F2.4.3 aggregation), scores via `CommonMetricsBundle`, writes `<run_dir>/output/eval_runs/<TAG>.json` (default TAG = `eval_<UTC_timestamp>`).
- **`EvalRunRecord`** domain model in `domain/models/eval_run.py` — mirrors `ExperimentResult` (flat metrics dict on the wire, list[MetricRecord] in memory) plus eval-only fields: `eval_id`, `eval_timestamp`, `policy_checkpoint`.
- **No capability flag** — eval-only is by-design a local-machine action even when the original training ran on OVH (the user pulls results down first, then evals locally). The CLI just verifies the run-dir has the artefacts; original `cfg.runtime.runner.type` is irrelevant.
- **`LocalStorageAdapter.save_eval_run`** added (off the Storage Protocol per F1.9 minimalism).
- **Multiple eval runs coexist** as separate files keyed by tag (e.g. `post_hoc.json` + `eval_20260507_1500.json` + `eval_20260507_1600.json` side-by-side).
- Tests: 5 (2 refusal: missing config / missing checkpoint; 3 slow happy paths: full e2e with `--episodes` override, default timestamped name, multiple coexisting eval runs).
- **Out of scope (deferred):** OOD overrides like `--scenario-params n_targets=10` (F5.8.1 if needed); multi-checkpoint eval (eval at multiple training stages); metric-bundle re-scoring.

---

### Phase 6 — OVH runner & secrets

#### F6.1 — `FernetSecretsAdapter` — S ✅

- `FernetSecretsAdapter` (`adapters/secrets/fernet.py`): Fernet (AES-128-CBC + HMAC-SHA256) with PBKDF2-HMAC-SHA256 (100k iters, fixed salt) deriving the 32-byte key from a passphrase. Methods: `encrypt(secrets, passphrase) -> str`, `decrypt(blob, passphrase) -> dict`, `encrypt_for_env(secrets, passphrase) -> dict[str, str]` (returns the `MS_ENCRYPTED_SECRETS` / `MS_SECRETS_PASSPHRASE` env-var pair), `decrypt_from_env() -> dict` (reads from `os.environ`; does NOT mutate it — caller decides how to inject).
- **Generic, not LLM-specific.** Naming dropped rendezvous_comm's `LERO_*` env vars — secrets layer is reusable for any future remote-job credential.
- **Threat model documented in module docstring:** protects against bystanders glancing at job specs / S3 / `ovhai job get` output, NOT against malicious cloud providers (they see both blob + passphrase). Real defence = rotate passphrase per job. For stronger threat models, plug in a different adapter that talks to a real KMS — the interface is small enough to swap.
- **`cryptography>=41`** added to `pyproject.toml` deps.
- **Port from rendezvous_comm:** `secrets_util.py` (refactored to a class; dropped the `.env`-shape filter — caller decides which keys to encrypt).
- Tests: 7 covering round-trip, wrong passphrase, empty dict, env-var key names, env-pipeline round-trip with monkeypatched `os.environ`, missing-blob → `{}` no-op, missing-passphrase → clear error.

**LLM context (worth being explicit):** the framework has **zero LLM code so far**. F6.1 is pure secrets infrastructure laid down before its consumer (LERO at Phase 9) lands, so OVH job submission (F6.2) has the credential-shipping plumbing ready. LLM client / prompt registry / disk cache / scenario_patch.exec all stay parked at Phase 9.

#### F6.2 — Port `ovh.py` (cleaned) — M

- Adapter implementing `Runner`. Builds the job spec, uploads code via S3, submits via `ovhai`, polls for completion. **Trailing-slash fix and per-experiment S3 prefix already baked in** (known gotchas, retain).
- **`supports_resume = False`** (capability flag from F5.7) — OVH-resume is intentionally out of scope; the `multi-scenario resume` CLI refuses with a helpful message. If a real long-running OVH crash pattern emerges later, F6.8 would add it.

#### F6.3 — `S3StorageAdapter` — S

- Mirror local layout under `s3://<bucket>/<prefix>/experiments/...`. TDD with `moto` (mocked S3).

#### F6.4 — Code uploader — S

- Rsync-style upload of `src/` + `experiments/<scenario>/<exp_type>/configs/` to S3 before job submit.

#### F6.5 — End-to-end OVH smoke — S (manual)

- Submit `discovery_mappo_smoke.yaml`; verify results land at the right prefix; pull back via `S3StorageAdapter`.
- **Manual demo** — gated on user confirmation that they want to spend an OVH credit.

#### F6.6 — Headless video handling + `regenerate-videos` CLI — S

**Background:** F2.11 records before/after MP4s inline during training using VMAS Pyglet rendering. OVH AI Training containers are headless (no OpenGL/X11) → any non-smoke run on OVH would crash inside `VideoRecorder.record()` (confirmed `pyglet.gl` import error in `rendezvous_comm/results/.../run.log`). This feature makes OVH runs complete cleanly and reproduces the videos locally after pulling results back.

**Two bundled changes (each XS, single feature for tight coupling):**

> **Checkpoint enabling moved to F5.7.** F5.7 owns the `bm.checkpoint_interval` + `bm.checkpoint_at_end = True` plumbing for non-smoke runs (it's load-bearing for resume). F6.6 inherits that — by the time F6.6 is implemented, checkpoints will already be written for non-smoke runs, and the regenerate-videos command can load them.

1. **Fail-soft `VideoRecorder` invocation** — wrap each `VideoRecorder().record(...)` call in `BenchmarlBaseAdapter.train()` with try/except. On failure, emit a warning: `"Video {before|after}_training skipped on headless host: <error>. Regenerate locally with 'multi-scenario regenerate-videos <run_dir>' after pulling results."` Training completes; `report.links.videos.{before,after}_training` resolves to `null`.
2. **`multi-scenario regenerate-videos <run_dir>` CLI command:**
   - Reads `<run_dir>/input/config.json` (cfg) and the BenchMARL checkpoint at `<run_dir>/output/benchmarl/<bm_run>/.../checkpoints/*.pt` (latest by mtime).
   - Rebuilds the experiment with the same seed via `BenchmarlBaseAdapter.build_experiment(cfg, run_dir)` — random-init policy → records `before_training.mp4`.
   - Loads the checkpoint state dict into `experiment.policy` → records `after_training.mp4`.
   - Re-runs `ReportBuilder.build` and overwrites `report.json` so `videos.{before,after}_training` populate.

**Optional polish (lower priority):** auto-detect OVH via `OVH_AI_TRAINING_*` env vars and short-circuit the recorder up front, so we skip the pyglet crash overhead on each OVH run.

**Reproducibility caveat to document:** post-hoc regeneration is faithful for MLP (deterministic given seed) but may diverge subtly for algorithms with eval-time stochasticity that BenchMARL handles internally. The "after" video reflects the *saved* checkpoint; if `checkpoint_at_end=True` saves slightly before the final eval (e.g., during cleanup), the policy may be one update behind the run's final eval metrics. Acceptable trade-off; document it.

**TDD:**

- Unit: a fake `VideoRecorder` that raises → `train()` returns the experiment, warning logged, training metrics unaffected.
- Integration: regenerate-videos against a fixture run folder (with checkpoint) → both MP4s land + `report.json` updated.
- OVH smoke (manual, F6.5 follow-up): submit a non-smoke config, confirm crash-free completion, pull back, run `regenerate-videos`, confirm both videos generated.

#### F6.7 — Parallel sweep on OVH — S

**Background:** F5.6's `multi-scenario sweep <input>` runs cells **sequentially** locally — one cell at a time through `LocalRunner`. On OVH, each cell is naturally a separate AI Training job and would block-by-block waste credits when run sequentially. F6.7 lifts F5.6's expansion to OVH-parallel.

**Scope when activated:**

- New CLI flag on `sweep`: `--runner ovh` (default `local`). When set, each expanded cell becomes one OVH AI Training submission via `OvhRunner` (F6.2) instead of an in-process `LocalRunner.run`.
- **Submission mode:** "fire and forget" — submit all cells, print job IDs, return. The user polls / pulls results via separate commands (or the Streamlit dashboard reads them from S3).
- **Per-cell isolation:** each cell gets its own S3 prefix (`s3://<bucket>/<prefix>/experiments/<scenario>/<exp_type>/<run_id>__<ts>/`). Avoids the trailing-slash collision gotcha (project memory).
- **Concurrency cap:** new flag `--max-parallel N` (default unlimited). When set, batches the submissions so no more than N jobs are queued at OVH at any time. Useful for credit budgeting.
- **Optional follow-mode:** `--follow` polls the OVH job statuses and prints progress; without it, exits as soon as all jobs are submitted. Polling cadence configurable (`--poll-interval 30`).
- **Validation:** before any submission, verify `OvhRunner` is configured (env vars / config file). Print the cell count + estimated cost (cells × per-job-cost-estimate) and require explicit `--yes` to confirm submissions over a configurable cost cap (default 10 credits).

**Tests:**

- Unit: dry-run with `--runner ovh --dry-run` prints submission plan (cell count, per-cell S3 prefix, estimated cost) without actually submitting; mocks `OvhRunner` to assert no real network calls.
- Integration (mocked): `--runner ovh` submits N cells via a fake `OvhRunner` that records calls; assert N submissions with distinct S3 prefixes.
- Manual OVH smoke: 2-cell sweep with real `OvhRunner`; verify both runs land at expected S3 prefixes.

**Out of scope (deferred):**

- Auto-retry on OVH job failures.
- Cross-cell dependencies (one cell's output feeding another's input).
- Live result streaming back during execution (today's pull-on-completion is fine).

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
- **Prerequisite check:** if any matrix entry needs non-default MLP `num_cells`, `activation_class`, or a separate critic config, **F2.4.2 must be implemented first** (it's a deferred placeholder until that need is real).

#### F8.3 — Run the matrix — M (compute-bound)

- **Prerequisite:** verify **F2.4.2** is done if matrix configs use any non-default model architecture knobs. F2.4.2 is a deferred placeholder; without it, BenchMARL will use its built-in MLP defaults (which may or may not match your matrix definition).
- Locally for tiny smoke; OVH for real. Collect to one master CSV.

#### F8.4 — Comparison report — S

- Streamlit page or notebook → per-scenario leaderboard, best baseline per scenario.
- **Output:** identifies the best baseline candidate per scenario → input to Phase 9 LERO.
- **Prerequisite check:** if the report needs cross-run training-curve comparisons (e.g. "compare MAPPO vs IPPO M2 at iter 50 across seeds in one table/plot"), **F5.2.1 must be implemented first**. F5.2 ships only `record_type=final` rows; the eval-step rows live in F5.2.1 and are a deferred placeholder. Per-run training scalars are still readable directly from each run's `output/benchmarl/.../scalars/eval_*.csv` for single-run views — only cross-run aggregation needs F5.2.1.

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

Final user-facing project docs. Replaces the placeholder stubs from F0.5 / F0.6 with the real thing once everything else is stable.

- **`README.md`** — quick-start (install, run a smoke config, browse results in Streamlit), 1-paragraph project description, links into `docs/`. Replaces the F0.6 stub.
- **`docs/architecture.md`** — the hexagonal layout (ports / adapters / application / domain), what each layer can and can't depend on (the F1.12 isolation rule), the lifecycle of one experiment run (F1.11 → F2.x slice).
- **`docs/scenarios.md`** — one section per scenario (discovery / navigation / flocking / transport): goal, agents, default params, M1 semantics, M6 semantics where applicable, special cases (e.g. flocking has no natural M1 — see F4.2).
- **`docs/run_layout.md`** — §3.5.x conventions formalised: run_id format, folder layout per run (the §3.5.2 tree), cross-run files (`runs.csv` / `runs.json`), provenance fields, run-state machine.
- **`docs/cli.md`** (or expand README) — every `multi-scenario` subcommand with one example each: `version`, `run`, `validate`, `consolidate`. Cross-reference what each emits.
- **Cross-link audit:** every `docs/*.md` should link to `README.md` and to its sibling docs; `README.md` should link to all of `docs/`. No orphans.

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

#### F10.6 — Scaffolding cleanup pass — XS

Some artefacts produced *during* development serve a one-time purpose — informing a decision, generating empirical numbers for a sign-off, validating a deferred choice — and become dead weight once the final code + F10.3 docs land. Sweep them out before extraction (F10.4).

**Removal candidates (review each before deleting; delete only when the final state preserves the information they carried):**

- **`docs/csv_format_decision.md`** + **`scripts/f5_5_format_decision.py`** — the F5.5 long-vs-summary decision. Once `long_format` defaults are locked in code and `docs/run_layout.md` documents the artefact set, the empirical doc is redundant. Re-runnable scripts for one-off decisions don't justify a permanent home.
- **Sub-feature placeholder sections in `implementation_plan.md`** (F2.4.2, F2.10.1, F5.2.1, F6.6) — once each is implemented (or definitively skipped), fold the resulting state into the relevant final doc and remove the placeholder.
- **Any `_<exp_id>` scratch folders under `experiments/<scenario>/<exp_type>/`** (e.g. `_f5_5_decision`) — these are temp dirs from reproducer scripts; should already be auto-cleaned, but spot-check.
- **Stale TODO comments** referencing deferred features whose triggers have since fired or been cancelled.

**What to keep** (do *not* remove just because they're "phase-tagged"):

- Smoke YAMLs under `experiments/<scenario>/<exp_type>/configs/<algo>_smoke.yaml` — these are CI fixtures, not dev scaffolding.
- The `.pre-commit-config.yaml` / `.markdownlint.json` / `.gitignore` / pyproject.toml — production tooling.
- All test fixtures, even the ones that were written to drive a single feature (regression value).

**Demo:**

- `find docs scripts -maxdepth 2 -type f \( -name '*decision*' -o -name 'f5_*' \)` — should return no surviving dev-time decision artefacts.
- `find experiments -type d -name '_*'` — should return no scratch run folders.
- `grep -rn 'TODO\|FIXME' src tests` — every remaining occurrence is justified or also deleted.

**Order:** run after F10.5 (comment cleanup) and after F10.3 (documentation pass) — those passes formalise what's worth preserving, so anything they didn't promote to canonical state is fair game here.

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
