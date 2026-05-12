# Multi-Scenario Cooperative MARL — Implementation Plan

> **Status:** draft v3 (2026-05-09) — Phases 0-7 implemented; Phases 8-11 reframed for reproducibility-then-LERO-then-extraction.
> **Companion docs:** [`plan.md`](plan.md) (scenario rationale & descriptions); [`docs/_drafts/F8_F11_plan_draft.md`](docs/_drafts/F8_F11_plan_draft.md) (1012-line agent-generated draft with full architecture deep-dive + self-criticism).
> **Folder name `multi_scenario/` is a placeholder** — will be renamed to **`coopvmas`** at F10.6.
> **Locked decisions (2026-05-09):** see `~/.claude/.../memory/project_coopvmas_decisions.md`. Headlines: name=coopvmas, license=GPL-v3, broker=LiteLLM, docs=mkdocs-material, GitHub=personal account (afin), fresh-import extraction.
> **Changes from v2:** Phase 8 narrowed from "ER1 across 4 scenarios" to "reproduce ER1+S3b-local on discovery"; Phase 9 lifted from "placeholder" to full LERO implementation; Phase 10 expanded to docs+naming+extraction; new Phase 11 holds the cross-scenario campaign (deferred scope).

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

#### F6.2.1 — OVH submission command-construction corrections — XS ✅

Found during pre-F6.5 prep: F6.2's mock-only tests didn't catch real-world `ovhai` syntax errors. Five corrections applied:

- **`--gpu V100S`** → **`--flavor ai1-1-gpu`** (separate `--gpu N` for count). Added `OvhJobConfig.flavor: str = "ai1-1-gpu"`; `gpu_type` stays for cost-registry display only.
- **`:RO` / no permission for results** → **`:ro` (lowercase)** / **`:rwd`** (writable + deletions sync back).
- **Raw runner command** → **wrapped in `bash -c "..."`** with `export HOME=/tmp && pip install -e {mount_code} && cd {mount_code} && python -m multi_scenario.cli run {yaml_path_in_container}`. `HOME=/tmp` is mandatory — pip can't write to /workspace mount on OVH.
- **Volume mounted at code root with no path** → mounted at `{mount_code}` and `{mount_results}` substituted into the template.
- **Required `yaml_path_in_repo`** constructor arg on `OvhRunner` — points at the experiment YAML relative to uploaded code root (e.g. `experiments/discovery/baseline/configs/mappo_smoke.yaml`); container resolves it under `mount_code`. Missing at submit-time → clear `OvhJobError`.

Tests updated: `test_ovh.py` now asserts the corrected arg shape (`--flavor`, `:rwd`, `bash -c`, `HOME=/tmp`, `pip install -e /workspace/code`) plus a new test that omitting `yaml_path_in_repo` raises. `configs/ovh.yaml.example` updated with `flavor` field + new template runner.

**Pyproject deps audit (also done here):** added `imageio[ffmpeg]>=2.30`, `torchrl`, `tensordict` as direct deps. Previously these were only transitively installed via `benchmarl`/`vmas` — the audit caught them as direct imports in F2.11 (video), F5.4 (long-format), and benchmarl_base; declaring direct usage prevents F10.4 extraction breakage.

#### F6.2 — Port `ovh.py` (cleaned) — M ✅

Three coupled deliverables landed together (framework wiring + submit/poll plumbing). Result-sync from S3 (F6.3) and code upload to `bucket_code` (F6.4) deferred — `OvhRunner.run()` reads `<run_dir>/output/metrics.json` directly, assuming someone has synced it.

- **`OvhJobConfig`** (`domain/models/ovh_job_config.py`) — strict Pydantic model loaded from `configs/ovh.yaml`. Holds region / image / GPU / buckets / mounts / poll cadence / timeout / known-GPU registry with cost estimates.
- **`OvhClient`** (`adapters/runners/ovh_cli.py`) — thin subprocess wrapper around the `ovhai` binary: `submit / get / list_jobs / logs / stop / check_available`. Mockable via an injected `runner` callable. Parses both plain-text and JSON `ovhai` output; recognises `DONE / FAILED / KILLED / ERROR` as terminal states.
- **`OvhRunner`** (`adapters/runners/ovh.py`) — implements `Runner`; `name="ovh"`, `supports_resume = False` (per F5.7). `run()` builds the `ovhai job run` arg list (per-experiment S3 prefix isolation, no trailing slash), submits, polls until terminal, raises `OvhJobError` on non-DONE with logs tail, otherwise loads `ExperimentResult` from disk. Encrypted secrets via F6.1's `FernetSecretsAdapter.encrypt_for_env` ride along as `--env` flags when configured.
- **`configs/ovh.yaml.example`** — commented template for the deployment config.
- Tests: 17 mock-only (10 OvhClient subprocess-mocked tests + 7 OvhRunner orchestration tests). No real OVH calls; F6.5 covers the manual end-to-end smoke when the user OKs spending a credit.
- **Out of scope (folded forward):**
  - **Code upload** to `bucket_code` → F6.4 (rsync helper).
  - **S3 result sync** back to local `run_dir` → F6.3 (`S3StorageAdapter`).
  - **OVH resume** → not implemented; capability flag refuses cleanly.

#### F6.3 — `S3StorageAdapter` — S ✅

- `S3StorageConfig` (`domain/models/s3_storage_config.py`) — Pydantic strict: `bucket`, `prefix`, `region`, optional `endpoint_url` (set to OVH Object Storage endpoint when targeting OVH; left None for AWS S3). `from_yaml` loader.
- `S3StorageAdapter` (`adapters/storage/s3.py`) — implements the 8-method `Storage` Protocol via boto3. Keys map to `<prefix>/<run_dir.name>/<rel>` so the §3.5.2 layout is preserved one-to-one under S3.
- **Sync helpers** (off-Protocol per F1.9):
  - `sync_to_local(run_dir, local_dir)` — paginated `list_objects_v2` + per-key `get_object` writes; recreates the per-run folder tree locally.
  - `sync_from_local(local_dir, run_dir)` — symmetric upload (used by F6.4 code uploader).
- **`OvhRunner` extension:** new optional `s3_storage: S3StorageAdapter | None` constructor arg. When wired, `run()` calls `s3_storage.sync_to_local(run_dir, run_dir)` before reading `metrics.json`. Without it, behaviour is unchanged from F6.2 (user hand-syncs).
- **Deps:** added `boto3>=1.30` (runtime), `moto>=5.0` (dev) to `pyproject.toml`.
- Tests (`tests/integration/storage/test_s3.py`, 9): protocol satisfaction, key construction, round-trip per artefact, sync-to-local + sync-from-local, YAML round-trip. All moto-mocked S3 — no AWS calls.
- **Out of scope:** multipart upload (run-folder files are small); `make_storage("s3")` factory wiring (direct construction only); `save_report` / `save_eval_episodes` etc. on S3 (same F1.9 minimalism rule — add when needed).

#### F6.4 — Code uploader — S ✅

- `CodeUploader` (`adapters/storage/code_uploader.py`) — walks a curated include set under the repo root, applies fnmatch exclude patterns, uploads each surviving file to `s3://<bucket>/<prefix>/<rel-from-repo-root>` via `S3StorageAdapter.put_file`.
  - Defaults: `include_dirs=("src/multi_scenario", "experiments", "configs")`, `include_files=("pyproject.toml", "README.md")`. All overridable.
  - Excludes: `__pycache__`, `*.pyc/.pyo`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, `*.egg-info`, `*/results/*`, `*/output/*`, `*/logs/*`, per-run folders (`<run_id>__<timestamp>` pattern), `.DS_Store`.
  - `dry_run=True` returns the would-upload list without touching S3.
- `S3StorageAdapter.put_file(key, body)` — flat-upload helper used by the code uploader (no run-dir transform).
- New CLI: `multi-scenario upload-code <s3-config.yaml> [--repo-root PATH] [--dry-run]`.
- **Decoupled from job submission** — the user runs `upload-code` once per code change; submitted jobs reuse the already-uploaded code in `bucket_code`. Avoids re-uploading on every job submit.
- Tests: 7 CodeUploader unit tests (curated set / pycache+results excludes / per-run-folder excludes / dry-run no-op / empty repo / custom includes / pattern sanity) + 2 CLI tests (`--dry-run` lists files without S3 calls; full upload puts files at expected keys). All moto-mocked.
- **Out of scope (deferred):** real rsync diffing (hash-based skip-unchanged); compression / tar-and-upload; per-experiment subset uploads.

#### F6.5 — End-to-end OVH smoke — S (manual) ✅

**Verified end-to-end on real OVH (2026-05-07):** mappo discovery 1-iter smoke submitted via `ovhai job run`, ran in `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`, results synced back via OVH FINALIZING, every artefact in §3.5.2 layout downloaded + parsed cleanly. ~90s wall time on `ai1-1-cpu`, ~4 jobs × cpu-flavor ≈ negligible cost.

- **Procedure documented in `docs/ovh_smoke_checklist.md`** (F6.5 deliverable). One-time prereqs in `docs/ovh_setup.md` (install `ovhai` Go binary, `ovhai login`, create buckets, generate S3 keys for boto3 / or skip them when using `ovhai bucket` directly). Both evergreen — F10.3 absorbs them into the canonical docs.
- **Two bugs caught + fixed during F6.5** (would have blocked any production OVH run):
  - `pyproject.toml`: `requires-python = ">=3.11"` → `">=3.10"`. The `pytorch/pytorch:*-runtime` images ship Python 3.10; we use no 3.11-only syntax.
  - `cli.py`: missing `if __name__ == "__main__": main()` — `python -m multi_scenario.cli` was a no-op (typer app never invoked, exit 0). Added the standard guard.
- **Per-experiment S3 prefix isolation confirmed working:** results landed at `rendezvous-results@GRA/multi_scenario_smoke/<run_id>__<ts>/...` — no trailing slash, no collision with other prefixes.
- **OVH FINALIZING auto-sync confirmed working:** the volume mount `:rwd` permission ports the entire local `/workspace/results` tree back to S3 at job end. We did NOT need F6.3's `S3StorageAdapter.sync_to_local` for this smoke (downloaded via `ovhai bucket object download` directly). F6.3's boto3 path remains the option for users who set up AWS credentials.
- **Smoke artefact:** `experiments/discovery/baseline/configs/mappo_ovh_smoke.yaml` (added in this feature) — same shape as `mappo_smoke.yaml` but `runtime.storage.path: /workspace/results` so the run-folder writes hit the rwd-mounted volume.

#### F6.6 — Headless video handling + `regenerate-videos` CLI — S ✅ (cross-device bug → F6.6.1)

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

#### F6.6.1 — Cross-device checkpoint loading for regenerate-videos — moved to F8.2.B

Bug discovered during the ER1 dry-run; re-scoped under F8.2.x as a hard prerequisite for the official multi-seed run. See **F8.2.B** below for the full spec.

**TDD:**

- Unit: a fake `VideoRecorder` that raises → `train()` returns the experiment, warning logged, training metrics unaffected.
- Integration: regenerate-videos against a fixture run folder (with checkpoint) → both MP4s land + `report.json` updated.
- OVH smoke (manual, F6.5 follow-up): submit a non-smoke config, confirm crash-free completion, pull back, run `regenerate-videos`, confirm both videos generated.

#### F6.7.1 — Friendly error when `ovhai` CLI missing — XS ✅

User-flagged usability gap: previously, missing `ovhai` binary surfaced as a bare `FileNotFoundError` traceback. Now:

- **`OvhClient.ensure_available()`** — calls `check_available()`; raises `OvhCliError` with the canonical install-instructions message (`curl -sSf https://cli.bhs.ai.cloud.ovh.net/install.sh | bash` → `ovhai login` → `docs/ovh_setup.md`).
- Called at the top of every CLI/runner entry point that needs the binary: **`OvhRunner.submit()`** (covers programmatic use) and **`_sweep_run_ovh`** in the CLI (covers `multi-scenario sweep --runner ovh`).
- Tests: `OvhClient.ensure_available()` raises with install URL when missing / no-op when present; CLI sweep with `check_available` mocked False → exit 2 + `cli.bhs.ai.cloud.ovh.net/install.sh` in stderr.
- `docs/ovh_setup.md` updated with a note explaining the friendly-error fallback.

#### F6.7 — Parallel sweep on OVH — S ✅

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

#### F7.3.1 — Educational scalar/metric explanations — moved to F8.2.E

FE plumbing for in-app metric/scalar tooltips. Re-scoped under F8.2.x because students/collaborators need it before the official multi-seed run lands so they can self-serve. See **F8.2.E** below.

#### F7.3.2 — In-app documentation hub — moved to F8.2.F

"📖 Docs" sidebar entry + inline doc deep-links. Same rationale as F7.3.1. See **F8.2.F** below.

#### F7.4 — Page 3: cross-experiment comparison — M

- Bar / box plots comparing algorithms across scenarios. Selectable metric, scenario, algos.
- **Port:** `pages/4_*.py` (cross-exp comparison).

#### F7.5 + F7.6 — Submit page (merged) — M

> The original separate F7.5 (local submission) + F7.6 (OVH submission) were merged into a single **Submit** page under the Experiments parent group. The form is 90% shared between runners; splitting forced "pick page first, then change runner" which is backwards. A runner radio toggle inside the page reveals OVH-specific fields when relevant.

**Workflow shape (5 always-visible step cards):**

1. **Pick** — scenario / folder / config cascade picker over `experiments/<scenario>/<folder>/configs/*.yaml`.
2. **Inspect & edit** — pre-filled form (Identity / Scenario / Algorithm / Training / Evaluation / Runner / Storage) inside an expander; dirty detection + "Modified fields:" summary.
3. **Save** — auto-skipped if clean; "Save as new" forced if edits exist (never overwrites the source).
4. **Preflight** — runner-aware LED panel (config schema valid · storage path writable · OVH CLI installed · results bucket reachable · **code matches OVH bucket** · per-run prefix not occupied · cost cap not exceeded).
5. **Submit** — gated until preflight all-green AND no unsaved edits.

**Phasing:**

- **Phase A (DONE)**: workflow shell, all forms, validation banner, preflight LEDs (mocked), Download YAML button.
- **Phase B**: real local-runner preflight checks (config schema, storage writable) + Submit button wired to `LocalRunner` (synchronous v1 with `st.spinner`; threaded log-tail v2 if pain).
- **Phase C**: OVH submission via `OvhRunner.submit()` + status polling + auto-regen videos (already wired in `OvhRunner.run()` post-pullback) + real OVH preflight checks (`OvhClient.ensure_available`, `boto3.head_bucket`, code-hash compare against the `.code_hash` blob `upload-code` writes).

State machine: `SubmitState` dataclass with derived `is_dirty`, `has_preflight_passed`, `active_step`, `can_submit` properties; session-state-backed.

**Save policy**: writes go alongside the source YAML in the same `configs/` dir; never overwrites the source (button disabled if name matches). Default new name: `<original_stem>_v2.yaml`.

**Code-vs-bucket consistency check (Phase C)**: hashes local `multi_scenario/` package via the existing `CodeHasher` (F2.7), compares to a `.code_hash` blob written by `multi-scenario upload-code` to the OVH code bucket. Mismatch → 🔴 with the exact `multi-scenario upload-code` command embedded in the error.

#### F7.7 — Frontend hex-architecture compliance, extensibility, validation, tests — M

The original F7.7 was a small "smoke-data + visual nits" pass. After F7.5/F7.6
landed the Submit workflow it became clear the frontend needed a deeper review
to meet the project's clean-code + hex-architecture bar. F7.7 grew to cover
that — the original polish work moves to Phase 6 below.

**Phase 1 — Hex-architecture compliance** (highest priority):
- **F7.7.A1** — Extend `OvhClient` with bucket verbs (`bucket_list`,
  `bucket_list_objects`, `bucket_object_exists`, `bucket_get_object`) that
  shell through `ovhai`. Reuses existing `_run` / `JobInfo` patterns; no new
  ports.
- **F7.7.A2** — Refactor the 4 OVH preflight probes (`_probe_results_bucket`,
  `_probe_code_hash`, `_probe_yaml_in_bucket`, `_probe_prefix_collision`) to
  call `OvhClient` instead of `boto3` directly. Drops the AWS-credentials
  requirement for preflight and the `boto3` import from the frontend layer
  entirely (the frontend should never import boto3 — that's a hex violation).
- **F7.7.A3** — Drop `OvhJobConfig.s3_endpoint_url` + `s3_endpoint()`. After
  A2 they have no callers; `S3StorageAdapter` reads its endpoint from a
  separate `S3StorageConfig` (right place). STRICT model rejects stale
  YAMLs with a clear migration error.

**Phase 2 — Extensibility**:
- **F7.7.B1** — Backend listing API in `application/factories.py`:
  `available_scenarios()`, `available_algorithms()`, `available_storages()`,
  `available_runners()`. Add `Algorithm.default_params()` to the port so
  every algorithm declares its UI-visible knobs. Pure additions; no
  behaviour change.
- **F7.7.B2** — Data-driven `forms.py`: replace `SCENARIO_FORMS` /
  `ALGORITHM_FORMS` dispatch tables with one generic
  `render_params_from_defaults(schema, overrides)` that picks the widget
  from the default value's Python type. Submit page reads
  `available_scenarios()` etc. dynamically. Adding a new scenario now
  requires zero frontend changes.

**Phase 3 — Test coverage**:
- **F7.7.C1** — `streamlit.testing.v1.AppTest` end-to-end tests for the
  Submit workflow (no browser). Covers pick → edit → save → preflight →
  submit for both runner targets. Local + OVH happy paths + cascade-on-
  missing-ovh.yaml all reachable through `at.session_state`.
- **F7.7.C2** — Add `pytest-bdd` to dev deps; one `submit.feature` Gherkin
  file with the two highest-value journeys (OVH submit happy path; missing-
  credentials error). Step definitions reuse the C1 fixtures.
- **F7.7.C3** — Per-probe unit-coverage audit. Every probe gets happy +
  failure paths against a fake `OvhClient`. Target >85% coverage on
  `preflight.py`.

**Phase 4 — Config validation hardening**:
- **F7.7.D1** — Field validators (`gt`/`ge`/`le`) on every numeric in
  `domain/models/config.py`; `Literal["cpu", "cuda"]` on `device`; pattern
  on `experiment.id`; cross-field `model_validator` enforcing
  `minibatch_size <= frames_per_batch`. Registry-aware type checks
  (`scenario.type ∈ available_scenarios()`, etc.) so an unknown type fails
  at parse time, not at runner-instantiation time.

**Phase 5 — Audit**:
- **F7.7.E1** — Mock / placeholder / phase-marker cleanup. Update stale
  "Phase A" / "Phase B" / "Phase C" wording in `preflight.py`,
  `submit.py`, `code_uploader.py`. Delete the placeholder
  `render_navigation_params` / `_transport_params` / `_flocking_params`
  shells (subsumed by B2). CI guard test asserts no `Phase B`/`Phase C`
  markers remain in `src/`.

**Phase 6 — Visual & data polish** (original F7.7 scope, retained):
- **F7.7.K** — Smoke-data regen: regenerate the canonical demo run with a
  longer config (e.g. `max_iters: 5`, `record_video: true`) so the
  dashboard ships with non-degenerate charts/videos out of the box. Drop
  the smoke runs that produced flat M-values once a richer reference exists.
- **F7.7.L** — Sweep visual nits surfaced during F7.1–F7.6 builds — one
  consolidated polish round rather than per-page PRs.

> **Auto-regen videos on OVH pullback (landed early as part of F7.4 review):**
> ``OvhRunner.run()`` now auto-invokes ``application.regenerate_videos`` on
> the local machine after results sync back from S3 — but only when
> ``cfg.runtime.runner.params.record_video`` is true AND no MP4s came back
> from the container (the in-job Pyglet renderer fails fail-soft on
> headless OVH hosts per F6.6). Failure is swallowed with a logger.warning;
> training success ≠ video success. CLI ``regenerate-videos`` shares the
> same ``application.regenerate_videos`` core so behaviour stays consistent.

---

### Phase 8 — Reproducibility validation (discovery only)

> **Scope reset (2026-05-09).** Phase 8 was originally "ER1 across 4 scenarios + heuristic baselines". After the F8/F9/F10/F11 planning round, that scope moved to **F11**; F8 narrows to *reproducing the rendezvous_comm headline numbers on `discovery`* (ER1 baseline + S3b-local LERO). The full draft is at `docs/_drafts/F8_F11_plan_draft.md` (1012 lines). User-locked decisions are in `~/.claude/.../memory/project_coopvmas_decisions.md`.
>
> **Reproducibility threshold (locked):** ±10% absolute on M1 AND within 1.5σ of rendezvous_comm seed-mean. LERO reproducibility is the success gate; ER1 is the reference baseline.
>
> **A note on `runner.type` and OVH.** Every YAML in F8 has `runtime.runner.type: local` even when we plan to submit it to OVH. That's the F7.7.A2 hex-architecture rule: `runner.type` describes what runs *inside* the host (LocalRunner reads the YAML and drives BenchMARL); the OVH-vs-local *submit* choice is a separate, runtime-level decision (`multi-scenario sweep --runner ovh ...` or the Submit page's submit-target radio). Same YAML, different orchestrator. F8 sub-phases default to OVH submission for compute-cost reasons (10M-frame CPU runs are slow on a laptop), but every YAML stays runner-agnostic and can run either way.

#### F8.0 — Optional: rendezvous_comm self-replication — XS

Default: **skip**. If F8.4 shows an unexpected delta vs the rendezvous_comm doc, come back here and run `rendezvous_comm/configs/{er1/single_al_lp_sr_cr035, lero/s3b_local_replicate_s{0,1,2}}.yaml` in that repo to set fresh reference numbers. Tabled in `docs/reproducibility/reference_numbers.md` if executed.

#### F8.1 — Port ER1 config to coopvmas YAML schema — S

- Translate `rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml` → `experiments/discovery/baseline/configs/baseline.yaml` (final name TBD with user; suggested: `baseline.yaml` for the canonical reference).
- Tests in `tests/reproducibility/test_er1_config_parity.py` — parametric per-field assertions against the rendezvous_comm source so silent drift is caught.
- Done: `multi-scenario validate experiments/discovery/baseline/configs/baseline.yaml` exits 0; parity test green.

#### F8.2 — Run ER1 ×3 seeds, validate — M

The baseline YAML is **runner-agnostic** by design (per F7.7.A2): `runtime.runner.type: local` means *LocalRunner reads the YAML and drives BenchMARL inside whatever host it lands on*. The local-vs-OVH choice happens at the *submit* layer — same YAML, different orchestrator.

- `scripts/run_er1_reproducibility.py` — thin wrapper over `multi-scenario sweep --seeds 0 1 2 --runner {local|ovh} baseline.yaml`. Default `--runner ovh` because ER1 at 10M frames × 600 envs is ~6-12h CPU per seed locally vs ~3-4h on V100S in parallel; users with beefier local machines can override.
- `scripts/compare_to_reference.py` — reads our `runs.csv` + the hardcoded reference dict (ER1 M1≈0.405); prints PASS/FAIL per the F8 threshold (±10% absolute on M1 AND within 1.5σ of rendezvous_comm seed-mean).
- Streamlit reproducibility page (F8.5.B) shows the same comparison side-by-side.

**Compute budget reminder.** OVH cost (3 seeds × ~3h V100S × €2.10/h) ≈ **€19**. Local-CPU cost is wall-clock (a day-ish) but no money out. Either is fine; pick at run-time, not at YAML-edit time.

**Pre-F8.2 dry-run (2026-05-10) — single-seed reference data.** Before the official ×3-seed run we fired one seed-0 ER1 to validate the post-Stage-1/2/3 OVH pipeline end-to-end on a real workload. Result is recorded here so the multi-seed run has a known baseline to compare against; if the seed-0 metric doesn't reproduce in the official run we'll know something between submit and result-extraction has drifted.

| Field | Value |
|---|---|
| Job ID | `b9f33c8d-b289-4a2b-a352-df51554e706b` |
| S3 prefix | `ms-results@GRA/er1_cr035_s0__20260510_144225` |
| Wall time | ~67 min on V100S (incl. ~10 min pip install) |
| Cost | ≈ €2.5 |
| `M1_success_rate` | **0.3250** (target single-seed: 0.405; miss 19.75% > ±10% threshold) |
| `M2_avg_return` | 5.96 |
| `M3_steps` | 200.0 (= max_steps; episodes ran to limit) |
| `M4_collisions` | 6.17 / episode |
| `M6_coverage_progress` | 0.866 (86.6% targets covered on average) |

**Eval reward trajectory (excerpt):**

```text
iter   0:  −1.528  (random init)
iter  39:   0.780  (learning kicks in)
iter  79:   1.565  (first plateau)
iter 125:   1.688  ← peak
iter 139:   0.998  (regression)
iter 165:   1.172  (final, used for M1 = 0.325)
```

**Diagnosis:** the agent **did** learn (eval reward climbed −1.5 → peak 1.69 — pipeline + hyperparameters fundamentally healthy). The M1 miss is a **checkpoint-selection artefact**: training peaked at iter 125 then oscillated/regressed in the last ~40 iters; the `final` checkpoint that drove the M1 number was at a local trough relative to peak. This is exactly the eval-vs-final-degradation gap that **F8.5.D (Best-checkpoint policy callback)** is designed to fix.

**Open observation — checkpoint retention.** YAML configured `checkpoint_interval_iters: 10` (so we expect ~16 snapshots), but only 3 made it back to the bucket: 9.0M / 9.6M / 10.02M frames (≈ iters 150 / 160 / 167). Earlier checkpoints (incl. the iter-125 peak) appear to have been overwritten by BenchMARL's rolling retention before sync. Implications for the multi-seed run:

- F8.5.D's "best-checkpoint callback" must be in place before the official run, otherwise we re-discover this on every seed.
- Until F8.5.D lands, increasing BenchMARL's `keep_n_checkpoints` (or whatever the retention knob is) is a cheap interim mitigation worth doing in F8.2 directly.

**Late-training oscillation hypotheses (to revisit if multi-seed shows the same shape):**

- LR=5e-5 may be too aggressive once the value function has converged; LR scheduling (cosine or step decay) would dampen this.
- PPO clip-ratio drift / entropy bonus decay schedule.
- CUDA-vs-CPU float-order on tight-margin tasks (we use V100S; rendezvous_comm headline used CPU). Typically ±2-3% impact, not 30% reward swing — so probably not the driver.

**What this means for the official ×3-seed F8.2 run.** Don't tune yet — collect data first. The official run should: (a) keep all checkpoints (interim or via F8.5.D), (b) record per-iter eval-reward + M1 trajectories, (c) use the seed-0 dry-run above as the "did anything regress?" anchor (if seed 0 in the official run doesn't land near M1=0.325, we know something between dry-run and official-run has changed). After the run, F8.5.D's best-checkpoint policy gives us the headline M1 number and the per-iter curves answer the "is late-training oscillation systemic?" question across seeds.

##### F8.2.A — Default-on before/after videos — XS

**Goal:** flip `record_video` default to `True` for non-smoke configs so every official run produces the existing 2-video pair (random-init "before" + trained "after") without needing per-YAML opt-in. Smoke YAMLs keep `False` explicitly so test runs stay fast.

**Why now:** the seed-0 ER1 dry-run produced no videos because `baseline.yaml` had `record_video: false`, making post-hoc visual analysis impossible. For the multi-seed run we want every seed's pair so we can eyeball whether different seeds learned visibly different policies (e.g., different rendezvous strategies, different collision patterns).

**Out of scope for this sub-phase:** per-eval video recording, multi-episode video, cross-seed montages. The **F8.2.C** safety cap protects against future feature creep down those roads, but today's deliverable is purely flipping the default.

**Scope:**

- `domain/models/runner.py`: change the `record_video` field's default from `False` to `True`. Smoke YAMLs that need `False` set it explicitly.
- `experiments/discovery/baseline/configs/baseline.yaml`: drop the `record_video: false` line. Update the in-YAML comment from "200 episodes × 200 steps × 4 agents — too slow" to "default ON; produces ~30s of MP4 (before + after only) on V100S".
- Smoke YAML (`*_smoke.yaml`) audit: every smoke YAML must set `record_video: false` explicitly. Add a CI check (parametrized test over all `*_smoke.yaml`) so a future smoke addition can't accidentally inherit the True default and slow CI.
- `regenerate-videos` is the recovery path for OVH (containers are headless; videos materialise post-pullback). The fail-soft pattern from F6.6 is already in place.

**TDD:**

- Unit: `OvhJobConfig(...)` (or whatever holds the field) — assert default is `True`.
- Unit: parametrized over every `*_smoke.yaml` in `experiments/`, assert `record_video: false` is set explicitly.
- Integration: a non-smoke local run produces both `before_training.mp4` and `after_training.mp4`.

##### F8.2.B — Cross-device checkpoint loading (CUDA→CPU) — XS

**Bug:** `regenerate-videos` against an OVH-trained run failed on a Mac (Smoke + ER1 dry-run, 2026-05-10) with `AssertionError: Torch not compiled with CUDA enabled`. Root cause: BenchMARL's `Experiment.reload_from_file(...)` honours the saved cfg's `device: cuda`; on a CUDA-less host this crashes during `torch.cuda.current_device()`. Hits the typical workflow: train on OVH GPU, regen videos locally on a Mac.

**Fix (two coupled changes):**

1. **BEFORE-video path** — already builds a fresh experiment using a `cfg` we read from `input/config.json`. Mutate `cfg.training.device = "cpu"` when `not torch.cuda.is_available()` before calling `adapter.build_experiment(cfg, ...)`. One-line cfg patch.
2. **AFTER-video path** — `Experiment.reload_from_file(checkpoint)` doesn't expose a `map_location`. Two viable approaches:
   - Monkey-patch `torch.load` with `map_location="cpu"` for the duration of the reload (cleanest).
   - Mutate the on-disk YAML inside the BenchMARL run-folder before reload, then restore — fragile.

   Going with the monkey-patch (a context manager `_force_cpu_load()` that sets `torch.load.__defaults__` then restores). Guarded by `torch.cuda.is_available()` so a CUDA host still uses GPU.

**TDD:**

- Integration: write a fake checkpoint with CUDA tensors via `torch.save({"x": torch.zeros(1).cuda()}, ...)` (skip when `torch.cuda.is_available()` is False — unfortunately we can't manufacture a "cuda checkpoint" on a CPU-only test runner; instead, build a checkpoint with `torch.save(..., _use_new_zipfile_serialization=True)` containing CUDA-flavoured metadata and assert the loader path picks the CPU branch).
- Pragmatic alternative: a unit test stubbing `torch.cuda.is_available()` to False, then verifying `cfg.training.device == "cpu"` after the wrapper runs and that the monkey-patch is applied/restored cleanly.

##### F8.2.C — Video count safety cap (≤10) — XS

**Goal:** prevent runaway disk usage if a future feature (per-eval video, multi-seed comparison montages, etc.) enables many recordings. Today's 2-video flow never hits this; the cap is a forward-looking guard so we don't have to retrofit it later.

**Scope:**

- Hard cap constant: `_MAX_VIDEOS_PER_RUN = 10` in `adapters/video/recorder.py`.
- `VideoRecorder` keeps a per-instance `_n_recorded` counter; the 11th `record(...)` call logs `logger.warning("video cap reached (10) — skipping further recordings")` and is a no-op.
- Cap is per-`VideoRecorder` instance (not per-process), so a regen run starts fresh after pullback. Deliberate: the only way to exceed 10 is to wire a multi-video mode that constructs many recorders, which is out of today's scope.

**TDD:**

- Unit: instantiate `VideoRecorder()`, call `record()` 11 times against a tiny env, assert the 11th call is a no-op (no MP4 file written) and that exactly one warning was logged.

##### F8.2.D — Post-pullback auto-regen on `sweep --follow` — XS

**Today's gap:** Stage 3 (Streamlit Refresh) auto-regens videos after pullback. CLI `sweep --follow` (Stage 2) does not — it only pulls files and prints success. So a researcher running `multi-scenario sweep --follow --runner ovh ...` ends up with no videos until they manually click Regenerate per run.

**Scope:** in `cli/sweep.py::_follow_ovh_jobs`, after each successful `pullback_run_dir(...)`, invoke `regenerate_videos(run_dir)` from `application/regenerate_videos.py`. Same fail-soft pattern as the Streamlit path: subprocess failure prints a warning but doesn't abort the sweep (multi-cell sweeps must keep going).

**TDD:**

- Add to `tests/integration/cli/test_sweep.py`: parametrized variant of `test_follow_ovh_jobs_calls_pullback_for_done_jobs` that asserts `regenerate_videos` is invoked after a successful pullback when the destination has no `videos/` dir.
- A second variant that asserts a regen failure does NOT abort the sweep loop (the pullback succeeded, results ARE on disk, regen is best-effort).

##### F8.2.E — Educational scalar/metric tooltips — S

[Originally drafted as F7.3.1; re-scoped here as a hard prerequisite for the multi-seed run so collaborators can self-interpret the results.]

**Why this matters:** the Run Detail page is the first surface a student or external collaborator hits to understand what the framework actually measures. Today the M1–M9 metric tiles and the BenchMARL scalar selector show raw names with no explanation — readable to the original author, opaque to anyone else. Goal: every quantity on the page is one click from "what is this and why does it matter?"

**Scope (two coupled deliverables):**

1. **Metric-tile tooltips (M1–M9) — `❓` info icon next to each tile.** Hovering or clicking shows a 1-2 sentence plain-English explanation + a "Learn more →" link to the corresponding mkdocs section (F10.1) when the docs site is reachable. Source of truth: a single registry in `frontend/metrics_glossary.py` (dict keyed by metric ID); both Streamlit and mkdocs read from it so the wiki and the UI never drift. Coverage: all M1–M9 with units (M1 is a fraction in [0,1], M3 is steps, M4 is collisions/episode, etc.).
2. **BenchMARL scalar-selector hints — `❓` per option in the multi-select.** When the user opens "Pick scalars to plot", each option (`collection_reward_episode_reward_mean`, `eval_reward_episode_len_mean`, `counters_iter`, …) shows a 1-line hint of what it actually is. Same single-source-of-truth pattern: `frontend/benchmarl_scalars_glossary.py`.

**TDD:**

- Unit: every metric ID in `domain.metrics.METRIC_IDS` has an entry in `metrics_glossary.py` (compile-time check via parametrized test).
- Integration (Streamlit AppTest): tooltips render without error; "Learn more →" link points at the expected mkdocs URL.

##### F8.2.F — In-app documentation hub — S

[Originally drafted as F7.3.2; re-scoped here.]

**Goal:** make the framework's docs (mkdocs from F10.1) discoverable from inside Streamlit. A student running the dashboard should never have to leave it to read an explanation.

**Two layers:**

1. **"📖 Docs" sidebar entry** — top-level Streamlit page that embeds (or links to) the mkdocs site. Local dev (`mkdocs serve` on :8000) → iframe-embed; deployed (GitHub Pages from F10.1) → link out. Detection via `MULTI_SCENARIO_DOCS_URL` env var; default to the deployed URL.
2. **Inline doc deep-links across pages.** Metric tiles / scalar selectors / preflight checks / scenario fields show a `📖` icon that deep-links to the relevant mkdocs anchor. Helper: `doc_link(slug: str) -> str` that builds `<docs_base_url>/<slug>` consistently; each glossary entry from F8.2.E carries a `doc_slug` field.

**Code-docs-from-Streamlit follow-on (deferred to F10.1.1):** mkdocs-material's `mkdocstrings` plugin auto-generates Python-API docs from docstrings. Once added in F10.1's mkdocs config (one plugin entry), the same in-app Docs hub also exposes class/function reference — clicking "OvhRunner" in a config display jumps to its docstring.

**TDD:**

- Unit: `doc_link("scenarios/discovery#m1-success-rate")` returns the expected URL given a fake env var.
- Integration: AppTest runs the Docs page with a mocked `MULTI_SCENARIO_DOCS_URL`; confirms iframe element renders.

##### F8.2.G — BenchMARL checkpoint retention — XS

**Bug from F8.2 dry-run (2026-05-10):** `checkpoint_interval_iters: 10` configured ~16 snapshots, but only 3 made it back to the bucket: 9.0M / 9.6M / 10.02M frames (≈ iters 150 / 160 / 167). The iter-125 peak — which is what F8.5.D's "best-checkpoint policy" must identify — was overwritten by BenchMARL's default rolling retention before the bucket sync.

**Fix:** propagate BenchMARL's checkpoint-retention knob (`bm.keep_n_checkpoints` or equivalent — verify against current BenchMARL API) through F2.4.1's training-knob propagation. Default for non-smoke runs: `null` (keep ALL). Smoke runs keep the rolling default (no point retaining 16 checkpoints from a 1-iter run).

**Compute / disk impact:** ~16 checkpoints × ~400KB each = ~6MB per run. Sweep ×3 seeds = ~18MB total. Negligible vs the run's other artefacts (BenchMARL logs are larger).

**TDD:**

- Unit: assert the BenchMARL config dict produced by `propagate_training_knobs` has the retention knob set to the expected value for non-smoke runs and the rolling default for smoke runs.
- Integration: a 5-iter smoke run with retention disabled-but-overridden-to-keep-all confirms all 5 checkpoint files survive at `output/benchmarl/.../checkpoints/`.

#### F8.3 — LERO architecture lands — block dep on F9.0–F9.6

Block dependency only — no work in F8.3 itself. F9.0–F9.6 must complete before F8.4 starts.

#### F8.4 — Port S3b-local config + run ×3 seeds + validate — M (OVH-bound)

- Translate `rendezvous_comm/configs/lero/s3b_local.yaml` → `experiments/discovery/lero/configs/lero_obs_only_local.yaml` (final name TBD; suggested for clarity).
- Add `lero: LeroSection | None = None` to `domain/models/config.py` (backwards-compat: existing baseline configs unchanged).
- Run ×3 seeds; compare to S3b-local reference (M1≈0.88 single-seed; threshold = mean ≥ 0.70 AND best ≥ 0.80).

#### F8.5 — Deep data-saving gap audit — M

Make every run-dir auditable end-to-end. Sub-phases:

- **F8.5.A — Per-step rollouts opt-in writer** (S): `runtime.storage.save_rollouts: bool = False` default; when on writes parquet under `output/rollouts/`.
- **F8.5.B — Reproducibility Streamlit page** (S): `pages/5_Reproducibility.py` reads runs.csv vs hardcoded reference dict; renders side-by-side table with PASS/FAIL.
- **F8.5.C — `runs.csv` LERO row schema** (S): add `record_type=lero_candidate` rows (cols: `iter, candidate_idx, fitness_rank, fallback_outcome`).
- **F8.5.D — Best-checkpoint policy callback** (S): BenchMARL writes `output/benchmarl/*/checkpoints/checkpoint_peak_M1.pt` whenever eval-M1 sets a new high. Fixes the eval-vs-final degradation gap rendezvous_comm flagged.
- **F8.5.E — `multi-scenario inspect-lero <run_dir>` CLI** (S): pretty-prints `best_reward.py` + `best_obs.py` with diff vs prior winner. Doc'd in `docs/results_analysis/lero_traces.md`.
- **F8.5.F — DuckDB index over LERO traces** (S): `multi-scenario index-traces` builds `<exp_root>/lero_traces.duckdb` for cross-run queries. Tables: `runs / candidates / llm_calls`.

---

### Phase 9 — LERO core implementation

> Hex-clean rebuild of LERO from the rendezvous_comm reference. **Locked decisions:** broker=LiteLLM, settings in YAML (`cfg.llm`), keys in project-root `.env`, cost cap **€10/day + €100/month rolling** (host-wide, persistent ledger) configurable + **must log when reached**, cache implemented but `enabled=false` default, `evolve_reward + evolve_observation` flag-controlled, **meta-prompting designed from day one but disabled by default**, reward_clip=±50, best-checkpoint enabled, whitelist_strict on for local mode.

#### F9.0 — Domain models + LeroSection / LlmSection — S

- `domain/models/config.py`: `LeroSection`, `LlmSection`. Both Optional on `ExperimentConfig`. STRICT mode; `lero` requires `llm` (no XOR).
- `domain/lero/`: `Candidate`, `CandidateMetrics`, `CandidateResult`, `PromptTrace`, `ResponseTrace`, `ReasoningTrace`, `LlmCompletion` (model output only — separate from our trace metadata), `LeroRunSummary`. All Pydantic; no torch/litellm imports.

#### F9.1 — LLM port + LiteLLM adapter + cost cap (host-wide rolling) — M

- `domain/ports/llm.py`: `LlmClient` Protocol. `generate(messages, n, seed) -> list[LlmCompletion]`.
- `adapters/llm/litellm_adapter.py`: real adapter wrapping LiteLLM (OpenAI / Anthropic / OVH endpoints). LiteLLM-native costs are USD; the adapter records `LlmUsage.estimated_cost_usd` from LiteLLM's price tables.
- `adapters/llm/cost_cap.py`: `CostCapDecorator` wraps any `LlmClient`. **Rolling-window semantics (€10/day + €100/month):** every call queries the persistent cost ledger for spend within the last 24h / 30 days, converts to EUR via `cfg.llm.usd_to_eur_rate`, raises `LlmCostCapExceeded` if either rolling sum + the new call's cost would exceed the cap, AND emits `logger.warning("cost cap reached: €X.XX > €Y.YY (window=…)" )` with the cap dict in extra fields. The decorator is composable — wrap a cached client to enforce budget on cache misses only, etc.
- `domain/ports/cost_ledger.py`: `CostLedger` Protocol with `record(timestamp, cost_eur, model, …)` and `sum_window(window=timedelta) -> float`.
- `adapters/llm/filesystem_cost_ledger.py`: JSONL-appender at `~/.multi_scenario/cost_ledger.jsonl` (overridable via `MULTI_SCENARIO_COST_LEDGER` env var for tests / sandbox). Atomic appends, prune-on-read for entries older than 31 days. `InMemoryCostLedger` for tests.
- `adapters/llm/disk_cache.py`: optional response cache, `enabled=false` by default. Cache key = SHA(model, messages, seed, response_format). Wraps `LlmClient` like the cost cap (decorator composes).
- `adapters/llm/fake_adapter.py`: in-memory canned-response adapter for tests (registered via `MULTI_SCENARIO_LLM_OVERRIDE=fake`). Returns predetermined `LlmCompletion` objects keyed by message-prefix matchers.

**Rationale for host-wide vs per-run caps:** the user's billing surface is the OpenAI/Anthropic API key, which is host-wide and invoiced monthly. A €5/run cap doesn't prevent a runaway sweep from spending €100 in 12 hours; €10/day + €100/month does. Persistent ledger means a crashed run that already spent €4 doesn't reset the budget on retry — accidental overspend is structurally impossible without flipping the cap fields explicitly.

**Decorator composition order (outermost first):** `CostCapDecorator → DiskCacheDecorator → LiteLlmClient`. The cap sees the cost of every uncached call (cache hits are free); the cache sees every request the cap allowed through.

#### F9.2 — Prompt registry (Jinja-based) + byte-parity vs rendezvous_comm — M

**Important note on the rendezvous_comm port (2026-05-10 audit):** the
reference uses Python ``string.Template`` syntax (``$variable``) in
``.txt`` files, NOT Jinja. We standardise on Jinja anyway so future
meta-prompt composition can use loops / conditionals / filters; the
syntax migration is mechanical.

- `adapters/prompts/<version>/{system.j2, initial_user.j2, feedback.j2, meta.yaml}` for the prompts F8.4 needs to reproduce: `v1`, `v1_global`, `v2`, `v2_min`, `v2_fewshot`, `v2_twofn`, `v2_fewshot_k2_local`. Content ported from rendezvous_comm; syntax translated `$X` → `{{ X }}`. The `meta.yaml` carries version / author / description (used by the registry's discovery + the FE's prompt picker).
- `adapters/prompts/jinja_renderer.py`: `JinjaPromptRenderer` implements `PromptRenderer` Protocol. Configured with `keep_trailing_newline=True` and default whitespace settings (no `trim_blocks`, no `lstrip_blocks`) so Jinja-rendered output matches `string.Template`-rendered output character-for-character. `StrictUndefined` so a missing context key fails loudly instead of silently rendering empty.
- **Load-bearing byte-parity test:** `test_v2_fewshot_k2_local_byte_parity.py` renders our Jinja template AND the rendezvous_comm `string.Template` against the same context dict; asserts the two byte sequences are equal. If this test ever drifts, F8.4's S3b-local replication will silently change too — that's the failure mode this test catches early.
- Future-prompt extension point: any *new* prompt versions (created post-port) can use full Jinja features (`{% if %}`, filters, includes); only the rendezvous_comm-ported set needs to stay byte-equivalent.

#### F9.3 — TraceWriter port + filesystem adapter — S

- `domain/ports/trace_writer.py`: `TraceWriter` Protocol. Methods write_prompt / write_response / write_reasoning / write_candidate / write_evolution_history / write_fallback_chain / write_summary.
- `adapters/lero/filesystem_trace_writer.py`: writes the canonical layout under `<run_dir>/output/lero/iter_<n>/cand_<m>/attempt_<a>/{prompt.json, response.json, reasoning.json}`, plus aggregate files `evolution_history.json`, `fallback_chain.json`, `best_reward.py`, `best_obs.py`, `final_metrics.json`, `llm_provenance.json`.
- Atomic write-rename to survive interrupts.

#### F9.4 — Code generation + safety — S

- `domain/lero/codegen.py`: `extract_candidates(response_text, evolve_reward, evolve_observation) -> CandidateCode | None`; `validate_function(source, ...) -> ValidationResult`.
- `ALLOWED_IMPORTS = {"torch", "math", "numpy"}`.
- **Byte-parity test** vs rendezvous_comm's `codegen.py::extract_candidates` on the same response text.

#### F9.5 — Scenario patching (Discovery first) — M

- Extend `Scenario` Protocol with optional `patch_with_llm_code(reward_source, obs_source, lero_section)`.
- `adapters/scenarios/_lero_patch_helpers.py`: ports rendezvous_comm helpers — `_build_reward_state`, `_build_obs_state`, `_compile_function`, `_sanitize_reward` (nan_to_num + clamp ±50), `AllowedKeysDict` (whitelist-strict mode), `FairnessViolation` exception.
- Patched class overrides `info()` to return per-agent `covering_reward` (M8 unblocker, rendezvous_comm bug §3.3).
- Per-scenario regression tests: patch closure bug, reward clip, NaN-to-zero, whitelist strict, per-agent info.

#### F9.6 — Evolutionary loop orchestrator — L

- `application/lero_orchestrator.py`: 8-port-injected use-case. Splits as `_run_iteration / _evaluate_candidate / _full_training_with_fallback` (each privately tested).
- `application/prompt_composer.py`: `PromptComposer` Protocol + `InitialAndFeedbackComposer` (default impl). `compose(iteration, history)` returns the messages list.
- **Resume support:** `LeroOrchestrator.resume(run_dir)` reloads existing iter_<n>/ subdirs into history.
- **`experiment_service.py` branch:** if `cfg.lero is not None`, delegate to `LeroOrchestrator.run()`.
- **Discharged-candidates note (user TBD at implementation time):** plan documents both interpretations. (A) within-run re-rank by post-full-training M1; (B) across-run no seeding from prior discharged candidates. User picked (B) with "review when we implement". Implementation review at F9.6 kickoff.

#### F9.7 — Meta-prompting seam (stub now) + full design notes (deferred) — XS now / L deferred

> **Scope split (locked 2026-05-10):**
>
> - **F9.7.A — Now (XS):** ship the seam — `PromptComposer` Protocol contract + a no-op `MetaPromptComposer` stub + the contract test that proves the orchestrator works with either composer. This is the only part needed before F8.4 fires.
> - **F9.7.B — Deferred (L):** the full meta-prompting implementation (Strategist + Editor + Critic round-table) is **deferred until after the GitHub-extraction migration (F10.4) AND after all reproducibility experiments (F8.2, F8.4) have been run**. Reason: meta-prompting is a research extension that builds on the validated baseline; locking the baseline first lets us measure meta-prompt deltas cleanly. We document the design now so the seam in F9.7.A is shaped to fit it.

##### F9.7.A — Composer-Protocol seam (executable now)

- Keep `PromptComposer` Protocol broad enough that the meta-prompt round-table plugs in as a different composer (no orchestrator changes).
- Ship a stub `MetaPromptComposer` that returns a trivially-mutated initial prompt (e.g., appends `"\n\n[meta-prompt placeholder]"`) so the orchestrator path can be exercised end-to-end without a real Strategist / Editor / Critic.
- `test_orchestrator_with_meta_composer.py`: full LERO loop with `cfg.lero.meta_prompting=true`, `FakeLlmClient`, asserts the placeholder lands in the recorded `prompt.json` traces.
- Default behaviour: `cfg.lero.meta_prompting=false` → `InitialAndFeedbackComposer` is used. `=true` → `MetaPromptComposer`. Field defaults to `false` in `LeroSection`.

##### F9.7.B — Full meta-prompting design (post-extraction + post-experiments)

Findings ported from rendezvous_comm's `src/lero/meta/v4_*` prototype (≈1500 lines, Phase 4 ablations 2026-04). These notes are the *spec* the deferred implementation will follow — recorded here so we don't lose context during the F10.x extraction.

**Three-role round-table architecture.** Today's inner loop is one LLM role (the *code generator*). Meta-prompting adds three more outer roles that **mutate the inner-loop prompts between iterations**:

1. **Strategist** — reads recent inner-loop history (candidate metrics + verdicts), decides *which sub-slot* of the next prompt to edit and *with what focus*. Emits a `StrategyCard`:

   ```text
   target_domain: "reward" | "observation" | "shared" | "both"
   target_slot:   "guidance_shared" | "guidance_reward" | "guidance_observation"
   focus:         List[str]  # 1–2 specific patterns to encourage
   avoid:         List[str]  # patterns that scored regression/collapse
   confidence:    "small" | "medium" | "large"
   include_signals: List["scalar" | "fingerprint" | "curve_shape"]
   rationale:     str  # 2–4 sentences citing specific evidence
   ```

2. **Editor** — given a `StrategyCard`, produces the new text for the chosen slot (`EditorOutput.new_slot_content`).
3. **Critic** — TextGrad-style second-pass review over the Editor's draft. Three verdicts: ``keep`` (accept), ``revise`` (re-invoke Editor with notes), ``reject`` (graceful-stop the round, keep prior prompt). Drives a 1–2 round critique-revise loop in `meta/critique.py`.

**Sub-slot model on the inner prompt.** The inner-prompt template has three editable sub-slots — `{{ guidance_shared }}`, `{{ guidance_reward }}`, `{{ guidance_observation }}` — initially empty. Each meta-iteration the Strategist picks one slot, the Editor rewrites it, the Critic vets it, and the new content is injected into the next inner-loop iteration's initial prompt. Slots are non-overlapping by design: the Strategist's `target_slot` is a single value per round.

**Behavioral signal tiering** — meta-prompting needs richer feedback than M1/M2/M3, but flooding the Strategist with raw rollouts is noisy. Three tiers:

- **scalar** (default): just M1/M2/M3 + verdict per candidate.
- **fingerprint**: per-candidate `BehavioralSummary` — collision rate, coverage curve shape, agent-utilization CV, dispersion stats (rendezvous_comm's `meta/behavioral_summary.py`).
- **curve_shape**: per-candidate eval-M1 trajectory (sparse samples, normalized).

The Strategist controls which tier feeds back via `include_signals`. Default is `["scalar"]` — only escalate when scalar evidence isn't enough to choose a slot.

**Trigger / fairness / failmode signals** (rendezvous_comm `meta/trigger.py`, `meta/fairness.py`, `meta/failmode.py`):

- **Trigger** — when to fire a meta-iter at all. Today's stub always fires; the real version skips meta-iters when the inner loop is converging cleanly (don't disturb a working trajectory).
- **Fairness restatement check** — every Editor draft must restate the fairness clause (no oracle / global-state access) verbatim. The Critic enforces this; missing → `revise`. Prevents subtle prompt drift that re-introduces global-state reward hacks.
- **Fail-mode catalogue** — pattern-match common reward-hacking failure modes (NaN actions, M2 explosion with M1 collapse, etc.) and surface them as `avoid:` candidates for the Strategist.

**Mutation log** (`meta/mutation_log.py`): every applied edit is recorded as `(meta_iter, slot, old_content_hash, new_content_hash, strategy_card, critic_verdict, downstream_M1_delta)`. Lets us answer "did this slot edit help?" post-run, and lets the Strategist `avoid:` patterns that failed before.

**Peak-checkpoint pinning** (`meta/peak_checkpoint.py`): the meta-loop tracks the best-prompt-config seen so far across all meta-iters. If a later meta-iter regresses, we can bail back to the peak prompt without losing it. This is the meta-loop's analogue of F8.5.D's best-checkpoint policy.

**Strict-mode JSON schemas.** OpenAI's structured-outputs API requires `extra="forbid"` + every field `required`. The rendezvous_comm `schemas.py` ships `StrategyCard` / `EditorOutput` / `EditorCritique` Pydantic models that round-trip through OpenAI structured outputs cleanly. Keep these; port verbatim to `domain/lero/meta_schemas.py` when F9.7.B lands.

**Code layout (deferred)** — when F9.7.B is implemented, layer it like this:

```text
domain/lero/
├── meta_schemas.py            ← StrategyCard, EditorOutput, EditorCritique
└── meta_signals.py            ← BehavioralSummary, FailmodeMatch (Pydantic, no torch)

application/
└── meta_prompt_orchestrator.py ← outer-loop driver: Strategist → Editor → Critic → mutate prompt → run inner LeroOrchestrator

adapters/prompt_composers/
├── meta_prompt.py              ← real MetaPromptComposer (replaces F9.7.A stub)
└── meta_helpers/               ← ports of rendezvous_comm/meta/{trigger,fairness,failmode,…}.py
```

**Why deferred to post-F10.4 + post-experiments:** meta-prompting is a *research mutation* on top of a working baseline. Locking the baseline (F8.2 ER1 ×3, F8.4 S3b-local ×3) before adding meta-prompting means the deltas we measure are clean. Doing it before would conflate "did meta-prompting help?" with "did porting LERO break something?" Also: F10.4's GitHub extraction is a natural inflection point — the meta layer lands in the new repo where the baseline is already validated, with a single PR that says "here's meta-prompting, here are the inner-loop numbers it improves on".

**Lift-trigger:** earliest of (a) F8.4 S3b-local ×3 seeds passing the reproducibility threshold, (b) F10.4 extraction complete, (c) a research need to compare meta-prompt vs fixed-prompt deltas on a new task. Until then, F9.7.A's stub is enough for the seam to stay honest.

#### F9.8 — CLI + Submit page integration — S

- `multi-scenario run <lero_yaml>` Just Works (the YAML drives the experiment_service branch).
- `multi-scenario inspect-lero <run_dir>` (per F8.5.E).
- Submit page: `frontend/forms.py` renders `LeroSection` + `LlmSection` widgets when YAML includes them. Preflight adds an OPENAI_API_KEY-presence check when `cfg.lero is not None`.

---

### Phase 10 — Docs, naming, extraction

> **Locked:** new name = `coopvmas`, GitHub personal account, license = GPL-v3 (matches parent VMAS), fresh-import extraction (no history preservation), Streamlit FE stays in same repo, .env at project root.
>
> **Audit update (2026-05-12):** plan re-validated after F8.4 Phase 6 + Phase 9 (F9.x including F9.8 widgets) + recent test work (dispatch_matrix, --json CLI, Playwright opt-in) landed. Key adjustments below: F10.3 names updated, F10.7 effort upgraded from XS to M (609 refs found), F10.1 scope expanded with new docs, new sub-features F10.9 (rendezvous_comm history port) + F10.10 (results gitignore). User-review checkpoints flagged on F10.2/F10.3/F10.10.

#### F10.1 — mkdocs-material wiki — M

- `mkdocs.yml` + `docs/` reorganised into topic-per-file structure under `docs/{getting_started, concepts, scenarios, cli, frontend, operations, results_analysis, ports, reproducibility}/`. Full file list in `docs/_drafts/F8_F11_plan_draft.md` Section D.
- `mkdocs build --strict` runs in CI to catch broken links.
- `docs/concepts/lero.md` ported and adapted from `rendezvous_comm/docs/lero.md` for the coopvmas codebase + Section C of the draft (architecture deep-dive).
- **New (post-F8.4/F9):** fold in `docs/f8_4_phase6_comparison.md` → `docs/reproducibility/lero_s3b_local.md`; add a `docs/getting_started/submitting_experiments.md` documenting the Streamlit Submit workflow as the canonical entry point (user locked Streamlit-only post-extraction); add `docs/cli/run.md` covering `--json` output for scripted submissions; add `docs/frontend/submit_page.md` covering F9.8 LERO/LLM widgets + auto-poll + pullback lifecycle.

#### F10.2 — Rewrite README as wiki landing page — S [USER-REVIEW CHECKPOINT]

Replaces the F0.6 stub. README links to every top-level `docs/` section; no orphans.

**User-review gate**: I draft, user reviews in depth before extraction. Must surface every doc — no orphaned wiki pages.

#### F10.3 — Pre-extraction YAML cleanup — XS [USER-REVIEW CHECKPOINT]

User asked: before extracting to coopvmas, **delete every per-experiment YAML except the canonical references** that prove the reproducibility story.

**Canonical references after F8.4 (names settled, plan-original `lero_obs_only_local.yaml` superseded):**

- `experiments/discovery/baseline/configs/baseline.yaml` — ER1 baseline reference (non-LERO).
- `experiments/discovery/lero/configs/lero_s3b_local.yaml` — rendezvous_comm S3b-local port (cr=0.25, ms=400); Phase 5a hit M1=0.795 vs rendezvous_comm 0.88.
- `experiments/discovery/lero/configs/lero_s3b_local_er1params.yaml` — Phase 5b apples-to-apples ER1-parameter LERO; M1=0.570 vs ER1 baseline 0.405 (+40% relative).

**User-review gate**: user confirms 2 vs 3 references (er1params variant is the F8.4 science result — keeping it preserves the comparison; dropping it makes the new repo lean) before any file deletion.

- Files to delete: every OTHER YAML under `experiments/<scenario>/*/configs/` (smoke variants, OVH-preflight tests, per-algorithm scratch, F8.4 Phase 3 smoke YAMLs).
- Smoke YAMLs used as CI fixtures stay (they're test fixtures, not dev scratch). If we have CI smoke YAMLs in `tests/fixtures/`, those are fine.
- Run dirs under `experiments/*/` (results from prior runs) — wipe (covered by F10.10's gitignore review).

#### F10.4 — CI pipeline — S

- GitHub Actions on the new repo: lint (pre-commit) + unit tests on push; smoke integration tests nightly. Coverage gate (start at 70%).
- `mkdocs build --strict` runs in CI for the docs site.

#### F10.5 — Reproducibility test (general, not LERO-specific) — S

- Run the same config with the same seed twice; assert all metrics agree within tolerance. Lives in `tests/reproducibility/`. Distinct from F8 (which is reproducing rendezvous_comm); F10.5 is general "same-config-same-seed → same numbers".
- **Audit note**: `tests/reproducibility/test_compare_to_reference.py` already exists but tests F8.2's `compare_to_reference.py` threshold-logic — different contract. F10.5's specific "same-config-same-seed-byte-equal" test still needs writing.

#### F10.6 — Repo extraction to coopvmas — M (manual)

**Procedure (locked: fresh import, no history preservation):**

1. **Tag** the multi_scenario folder state at the extraction commit so we can refer back: `git tag coopvmas-extracted-from`.
2. **I produce a zip** (`coopvmas-v0.1.0.zip`) plus a step-by-step `EXTRACT.md` that includes:
   - Pre-extraction checklist (F10.3 YAML cleanup confirmed; F10.4 CI green; F10.5 repro test green; mkdocs build --strict green).
   - Files included / excluded from the zip (e.g. drop `.venv/`, `__pycache__`, `experiments/*/results/`, `output/` artifacts).
   - Post-extraction setup steps: `cd coopvmas && git init && git add . && git commit -m "Initial import from VMAS monorepo"`, create GitHub repo `afin/coopvmas`, `git remote add origin … && git push`.
   - License file: `LICENSE` = GPL-v3 (matches parent VMAS).
   - Cleanup of monorepo-only constructs:
     - Remove `files: '^multi_scenario/'` from `.pre-commit-config.yaml`.
     - Change markdownlint `args: ["--config", "multi_scenario/.markdownlint.json"]` → `args: ["--config", ".markdownlint.json"]`.
     - Update setup.cfg per-file-ignores: drop the `multi_scenario/src/multi_scenario/cli/*.py` prefix → `src/multi_scenario/cli/*.py` (or rename `multi_scenario` package to `coopvmas`).
     - Search-replace `multi_scenario` → `coopvmas` package-wide (Python imports, paths, docs, README).
   - First-run validation in the new repo: `pre-commit run --all-files` green, `pytest` green, `mkdocs serve` works locally.
3. **User copies the zip + executes EXTRACT.md.** I'm not in the loop after step 2; user tells me when the new repo is live and we resume from there.

#### F10.7 — Comment cleanup pass — M [audit-updated from XS]

- Sweep all source files (everything *outside* the planning markdowns) for comments that reference phase or feature IDs (`F0.1`, `F2.4`, `Phase 9`, etc.) or section anchors from this plan (`§3.5.2`, etc.). These references are scaffolding from the build process — useful during development to trace which feature added what, useless and confusing once the project is extracted to its own repo (where this plan no longer lives).
- **Audit count (2026-05-12)**: 609 matches across `src/` `tests/` `pyproject.toml`. Effort bumped from "XS" to "M" — realistic ~3-4h scripted sweep + spot-check pass. Mechanise via `git ls-files | xargs sed` for trivial deletes; manual review for comments where the phase ref is interleaved with substantive prose.
- For each comment found: if there is a substantive WHY behind the reference, keep that prose and drop the phase pointer. If the comment was *only* a phase pointer, delete the comment entirely. Per project style (CLAUDE.md) — comments justify *why*, not *which-feature-added-this*.
- Files in scope: `src/**`, `tests/**`, `docs/**` (non-markdown), `pyproject.toml`, `.pre-commit-config.yaml`, `.markdownlint.json`, `.gitignore`, YAML configs under `experiments/**`. Out of scope: `plan.md`, `implementation_plan.md`, and any other markdown documents in `docs/` whose purpose is planning/architectural narrative — those are allowed to keep phase references.
- **Demo:** `grep -rnE 'F[0-9]+\.[0-9]+|Phase [0-9]+|§[0-9]' src tests pyproject.toml .pre-commit-config.yaml .markdownlint.json .gitignore docs experiments --include='*.py' --include='*.toml' --include='*.yaml' --include='*.yml' --include='*.json' --include='.gitignore' --include='.pre-commit-config.yaml'` returns no matches.

#### F10.8 — Scaffolding cleanup pass — XS

Some artefacts produced *during* development serve a one-time purpose — informing a decision, generating empirical numbers for a sign-off, validating a deferred choice — and become dead weight once the final code + F10.1/F10.2 docs land. Sweep them out before extraction (F10.6).

**Removal candidates:**

- **`docs/csv_format_decision.md`** + **`scripts/f5_5_format_decision.py`** — F5.5 decision artifacts.
- **Sub-feature placeholder sections in `implementation_plan.md`** (F2.4.2, F2.10.1, F5.2.1) — fold the resulting state into the relevant final doc and remove the placeholder.
- **Any `_<exp_id>` scratch folders under `experiments/`** — temp dirs from reproducer scripts; should already be auto-cleaned.
- **Stale TODO comments** referencing deferred features whose triggers have since fired.
- **`docs/_drafts/`** — once F8/F9/F10/F11 implementation is complete, drop the agent-generated draft (`F8_F11_plan_draft.md`).

**What to keep:**

- Smoke YAMLs under `tests/fixtures/` — CI fixtures, not dev scaffolding.
- `.pre-commit-config.yaml` / `.markdownlint.json` / `.gitignore` / `pyproject.toml` — production tooling.
- All test fixtures, even the ones that were written to drive a single feature (regression value).
- **`docs/operations/ovh_setup.md` + `docs/operations/ovh_smoke_checklist.md`** — operational user docs (evergreen).
- **`configs/ovh.yaml`** — user-editable production config.

#### F10.9 — Port rendezvous_comm experiment history into a reference doc — S [NEW, audit-added]

User asked: distil `rendezvous_comm/docs/lero.md` + the empirical results scattered across `rendezvous_comm/results/` into a SINGLE reference document inside this repo. User will cite from this doc when writing the new repo's reports (Phase 6 comparison, per-scenario campaign), then **delete the file** after reproducing the relevant results in the new repo.

- Source: `rendezvous_comm/docs/lero.md` (canonical single-source-of-truth doc per project memory 2026-04-16) + `rendezvous_comm/results/*` summary tables.
- Destination: `docs/_drafts/rendezvous_comm_history.md` (under `_drafts/` so F10.8 cleanup picks it up when the user signals "done with citations").
- Content: LERO Phase 4 results table (n=3 t=3 k=1 numbers, ms200/ms400 ablation), infra lessons (max_steps bug, env-leak Fernet, M8 covering_reward fix, etc.), pending experiments user planned but never ran.
- Format: markdown with citation-friendly section anchors (the user pulls quotes from it into new-repo reports).

#### F10.10 — Pre-extraction .gitignore review for results — S [NEW, user-review checkpoint]

User asked: review WHAT currently gets committed to git under `experiments/*/results/`, `output/`, and friends. Decide which artefact types are worth versioning vs ignoring in the new repo.

**My pass (proposal)**:

- Inventory every glob currently matched under `experiments/`, `output/`, `results/` across recent runs (sizes, file types, science-load-bearing or not).
- Propose `.gitignore` patterns: typically ignore everything under `experiments/*/results/`, `output/`, `*.pt` (checkpoints), `videos/` (regenerable). Keep YAML configs + `.gitkeep` placeholders for required dirs.
- Highlight grey-area items (eval_episodes.json, run_state.json, scalars CSVs) for user decision.

**User-review gate**: user reviews the proposed gitignore + decides what to keep on commits before extraction. Lives in `EXTRACT.md` (F10.6).

**Order (revised after audit):**

1. **F10.5** (repro test, smallest, no review needed) — closes the F10.5 missing-test contract first.
2. **F10.9** (rendezvous_comm history port) — write the reference doc the user wants to cite from in new-repo reports.
3. **F10.1** (mkdocs structure + ports docs).
4. **F10.2** (README rewrite) → **USER REVIEWS** before extraction.
5. **F10.4** (CI workflows template for new repo).
6. **F10.7** (comment cleanup, 609 refs).
7. **F10.8** (scaffolding cleanup).
8. **F10.10** (gitignore review) → **USER REVIEWS** before extraction.
9. **F10.3** (YAML cleanup, final canonical list) → **USER REVIEWS** before extraction.
10. **F10.6** (zip + EXTRACT.md, user-executed).

**Total Phase 10 effort (audit-revised)**: ~17-21h dev + 3 user-review checkpoints.

---

### Phase 11 — Per-scenario experiment campaign

> **Scope deferred (locked):** the per-scenario matrix (which algorithms, which seeds, which ablations) gets discussed in depth at F11 kickoff, not now. F11 is sketched here as placeholder structure; sub-phases land after F8 reproducibility validates and F10.6 extraction completes.

#### F11.1 — Discovery campaign — TBD

After F8 + F9 + F10 land in coopvmas, run the full discovery experiment campaign: ER1 ablation matrix + LERO sweep with multiple prompts/configs. Decide scope at F11.1 kickoff (which ablations to port from rendezvous_comm, how many seeds, etc.).

#### F11.2 — Navigation campaign — TBD

Adapt ER1 + LERO to navigation. Identify scenario-specific tweaks (different success_predicate semantics, different default params).

#### F11.3 — Transport campaign — TBD

#### F11.4 — Flocking campaign — TBD

Note: flocking has no natural M1 success rate; campaign uses M2/M9 instead.

#### F11.5 — Cross-scenario synthesis report — TBD

Streamlit page showing per-scenario leaderboards + LERO-vs-baseline deltas. Output informs publication / future research direction.

---

## 6. Open questions / decisions deferred

Locked decisions from the 2026-05-09 planning round are recorded in
`~/.claude/.../memory/project_coopvmas_decisions.md`. The table below tracks
items still open.

| Topic | Latest moment | Status |
| --- | --- | --- |
| Default CSV format (long vs summary) | F5.5 | locked (long) |
| Per-scenario success metric (nav, flocking, transport) | F4.1–F4.3 | locked |
| Final package name | before F10.6 | **locked: `coopvmas`** |
| Algorithm hyperparameters per scenario | F11.1 kickoff | open |
| Streamlit FE: keep in same repo or split? | F10.6 | **locked: same repo** |
| LERO architecture | F9 kickoff | **locked** (see memory) |
| OVH cost vs local smoke threshold | F6.5 | locked |
| Resume tolerance threshold (acceptable metric drift) | F5.7 | locked |
| Heuristic baseline complexity per scenario | F8.1 (was) | **dropped** (F11.1 may revisit) |
| Reproducibility threshold (ER1, S3b-local) | F8.0 | **locked: ±10% abs + 1.5σ** |
| LERO discharged-candidates handling (within-run vs across-run) | F9.6 kickoff | open ("review at impl time") |
| LERO experiment campaign matrix (Phase 11) | F11.1 kickoff | open ("discuss in depth then") |
| `coopvmas` license | F10.6 | **locked: GPL-v3 (matches parent VMAS)** |

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
