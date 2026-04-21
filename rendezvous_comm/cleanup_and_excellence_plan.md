# Cleanup & Excellence Plan — rendezvous_comm

**Date:** 2026-04-20
**Version:** v4
**Author:** planning draft, pending user review
**Goals:**

1. Put `rendezvous_comm/` on solid, student-friendly, well-documented, well-tested, crash-safe footing.
2. Re-run every baseline from scratch and verify against the archive before any new advanced technique.
3. Make cross-experiment comparison first-class.
4. **(New in v4) Make `rendezvous_comm/` a fully standalone Python package** — installable via `pip`/`uv`, depending on VMAS/BenchMARL as regular pinned dependencies, ready to extract into its own git repository at any time with zero loss of functionality.

> ⚠️ **This is a plan, not an action.** Nothing in `docs/`, `results/`, or the OVH S3 bucket will be moved or deleted until you explicitly approve each phase.

---

## 0. Executive Summary

Thirteen sequential phases. Each phase lands a separate, revertible commit. Phases are ordered so we never build on top of code we don't fully trust, never compare against data we haven't re-verified, and never start a new technique until the baseline reproduces.

| # | Phase | Deliverable | Effort |
|---|-------|-------------|--------|
| 0 | Archive & clean slate | Old docs/results in `archive01/`, OVH S3 empty | 0.5 d |
| 1 | Environment & hyperparameter docs | `docs/environment.md`, `docs/hyperparameters.md` | 1 d |
| 2 | CSV schema + I/O contract | Enforced schema, 4 new tests, `docs/io_contract.md` | 1 d |
| 3 | RL techniques theory + impl docs | `docs/rl/` folder | 1.5 d |
| 4 | LERO flow diagrams | Mermaid diagrams in `docs/rl/lero.md` | 0.5 d |
| 5 | Clean architecture **+ standalone packaging** | `pyproject.toml`, `uv.lock`, src-layout, layering tests | 2 d |
| 6 | Parallel safety & crash recovery | Globally-unique paths, resume-from-checkpoint, write-once I/O | 1 d |
| 7 | Src/tests audit + regression suite | `docs/audit_report.md`, `tests/regressions/`, coverage gate | 1.5 d |
| 8 | Streamlit UX + multi-exp comparison | Parallel OVH jobs, auto videos, cross-experiment rewrite | 1 d |
| 9 | Candidates + ablation pages | `docs/candidates.md`, `docs/rl/ablations/` | 0.5 d |
| 10 | QA gate / test plan | `docs/qa_plan.md`, CI-runnable suite | 0.5 d |
| 11 | Sequential baseline re-runs | ER1 → ER2 → ER3 → LERO verified vs archive01 | 2–3 d |
| 12 | Advanced technique extensions | Per-technique checklist, scaffolder, first new family | 1+ d |

**Total:** ~13–15 working days to Phase 11 done; Phase 12 is open-ended.

---

## 1. Current State Snapshot

### 1.1 Code layout

- `src/config.py` — `TaskConfig`, `TrainConfig`, `SweepConfig`, `ExperimentSpec`, YAML loader.
- `src/runner.py` (1.6 k lines) — `build_experiment()`, `run_single()`, `run_lero()`, `run_experiment()`; wraps BenchMARL `Experiment`.
- `src/lero/` — `loop.py`, `scenario_patch.py`, `codegen.py`, `llm_client.py`, 9 prompt variants.
- `src/metrics.py` — `EpisodeMetrics` producing M1–M9 (+ M7 post-hoc).
- `src/storage.py` — per-run timestamped folder; `input/config.yaml`, `output/metrics.json`, `output/benchmarl/`, `output/policy.pt`, videos.
- `src/ovh.py` — `ovhai` CLI wrapper.
- `src/consolidate.py` — builds `sweep_results_{ts}.csv`.
- `pages/*` + `Dashboard.py` — Streamlit (use `sys.path.insert`).
- `tests/` — 21 files; ~1 k lines.

### 1.2 Current coupling to the parent repo

- **No `pyproject.toml`** in `rendezvous_comm/`. `requirements.txt` exists but isn't authoritative.
- **`vmas` import is already via pip package** (not relative), so dependency-wise VMAS can be pinned remotely.
- **However, parent repo's `setup.py develop` installs `vmas` in editable mode** from the parent source tree — swapping to a released `vmas` version would be a behavioural change we must verify.
- **9 files use `sys.path.insert`** (all Streamlit pages, `Dashboard.py`, `src/lero/loop.py`, some notebooks) — a src-layout converts these to real package imports.
- **No published package name, version, or entry points.**
- **No `.github/workflows/`**, no Makefile, no standalone README-for-install.

This is what makes extraction hard today and what Phase 5 fixes.

### 1.3 Scenario defaults

VMAS `discovery.py` via `VmasTask.DISCOVERY`. `TaskConfig`: `n_agents=5, n_targets=7, agents_per_target=2, lidar_range=0.35, covering_range=0.25, agent_radius=0.05, x/y_semidim=1.0, max_steps=200, dt≈0.1, substeps=2, dim_c=0, targets_respawn=False, shared_reward=False`.

### 1.4 Results footprint

`er1/ 242M, er2/ 120M, er3/ 31M, lero/ 13M, e1/ 12M, er1_abl_*/ ~12M, er2_*/ ~2.5M, lero_rescued/ 604K` — ≈ 430 MB local, mirrored in OVH S3.

### 1.5 Experiment families

`e1/`, `e1_ablations/`, `e2/`, `er1/`, `er2/`, `er3/`, `lero/` (14 LERO configs).

---

## 2. Critical Analysis

Stable IDs (**R1**, **R2**, …) so later phases can reference them. Severity: 🔴 blocks excellence, 🟠 important gap, 🟡 doc gap, 🟢 minor.

### 2.1 Correctness

- 🔴 **R1** — CSV schema is implicit; regressions silent.
- 🔴 **R2** — Hyperparameters not guaranteed in CSV.
- 🔴 **R3** — `run_id` format untested for stability / uniqueness.
- 🔴 **R4** — LERO `scenario_patch` `exec`s LLM code without compile + smoke contract.
- 🔴 **R5** — M7 heuristic (80% threshold) untested against reference curve.
- 🔴 **R6** — `targets_respawn=False` load-bearing but only by convention.
- 🔴 **R7** — `runner.py` is 1.6 k-line god module.
- 🔴 **R8** — Global singleton `_cfg._cache` in `ovh.py`.

### 2.2 Parallel safety & recovery

- 🔴 **R9** — Same-`exp_id` FINALIZING race still possible.
- 🔴 **R10** — Policy only saved at run end; crashes lose artifacts.
- 🔴 **R11** — No `run_state.json`; FS can't tell RUNNING / CRASHED / DONE.
- 🔴 **R12** — No resume-from-checkpoint.
- 🔴 **R13** — Logs are plain text (not machine-queryable).
- 🔴 **R14** — Streamlit submit has no idempotency.

### 2.3 Packaging & standalone readiness (v4)

- 🔴 **R37** — No `pyproject.toml`; not pip-installable as a package.
- 🔴 **R38** — Flat folder, not src-layout; 9 files use `sys.path.insert`.
- 🔴 **R39** — No lock file; library versions can silently drift between local and OVH.
- 🔴 **R40** — No CI (no `.github/workflows/`).
- 🟠 **R41** — No LICENSE file inside `rendezvous_comm/` (inherits parent implicitly).
- 🟠 **R42** — No CONTRIBUTING / CHANGELOG.
- 🟠 **R43** — No entry-point scripts (`rendezvous-train`, etc.).

### 2.4 Testing gaps

- 🟠 **R15** — No `test_ovh.py`.
- 🟠 **R16** — No end-to-end runner integration test.
- 🟠 **R17** — No hyperparameter-propagation test.
- 🟠 **R18** — No LERO prompt-registry test.
- 🟠 **R19** — No `max_steps` guardrail test.
- 🟠 **R20** — No regression suite.
- 🟠 **R21** — No coverage gate.
- 🟠 **R22** — No mutation or golden-fixture tests.

### 2.5 Documentation gaps

- 🟡 **R23** — No environment-spec doc.
- 🟡 **R24** — No hyperparameter catalogue.
- 🟡 **R25** — No theory + impl + test triad per technique.
- 🟡 **R26** — LERO code flow not diagrammed.
- 🟡 **R27** — Redundant docs.
- 🟡 **R28** — No naming-conventions doc.

### 2.6 UX & infrastructure

- 🟠 **R29** — OVH Jobs page has no parallelism cap.
- 🟠 **R30** — Results-page video generation is manual.
- 🟠 **R31** — Streamlit reloads full CSVs every time.
- 🟠 **R32** — No "diff vs default" view.
- 🟠 **R33** — Cross-experiment page is thin.

### 2.7 Minor

- 🟢 **R34** `.coverage`, `.DS_Store` tracked.
- 🟢 **R35** empty `checkpoints/` tracked.
- 🟢 **R36** `consolidate.py` has no CLI.

**Headline:** project works but rests on implicit contracts, unsafe parallelism, and folder-not-package structure. Every R-item is resolved in a named phase.

---

## 3. Phase 0 — Archive & Clean Slate

**Resolves:** R34, R35 (partially).

### 3.1 Local filesystem

Create `rendezvous_comm/archive01/`:

```text
archive01/
├── README.md
├── docs/          # all current docs/ except this plan
├── results/       # ~430 MB
└── configs/       # frozen reference copies
```

After: `docs/` has only this plan + new docs; `results/` empty; `configs/` stays live; `checkpoints/` deleted.

### 3.2 OVH S3

Only `rendezvous-results`; never touch `rendezvous-code`.

1. Verify local `archive01/results/` exists and is intact.
2. Optionally copy remote → `archive01/results_ovh/`.
3. Delete all prefixes under `rendezvous-results@GRA/` via `scripts/clean_ovh_results.py --dry-run` default.

⚠️ Confirmation gate — commands shown, dry-run first.

### 3.3 `.gitignore`

Add `rendezvous_comm/archive01/results/**`, `.coverage`, `.DS_Store`, `__pycache__/`.

---

## 4. Phase 1 — Environment & Hyperparameter Documentation

**Resolves:** R23, R24, R28.

### 4.1 `docs/environment.md`

1. Big picture — Discovery scenario; "rendezvous" definition.
2. Spatial scales — `2.0 × 2.0` units; agent radius `0.05`; LiDAR `0.35`.
3. Step size / dynamics — VMAS `dt` (measured via `scripts/measure_scenario.py`); substeps, drag, collision force; derived *steps to cross arena*, *steps to exit LiDAR*.
4. Action space — 2 D force; shape, bounds, dynamics class.
5. Observation space — `(self_pos, self_vel, entity_lidar[15], [agent_lidar[12]], [comm[dim_c]])`.
6. Reward formula — explicit per-agent-per-step; team aggregation under `shared_reward=True`.
7. Termination / truncation — `max_steps`; `targets_respawn=False` + cumsum.
8. Batch dimension — leading `num_envs` (600 during training).
9. Scales cheat sheet.

### 4.2 `docs/hyperparameters.md`

Table: name, where set, default, tested range, consumer, typical effect, hidden coupling. Subsection for BenchMARL implicit defaults.

### 4.3 `docs/naming_conventions.md`

`run_id`, `config_tag`, `exp_id`, `family`, `ovh_job_id`: construction, consumers, breakage surface.

---

## 5. Phase 2 — CSV Schema & I/O Contract

**Resolves:** R1, R2, R15, R17, R19.

### 5.1 Freeze schema

Add `rendezvous_comm/domain/schema.py`:

- `HYPERPARAMS_COLUMNS` — all `TaskConfig` + `TrainConfig` fields flattened.
- `METRICS_COLUMNS` — M1–M9 + M7 sample efficiency.
- `IDENTITY_COLUMNS` — `run_id`, `exp_id`, `algorithm`, `seed`, `timestamp`, `git_sha`, `ovh_job_id`, `run_state`, `config_hash`.
- `SWEEP_RESULTS_SCHEMA = IDENTITY + HYPERPARAMS + METRICS`.

Storage / consolidation produce rows matching schema; missing metric → `NaN`; unknown key → `ValueError`.

### 5.2 New tests

1. `tests/test_schema.py` — field ↔ column coverage; round-trip.
2. `tests/test_run_id.py` — stable snapshot; uniqueness.
3. `tests/test_config_propagation.py` — YAML → VMAS scenario attr; guards R19.
4. `tests/test_ovh_submit.py` — prefix correctness; env encryption; command shape.

### 5.3 `docs/io_contract.md`

Folder shape; consolidated-CSV columns; invariants.

---

## 6. Phase 3 — RL Techniques: Theory + Implementation + Tests

**Resolves:** R25.

```text
docs/rl/
├── README.md                 # index + decision tree
├── mappo.md
├── ippo.md
├── qmix.md
├── maddpg.md
├── gnn_policies.md
├── communication.md
├── lero.md
└── ablations/
    ├── mappo_ablations.md
    ├── communication_ablations.md
    └── lero_ablations.md
```

Per-page: theory / implementation / testing / most-promising candidate / ablations link.

Wiring tests: `test_algorithm_wiring.py`, `test_gnn_wiring.py`, `test_communication_wiring.py`.

---

## 7. Phase 4 — LERO Flow Diagrams (Mermaid)

**Resolves:** R26, R27.

Four Mermaid diagrams in `docs/rl/lero.md`: high-level loop, code map, data flow (reward full-state vs observation local), failure modes. Docstring pointers at top of `loop.py`, `scenario_patch.py`, `codegen.py`. Delete stub `docs/lero_phase4_results.md`.

---

## 8. Phase 5 — Clean Architecture **+ Standalone Packaging**

**Resolves:** R7, R8, R37, R38, R39, R40, R41, R42, R43.

This phase has two halves. Half A is packaging (pip-installable, lock file, CI). Half B is the clean-arch refactor, happening *inside* the new package layout so we don't migrate twice.

### 8.1 Half A — Standalone packaging

#### 8.1.1 Target top-level layout

```text
rendezvous_comm/                 # (becomes the standalone repo root)
├── pyproject.toml               # PEP 621 metadata, deps, tool config
├── uv.lock                      # pinned resolution (or requirements.lock)
├── README.md                    # quickstart: install, run, compare
├── LICENSE                      # match parent (GPL-3.0) or explicit choice
├── CHANGELOG.md                 # per-PR entries linked to tests
├── CONTRIBUTING.md              # dev setup, test layers, PR template reference
├── Makefile                     # make test, make qa, make docs, make ci-local
├── .github/
│   └── workflows/
│       ├── ci.yml               # lint, type, unit, smoke
│       └── nightly.yml          # mutation, goldens, L6+L7
├── .pre-commit-config.yaml
├── .gitignore
├── .env.example
├── src/
│   └── rendezvous_comm/         # src-layout: importable package
│       ├── __init__.py          # version constant
│       ├── domain/              # pure: config, schema, metrics, naming
│       ├── application/         # orchestration: runner/, lero/
│       ├── infrastructure/      # effects: storage/, ovh/, llm/, logging_setup
│       └── ui/                  # pages/, theme, ui_utils
├── tests/
├── docs/
├── configs/
├── scripts/
├── notebooks/
└── archive01/                   # gitignored; local-only
```

#### 8.1.2 `pyproject.toml` sketch

```toml
[project]
name = "rendezvous-comm"
version = "0.1.0"
description = "Learning communication protocols for multi-robot rendezvous"
readme = "README.md"
requires-python = ">=3.11"
license = {file = "LICENSE"}
authors = [{name = "Afin"}]
dependencies = [
    "vmas>=1.4",           # pin exact in uv.lock
    "benchmarl>=1.3",
    "torchrl>=0.4",
    "tensordict>=0.4",
    "torch>=2.1",
    "pyyaml",
    "pandas",
    "scipy",
    "numpy",
    "tqdm",
    "imageio",
    "imageio-ffmpeg",
    "matplotlib",
]

[project.optional-dependencies]
lero = ["litellm>=1.40", "cryptography>=41.0"]
ui = ["streamlit>=1.30", "ipywidgets", "markdown"]
ovh = ["python-dotenv"]
dev = [
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "hypothesis",
    "mypy",
    "ruff",
    "pre-commit",
    "mutmut",
    "radon",
]

[project.scripts]
rendezvous-train   = "rendezvous_comm.application.runner.cli:main"
rendezvous-inspect = "rendezvous_comm.scripts.inspect_run:main"
rendezvous-compare = "rendezvous_comm.scripts.compare_vs_archive:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
files = ["src/rendezvous_comm/domain", "src/rendezvous_comm/application"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks integration tests (deselect with -m 'not integration')",
    "slow: marks slow tests",
]
```

Dev install becomes `pip install -e ".[dev,lero,ui,ovh]"`.

#### 8.1.3 Lock file

Choose `uv` (fast, modern) — answer in §19. Command: `uv lock`. Committed; OVH Docker image `pip install -r <(uv export --format requirements-txt)` so local and OVH run the *same* resolved versions.

#### 8.1.4 Sever cross-directory coupling

Audit (already done — 9 files use `sys.path.insert`):

- `Dashboard.py`, `pages/1_Experiments.py`, `pages/2_OVH_Jobs.py`, `pages/3_Results.py`, `pages/4_Cross_Experiment.py` — replace with proper `from rendezvous_comm.ui.pages import ...` after src-layout.
- `src/lero/loop.py` — remove local `sys.path` hack; use package imports.
- Notebooks — `sys.path.insert` replaced with `%pip install -e ..[dev]` cell (or just rely on dev install done once per env).

VMAS import: verify `vmas` is imported only via public API (`from vmas import make_env, …`, `from vmas.simulator.scenario import BaseScenario`, etc.). Any reach into `vmas.scenarios.debug.*` or private modules must be replaced with public equivalents or forked into `rendezvous_comm/domain/scenarios/`.

#### 8.1.5 Entry-point scripts

`src/rendezvous_comm/application/runner/cli.py` replaces top-level `train.py`:

```python
def main() -> None:
    ...  # argparse, load ExperimentSpec, run_single / run_experiment / run_lero
```

Old `train.py` becomes a one-line shim calling the new CLI, preserved until OVH Docker image is updated.

#### 8.1.6 CI

`.github/workflows/ci.yml`:

- Matrix: `python-version: ["3.11"]` (can extend to 3.12).
- Steps: checkout → `uv sync --all-extras` → ruff → mypy → `pytest -m "not integration" --cov` → coverage upload.
- Separate job: build OVH Docker image variant; run L5 integration smoke.

`.github/workflows/nightly.yml`:

- Mutation testing (`mutmut run`) on `domain/`.
- Golden fixture re-run.
- L6 (LERO crash-resume smoke) + L7 (Streamlit Playwright).

#### 8.1.7 Extraction-readiness

At the end of Phase 5 Half A, extraction becomes a one-liner:

```bash
# Future, when you want a standalone repo:
git subtree split --prefix=rendezvous_comm HEAD -b rendezvous-standalone
git push git@github.com:afin/rendezvous-comm.git rendezvous-standalone:main
# ... or ...
cp -r rendezvous_comm ~/Code/rendezvous-comm && cd $_ && git init && git add . && git commit
```

At that point, VMAS is a regular pip dep — no surgery required.

### 8.2 Half B — Clean-architecture layering

Three layers inside `src/rendezvous_comm/`:

- `domain/` — pure logic: `config.py`, `schema.py`, `metrics.py`, `naming.py`. No I/O; imports only stdlib + torch + vmas public types. `mypy --strict`.
- `application/` — orchestration: `runner/{build, run_single, run_sweep, callbacks, lero_adapter}.py`, `lero/{loop, codegen, scenario_patch}.py`. Depends on `domain/` and on Protocols; not on `infrastructure/` concrete classes. `mypy --strict` (stretch).
- `infrastructure/` — effects at the edge: `storage/` (RunStorage, consolidate), `ovh/` (client, paths, jobs), `llm/` (llm_client), `logging_setup.py`.
- `ui/` — `pages/`, `theme.py`, `ui_utils.py` (pure helpers, testable without Streamlit).

#### 8.2.1 Principles enforced

- Single responsibility: ≤ 400 lines per module.
- Dependency inversion via `Protocol`s: `StoragePort`, `OvhPort`, `LlmPort`. Tests pass fakes from `tests/fakes/`.
- No global singletons. Kill `_cfg._cache`.
- Typed boundaries (`dataclass`, `TypedDict`, `Protocol`).
- Pure core; effects at the edge.
- Small functions ≤ 40 lines; cyclomatic ≤ 10 (`radon cc`).

#### 8.2.2 Migration steps (no behaviour change)

Sequential, one module per commit: (1) move to new path, (2) update imports, (3) full suite green, (4) commit. Re-export shims preserve public imports during transition. Codemod `scripts/migrate_imports.py`. Shims deleted at end; grep for stale imports.

#### 8.2.3 Architecture tests

- `tests/architecture/test_layering.py` — import-graph: `domain/` imports no `application/` or `infrastructure/`; `application/` imports no `ui/`.
- `tests/architecture/test_module_size.py` — ≤ 400 lines per file.
- `tests/architecture/test_protocols.py` — every port has a fake in `tests/fakes/`.
- `tests/architecture/test_no_sys_path.py` — no `sys.path.insert` in any shipped module.
- `tests/architecture/test_public_imports.py` — `rendezvous_comm` imports only from the declared `pyproject.toml` dependency list (uses `importlib.metadata`).
- `tests/architecture/test_package_metadata.py` — `rendezvous_comm.__version__` matches `pyproject.toml` version.

### 8.3 Order within Phase 5

1. Scaffold `pyproject.toml`, `uv.lock`, CI, Makefile (Half A-1..A-3, A-6). No code moves yet; old layout still works via existing `sys.path` hacks.
2. Adopt src-layout (`rendezvous_comm/` → `src/rendezvous_comm/`). Update tests. Delete `sys.path.insert` lines.
3. Layering refactor (Half B). One sub-module at a time.
4. Entry-point scripts (A-5). `train.py` becomes shim.
5. Extraction dry-run: `cp -r` into `/tmp/`, `pip install -e .`, run `pytest -m "not integration"` — must pass. This proves the standalone claim.

---

## 9. Phase 6 — Parallel Safety & Crash Recovery

**Resolves:** R9–R14.

### 9.1 Globally-unique paths

Invariant: every write lands at `bucket/<exp_id>/<job_id>/<run_timestamp>__<run_id>/...`.

`<job_id>` = `OVH_JOB_ID` env var on OVH, `uuid.uuid4().hex[:12]` locally (persisted to `input/job_id.txt`).

OVH volume: `{bucket_results}@{region}/{exp_id}/{OVH_JOB_ID}:{mount_results}:rwd`.

Two parallel same-exp jobs now live in different `job_id` folders — cannot collide.

### 9.2 Write-once semantics

- `.tmp/<name>` + `os.rename` on success.
- `output/MANIFEST.json` lists every path + SHA256; consolidation verifies.
- Re-running same config + seed → new `<run_timestamp>` folder. Never overwrites.

### 9.3 Incremental persistence

- `run_state.json` with `status ∈ {RUNNING, DONE, DONE_PARTIAL, CRASHED}`, updated per eval.
- `policy_latest.pt` every `save_interval_frames` (default 250 000).
- BenchMARL CSV already streams — kept.
- Structured logs: `logs/run.jsonl`, `flush=True` per line; plain `.log` tee for humans.
- `scripts/sync_results_to_s3.py` background thread on OVH pushes `logs/` + `output/benchmarl/` every 5 min — even killed jobs leave partial results in S3 before FINALIZING.

### 9.4 Resume-from-checkpoint

`rendezvous-train --resume <run_dir>` loads `policy_latest.pt` + last BenchMARL CSV via `Experiment.load_from_file()`. Config hash mismatch aborts.

### 9.5 Streamlit idempotency

Submit button locked 5 s after click; `submit_training_job(idempotency_key=...)` refuses same key within session; batch queue persisted to `~/.streamlit/rendezvous_batch_state.json`.

### 9.6 Tools

- `scripts/inspect_run.py <run_dir>` — summary of partial state.
- `scripts/rescue_ovh.py <job_id>` — download current S3 state even mid-run.
- `scripts/promote_latest.py <run_dir>` — promote `policy_latest.pt` → `policy.pt`; `status = DONE_PARTIAL`.

### 9.7 What BenchMARL / VMAS give us

- BenchMARL: `Experiment.save_state()`, `restore_env_path` — wired explicitly.
- VMAS: stateless; recreate from seed.
- torchrl: off-policy replay buffers persisted via `buffer.dumps(path)` at each eval (only matters when we introduce MADDPG/QMIX at scale).

### 9.8 Tests

`test_parallel_paths.py`, `test_write_once.py`, `test_run_state_lifecycle.py`, `test_resume_from_checkpoint.py`, `test_manifest.py`, `test_idempotency.py`.

### 9.9 `docs/crash_recovery.md`

Operator decision tree: symptoms → inspection → retry/resume/promote/discard.

---

## 10. Phase 7 — Src & Tests Audit + Regression Suite

**Resolves:** R16, R18, R20, R21, R22, R36.

### 10.1 Audit

1. Every function in `src/rendezvous_comm/**/*.py`: tested? branches? edges?
2. `pytest --cov=src/rendezvous_comm --cov-fail-under=80`; list modules < 80 %.
3. Skipped tests audited; each gets reason + expiry.
4. MEMORY.md incidents → regression tests.

### 10.2 `tests/regressions/`

One file per past incident, dated:

```text
tests/regressions/
├── test_20260322_max_steps_pop.py
├── test_20260416_ovh_prefix_trailing_slash.py
├── test_20260416_lero_closure.py
├── test_20260416_m8_shared_reward.py
├── test_20260416_pip_vmas_drift.py
├── test_20260312_m1_m3_m6_targets_covered.py
└── test_20260321_m5_token_extraction.py
```

### 10.3 No-regression workflow

Eleven rules (numbering restarts here):

1. **Pin versions** — `uv.lock` authoritative for local and OVH.
2. **Golden fixture tests** — canonical 100-frame MAPPO CSV committed; exact on hyperparams, within 3 σ on stochastic metrics.
3. **Coverage gate** — `--cov-fail-under=80` in CI.
4. **Pre-commit** — L1 + L2 fast.
5. **Pre-push** — unit + architecture.
6. **CI on PR** — unit + smoke + coverage + `mypy --strict` on `domain/`.
7. **Mutation testing** — weekly cron on `domain/metrics.py`, `domain/schema.py`.
8. **Property-based tests** — `hypothesis` for `iter_runs()` and metric invariants.
9. **PR template** — "new test for the behaviour changed? link it."
10. **CHANGELOG.md** — entry per PR with test links.
11. **OVH Docker CI runs** — catch env drift the lock file alone can't.

---

## 11. Phase 8 — Streamlit UX + Multi-Experiment Comparison

**Resolves:** R29, R30, R31, R32, R33.

### 11.1 OVH Jobs — parallel submission with cap

Multi-select configs; `max_parallel` slider (default 4, in `configs/ovh.yaml:launch.max_parallel`); "Submit Batch" dispatches and polls; "Kill Batch" cancels queued; idempotency key.

### 11.2 Results — auto videos / auto consolidate

Run Detail auto-generates videos on entry when stale. Overview auto-rebuilds CSVs when run folders are newer than the latest CSV. Manual buttons remain.

### 11.3 Smoothness

`@st.cache_data(ttl=30)`, sidebar last-refresh + "Refresh all", palette via `ui/theme.py`.

### 11.4 Experiments — diff vs default

YAML alongside human-readable diff vs default `TaskConfig` / `TrainConfig`.

### 11.5 Cross-experiment page — full rewrite

1. Selector: multi-select at config granularity; "include archive01" toggle; tag filters (algo, dim_c, model_type).
2. Aggregation toggle: per-seed / per-config / per-family.
3. Plots: M1/M3/M6 overlay with 95 % bootstrap CI; radar on normalized M1–M9; Pareto (M1 vs M3 / M5 / cost); training-curve overlay.
4. Stats: paired Wilcoxon, Mann-Whitney U, Cliff's delta.
5. Export: Markdown booktabs / LaTeX / CSV.
6. Interactive YAML + metric diff between A and B.
7. One-click "Compare current vs archive01" runs §14 acceptance.

### 11.6 Tests

Pure helpers in `ui/ui_utils.py` unit-tested; Playwright smoke on all four pages.

---

## 12. Phase 9 — Candidate Selection & Ablation Pages

**Resolves:** R25 (completion), R27.

- `docs/candidates.md` — most-promising config per technique (YAML path, headline metrics, cost, "when to pick").
- `docs/rl/ablations/*.md` — re-surface archived experiments via `scripts/rebuild_ablation_tables.py`.

---

## 13. Phase 10 — QA Gate / Test Plan

**Resolves:** R21 (gate), R22 (completion).

### 13.1 Layers

- **L1** unit (`pytest -m "not integration"`) — pre-commit.
- **L2** schema + architecture — pre-commit.
- **L3** regressions — PR.
- **L4** goldens — PR.
- **L5** integration smoke (100-frame MAPPO × 2 seeds, ~2 min CPU) — PR.
- **L6** LERO smoke with injected crash + resume — PR.
- **L7** Streamlit Playwright — PR.

### 13.2 Gate

`make qa` = L1–L7; pre-commit = L1+L2; CI = L1–L6; nightly = all + mutation.

### 13.3 Excellence checklist

Living checklist mapping every R-item to resolving PR. Exit only when all 🔴 resolved; all 🟠 fixed or explicitly deferred.

---

## 14. Phase 11 — Sequential Baseline Re-runs & Archive Verification

**This is the phase where we prove everything above works on real data.**

After Phase 10 is green, re-run canonical configs one at a time; each must reproduce the archived result within tolerance before the next starts.

### 14.1 Sequence

| Order | Experiment | Canonical config | Seeds | Est. V100S time |
|---|---|---|---|---|
| 1 | ER1 — no comm | `er1/single_al_abl_combined.yaml` | 5 | ~4 h × 5 |
| 2 | ER2 — with comm | `er2/single_al_lp_sr_bc_dc8_ms400.yaml` | 5 | ~5 h × 5 |
| 3 | ER3 — GNN comm | `er3/single_al_lp_sr_gatv2.yaml` | 5 | ~6 h × 5 |
| 4 | LERO Phase 4 | `lero/l21.yaml` | 3 | ~18 h × 3 |

~6 days OVH wall-clock at `max_parallel=4`.

### 14.2 Per-experiment protocol

1. Pre-flight: `make qa` green.
2. Submit: Streamlit batch, `max_parallel=4`, idempotency key `<exp_id>-<date>-rerun`.
3. Monitor: `scripts/inspect_run.py` on partial results.
4. Rescue on crash: `scripts/rescue_ovh.py`, `--resume`, or `scripts/promote_latest.py`. Resume recovery does not count as a failed reproduction.
5. Consolidate: `consolidate.build_sweep_csv(exp_id)`.
6. Verify: `rendezvous-compare <exp_id> --archive archive01/results/<exp_id>/`.
7. Report: `docs/rerun_reports/<exp_id>.md` auto-generated (tables + plots + verdict).
8. Gate: PASS → next experiment; FAIL → stop and investigate.

### 14.3 Acceptance criteria

Per seed:

- `|M1_new − M1_archive| ≤ 0.10` **or** 95 % bootstrap CIs overlap.
- `|M3_new − M3_archive| / M3_archive ≤ 0.15`.
- `|M4_new − M4_archive| ≤ 0.5`.

Per sweep:

- Paired Wilcoxon on M1: `p > 0.05`.
- Mean M6 within `[archive − 0.05, archive + 0.05]`.

### 14.4 Failure protocol

Stop; investigate in order:

1. Version drift — check lock file honoured by OVH image.
2. Code regression — `git bisect` against archived `provenance.json` commit sha.
3. Seeding drift — raw per-step rewards.
4. Infra drift — CUDA / driver / batch differences; CPU isolation run.

Findings recorded in `docs/rerun_reports/<exp_id>.md` even on failure.

### 14.5 Deliverables

- `scripts/compare_vs_archive.py` (exposed as `rendezvous-compare`).
- `docs/baseline_verification_protocol.md`.
- `docs/rerun_reports/{er1,er2,er3,lero}.md`.
- Updated `docs/candidates.md` if headline numbers shift.

---

## 15. Phase 12 — Advanced Technique Extensions

Only after all Phase 11 verdicts are PASS.

### 15.1 Candidates (pick one per cycle)

Attention-based communication, transformer policy, graph transformer, curriculum learning, intrinsic motivation (RND), hierarchical policy, population-based training, offline pretraining + online fine-tune.

### 15.2 Onboarding checklist (PR-enforced)

1. `docs/rl/<technique>.md` — theory + impl + tests + most-promising candidate.
2. `src/rendezvous_comm/application/<module>.py` — passes architecture tests.
3. `configs/<family>/<technique>_baseline.yaml` + ablation subdir.
4. `tests/test_<technique>_wiring.py` — builds; 1 iteration without NaN.
5. Regression tests for bugs fixed during development.
6. Golden fixture: 100-frame run in `tests/goldens/`.
7. First real OVH run; comparison vs ER1/ER2/ER3 via §11.5.
8. `docs/candidates.md` updated if this becomes a promising contender.
9. Crash-safety proof: SIGKILL one seed at 50 % frames → resume recovers to DONE, no artifact loss.
10. `docs/rl/ablations/<technique>_ablations.md`.

### 15.3 Naming

New families: `configs/a1/`, `a2/` (`a` = advanced) — visually distinct from `er1/er2/er3/` baselines.

Results: `results/a1/<OVH_JOB_ID>/<timestamp>__<run_id>/`.

### 15.4 Comparison protocol

Every report must include vs-baseline comparison with Wilcoxon p-values, Cliff's delta, Pareto placement, and honest "when this wins / loses" prose.

### 15.5 Deliverables

- `docs/advanced_technique_template.md`.
- `scripts/new_technique.py` — scaffolder (`python -m rendezvous_comm.scripts.new_technique attention-comm a1`).
- Per-extension: `docs/reports/<family>.md`.

---

## 16. Deliverables Checklist

### 16.1 Packaging / infrastructure (new in v4)

- `pyproject.toml`
- `uv.lock`
- `LICENSE`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `.github/workflows/{ci,nightly}.yml`
- `.pre-commit-config.yaml`
- `Makefile`

### 16.2 Docs

- `docs/environment.md`, `docs/hyperparameters.md`, `docs/io_contract.md`, `docs/naming_conventions.md`
- `docs/rl/{README,mappo,ippo,qmix,maddpg,gnn_policies,communication,lero}.md`
- `docs/rl/ablations/*.md`
- `docs/candidates.md`, `docs/audit_report.md`, `docs/qa_plan.md`, `docs/crash_recovery.md`
- `docs/baseline_verification_protocol.md`
- `docs/rerun_reports/{er1,er2,er3,lero}.md`
- `docs/advanced_technique_template.md`
- `docs/reports/<family>.md` per advanced technique

### 16.3 Deleted / archived

- `docs/lero_phase4_results.md` → merged
- `docs/archive/*` → `archive01/docs/archive/*`
- `results/*` → `archive01/results/*`
- OVH `rendezvous-results` → emptied

### 16.4 Code

- `src/rendezvous_comm/{domain,application,infrastructure,ui}/`
- `src/rendezvous_comm/application/runner/cli.py` (replaces `train.py`)
- `scripts/clean_ovh_results.py`, `scripts/qa.sh`, `scripts/measure_scenario.py`
- `scripts/inspect_run.py`, `scripts/rescue_ovh.py`, `scripts/promote_latest.py`, `scripts/sync_results_to_s3.py`
- `scripts/compare_vs_archive.py`, `scripts/new_technique.py`, `scripts/rebuild_ablation_tables.py`
- `scripts/migrate_imports.py` (temporary)

### 16.5 Tests

- Schema / run_id / config propagation / ovh submit: Phase 2.
- Algorithm / GNN / communication wiring: Phase 3.
- LERO smoke, idempotency, parallel paths, write-once, run-state, resume, manifest: Phase 6.
- Architecture (layering, size, protocols, no-sys-path, public-imports, package-metadata): Phase 5.
- Regressions (one per MEMORY.md incident): Phase 7.
- Goldens (MAPPO 100-frame): Phase 7.
- Integration (smoke, LERO crash-resume, Streamlit Playwright): Phase 10.
- UI utils / comparison stats / comparison export: Phase 8.

### 16.6 Streamlit

- `src/rendezvous_comm/ui/pages/1_Experiments.py` — diff view
- `src/rendezvous_comm/ui/pages/2_OVH_Jobs.py` — batch + cap + idempotency
- `src/rendezvous_comm/ui/pages/3_Results.py` — auto videos + auto rebuild
- `src/rendezvous_comm/ui/pages/4_Cross_Experiment.py` — full rewrite

---

## 17. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Archiving breaks notebook paths | Med | Grep hardcoded paths; `sys.path` disappears after Phase 5. |
| OVH S3 delete irreversible | High | Dry-run; optional backup; confirmation gate. |
| Streamlit batch spams OVH | Med | `max_parallel`, queue, idempotency key. |
| Packaging refactor breaks behaviour | Med | `cp -r` extraction smoke test; full suite green per commit; goldens. |
| Pinning `vmas` to a release breaks an unknown coupling | Med | Verify with `cp -r` smoke test before deleting parent's editable install. |
| Resume loads mismatched config | High | Config hash in `run_state.json`. |
| S3 sync races FINALIZING | Low | SIGTERM pause; `--no-overwrite`. |
| Baseline re-runs don't reproduce | Med | Wide-then-narrow tolerance; `git bisect` protocol; lock file. |
| Advanced-technique PR skips checklist | Med | PR template; CI fails on missing items. |

---

## 18. Extraction to Standalone Repo (future)

After Phase 5 completes, extraction is optional but frictionless:

```bash
# Option A — preserve history
git subtree split --prefix=rendezvous_comm HEAD -b rendezvous-standalone
git push git@github.com:afin/rendezvous-comm.git rendezvous-standalone:main

# Option B — fresh history
cp -r rendezvous_comm ~/Code/rendezvous-comm
cd ~/Code/rendezvous-comm && git init && git add . && git commit -m "Initial import"
```

Then on a fresh machine:

```bash
git clone git@github.com:afin/rendezvous-comm.git
cd rendezvous-comm
uv sync --all-extras         # installs vmas, benchmarl, torchrl, ... from pyproject.toml
make qa                      # L1–L7 pass out of the box
rendezvous-train configs/er1/single_al_abl_combined.yaml
```

No parent repo, no `setup.py develop`, no `sys.path` hacks.

---

## 19. Open Questions

1. Archive naming — `archive01/` or `archive_2026-04-20/`?
2. OVH backup before wipe — full / CSVs only / skip?
3. Configs — copy-archive only (recommended) or full archive?
4. `max_parallel` default — 4, in `configs/ovh.yaml:launch.max_parallel`?
5. LERO canonical doc — `docs/rl/lero.md` (recommended) or keep `docs/lero.md`?
6. Technique scope — MAPPO, IPPO, QMIX, MADDPG, GNN, communication, LERO; add more?
7. QA gate blocking or advisory?
8. Phase ordering as listed; swaps?
9. **Lock-file tool — `uv` (recommended), `pip-tools`, or `poetry`?**
10. Golden fixture — every PR or nightly?
11. `mypy --strict` — `domain/` only or also `application/`?
12. S3 sync cadence — 5 min default?
13. Checkpoint interval — 250 k frames default?
14. Baseline rerun seeds — 5 / 5 / 5 / 3 for ER1 / ER2 / ER3 / LERO?
15. Acceptance tolerance — M1 ±0.10 / M3 ±15 % / M4 ±0.5 / Wilcoxon p > 0.05?
16. First advanced technique — attention / transformer / curriculum / …? (can defer after Phase 11)
17. Archive comparison in Streamlit — default on or opt-in?
18. Sequential re-runs — strict one-at-a-time, or allow next to start at 80 % of current?
19. **Package name — `rendezvous-comm` (PyPI-style) or `rendezvous_communication_learning`?**
20. **License — match parent GPL-3.0 or choose (MIT / Apache-2.0)?**
21. **Python minimum — 3.11 (modern) or 3.10 (wider compat)?**
22. **Entry-point names — `rendezvous-train` / `rendezvous-inspect` / `rendezvous-compare`? Any preferences?**
23. **Extraction timing — extract to standalone repo immediately after Phase 5, after Phase 10, or after Phase 11?**

---

## 20. What I Will NOT Do Without Further Approval

- Delete anything outside `archive01/`.
- Touch the OVH `rendezvous-code` bucket.
- Modify any YAML in `configs/` (copy to `archive01/` only).
- Rename existing schema fields (only additions / freezing).
- Launch any training job (local or OVH).
- Push to `main` without a green `make qa`.
- Start Phase 12 before Phase 11 PASS.
- Actually extract to a standalone repo — the groundwork lands; the `git subtree split` / repo push is your call.

---

**Next step:** read through, mark up, answer §19 (now 23 questions). Once approved, I start Phase 0 with the S3 dry-run listing + the local archive move — both fully reversible if we stop immediately after.
