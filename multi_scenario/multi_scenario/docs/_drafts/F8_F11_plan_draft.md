I have a complete picture. Now I'll write the comprehensive plan. Given the request asks for 1500-2500 lines of detailed planning, I'll structure this carefully and exhaustively.

---

# Multi-Scenario — Phase 8/9/10/11 Plan Update

> Companion update to `multi_scenario/implementation_plan.md` covering five workstreams: (1) reproducing rendezvous_comm headline numbers on `discovery`, (2) building LERO from scratch hex-cleanly, (3) wiki-style docs, (4) project naming + repo extraction, (5) per-scenario experiment campaigns. Read Section A first — it lists every decision I need from you before I lock anything in.

---

## Section A — Open questions for the user (read first)

These are gates. Each one blocks or shapes one or more sub-phases below. Where I have a recommendation, I label it `Recommend:`; where I genuinely need input, I label it `Need input:`.

### A.1 Reproducibility scope and threshold

1. **Reproducibility threshold for ER1.** rendezvous_comm reports ER1 M1 ≈ 0.405 on `er1_al_lp_sr_cr035` (k=2, n=4, t=4, ms=200, cr=0.35) at 10M frames, seed 0. Bit-exact reproduction is impossible (different process trees, different BenchMARL build, possibly different VMAS pip vs source). What threshold counts as "reproduced"?
   - Recommend: ±10% absolute on M1 at single seed (so 0.365 ≤ M1 ≤ 0.445 passes); OR ±2σ across seeds [0,1,2] if multi-seed budget exists. Pick one; do not let me invent it.
2. **Reproducibility threshold for S3b-local.** Headline is M1 ≈ 0.88 (single seed). The doc itself flags only one run exists.
   - Recommend: run seeds [0,1,2], pass if **mean ≥ 0.70 AND best ≥ 0.80**. The 0.88 number is fragile (the LLM has to evolve the `hold_signal` feature in iter 3); seed variance could legitimately put a single run at 0.40. Need your confirmation.
3. **Do we replicate the rendezvous_comm `s3b_local_replicate_s{0,1,2}.yaml` runs for cross-validation first** (using the rendezvous_comm code), to set our reference numbers? Without that, we don't know what the actual seed-variance is.
4. **Is k=2 obs-only the only LERO target, or do we also need k=1 (S1, L8) and the failed reward-design baselines (P1, L1, L4) for completeness?** I recommend only S3b-local (the headline) plus L8 (k=1, easy win, quick smoke). Need your call.
5. **Frames budget per LERO candidate.** rendezvous_comm uses 1M-frame eval × 3 candidates × 4 iterations = 12M frames per LERO run, plus 10M frame final training = 22M frames per run. At 1M frames ≈ 8 min on 8-core CPU OVH = ~3h per LERO run. Are we OK with that on OVH cost? If not, we drop to 0.5M eval (M1 sensitivity gets noisier).

### A.2 LLM provider, keys, cost cap

6. **Which LLM provider?** rendezvous_comm uses LiteLLM as the broker, defaulting to `gpt-5.4-mini` (S3b-local config) or Claude. Three options:
   - (a) **OpenAI** (gpt-5.4-mini for parity with the headline result).
   - (b) **Anthropic** (claude-sonnet-4-7 — likely better code quality but no rendezvous_comm reference numbers; reproducibility argument weakens).
   - (c) **OVH-hosted endpoint** (you have OVH already; no per-token spend if it's a flat-rate node).
   - Recommend: (a) for the **reproducibility runs** (must match headline LLM), then expose all three behind the LLM port for future use. Need confirmation.
7. **API keys.** Where do they live?
   - Recommend: `~/.multi_scenario/.env` (user-level, gitignored), loaded by the LLM adapter. Backwards-compatible with the existing `.env` already in the project root. Need confirmation it's OK to keep `.env` at project root for now.
8. **Cost cap.** A LERO run with `gpt-5.4-mini` at ~3000 tokens/candidate × 12 candidates × 4 iter ≈ 150k tokens ≈ $0.05 per LERO run. Plus retries. Default cap?
   - Recommend: **$5 per single LERO run, $50 per sweep**, hard-stop in the LLM adapter with a clean exception. Need confirmation.
9. **Disk-cache the LLM responses?** rendezvous_comm has `llm_cache.py` on disk. With identical (model, messages, seed) → identical response, this is free reproducibility. But it shadows real changes if you forget to invalidate.
   - Recommend: ship the cache, default `mode=write_only` (always write, never read) so a re-run regenerates fresh. `mode=read_write` is opt-in for reproducibility tests. Need confirmation.

### A.3 LERO design choices

10. **Reward + obs evolution, or obs-only by default?** S3b-local is obs-only; reward LERO reward-hacks at k=2. Default in the new code:
    - Recommend: **`evolve_reward=False, evolve_observation=True` as the YAML default** (reflects the actual finding). Reward LERO stays available behind a flag. Need confirmation.
11. **Meta-prompting (LERO-MP / outer loop) — included in F9 or deferred to F12?**
    - Recommend: design F9 with a clean extension point (`PromptComposer` strategy interface), but ship F9 with only the inner loop. Wire meta-prompting in F12 once inner loop is validated. Need your call on whether to attempt both at once.
12. **Per-candidate seed derivation.** rendezvous_comm derives a per-candidate OpenAI seed via SHA(run_id, seed, iter, cand_idx). Keep that, or use simple `seed * 1000 + iter * 100 + cand_idx`?
    - Recommend: keep the SHA derivation (cleaner failure mode if any term is shadowed). Need confirmation.
13. **Reward-clip default.** Paper says raw rewards; rendezvous_comm clips to ±50 for stability.
    - Recommend: default ±50 in `LeroConfig` (matches headline run). Disable via `reward_clip: null` for paper-faithful runs. Need confirmation.
14. **Best-checkpoint policy saving.** rendezvous_comm `lero.md §5.1` flags this as a missing feature; LERO-MP added it. Should F9 include best-checkpoint from day one?
    - Recommend: yes, include it. Cheap to add and fixes the eval-vs-final degradation problem (S3a_gpt5: 0.86 → 0.09). Need confirmation.
15. **Whitelist-strict mode (`whitelist_strict=true`).** LERO-MP default. Forces fairness violation at runtime when the LLM-generated obs function reads forbidden global keys.
    - Recommend: default `true` for `obs_state_mode=local`. The reproducibility headline run was technically `false` — but turning it on doesn't change the result and prevents future drift. Need confirmation.

### A.4 Saving & traceability

16. **Per-step rollouts: save them or not?** Discovery k=2 with 200 envs × 400 steps × 100 eval episodes × 4-dim per-agent state = ~50 MB per eval. Across 12 candidates × 4 iter × 1 LERO run = ~2.5 GB.
    - Recommend: save them only when `cfg.runtime.storage.save_rollouts: true` (default false). Always save the eval-episode summary (M1-M9 per episode + final reward). Need confirmation.
17. **LLM trace verbosity.** Every prompt/response/reasoning gets saved per-candidate. Two questions:
    - (a) Save the **full conversation** (system + all messages) per call, or just the delta added that turn? Recommend: full snapshot per call (easier to grep, costs ~20 KB per file). Need confirmation.
    - (b) For Anthropic-style "reasoning" / "thinking" tokens (extended thinking enabled), save them in a separate `reasoning.json`. Need confirmation we want this.
18. **Where do LERO traces live in the run dir?**
    - Recommend the layout under `<run_dir>/output/lero/iter_<n>/cand_<m>/{prompt.json, response.json, reasoning.json, generated_code.py, eval_metrics.json, fitness.json}` plus `<run_dir>/output/lero/{evolution_history.json, fallback_chain.json, final_metrics.json, best_reward.py, best_obs.py}`. Need confirmation.
19. **What gets aggregated into `runs.csv`?** Currently `record_type ∈ {final, eval}`. For LERO add `record_type=lero_iter_candidate` with columns `iter, candidate_idx, eval_M1..M9, fitness_rank, fallback_outcome` so the cross-run CSV is the single queryable source for LERO post-hoc.
    - Recommend: yes, add it. Need confirmation it's worth the schema bump.

### A.5 Docs format

20. **mkdocs vs plain MD?** The user asked for a recommendation.
    - Recommend: **mkdocs-material**. Reasons: (a) you get search out of the box; (b) `mkdocs serve` gives live preview at zero CI cost; (c) GitHub Pages deployment is one workflow file; (d) topic-based navigation is declarative in `mkdocs.yml`; (e) hyperlink validation via `mkdocs build --strict` catches dead links in CI; (f) survives moving to a new repo because all deps are declared. Cost: one extra dev dep + ~30 min setup.
    - Counter-argument for plain MD: zero-tooling, every rendering surface (GitHub, IDE, VS Code preview) handles it. If you're allergic to docs-tooling sprawl, plain MD is fine; we just lose search and link validation. Need your call.
21. **Hosting.** If mkdocs: GitHub Pages (free, sufficient) vs Read the Docs (more features, but the project doesn't need them). Recommend GitHub Pages.

### A.6 Project naming + extraction

22. **Project name.** `multi_scenario` is a placeholder. Name candidates (judge against: short, pronounceable, evocative of the goal, not already a PyPI package, not generic):
    - **`coopvmas`** — explicit (cooperative + VMAS); ugly stem; PyPI-free.
    - **`marlbench`** — descriptive; risk of confusion with BenchMARL (the upstream).
    - **`herd`** — evocative (multi-agent), short, pronounceable; PyPI has a `herd` package (event-sourcing, abandoned 2017 — checkable).
    - **`flock`** — overlaps with the `flocking` scenario name; confusing.
    - **`covenant`** — multi-agent agreement metaphor; pretentious.
    - **`coordbench`** — coordination benchmark; terse, available.
    - Recommend: **`coordbench`** if you want descriptive; **`herd`** if you want evocative. Need your call OR your own suggestion.
23. **GitHub org for the new repo.** Personal account `@afin` vs an org? Need input.
24. **History preservation strategy.** Three options:
    - (a) **`git filter-repo --path multi_scenario/ --path-rename multi_scenario/:`** — preserves all commits that ever touched `multi_scenario/`, rewrites paths so the new repo root has no `multi_scenario/` prefix. Loses commits that *only* touched the parent VMAS code (those weren't relevant anyway).
    - (b) **`git subtree split --prefix=multi_scenario --branch=extracted`** — slower (re-walks history), produces a branch you can push to a fresh remote.
    - (c) **Fresh import** — `cp -r multi_scenario /tmp/new_repo && cd /tmp/new_repo && git init && git add . && git commit -m "Initial import from VMAS monorepo"`. Loses all history but is one-command and clean.
    - Recommend: (a) `git filter-repo`. Modern, fast, well-supported. Need your call.
25. **License.** Defer was already noted in F10.4. Options: GPL-v3 (parent VMAS — copyleft, ensures derivatives stay open), MIT (permissive, max adoption), Apache-2.0 (permissive + patent grant). Need input. If we're depending on VMAS at runtime (not vendoring), GPL parent doesn't force GPL on us; we have free choice.
26. **Are we keeping the Streamlit FE in the new repo or splitting it?** Open question already raised in §6 of the existing plan; needs to be resolved before F10.4. Recommend: keep in the same repo — the FE is the primary user surface, splitting buys nothing.

### A.7 Experiment campaign scope

27. **How many seeds per (scenario, algorithm) cell** for the Phase 11 baselines? rendezvous_comm runs single seed for most; their headline ER1 is single-seed.
    - Recommend: **3 seeds for headline cells, 1 seed for ablations**. Need confirmation.
28. **Which algorithms per scenario?** All 6 (MAPPO/IPPO/MADDPG/IDDPG/ISAC/MASAC), or a subset? Compute multiplier matters.
    - Recommend: 4 scenarios × {MAPPO, IPPO, MADDPG} × 3 seeds = 36 runs for the baseline matrix. Adds {ISAC, MASAC, IDDPG} only if MAPPO/IPPO show distinct failure modes per scenario. Need confirmation.
29. **Heuristic baseline (F8.1 in the existing plan) — keep, or drop?** rendezvous_comm doesn't have one. Adds complexity, value is marginal once we have ER1 trained.
    - Recommend: drop. Need confirmation.
30. **Cross-scenario synthesis report (F11.5) — Streamlit page or Jupyter notebook?**
    - Recommend: Streamlit page (consistent with existing dashboard). Need confirmation.

### A.8 Data analysis depth

31. **What's the canonical "did we reproduce the result" artifact?** A markdown report? A Jupyter notebook? A Streamlit page?
    - Recommend: a Streamlit page `pages/5_reproducibility.py` that reads `runs.csv` for `experiments/discovery/baseline/` + `experiments/discovery/lero/`, computes mean/std M1 per exp_id, and shows a side-by-side table vs the rendezvous_comm reference numbers (hardcoded as a Python dict). Need confirmation.
32. **Should LERO traces be queryable via SQL** (DuckDB over the JSON traces)? Or is grep + jq enough?
    - Recommend: not at first. Filesystem layout is queryable enough. Add DuckDB if/when we run >100 LERO runs and the manual grep gets painful.

### A.9 Backwards compatibility

33. **Adding the LERO `lero:` config block — STRICT-mode rejection on existing YAMLs?** Pydantic-strict configs reject unknown keys today. The new `lero:` section needs to be Optional on `ExperimentConfig`.
    - Recommend: `lero: LeroSection | None = None`. Existing baseline configs continue to parse; only LERO configs include the section. Need confirmation.
34. **Run-folder layout for LERO runs.** F8/F11 plan currently has `experiments/<scenario>/<exp_type>/<run_id>__<timestamp>/`. For LERO, do we add a sub-prefix (`<run_id>__lero__<timestamp>/`) or rely on the `lero/` subdir of `output/` to discriminate?
    - Recommend: the latter — keep the same folder structure; LERO-specific artifacts live under `output/lero/`. Streamlit shows a "LERO" badge if `output/lero/` exists. Need confirmation.

---

## Section B — Renumbered/expanded plan sections

### F8 — Reproducibility validation (NEW SCOPE)

> **Status change.** F8 was originally "First cross-scenario baseline ablation". The user's new scope replaces that: F8 becomes pure reproducibility (single scenario = discovery, two configs = ER1 + S3b-local). The cross-scenario campaign moves to **F11**. The original F8.1 (heuristics) is dropped pending A.29.
>
> **Hard prerequisite.** F8 cannot start until F2.4.2 (BenchMARL model architecture knobs) lands — without it, our MAPPO uses BenchMARL's default MLP, which may differ from what rendezvous_comm built. **Action:** confirm `MlpConfig.get_from_yaml()` returns the same defaults locally; if not, F2.4.2 is a hard blocker, not deferred.

#### F8.0 — Pre-flight: verify rendezvous_comm headline still reproduces in its own repo (XS, ½ day)

**Goal:** before we start porting, run `rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml` and `rendezvous_comm/configs/lero/s3b_local_replicate_s{0,1,2}.yaml` in *their own repo* on the OVH stack. Establish the reference numbers we're aiming at (rather than trusting the doc).

- Files to edit: none (use rendezvous_comm as-is).
- Files to create (in this repo): `multi_scenario/docs/reproducibility/reference_numbers.md` with a table `{exp_id, seed, M1, M2, M6, run_url_in_rendezvous_comm}`.
- Done-criteria: 1 ER1 run + 3 S3b-local runs (3 seeds) executed in rendezvous_comm; numbers tabled.
- Dependency: A.1 (threshold), A.3 (which runs to replicate), A.6 (LLM provider).

#### F8.1 — Port ER1 config to multi_scenario YAML schema (S, ½ day)

**Goal:** translate `rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml` → `multi_scenario/experiments/discovery/baseline/configs/er1_cr035.yaml` using our nested `experiment / scenario / algorithm / training / evaluation / runtime` schema.

- Files to create:
  - `multi_scenario/experiments/discovery/baseline/configs/er1_cr035.yaml`
  - `multi_scenario/experiments/discovery/baseline/configs/er1_cr035_s1.yaml` (seed 1)
  - `multi_scenario/experiments/discovery/baseline/configs/er1_cr035_s2.yaml` (seed 2)
- Files to edit: none (assuming `evaluation_episodes=200` already supported; verify by a config-validation test).
- Tests to add (`tests/reproducibility/test_er1_config_parity.py`):
  - One parametric test per task field (`n_agents`, `n_targets`, `agents_per_target`, `lidar_range`, `covering_range`, `use_agent_lidar`, `n_lidar_rays_entities`, `n_lidar_rays_agents`, `targets_respawn`, `shared_reward`, `agent_collision_penalty`, `covering_rew_coeff`, `time_penalty`, `max_steps`) asserting our parsed config equals the rendezvous_comm field. Reference: `rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml`.
  - One test per training field (`max_n_frames` → `max_iters * frames_per_batch`; `gamma`, `lr`, etc.).
  - The CR035 training field mapping is the load-bearing one — `max_n_frames=10_000_000`, `frames_per_batch=60_000`, → `max_iters=167`. Validator must fire if these drift.
- Done-criteria: `pytest tests/reproducibility/test_er1_config_parity.py -x` green; `multi-scenario validate experiments/discovery/baseline/configs/er1_cr035.yaml` exits 0.
- Hex concerns: none (config is data).
- Dependencies: F8.0; A.33 (schema accepts the section).

#### F8.2 — Run ER1 and validate reproducibility (M, 2 days OVH-bound)

**Goal:** execute `er1_cr035.yaml` × 3 seeds on OVH; compare M1 to F8.0's reference numbers; pass/fail per A.1's threshold.

- Files to create:
  - `multi_scenario/scripts/run_er1_reproducibility.py` — a thin sweep launcher that submits the 3 OVH jobs, waits, pulls results, runs the comparison script.
  - `multi_scenario/scripts/compare_to_reference.py` — reads our `runs.csv` + the F8.0 reference dict; prints a pass/fail table; non-zero exit on fail.
- Files to edit: none.
- Tests to add (`tests/reproducibility/test_compare_to_reference.py`): unit-test the comparison logic with a fixture `runs.csv` (mock-OVH passing case + mock-OVH failing case).
- Done-criteria: `python scripts/compare_to_reference.py --exp er1_cr035` prints `PASS` and exit 0. The Streamlit reproducibility page (F8.6) shows the same numbers.
- Hex concerns: scripts are entry points; they call into application/CLI layer only.
- Dependencies: F8.1; F9.0–F9.6 NOT required (ER1 is pure baseline, no LLM); A.1, A.7.

#### F8.3 — Port LERO architecture (gates F8.4)

**Block dependency only.** F8.3 = "land F9.0 through F9.6". See F9 below for sub-phases. Until F9.6 completes (evolutionary loop produces a candidate that trains successfully), F8.4 cannot start.

#### F8.4 — Port S3b-local config + run + validate (M, 3 days OVH-bound)

**Goal:** translate `rendezvous_comm/configs/lero/s3b_local.yaml` → `multi_scenario/experiments/discovery/lero/configs/s3b_local.yaml`; run ×3 seeds; compare to F8.0's S3b-local reference numbers per A.2's threshold.

- Files to create:
  - `multi_scenario/experiments/discovery/lero/configs/s3b_local.yaml` (single-seed 0)
  - `multi_scenario/experiments/discovery/lero/configs/s3b_local_s1.yaml`
  - `multi_scenario/experiments/discovery/lero/configs/s3b_local_s2.yaml`
- Files to edit:
  - `multi_scenario/src/multi_scenario/domain/models/config.py` — add `lero: LeroSection | None = None`. **Strict-mode** check: `lero` is disallowed unless `algorithm.type` is the LERO meta-algorithm OR a `lero:` block is explicitly present.
- Tests to add (`tests/reproducibility/test_s3b_local_config_parity.py`): parametric tests per LERO field (`n_iterations`, `n_candidates`, `top_k`, `eval_frames`, `eval_episodes`, `full_frames`, `evolve_reward`, `evolve_observation`, `reward_mode`, `obs_state_mode`, `bonus_scale`, `reward_clip`) asserting parity with `rendezvous_comm/configs/lero/s3b_local.yaml`.
- Done-criteria: 3 LERO runs complete on OVH; `compare_to_reference.py --exp s3b_local` prints PASS per A.2.
- Hex concerns: `LeroSection` lives in `domain/models/config.py` (parsed-data class only, no LLM imports).
- Dependencies: F8.3 (so F9.0–F9.6); A.1, A.2, A.7.

#### F8.5 — Deep data analysis: gap-fill the saving layer (M, 2 days)

**Goal:** make the run dir auditable end-to-end. Define exactly what "saved everything relevant" means and add anything missing.

**The saved-data inventory (canonical list):**

| Artifact | Path under run dir | Source (which adapter writes) | Schema doc |
|---|---|---|---|
| Resolved YAML config | `input/config.json` | `LocalStorageAdapter.save_input` | `docs/run_layout.md §config` |
| Provenance | `input/provenance.json` | `provenance/writer.py` | `docs/run_layout.md §provenance` |
| Run lifecycle log | `logs/run.log` | `file_logger.py` | (free-form) |
| Per-eval metrics summary | `output/metrics.json` | `ExperimentService` | `docs/run_layout.md §metrics` |
| Per-eval per-episode raw | `output/eval_episodes.json` | `LocalStorageAdapter.save_eval_episodes` | `docs/run_layout.md §eval_episodes` |
| BenchMARL native scalars | `output/benchmarl/*/scalars/*.csv` | BenchMARL itself | `docs/run_layout.md §benchmarl` |
| Trained policy weights | `output/benchmarl/*/checkpoints/checkpoint_*.pt` | BenchMARL itself | (referenced via `report.json`) |
| Best-checkpoint policy (peak M1) | `output/benchmarl/*/checkpoints/checkpoint_peak_M1.pt` | F9.x peak callback | `docs/run_layout.md §peak_checkpoint` |
| Manifest | `output/report.json` | `report_builder.py` | `docs/run_layout.md §report` |
| Eval videos | `output/videos/{before,after}_training.mp4` | `video/recorder.py` | (binary) |
| Run state | `run_state.json` | `local_storage` | `docs/run_layout.md §run_state` |
| **NEW** Per-step rollout (opt-in) | `output/rollouts/eval_*.parquet` | F8.5.A new writer | `docs/run_layout.md §rollouts` |
| **NEW** LERO iter/cand artifacts | `output/lero/iter_<n>/cand_<m>/*` | F9.3 trace writer | `docs/lero/trace_layout.md` |
| **NEW** LERO evolution history | `output/lero/evolution_history.json` | F9.6 orchestrator | same |
| **NEW** LERO fallback chain | `output/lero/fallback_chain.json` | F9.6 orchestrator | same |
| **NEW** LERO best functions | `output/lero/best_reward.py`, `best_obs.py` | F9.6 orchestrator | same |
| **NEW** LERO final metrics | `output/lero/final_metrics.json` | F9.6 orchestrator | same |
| **NEW** LERO LLM provider info | `output/lero/llm_provenance.json` | F9.1 LLM adapter | model + version + system_fingerprint + library_version |

**Sub-phases:**

- **F8.5.A — Per-step rollouts opt-in writer** (S, ½ day):
  - Files to create: `multi_scenario/src/multi_scenario/adapters/storage/rollout_writer.py` writing parquet (uses `pyarrow`; add to deps). Schema: `(env_idx: int, step: int, agent_idx: int, action: float[A], reward: float, obs: float[O], info_*: ...)` long-form.
  - Files to edit: `domain/models/config.py` (add `runtime.storage.save_rollouts: bool = False`); `application/experiment_service.py` (call writer if flag on); `adapters/algorithms/benchmarl_base.py::evaluate` (yield the raw rollout TD).
  - Tests: `tests/unit/test_rollout_writer.py` round-trip; `tests/integration/test_save_rollouts_smoke.py` smoke run with flag on.
  - Hex concerns: `RolloutWriter` is an adapter; protocol stays in `domain/ports/storage.py` as an optional method (or as a separate `RolloutSink` Protocol — recommend the latter, it's cleaner).
  - Done: smoke run with `save_rollouts: true` produces `output/rollouts/eval_0.parquet` with the documented schema.

- **F8.5.B — Reproducibility Streamlit page** (S, ½ day):
  - Files to create: `multi_scenario/src/multi_scenario/frontend/pages/5_Reproducibility.py` reading our runs.csv vs the hardcoded reference dict.
  - Files to edit: `frontend/streamlit_app.py` (page registration).
  - Tests: `tests/unit/test_reproducibility_page.py` using `streamlit.testing.v1.AppTest` (consistent with F7.7.C1).
  - Done: `streamlit run frontend/streamlit_app.py` → page 5 shows the comparison table.

- **F8.5.C — `runs.csv` schema bump for LERO rows** (S, ½ day):
  - Files to edit: `adapters/storage/runs_csv.py` (add `record_type=lero_candidate` rows + columns `iter, candidate_idx, fitness_rank, fallback_outcome`).
  - Tests: `tests/unit/test_runs_csv_lero_rows.py` golden-file test.
  - Done: a LERO run produces extra `record_type=lero_candidate` rows in `runs.csv`; baseline runs unaffected.

- **F8.5.D — `docs/run_layout.md` rewrite** (XS, 1h):
  - Files to edit: `docs/run_layout.md` (add the inventory table from above; document each schema section).
  - Done: every artifact in the inventory has a documented schema.

- **F8.5.E — Run-dir audit script** (XS, 1h):
  - Files to create: `multi_scenario/scripts/audit_run_dir.py` that takes a `<run_dir>` and asserts every documented artifact exists (LERO ones only if `output/lero/` exists). Used in CI on every smoke run output.
  - Tests: `tests/integration/test_audit_run_dir.py` red on a corrupted run dir, green on a clean one.
  - Done: CI nightly runs the script over all latest smoke runs.

**F8.5 done-criteria:** an analyst with no prior context can take any run dir, run `audit_run_dir.py`, get a green and then read the documented schemas to extract everything they need. No tribal knowledge required.

#### F8.6 — Reproducibility validation gate (XS, 1h)

- Validation that all of F8.1–F8.5 sign off as a unit before F11 starts.
- Done: F8.0 reference numbers, F8.2 ER1 runs, F8.4 S3b-local runs all pass A.1/A.2 thresholds; F8.5 audit script green; you sign off.

---

### F9 — LERO core implementation

> Hex-architecture-clean re-design. The rendezvous_comm `src/lero/` is the *reference for what worked*, not a code blueprint. Domain knows nothing about LLMs; LLM is one adapter behind a port. The key innovation vs rendezvous_comm: every layer (LLM call, prompt rendering, code extraction, scenario patching, evolutionary loop, trace writing) sits behind a Protocol so the inner loop can be unit-tested in isolation against fakes.

#### F9.0 — Domain models (M, 1 day)

**Goal:** define the data classes LERO operates on. No LLM imports, no I/O.

- Files to create:
  - `src/multi_scenario/domain/models/lero/__init__.py`
  - `src/multi_scenario/domain/models/lero/config.py`:
    - `LeroSection(BaseModel)` — Pydantic-strict, mirrors `rendezvous_comm/src/lero/config.py::LeroConfig`. Fields: `n_iterations: int (gt=0)`, `n_candidates: int (gt=0)`, `top_k: int (ge=1)`, `eval_frames: int (gt=0)`, `eval_episodes: int (gt=0)`, `full_frames: int (gt=0)`, `evolve_reward: bool = False`, `evolve_observation: bool = True`, `reward_mode: Literal["replace", "bonus"] = "replace"`, `obs_state_mode: Literal["global", "local"] = "local"`, `bonus_scale: float (gt=0) = 0.5`, `reward_clip: float | None = 50.0`, `whitelist_strict: bool = True` (per A.15), `peak_checkpoint: bool = True` (per A.14), `cache_mode: Literal["off", "read_write", "read_only", "write_only"] = "write_only"` (per A.9).
    - `LlmSection(BaseModel)` — fields: `model: str`, `temperature: float = 0.8`, `max_tokens: int | None = None`, `api_base: str | None = None`, `api_key_env: str = "OPENAI_API_KEY"` (note: never store the key in YAML — only the env-var name), `prompt_version: str = "v2_fewshot_k2_local"`, `max_retries: int = 3`, `retry_delay: float = 2.0`, `cost_cap_usd: float = 5.0` (per A.8).
  - `src/multi_scenario/domain/models/lero/candidate.py`:
    - `Candidate(BaseModel, frozen=True)` — `iteration: int`, `candidate_idx: int`, `reward_source: str | None`, `obs_source: str | None`, `raw_response: str`, `prompt_hash: str`, `response_hash: str`, `seed_used: int`, `created_at: datetime`.
    - `CandidateMetrics(BaseModel, frozen=True)` — `M1..M9: float | None`, `train_frames_used: int`, `train_failed: bool`, `error_msg: str | None`.
    - `CandidateResult(BaseModel, frozen=True)` — `candidate: Candidate`, `metrics: CandidateMetrics`, `fitness_rank: int | None` (set after iteration ranking), `fallback_outcome: Literal["unused", "tried_success", "tried_crashed"] = "unused"`.
  - `src/multi_scenario/domain/models/lero/trace.py`:
    - `PromptTrace(BaseModel, frozen=True)` — `iteration: int`, `candidate_idx: int`, `attempt: int` (for retries), `messages: list[dict]` (system + user/assistant turns), `prompt_version: str`, `template_render_context: dict`, `created_at: datetime`.
    - `ResponseTrace(BaseModel, frozen=True)` — `iteration: int`, `candidate_idx: int`, `attempt: int`, `model: str`, `system_fingerprint: str | None`, `seed_used: int`, `text: str`, `prompt_tokens: int`, `completion_tokens: int`, `total_cost_usd: float`, `latency_ms: float`, `created_at: datetime`.
    - `ReasoningTrace(BaseModel, frozen=True)` — `iteration: int`, `candidate_idx: int`, `attempt: int`, `reasoning_text: str | None` (Anthropic extended thinking; nullable for providers that don't expose it), `tool_calls: list[dict]` (if any), `created_at: datetime`.
    - `LeroRunSummary(BaseModel, frozen=True)` — `run_id: str`, `n_iterations_completed: int`, `n_candidates_total: int`, `n_candidates_valid: int`, `n_llm_calls: int`, `total_cost_usd: float`, `winning_candidate: CandidateResult`, `fallback_chain: list[CandidateResult]`, `peak_M1: float | None`, `final_M1: float`.
- Files to edit:
  - `src/multi_scenario/domain/models/__init__.py` — re-export the lero submodule.
  - `src/multi_scenario/domain/models/config.py` — `ExperimentConfig` adds `lero: LeroSection | None = None`, `llm: LlmSection | None = None`. Strict-mode validator: if `lero` is set, `llm` must be set; if `lero` is None, `llm` must also be None (and vice versa).
- Tests (all `tests/unit/lero/`):
  - `test_lero_config_strict.py` — strict mode rejects unknowns; required fields fail without value; literal fields reject typos.
  - `test_candidate_model_immutable.py` — frozen flag; equality semantics; hash works.
  - `test_lero_section_xor_invariant.py` — `lero` and `llm` must both be set or both None.
- Hex concerns: `domain/models/lero/` is data-only. No `import litellm`, no `import anthropic`, no `import openai`, no `import torch`. Enforced by the existing F1.12 isolation test (extend its allowlist if needed).
- Done: pytest green; `from multi_scenario.domain.models.lero import LeroSection, Candidate` works in isolation.

#### F9.1 — LLM port + first adapter (M, 1.5 days)

**Goal:** define `LlmClient` Protocol and ship one working adapter (recommend OpenAI per A.6).

- Files to create:
  - `src/multi_scenario/domain/ports/llm.py`:
    - `LlmClient(Protocol)`:
      - `def generate(messages: list[dict], n: int, seed: int, temperature: float, max_tokens: int | None) -> list[ResponseTrace]:` (returns N completions). Raises `LlmCostCapExceeded`, `LlmRateLimited`, `LlmInvalidResponse`.
      - `def name() -> str:` (e.g. `"openai_litellm"`).
      - `def estimated_cost_usd(messages, n, max_tokens) -> float:` (pre-flight cost estimate).
    - Exceptions: `LlmError`, `LlmCostCapExceeded(LlmError)`, `LlmRateLimited(LlmError)`, `LlmInvalidResponse(LlmError)`.
  - `src/multi_scenario/adapters/llm/__init__.py`
  - `src/multi_scenario/adapters/llm/openai_litellm.py` — wraps LiteLLM, mirrors the working `rendezvous_comm/src/lero/llm_client.py::LLMClient.generate()` semantics: per-candidate seed derivation (uses the helper from `domain/lero/seeding.py` — see below), retry-on-rate-limit, cost accounting, system_fingerprint logging.
  - `src/multi_scenario/adapters/llm/disk_cache.py` — `class LlmDiskCache:` with the `(model, messages, seed, response_format) → response` cache logic from `rendezvous_comm/src/lero/llm_cache.py`. Decoration via composition: `CachingLlmAdapter(inner: LlmClient, cache: LlmDiskCache, mode: str)`.
  - `src/multi_scenario/adapters/llm/fake.py` — `class FakeLlmClient(LlmClient):` for unit tests; takes a list of canned responses, returns them in order, records the messages passed in.
  - `src/multi_scenario/domain/lero/seeding.py` — pure helper `derive_per_call_seed(run_id, base_seed, iteration, candidate_idx, level) -> int` (port of `_derive_seed` from rendezvous_comm; lives in domain because it's pure math and used by both the orchestrator and the adapter).
- Files to edit:
  - `application/factories.py` — `make_llm(name: str, llm_cfg: LlmSection) → LlmClient` registry. Default `"openai_litellm"`.
  - `pyproject.toml` — add `litellm` to `[project.optional-dependencies].lero`.
- Tests (`tests/unit/lero/`):
  - `test_llm_port.py` — runtime_checkable confirms `OpenAiLitellmAdapter` and `FakeLlmClient` both satisfy.
  - `test_seeding.py` — derive_per_call_seed is deterministic; collision-free for n=12 cands × 4 iter × 10 base seeds.
  - `test_disk_cache.py` — write_only mode never reads; read_write returns cached on second call; mode=off bypasses; cache key changes when model changes.
  - `test_fake_llm.py` — records messages in order; raises on out-of-canned-responses.
  - `test_openai_adapter_unit.py` — patches `litellm.completion`; asserts payload, retry-on-rate-limit, cost cap.
- Hex concerns: `domain/ports/llm.py` is import-pure (only stdlib + pydantic for the response trace model). The adapter imports LiteLLM. Frontend never imports either.
- Dependencies: F9.0; A.6 (provider), A.8 (cost cap), A.9 (cache mode).
- Done: a unit test using `FakeLlmClient` calls `generate` and gets back N `ResponseTrace`s; the OpenAI adapter test passes (`litellm` patched).

#### F9.2 — Prompt registry & renderer (S, 1 day)

**Goal:** versioned prompts loadable by name; template substitution; cleanly extensible to meta-prompting (F12).

- Files to create:
  - `src/multi_scenario/domain/ports/prompt_renderer.py`:
    - `PromptRenderer(Protocol)`:
      - `def render(template_name: str, context: dict) -> str:` returns rendered text.
      - `def list_versions() -> list[str]:`.
      - `def slot_hashes(version: str) -> dict[str, str]:` for fairness pinning (A.15).
  - `src/multi_scenario/adapters/prompts/__init__.py`
  - `src/multi_scenario/adapters/prompts/jinja_renderer.py` — implements `PromptRenderer` over Jinja2 templates loaded from `src/multi_scenario/adapters/prompts/templates/<version>/`.
  - `src/multi_scenario/adapters/prompts/templates/v2_fewshot_k2_local/system.j2` — port from `rendezvous_comm/src/lero/prompts/v2_fewshot_k2_local/`.
  - `src/multi_scenario/adapters/prompts/templates/v2_fewshot_k2_local/initial.j2` — initial user prompt template.
  - `src/multi_scenario/adapters/prompts/templates/v2_fewshot_k2_local/feedback.j2` — feedback template per iteration.
  - `src/multi_scenario/adapters/prompts/templates/v2_fewshot_k2_local/meta.yaml` — slot definitions, frozen-slot hashes (for A.15).
  - `src/multi_scenario/adapters/prompts/templates/v2/...` — port the v2 paper-faithful prompt (for the evolve_reward path).
  - **Rule:** templates are loaded as **package-data** (declared in `pyproject.toml::tool.setuptools.package-data`) so they survive a `pip install`.
- Files to edit:
  - `pyproject.toml` — add `jinja2` to the `lero` extra; declare `multi_scenario.adapters.prompts.templates` as package data.
- Tests (`tests/unit/lero/`):
  - `test_jinja_renderer.py` — renders both templates with a fixture context; output contains expected markers; missing context-key fails loudly (not silent).
  - `test_prompt_versions_listed.py` — `list_versions()` returns ≥2; each can be rendered.
  - `test_frozen_slot_hash.py` — tampering with a frozen slot template raises (mirrors the `FrozenSlotMismatch` in rendezvous_comm meta loader).
  - `test_v2_fewshot_byte_parity.py` — render with the canonical context, assert byte-equal to the rendered output of `rendezvous_comm/src/lero/prompts/v2_fewshot_k2_local/...` rendered with the same context. **This is the load-bearing reproducibility guard.**
- Hex concerns: domain port has zero Jinja import; adapter does. Frontend never imports the adapter (it can call `available_prompt_versions()` via `application/factories.py`).
- Dependencies: F9.0; needs the rendezvous_comm prompt templates copied byte-for-byte.
- Done: `JinjaPromptRenderer().render("v2_fewshot_k2_local/initial.j2", ctx)` returns expected text; F8.4 reproducibility test depends on byte-parity test green.

#### F9.3 — Trace writer (S, 1 day)

**Goal:** every LLM call writes a structured artifact under the run dir. Auditable, queryable, schema-enforced.

- Files to create:
  - `src/multi_scenario/domain/ports/trace_writer.py`:
    - `TraceWriter(Protocol)`:
      - `def write_prompt(trace: PromptTrace, run_dir: Path) -> Path:` returns path written.
      - `def write_response(trace: ResponseTrace, run_dir: Path) -> Path:`.
      - `def write_reasoning(trace: ReasoningTrace, run_dir: Path) -> Path:`.
      - `def write_candidate(result: CandidateResult, run_dir: Path) -> Path:`.
      - `def write_evolution_history(history: list[CandidateResult], run_dir: Path) -> Path:`.
      - `def write_fallback_chain(chain: list[CandidateResult], run_dir: Path) -> Path:`.
      - `def write_summary(summary: LeroRunSummary, run_dir: Path) -> Path:`.
  - `src/multi_scenario/adapters/lero/__init__.py`
  - `src/multi_scenario/adapters/lero/filesystem_trace_writer.py` — implements TraceWriter to local disk under `<run_dir>/output/lero/`. **Layout exactly:**
    ```
    output/lero/
      iter_<n>/
        cand_<m>/
          attempt_<a>/
            prompt.json       # PromptTrace as JSON
            response.json     # ResponseTrace
            reasoning.json    # ReasoningTrace (may be {}, never absent)
          generated_code.py   # extracted reward + obs source
          eval_metrics.json   # CandidateMetrics
          fitness.json        # {fitness_rank, score_tuple}
      evolution_history.json  # list[CandidateResult]
      fallback_chain.json     # list[CandidateResult]
      best_reward.py          # winning candidate's reward source
      best_obs.py             # winning candidate's obs source
      final_metrics.json      # LeroRunSummary
      llm_provenance.json     # {model, system_fingerprint, prompt_version, library_versions}
    ```
  - `src/multi_scenario/adapters/lero/s3_trace_writer.py` — stub interface matching `TraceWriter` over S3 (deferred F12 unless needed sooner; ship the file with `NotImplementedError` placeholders so extension is signposted).
- Files to edit:
  - `application/factories.py` — `make_trace_writer(name: str) → TraceWriter` registry; default `"filesystem"`.
- Tests (`tests/unit/lero/`):
  - `test_filesystem_trace_writer.py` — round-trip every trace type to a tmp_path; re-load via `pydantic.model_validate_json`; assert deep-equal.
  - `test_trace_layout.py` — after a 1-iter × 2-cand × 1-attempt mock LERO run, every documented file exists under the documented path.
  - `test_no_partial_writes.py` — interrupt mid-iteration; on next start, no half-written JSONs (atomic write-rename).
- Hex concerns: TraceWriter port lives in `domain/ports/`. Filesystem adapter is the only one until S3 lands. Frontend reads via a separate `TraceReader` port (F9.8) so the FE stays decoupled.
- Dependencies: F9.0.
- Done: a fixture `LeroRunSummary` written to disk and re-read produces an equal object.

#### F9.4 — Code generation + safety (S, 1 day)

**Goal:** parse LLM response into validated Python; AST-check for forbidden imports; safely import.

- Files to create:
  - `src/multi_scenario/domain/lero/codegen.py`:
    - `extract_candidates(response_text, evolve_reward, evolve_observation) -> CandidateCode | None` — port of `rendezvous_comm/src/lero/codegen.py::extract_candidates` with our `Candidate` model.
    - `validate_function(source, expected_name, expected_args, allowed_imports) -> ValidationResult` — AST checks; returns structured result not bool (so the trace writer can record *why* it failed).
    - `class CandidateCode` (pure dataclass, mirrors rendezvous_comm).
    - `ALLOWED_IMPORTS = {"torch", "math", "numpy"}` — same as rendezvous_comm.
- Files to edit: none.
- Tests (`tests/unit/lero/`):
  - `test_codegen_extract.py` — happy: response with one reward + one obs block extracted. Sad: no code blocks, malformed Python, forbidden import (`import os`), wrong function name, wrong arg names. Each sad case produces a `ValidationResult` with `failure_reason` set.
  - `test_codegen_byte_parity.py` — feed the same response text into our `extract_candidates` and rendezvous_comm's; assert identical `reward_source` and `obs_source`. Reproducibility-critical.
- Hex concerns: pure-Python; no imports of torch/litellm/anything stateful. Lives in `domain/lero/` because it's reused by both the orchestrator and tests.
- Dependencies: F9.0.
- Done: parity test green vs rendezvous_comm.

#### F9.5 — Scenario patching (M, 1.5 days)

**Goal:** generic mechanism to splice an LLM-generated `compute_reward` and `enhance_observation` into a `Scenario` adapter, **respecting the F7.7.B1 factories pattern**. Discovery is the first concrete patcher; the abstraction must trivially extend to navigation/transport/flocking.

- Files to create:
  - `src/multi_scenario/domain/ports/scenario.py` — extend (not break) the existing `Scenario` Protocol with an optional method `def patch_with_llm_code(reward_source, obs_source, lero_section) -> "PatchedScenario":`. Adapters that don't implement it raise `NotImplementedError("LERO not supported on this scenario")`.
  - `src/multi_scenario/adapters/scenarios/_lero_patch_helpers.py`:
    - `_build_reward_state(scenario, agent, agent_idx) -> dict` — port from rendezvous_comm.
    - `_build_obs_state(scenario, agent, agent_idx) -> dict` — port.
    - `_compile_function(source, func_name, allowed_namespace) -> Callable` — port.
    - `_sanitize_reward(r, clip) -> Tensor` — port (`nan_to_num` + `clamp`).
    - `_maybe_wrap_obs_state(state, mode, whitelist_strict) -> dict | AllowedKeysDict` — port.
    - `class AllowedKeysDict(dict)` — port from `rendezvous_comm/src/lero/meta/fairness.py` for whitelist-strict mode; raises `FairnessViolation` on forbidden-key access.
    - Exception class `FairnessViolation(Exception)`.
  - `src/multi_scenario/adapters/scenarios/discovery.py` — extend `VmasDiscoveryAdapter` with `patch_with_llm_code(...)` that uses the helpers to produce a `PatchedDiscoveryScenario` subclass.
  - **Sanity:** the `info()` override that returns per-agent `covering_reward` (so M8 isn't structurally zero) is included in the patched class. Already documented in the rendezvous_comm bug log §3.3.
- Files to edit: none beyond `discovery.py`.
- Tests (`tests/unit/lero/`):
  - `test_scenario_patch_compile.py` — fixture LLM source compiles; missing function name raises.
  - `test_scenario_patch_reward_clip.py` — patched scenario's `reward()` clips to [-50, 50] when LLM returns ±1000.
  - `test_scenario_patch_nan_to_num.py` — `reward()` returns 0 (not NaN) when LLM source returns NaN.
  - `test_scenario_patch_whitelist_strict.py` — `enhance_observation` reading `targets_pos` in `local + strict` mode raises `FairnessViolation`.
  - `test_scenario_patch_obs_mode_closure.py` — regression for the §3.1 closure bug (`_obs_mode` NameError); assert `local` mode actually uses local state, not global.
  - `test_scenario_patch_per_agent_info.py` — `info(agent)` returns per-agent `covering_reward`, not the shared scalar.
- Hex concerns: helpers in `_lero_patch_helpers.py` import torch (allowed in adapter layer). The `FairnessViolation` exception class lives in the helpers; if it leaks into domain (caught by orchestrator), promote it to `domain/lero/exceptions.py`.
- Dependencies: F9.0, F9.4; existing F2.1 discovery scenario.
- Done: smoke test — patched scenario instantiated with the rendezvous_comm S3b-local winning obs source produces an env that runs one VMAS step without exceptions.

#### F9.6 — Evolutionary loop orchestrator (L, 2 days)

**Goal:** the application-layer use-case that ties everything together. Pure orchestration; all I/O behind ports.

- Files to create:
  - `src/multi_scenario/application/lero_orchestrator.py`:
    - `class LeroOrchestrator:` constructor takes `(prompt_renderer, llm_client, trace_writer, scenario, algorithm, metrics, storage, logger, prompt_composer)`. **Eight injected ports** (echoes `ExperimentService`); no class-level state.
    - `def run(cfg: ExperimentConfig, run_dir: Path, provenance: Provenance) -> LeroRunSummary:` orchestrates:
      1. For `iter in range(cfg.lero.n_iterations)`:
         - `messages = prompt_composer.compose(iter, history)`.
         - `responses = llm_client.generate(messages, n=n_candidates, seed=derive_seed(...))`.
         - For each response:
           - Write prompt + response + reasoning traces.
           - `cand = codegen.extract_candidates(resp.text)`.
           - If no valid candidate → record as `train_failed=True`, `error_msg=...`, continue.
           - Else: `patched = scenario.patch_with_llm_code(cand.reward_source, cand.obs_source, cfg.lero)`.
           - Eval-train: call `algorithm.train(patched, cfg, frames=cfg.lero.eval_frames)` → `eval_artifact`.
           - Evaluate: `algorithm.evaluate(eval_artifact, patched, cfg, episodes=cfg.lero.eval_episodes)` → rollout.
           - `metrics_dict = metrics.compute(rollout)`.
           - Write `eval_metrics.json` + `fitness.json`.
         - Rank iter's candidates by `(M1, M6, M2)`; persist iter's evolution_history.
      2. After loop: rank ALL candidates across iterations.
      3. Fallback chain: try rank 0 with `full_frames`; on Exception, fall back to rank 1, etc.
      4. Write fallback_chain.json, best_reward.py, best_obs.py, final_metrics.json, llm_provenance.json.
      5. Return `LeroRunSummary`.
  - `src/multi_scenario/application/prompt_composer.py`:
    - `class PromptComposer(Protocol):` — `def compose(iteration: int, history: list[CandidateResult]) -> list[dict]:` returns the messages list to send.
    - `class InitialAndFeedbackComposer(PromptComposer):` — default impl. `iter==0` → uses `initial.j2`. `iter>0` → uses `feedback.j2` with the top_k history. Mirrors rendezvous_comm's behavior.
  - `src/multi_scenario/application/lero_factories.py`:
    - `make_lero_orchestrator(cfg: ExperimentConfig, run_dir: Path) -> LeroOrchestrator:` — wires defaults from the registries.
- Files to edit:
  - `application/experiment_service.py` — add a branch: if `cfg.lero is not None`, delegate to `LeroOrchestrator.run()` and persist its summary as `output/metrics.json` (LERO writes the same final metric format so downstream consumers don't need changes). **Otherwise** unchanged.
  - `cli/run.py` — no changes (the same `multi-scenario run <yaml>` command picks the LERO branch automatically when `lero:` is present in the YAML).
- Tests (`tests/unit/lero/`):
  - `test_orchestrator_with_fakes.py` — wires `FakeLlmClient` (canned responses) + a fake `Scenario.patch_with_llm_code` (returns the unpatched scenario) + a fake `Algorithm.train` (returns a tagged artifact) + `metrics.compute` (returns canned M1). Asserts: 4 iter × 3 cand = 12 LLM calls; evolution_history has 12 entries; winning rank-0 candidate runs full_frames training.
  - `test_orchestrator_fallback_chain.py` — fake `Algorithm.train` raises on the first 2 candidates' full-train; assert fallback chain records two `tried_crashed` and one `tried_success`.
  - `test_orchestrator_zero_valid_candidates.py` — `FakeLlmClient` returns garbage; orchestrator raises a clean `LeroNoValidCandidates` exception; partial traces still written.
  - `test_orchestrator_cost_cap.py` — `FakeLlmClient` raises `LlmCostCapExceeded`; orchestrator stops gracefully, writes summary with `n_iterations_completed=<partial>`.
- Hex concerns: orchestrator lives in `application/`; uses only domain ports + domain models; **no torch, no litellm, no jinja imports** in this file (verify with the existing F1.12 test extended to cover `application/lero_orchestrator.py`).
- Dependencies: F9.0–F9.5.
- Done: `pytest tests/unit/lero/test_orchestrator_with_fakes.py -x` green; mock-only run of the orchestrator produces the canonical `output/lero/` layout.

#### F9.7 — Meta-prompting hook (XS, ½ day) — *deferred default; design only*

**Goal:** make sure `F12` can plug in a `MetaPromptingComposer` without restructuring `LeroOrchestrator`.

- Files to create: none in F9.7 itself; this is a **design-validation** sub-phase.
- Files to edit:
  - `application/prompt_composer.py` — confirm the `PromptComposer` Protocol is sufficient to express LERO-MP's outer loop. Specifically: outer loop *is itself* a different `PromptComposer` that internally runs an inner LERO and uses the inner's results to mutate its own prompt.
  - `docs/concepts/lero.md` — write down the extension-point contract:
    - "To add meta-prompting, implement a `PromptComposer` whose `compose()` runs an inner `LeroOrchestrator` and returns mutated messages. No changes to `LeroOrchestrator` required."
- Tests: `test_composer_protocol_extension.py` — a fake `MetaComposer` that wraps an inner composer is `runtime_checkable` against `PromptComposer`.
- Done: design doc written; the Protocol passes a runtime check on a meta wrapper; no F12 implementation yet.

#### F9.8 — CLI integration & frontend Submit-page support (S, 1 day)

**Goal:** `multi-scenario run <lero_yaml>` Just Works; the Submit page lets you pick a LERO config and submit it.

- Files to edit:
  - `src/multi_scenario/cli/run.py` — no changes if F9.6's `experiment_service.py` integration is in place (the YAML drives the branch).
  - `src/multi_scenario/cli/sweep.py` — already supports algorithm sweeps; verify that it iterates seeds for LERO configs the same way (test below).
  - `src/multi_scenario/frontend/forms.py` — extend the data-driven form to render the `LeroSection` and `LlmSection` widgets when the loaded YAML includes them. Mechanism: add `available_lero_defaults()` to `application/factories.py`; F7.7.B2's `render_params_from_defaults` already handles the rest.
  - `src/multi_scenario/frontend/submit_workflow.py` — preflight: add an OPENAI_API_KEY presence check when `cfg.lero is not None`.
- Files to create:
  - `tests/integration/test_cli_run_lero_smoke.py` — runs `multi-scenario run experiments/discovery/lero/configs/lero_smoke.yaml` (a tiny 1-iter × 2-cand × 5-frame fake-LLM smoke) end-to-end, asserts `output/lero/final_metrics.json` exists. Uses the `FakeLlmClient` registered via env-var override `MULTI_SCENARIO_LLM_OVERRIDE=fake` (added to `factories.make_llm`).
  - `experiments/discovery/lero/configs/lero_smoke.yaml` — the smoke fixture above.
  - `tests/integration/test_streamlit_submit_lero_form.py` — `streamlit.testing.v1.AppTest` loads the submit page with a LERO yaml; asserts the LERO section widgets render.
- Done: the CLI smoke test passes locally in <60s; the Streamlit submit page shows LERO widgets on a LERO YAML.

**F9 done-criteria:** every sub-phase green; `pytest tests/unit/lero -x` runs in <30s; `pytest tests/integration -k lero -x` runs in <2 min; F8.4 unblocked.

---

### F10 — Docs, naming, extraction

#### F10.1 — Reorganize `docs/` as a wiki (M, 1.5 days)

**Goal:** topic-per-file structure that survives the project moving repos. Decision on mkdocs vs plain MD per A.20.

- Files to create (assuming mkdocs-material per recommendation; if A.20 says plain MD, drop the mkdocs config but keep the file structure):
  - `mkdocs.yml` — site nav, theme (mkdocs-material), search plugin, link-check on `mkdocs build --strict`.
  - `docs/index.md` — landing page (replaces README's role inside the docs site; README links here).
  - `docs/getting_started/install.md` — pip / uv install; system requirements; OVH optional deps.
  - `docs/getting_started/first_run.md` — copy a smoke YAML, `multi-scenario run`, browse with `streamlit run`.
  - `docs/getting_started/ovh.md` — point to existing `docs/ovh_setup.md` (or migrate that content here).
  - `docs/concepts/vmas.md` — what is VMAS, why we use it, scenario lineage, link to upstream.
  - `docs/concepts/benchmarl.md` — what is BenchMARL, what we wrap from it, what we don't.
  - `docs/concepts/hex_architecture.md` — domain / application / adapters / frontend layering, the F1.12 isolation rule, why.
  - `docs/concepts/lero.md` — the LERO architecture doc (copy-and-adapt from rendezvous_comm/docs/lero.md, but rewritten in our voice for our codebase). Includes Section C below as content.
  - `docs/concepts/run_layout.md` — promoted from existing top-level `docs/run_layout.md`.
  - `docs/scenarios/discovery.md` — params, M1 semantics, M6, special cases.
  - `docs/scenarios/navigation.md` — same.
  - `docs/scenarios/transport.md` — same.
  - `docs/scenarios/flocking.md` — same; note no natural M1.
  - `docs/cli/index.md` — overview.
  - `docs/cli/run.md`, `docs/cli/validate.md`, `docs/cli/consolidate.md`, `docs/cli/sweep.md`, `docs/cli/resume.md`, `docs/cli/eval.md`, `docs/cli/upload-code.md`, `docs/cli/regenerate-videos.md` — one per command.
  - `docs/frontend/index.md` — Streamlit page guide (overview).
  - `docs/frontend/dashboard.md`, `docs/frontend/experiments.md`, `docs/frontend/run_detail.md`, `docs/frontend/comparison.md`, `docs/frontend/submit.md`, `docs/frontend/reproducibility.md` — one per page.
  - `docs/operations/ovh_setup.md` — moved verbatim from `docs/ovh_setup.md`.
  - `docs/operations/ovh_smoke_checklist.md` — moved verbatim.
  - `docs/operations/cost_management.md` — OVH cost caps, LLM cost caps (A.8).
  - `docs/results_analysis/results_layout.md` — anatomy of `runs.csv`, `runs.json`, per-run JSONs.
  - `docs/results_analysis/comparing_runs.md` — recipes (compare two seeds, compare two algorithms).
  - `docs/results_analysis/lero_traces.md` — how to read `output/lero/`; example queries with grep/jq.
  - `docs/ports/scenario.md`, `docs/ports/algorithm.md`, `docs/ports/metrics.md`, `docs/ports/storage.md`, `docs/ports/runner.md`, `docs/ports/logger.md`, `docs/ports/llm.md`, `docs/ports/prompt_renderer.md`, `docs/ports/trace_writer.md` — one file per port: contract, current adapters, how to add another.
  - `docs/architecture.md` — the high-level layering doc; cross-links every port doc.
  - `docs/contributing.md` — pre-commit, ruff, pytest, branching, validation gate template.
  - `docs/glossary.md` — moved from `implementation_plan.md::§8`.
  - `docs/changelog.md` — milestone log post-extraction.

  > **Mapping note:** existing `docs/csv_format_decision.md` and `docs/example_config.yaml` are dev-time artefacts; they migrate to `docs/_archive/` (kept for history) but are excluded from `mkdocs.yml` nav. F10.6 cleanup may delete them if A.20 + reviewer agree.

- Files to edit:
  - `pyproject.toml` — add `mkdocs`, `mkdocs-material`, `mkdocs-linkcheck` to `[project.optional-dependencies].docs`.
  - `.gitignore` — add `site/` (mkdocs build output).
- Tests:
  - `tests/unit/test_docs_links.py` — runs `mkdocs build --strict` in subprocess; fails on broken link or unknown nav reference.
- Done: `mkdocs serve` shows the wiki at http://127.0.0.1:8000 with all pages reachable from index; `mkdocs build --strict` exits 0.

#### F10.2 — Rewrite README as wiki landing page (XS, 1h)

- Files to edit:
  - `README.md` — replace existing content. New structure (terse):
    - Project name + 1-paragraph description.
    - Quick start: `pip install`, `multi-scenario run <smoke yaml>`.
    - Hyperlinks to: docs site (if mkdocs deployed), `docs/getting_started/install.md`, `docs/concepts/hex_architecture.md`, `docs/concepts/lero.md`, `docs/operations/ovh_setup.md`, `docs/contributing.md`.
    - One-line for each scenario with a link to its scenario doc.
    - License + citation footer.
- Done: README < 100 lines; every section is a one-or-two-line lead with a link out.

#### F10.3 — Naming proposal & decision (XS, defer to A.22)

- This is a *decision-point*, not a sub-phase. Once A.22 resolves: rename the package via `find/sed`-script — files to edit are programmatically computable.
  - `pyproject.toml` (`name`, `[project.scripts]` entry).
  - `src/multi_scenario/` → `src/<new_name>/` (rename).
  - All `from multi_scenario` imports → `from <new_name>` (sed).
  - `tests/**.py` (sed).
  - `experiments/**.yaml` (sed if `multi_scenario` appears as type discriminator; check first — likely doesn't).
  - `docs/**.md` (sed).
  - `.pre-commit-config.yaml` (the `multi_scenario/` prefix lines per existing F10.4 cleanup).
- Done: a green test suite after the rename; the package can be installed under the new name.

#### F10.4 — Repo extraction (M, 1 day)

**Goal:** extract the new-name package into its own repo, history-preserving per A.24.

- Procedure (assuming A.24 = `git filter-repo`):
  1. Pre-flight: confirm `git filter-repo` installed (`pip install git-filter-repo`).
  2. Clone parent VMAS repo to a fresh path `/tmp/extract/`.
  3. `cd /tmp/extract && git filter-repo --path multi_scenario/ --path-rename multi_scenario/:`.
  4. Confirm the resulting tree has files at root (no `multi_scenario/` prefix).
  5. Run F10.4 cleanup (already documented in current plan — drop `files: '^multi_scenario/'` from pre-commit, restore markdownlint config path).
  6. Add `LICENSE` per A.25.
  7. Update VMAS pin in `pyproject.toml` from path-relative install to a release version (or commit hash).
  8. `git remote add origin <new-repo-url>` per A.23; `git push -u origin main`.
  9. Run full pytest suite on the extracted repo. Must be green before sign-off.
  10. Open a parent-repo PR removing `multi_scenario/` (after extraction is verified). Add a top-level pointer file in parent VMAS repo `multi_scenario_moved.md` linking to the new repo URL.
- Files to create at extraction time:
  - `LICENSE` — per A.25.
  - `multi_scenario_moved.md` (in parent VMAS repo) — extraction notice.
- Files to edit at extraction time:
  - `.pre-commit-config.yaml`, `.markdownlint.json` — per existing F10.4 notes.
  - `pyproject.toml` — `dependencies` list updated; VMAS pinned.
  - All paths in docs that reference `multi_scenario/...` → root paths.
- Tests:
  - `tests/integration/test_extracted_install.py` — installable via `pip install -e .` from the new repo root; `multi-scenario --version` runs.
- Done: new repo cloned fresh, dev install works, full test suite green, F8 reproducibility configs still pass.

#### F10.5 — CI for the new repo (S, 1 day)

- Files to create:
  - `.github/workflows/ci.yml`:
    - Lint: `ruff check . && ruff format --check .`
    - Type: `mypy src/`
    - Unit tests: `pytest tests/unit -x --cov`
    - Integration tests (smoke only on push; full nightly): `pytest tests/integration -k smoke`
    - Coverage gate: 70% on push, 80% target by F12.
  - `.github/workflows/docs.yml`:
    - On push to main: `mkdocs build --strict` then deploy to GitHub Pages (per A.21).
  - `.github/workflows/nightly.yml`:
    - Cron 02:00 UTC: full integration tests including `tests/integration/test_save_rollouts_smoke.py` and the LERO smoke (with FakeLlmClient — no real API calls in CI).
- Files to edit:
  - `pyproject.toml` — add `mypy`, `pytest-cov` to dev deps.
- Done: a PR to the new repo triggers CI green; main-push deploys docs.

#### F10.6 — Docs deployment (XS, 30min)

- Per A.20/A.21:
  - If mkdocs + GitHub Pages: `gh-deploy` workflow handles it. Custom domain optional.
  - If plain MD: docs are browsable via GitHub's MD rendering; nothing to deploy.
- Files to create: `docs/_static/CNAME` (only if you add a custom domain).
- Done: docs URL works; published.

---

### F11 — Per-scenario experiment campaign (after F8 reproducibility validated)

> **Hard prerequisite:** F8.6 sign-off. We do not run F11 until ER1 + S3b-local both reproduce.

#### F11.1 — Discovery: full ablation matrix + LERO sweep (L, 1 week OVH-bound)

**Scope per A.27/A.28:**

- ER1 baseline matrix (already half-built from F8): MAPPO + IPPO + MADDPG, 3 seeds each, on `cr035` config = **9 runs**.
- ER2 (proximity comm) — need to port `rendezvous_comm/configs/er2/...` if it exists. Checkable in F11.1.A.
- ER3 (GNN) — port; needs the GNN model to be available in BenchMARL or our adapter. Likely deferred unless A.28 includes it explicitly.
- LERO sweep: S3b-local × 3 seeds (already in F8.4 = **3 runs**), plus S3b-global (oracle obs) × 1 seed for upper-bound = **1 run**, plus L8 (k=1, easy win) × 1 seed for sanity = **1 run**.
- **Total: 14 runs × ~3h each on OVH ≈ 42 OVH-hours.** Roughly a long weekend.

- Files to create: ablation-matrix YAMLs under `experiments/discovery/baseline/configs/` and `experiments/discovery/lero/configs/`.
- Files to edit: none in src.
- Reporting: a Streamlit page or notebook (per A.30) that consumes `experiments/discovery/baseline/runs.csv` + `experiments/discovery/lero/runs.csv` and renders per-method M1 table.
- Done: comparison report shows ER1 = ~0.4, S3b-local = ~0.8, S3b-global = ~1.0 (matches rendezvous_comm); reviewed and signed off.

#### F11.2 — Navigation campaign (M, 3 days OVH-bound)

- ER1 baseline on navigation: **3 algorithms × 3 seeds = 9 runs**.
- LERO obs-only on navigation: identify the analogous "rendezvous-like" task formulation. Navigation's success_predicate is "all agents reached their assigned goals" — different shape from discovery's "all targets covered ≥ k times". The LLM prompt must be adapted; this is the work:
  - F11.2.A: write `prompts/templates/v2_fewshot_navigation_local/...` (XS once we have the discovery template as reference).
  - F11.2.B: extend `_lero_patch_helpers.py` and `adapters/scenarios/navigation.py::patch_with_llm_code` to build navigation-specific reward/obs state dicts.
  - F11.2.C: write a navigation-specific obs whitelist (no oracle goal positions; only lidar + own goal idx + own pos/vel).
- LERO runs: 1 prompt × 3 seeds = **3 runs**.
- Done: report; document any scenario-specific tweaks in `docs/scenarios/navigation.md`.

#### F11.3 — Transport campaign (M, 3 days OVH-bound)

- Mirror F11.2 for transport. Transport has a movable package payload; success = package at goal. Reward design likely needs a different prompt (force/torque guidance).
- Same structure: F11.3.A/B/C.
- Done: report.

#### F11.4 — Flocking campaign (S, 1 day OVH-bound)

- Flocking has no natural M1. Per F4.2 in the existing plan, M1 is replaced by a "convergence to flock" metric (defined in `domain/ports/scenario.py` via `success_predicate` returning whatever the adapter computes).
- ER1 baseline only; LERO is questionable (LLM can't optimize what we can't measure cleanly). Recommend: ER1 baseline only, document why no LERO run.
- Done: ER1 baseline run for flocking; `docs/scenarios/flocking.md` updated with the M1-substitute rationale.

#### F11.5 — Cross-scenario synthesis report (S, 1 day)

- Per A.30, a Streamlit page `pages/6_CrossScenario.py`:
  - Heatmap rows = scenarios, cols = methods (ER1, S3b-local, etc.); cell = M1.
  - Drilldown: click a cell → opens the per-run page for the best seed.
  - Headline take-away: "obs-only LERO transferred to scenarios X, Y; failed on Z; here's the evidence".
- Done: page renders; report markdown in `docs/results_analysis/cross_scenario_findings.md`.

---

## Section C — LERO architecture deep-dive

### C.1 Layer assignment (the diagram)

```
┌──────────────────────────────────────────────────────────────────────┐
│ FRONTEND                                                             │
│  pages/Submit.py  — renders LeroSection + LlmSection widgets         │
│  pages/5_Reproducibility.py — reads runs.csv vs reference numbers    │
│  pages/run_detail.py — shows output/lero/ tree if present            │
│  imports: factories (listing), application (preflight)               │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────────┐
│ APPLICATION                                                          │
│  experiment_service.py  — branch on cfg.lero is not None             │
│  lero_orchestrator.py   — the evolutionary loop                      │
│  prompt_composer.py     — Initial+Feedback (default); meta-hook      │
│  lero_factories.py      — wires ports+adapters from cfg              │
│  imports: domain.* only                                              │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌──────────────────────────────────────────────────────────────────────┐
│ DOMAIN                                                               │
│  ports/                                                              │
│   ├─ llm.py          — LlmClient Protocol, LlmError exceptions       │
│   ├─ prompt_renderer.py — PromptRenderer Protocol                    │
│   ├─ trace_writer.py — TraceWriter Protocol                          │
│   └─ (existing) scenario, algorithm, metrics, storage, logger        │
│  models/lero/                                                        │
│   ├─ config.py       — LeroSection, LlmSection (Pydantic-strict)     │
│   ├─ candidate.py    — Candidate, CandidateMetrics, CandidateResult  │
│   └─ trace.py        — PromptTrace, ResponseTrace, ReasoningTrace,   │
│                        LeroRunSummary                                │
│  lero/                                                               │
│   ├─ codegen.py      — extract + AST validate (pure)                 │
│   ├─ seeding.py      — derive_per_call_seed (pure)                   │
│   └─ exceptions.py   — LeroNoValidCandidates, FairnessViolation      │
│  imports: stdlib + pydantic ONLY (extends F1.12 isolation rule)      │
└──────────────────────────────────────────────────────────────────────┘
                              ▲ implemented by
┌──────────────────────────────────────────────────────────────────────┐
│ ADAPTERS                                                             │
│  llm/                                                                │
│   ├─ openai_litellm.py    — LlmClient via LiteLLM (OpenAI default)   │
│   ├─ anthropic.py         — LlmClient via Anthropic SDK (future)     │
│   ├─ ovh_endpoint.py      — LlmClient via OVH HTTP (future)          │
│   ├─ disk_cache.py        — caching decorator                        │
│   └─ fake.py              — test double                              │
│  prompts/                                                            │
│   ├─ jinja_renderer.py    — PromptRenderer over Jinja2               │
│   └─ templates/<version>/ — packaged data                            │
│  lero/                                                               │
│   ├─ filesystem_trace_writer.py  — TraceWriter on local FS           │
│   └─ s3_trace_writer.py          — TraceWriter on S3 (future)        │
│  scenarios/discovery.py    — patch_with_llm_code()                   │
│  scenarios/_lero_patch_helpers.py — _build_*_state, _sanitize_reward,│
│                                     AllowedKeysDict, FairnessViolation│
└──────────────────────────────────────────────────────────────────────┘
```

### C.2 Trace flow (per LLM call)

```
LeroOrchestrator.run()
  for iter in range(N):
    messages = PromptComposer.compose(iter, history)   # pure
    seed     = derive_per_call_seed(run_id, base, iter, cand_idx, "outer")
    
    # Logged BEFORE call so we have it even if call crashes
    TraceWriter.write_prompt(PromptTrace(iter, cand_idx, attempt, messages, ...))
    
    try:
        responses = LlmClient.generate(messages, n=n_candidates, seed=seed, ...)
    except LlmCostCapExceeded as e:
        # Logged regardless: prompt is on disk, partial summary written
        TraceWriter.write_summary(LeroRunSummary(..., n_iterations_completed=iter))
        raise
    
    for cand_idx, response in enumerate(responses):
        TraceWriter.write_response(ResponseTrace(iter, cand_idx, attempt, response, ...))
        # Anthropic 'thinking' / OpenAI 'reasoning_content' if present
        TraceWriter.write_reasoning(ReasoningTrace(iter, cand_idx, attempt, ...))
        
        cand = codegen.extract_candidates(response.text, evolve_reward, evolve_observation)
        if cand is None:
            # invalid: still tracked
            metrics = CandidateMetrics(M1=None, ..., train_failed=True,
                                        error_msg="codegen failed: ...")
        else:
            patched = scenario.patch_with_llm_code(cand.reward_source, cand.obs_source, cfg.lero)
            try:
                artifact = algorithm.train(patched, cfg, run_dir/iter_n/cand_m, frames=eval_frames)
                rollout  = algorithm.evaluate(artifact, patched, cfg)
                metrics  = MetricsBundle.compute(rollout)
            except Exception as e:
                metrics = CandidateMetrics(M1=None, ..., train_failed=True, error_msg=str(e))
        
        TraceWriter.write_candidate(CandidateResult(cand, metrics, fitness_rank=None))
    
    # iter ranking
    rank_candidates(history[iter])  # mutates fitness_rank
    
    # cumulative
    TraceWriter.write_evolution_history(history)
```

### C.3 Why this layering makes meta-prompting trivial

LERO-MP is just a **different `PromptComposer`**:

```python
class MetaPromptComposer(PromptComposer):
    def __init__(self, inner_lero_orchestrator_factory: Callable[..., LeroOrchestrator],
                 base_composer: PromptComposer, mutator: Callable[[list[CandidateResult]], list[dict]]):
        ...
    
    def compose(self, iteration: int, history: list[CandidateResult]) -> list[dict]:
        # Run inner LERO; classify failmode; mutate prompt slots; return new messages
        if iteration == 0:
            return self._base.compose(0, [])
        inner_summary = self._inner_orch.run(...)
        if classify(inner_summary) == FailMode.REWARD_HACK:
            return self._mutator(inner_summary)
        return self._base.compose(iteration, history)
```

`LeroOrchestrator` doesn't need to know meta-prompting exists. The user wires a `MetaPromptComposer` in `lero_factories.py` instead of an `InitialAndFeedbackComposer`. F12 can land additively.

### C.4 Saved-data schema (summary, see F8.5 for full inventory)

Per LERO run:

```
output/lero/
├── llm_provenance.json       # {model, system_fingerprint, prompt_version, library_versions{litellm,anthropic,openai}}
├── evolution_history.json    # list[CandidateResult] — chronological
├── fallback_chain.json       # list[CandidateResult] with fallback_outcome set
├── final_metrics.json        # LeroRunSummary
├── best_reward.py            # the source that won (after fallback)
├── best_obs.py
├── iter_<n>/
│   ├── cand_<m>/
│   │   ├── attempt_<a>/
│   │   │   ├── prompt.json    # full messages list, prompt_version, render context
│   │   │   ├── response.json  # text + tokens + cost + model fingerprint
│   │   │   └── reasoning.json # extended thinking (may be {})
│   │   ├── generated_code.py  # extracted source
│   │   ├── eval_metrics.json  # CandidateMetrics
│   │   └── fitness.json       # rank within iter
│   └── benchmarl/             # BenchMARL native output for the iter's training runs
└── seed_provenance.json       # the per-call seeds used (for re-derivation)
```

### C.5 Why every LLM call is auditable

- **Order:** `prompt.json` written *before* the LLM call → even a network crash leaves a trace.
- **Provenance:** `response.json` includes `model`, `system_fingerprint` (changes invalidate reproducibility), `seed_used`, `prompt_tokens`, `completion_tokens`, `total_cost_usd`, `latency_ms`. The cache key is hashed from `(model, messages, seed, response_format)` — replaying the same trace from disk is one function call.
- **Cost:** running cost integral persisted per-iter in `llm_provenance.json`; cost-cap enforcement in the LLM adapter is the only place that can stop a run mid-flight.
- **Reasoning:** for providers that expose chain-of-thought (Anthropic extended thinking, OpenAI o1-style reasoning content), captured in `reasoning.json`.

---

## Section D — Doc wiki structure (file tree)

(Already enumerated in F10.1. Repeated as a single tree for ease of grok.)

```
docs/
├── index.md                                     # landing
├── architecture.md                              # high-level layering
├── glossary.md                                  # ported from impl_plan §8
├── changelog.md                                 # post-extraction milestone log
├── contributing.md                              # dev setup, validation gate
├── getting_started/
│   ├── install.md
│   ├── first_run.md
│   └── ovh.md                                   # link to operations/
├── concepts/
│   ├── vmas.md
│   ├── benchmarl.md
│   ├── hex_architecture.md
│   ├── lero.md                                  # Section C content
│   └── run_layout.md                            # promoted from existing
├── scenarios/
│   ├── discovery.md
│   ├── navigation.md
│   ├── transport.md
│   └── flocking.md
├── cli/
│   ├── index.md
│   ├── run.md
│   ├── validate.md
│   ├── consolidate.md
│   ├── sweep.md
│   ├── resume.md
│   ├── eval.md
│   ├── upload-code.md
│   └── regenerate-videos.md
├── frontend/
│   ├── index.md
│   ├── dashboard.md
│   ├── experiments.md
│   ├── run_detail.md
│   ├── comparison.md
│   ├── submit.md
│   └── reproducibility.md
├── operations/
│   ├── ovh_setup.md
│   ├── ovh_smoke_checklist.md
│   └── cost_management.md
├── results_analysis/
│   ├── results_layout.md
│   ├── comparing_runs.md
│   ├── lero_traces.md
│   └── cross_scenario_findings.md               # F11.5 output
├── ports/
│   ├── scenario.md
│   ├── algorithm.md
│   ├── metrics.md
│   ├── storage.md
│   ├── runner.md
│   ├── logger.md
│   ├── llm.md
│   ├── prompt_renderer.md
│   └── trace_writer.md
├── reproducibility/
│   └── reference_numbers.md                     # F8.0 output
└── _archive/                                    # excluded from mkdocs nav
    ├── csv_format_decision.md
    └── example_config.yaml
```

**Cross-link rules:**

- Every page must link to `index.md` and to its parent section's index.
- Every port doc cross-links to (a) the Protocol source file, (b) the test file, (c) every adapter that implements it.
- Every scenario doc cross-links to (a) the adapter source, (b) the canonical experiment configs, (c) `concepts/lero.md` if LERO supports the scenario.
- A CI test (`tests/unit/test_docs_links.py`) runs `mkdocs build --strict` to catch broken links.

---

## Section E — Reflect & refine

(Self-criticism of Sections A-D, then v2 deltas.)

### E.1 Weaknesses and risks of the v1 draft

1. **A.1's reproducibility threshold is hand-wavy.** "±10% absolute on M1" passes essentially any non-pathological run. M1=0.405 vs M1=0.50 would pass — but those are different policies. **Risk:** we declare reproduction success on noise, and downstream LERO experiments rest on a falsified premise. **Mitigation v2:** require both **absolute** (±10%) AND **relative** (within 1.5σ of the rendezvous_comm seed-mean) to pass. Run the rendezvous_comm seeds [0,1,2] in F8.0 to estimate σ.

2. **F8.0 assumes rendezvous_comm "still reproduces" — but does it?** rendezvous_comm itself might have bit-rotted; the headline numbers could be unreplicable in their own repo today. **Risk:** F8.0 fails and we have no reference baseline. **Mitigation v2:** treat F8.0 as gated — if rendezvous_comm doesn't reproduce its own headline within 1σ, *the question becomes "are we reproducing the doc or the code?"* and we need a user decision before proceeding. Add as a sub-decision.

3. **F9.0's `LeroSection` strict-mode XOR (lero ↔ llm) is brittle.** A user who wants to *evaluate* an LLM-trained policy without re-running LERO has `lero: None` but might want `llm: <set>` for some other purpose. **Risk:** future flexibility loss. **Mitigation v2:** drop the XOR. `lero` requires `llm`; `llm` standalone is allowed (no enforcement, future-proof).

4. **F9.6's orchestrator is too monolithic.** A 200-line `run()` method violates the SRP we preach. **Risk:** unit-test surface explodes; maintenance pain. **Mitigation v2:** split into `_run_iteration()`, `_evaluate_candidate()`, `_full_training_with_fallback()`, each privately-testable.

5. **No explicit story for handling partial / interrupted LERO runs.** What if the OVH job is killed at iter 2 of 4? Today's design loses iter 0/1 traces unless we re-load history on resume — which the orchestrator doesn't. **Risk:** wasted compute, broken reproducibility audit trail. **Mitigation v2:** add F9.6.B "resume LERO" — `LeroOrchestrator.resume(run_dir)` reads existing iter_<n>/ subdirs back into `history` and continues from the next iter. Aligns with the existing F5.7 resume pattern.

6. **F11's compute estimate is suspicious.** "3h per LERO run" — but discovery's 10M frames at MAPPO on 8-core CPU is closer to 4-6h based on rendezvous_comm reports. The 14-run F11.1 estimate of 42h is probably 60-80h. **Risk:** schedule slip. **Mitigation v2:** time-box F11.1 with measured first run, then update estimate before committing the rest.

7. **Trace storage costs not bounded.** A LERO run with full traces is ~50-200 KB JSON × ~50 candidates ≈ 10 MB. Times 100 LERO runs in F11 = 1 GB on disk plus the BenchMARL outputs (multi-GB each). Plus the optional rollouts at ~50 MB per eval = >100 GB. **Risk:** running out of disk on OVH or local. **Mitigation v2:** add a `runtime.storage.compress: bool` flag (gzip rollouts and per-iter benchmarl outputs), and add `scripts/prune_old_runs.py` to F8.5.E.

8. **Section D's docs structure has redundancy.** `getting_started/ovh.md` AND `operations/ovh_setup.md` AND README's quick-start all overlap. **Risk:** docs drift; users can't tell which is canonical. **Mitigation v2:** make `operations/ovh_setup.md` canonical; `getting_started/ovh.md` is a 5-line landing that links to it; README links to `getting_started/index.md` only.

9. **The `LlmClient.generate()` signature in F9.1 returns `list[ResponseTrace]` — but `ResponseTrace` mixes the LLM response with our trace metadata.** Conflates the model's output with our auditing layer. **Risk:** an adapter has to know about our trace model to satisfy the port. **Mitigation v2:** split — `LlmClient.generate()` returns `list[LlmCompletion]` (a domain model with just text + tokens + model + seed + cost + system_fingerprint), and the orchestrator constructs `ResponseTrace` from `LlmCompletion + iter/cand context`.

10. **F9.7 "design only" is too soft.** Without a failing test, the meta-prompting extension contract will silently break the first time we refactor `LeroOrchestrator`. **Risk:** F12 work double-cost. **Mitigation v2:** F9.7 must include a stub `MetaPromptComposer` *implementation* (returns trivial mutated prompts; not connected to anything real) and a test that the orchestrator runs end-to-end with it. Not just a doc.

11. **A.32's deferral of DuckDB might be wrong.** With LERO traces in JSON across 50+ runs, grep-and-jq doesn't compose well for "find the LLM call where M1 first crossed 0.5". **Risk:** post-hoc analysis takes 10× longer. **Mitigation v2:** reconsider — recommend shipping DuckDB-backed `lero_traces.duckdb` regenerated by a `multi-scenario index-traces` CLI in F8.5.F (new sub-phase). Cheap to add.

12. **No concrete plan for human review of LLM-generated code.** LERO writes code to disk; the user should *read* the winning candidate before deploying. Today's design has no review gate. **Risk:** deploying surprising code in `best_reward.py`. **Mitigation v2:** add `multi-scenario inspect-lero <run_dir>` CLI that pretty-prints the winning code with a diff against any prior winner; document in `docs/results_analysis/lero_traces.md`. Not blocking, but should be in F9.8.

### E.2 v2 plan (deltas only)

#### A — questions, additions

- **A.1 v2:** reproducibility threshold = absolute (±10% abs) AND relative (within 1.5σ of rendezvous_comm seed-mean). Run rendezvous_comm seeds [0,1,2] in F8.0 to estimate σ before our runs.
- **A.34 (new):** if F8.0's rendezvous_comm replication fails its own headline (within 1σ of doc), do we (a) freeze rendezvous_comm code and re-run with our latest VMAS / BenchMARL, (b) declare the doc the source of truth and accept the gap, or (c) abort the reproducibility effort? **Need input.**
- **A.35 (new):** trace storage compression default — yes/no? Recommend yes (gzip benchmarl outputs + rollouts; LERO traces stay uncompressed for grep-ability). **Need input.**

#### B — sub-phase deltas

- **F8.0 v2:** explicitly run 3 seeds in rendezvous_comm; record M1 mean+std; gate F8.2 on the std being usable.
- **F8.5.F (new):** `multi-scenario index-traces` builds DuckDB index over LERO traces. Tables: `runs(run_id, scenario, exp_id, seed)`, `candidates(run_id, iter, cand_idx, M1, ...)`, `llm_calls(run_id, iter, cand_idx, attempt, model, seed_used, total_cost_usd)`. Files to create: `application/index_traces.py`, `cli/index_traces.py`. Tests: `test_index_traces_query.py`. Done: `multi-scenario index-traces` produces `<exp_root>/lero_traces.duckdb`; a sample query returns expected rows.
- **F9.0 v2:** drop the XOR invariant on `(lero, llm)`. Keep `lero requires llm`.
- **F9.1 v2:** new `LlmCompletion` domain model (just LLM-output fields). `LlmClient.generate()` returns `list[LlmCompletion]`. Orchestrator constructs `ResponseTrace` separately. `ResponseTrace` field set is `LlmCompletion` fields + `iter, cand_idx, attempt`.
- **F9.6 v2:** split `run()` into `_run_iteration()`, `_evaluate_candidate()`, `_full_training_with_fallback()`. Each gets its own unit test in `tests/unit/lero/`.
- **F9.6.B (new):** `LeroOrchestrator.resume(run_dir)` — reads existing iter_<n>/ back into `history`; resumes from next iter. Mirrors F5.7. Tests: `test_orchestrator_resume.py`.
- **F9.7 v2:** ship a stub `MetaPromptComposer` in `application/prompt_composer.py` and a test `test_orchestrator_with_meta_composer.py` that the orchestrator runs end-to-end with the stub.
- **F9.8 v2:** add `multi-scenario inspect-lero <run_dir>` to `cli/inspect_lero.py`. Pretty-prints `best_reward.py` and `best_obs.py` with a diff against the last LERO run's winners (if any). Tests: `test_inspect_lero_cli.py`.
- **F8.5.A v2:** rollouts default off. When on, gzip the parquet (`pyarrow` supports `compression='snappy'` natively). Add `runtime.storage.compress: bool = True` (compresses BenchMARL outputs + rollouts). LERO traces stay uncompressed.
- **F11.1 v2:** time-box. Run **one** ER1 run first; measure wall-clock; multiply to estimate total; gate the rest of F11 on user re-confirming the budget.
- **F10.1 v2:** dedupe `getting_started/ovh.md` → 5-line shim that links to `operations/ovh_setup.md`. README → `getting_started/index.md` only.

#### C — architecture deltas

- C.2 trace-flow updated: `responses` is `list[LlmCompletion]`; orchestrator wraps each in `ResponseTrace(completion, iter, cand_idx, attempt)` before persisting.
- C.3 meta-prompting stub now exists (not just designed). Test asserts the contract.

#### D — doc deltas

- `docs/getting_started/ovh.md` becomes a stub.
- `docs/concepts/lero.md` includes a "How to read a LERO run" section with example DuckDB queries.
- `docs/results_analysis/lero_traces.md` references the DuckDB index + the `inspect-lero` CLI.

---

### Sign-off checklist (gate F8.6 + F9 + F10 + F11)

```
[ ] A.1–A.35 all answered (no item left as "ask later")
[ ] F8.0 reference numbers tabled
[ ] F8.1–F8.5 done, audit script green
[ ] F9.0–F9.6 done, all unit tests green
[ ] F9.7 stub in place, end-to-end test green
[ ] F9.8 CLI smoke green; Streamlit submit page renders LERO widgets
[ ] F8.4 S3b-local reproduces per A.2 threshold
[ ] F10.1 mkdocs serve renders all pages; --strict green
[ ] F10.2 README rewritten
[ ] F10.3 name decided + applied (or explicitly deferred to post-F11)
[ ] F10.4 extraction procedure tested in /tmp first
[ ] F10.5 CI green on the new repo
[ ] F11 campaign budget confirmed by F11.1 wall-clock measurement
```

---

### Critical Files for Implementation

The following files are the highest-leverage in implementing this plan (the path the user should keep open in editor while executing):

- `/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/multi_scenario/implementation_plan.md` — merge target for everything in Section B above.
- `/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/multi_scenario/src/multi_scenario/domain/models/config.py` — add `LeroSection` + `LlmSection` references; F9.0.
- `/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/multi_scenario/src/multi_scenario/application/factories.py` — add LLM / prompt-renderer / trace-writer registries; F9.1–F9.3.
- `/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/multi_scenario/src/multi_scenario/adapters/scenarios/discovery.py` — extend with `patch_with_llm_code`; F9.5.
- `/Users/afin/Documents/Studio/PHD/Code/VectorizedMultiAgentSimulator/rendezvous_comm/src/lero/scenario_patch.py` — primary reference for the helpers in `_lero_patch_helpers.py`; F9.5.
