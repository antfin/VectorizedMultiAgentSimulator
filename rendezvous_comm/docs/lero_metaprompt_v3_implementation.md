# LERO-MP v3 — Implementation Notes (2026-04-24)

Companion to `lero_metaprompt_v3_plan.md`. Documents what shipped per
step, test coverage, and known gaps. Read this before launching the
v3 dry-run so you know exactly which features are live and which
are fallback paths.

## Summary

All 10 steps from plan §8 implemented. 51 new v3 unit tests + 1 end-to-end
integration test pass; 182 legacy v2 tests still pass (no regressions).
Dry-run config `configs/lero_mp/mp_v3_dryrun_3m.yaml` is ready to fire.

## Per-step status

### Step 1 — Inner-LLM retry loop  (`src/lero/inner_llm.py`)

**What ships**

- `InnerLLM.generate(messages, seed_base)` wraps a single LLM candidate
  generation in a 3-attempt retry loop (`MAX_GEN_ATTEMPTS = 3`).
- On failure, the compiler/AST error is appended to the conversation
  as a new user message, asking for a fix. The retry sees the
  specific error string, not just "please try again".
- Each candidate records `.attempts` and `.attempt_records` as
  additional attributes on the returned `CandidateCode`. Persisted
  to `iter_<N>/candidate_<J>_attempts.json`.
- `CandidateGenerationFailed` raised after exhausting attempts; the
  outer loop catches + logs, the iteration proceeds with fewer
  candidates.

**How it's wired**

- `loop.py::LeroLoop.run` now iterates per-candidate with `InnerLLM`
  instead of batching `llm.generate(n=N)`. Each candidate gets its
  own derived OpenAI seed via `_derive_seed(run_id, seed, iter,
  cand_idx, level)`.

**Tests**: `tests/test_lero_retry.py` — 5 tests covering
pass-on-first, pass-on-retry, exhaust-retries, retry message format,
and per-candidate seed derivation.

### Step 2 — Conditional output_spec  (`src/lero/prompts/v2_fewshot_modular_v2/`)

**What ships**

- Three new slot files: `output_spec_both.txt`,
  `output_spec_obs_only.txt`, `output_spec_reward_only.txt`.
- `PromptLoader.render("initial_user.txt",
  output_spec_variant="obs_only")` selects the matching variant.
- `LeroLoop._build_initial_messages` derives the variant from
  `(lero_config.evolve_reward, lero_config.evolve_observation)`:
    - both True → `both`
    - reward only → `reward_only`
    - obs only → `obs_only`

**Effect**: `evolve_reward=false, evolve_observation=true` runs no
longer emit a reward signature the LLM would waste tokens filling in.

**Tests**: `tests/test_lero_output_spec.py` — 5 tests covering each
variant + unknown-variant fallback.

### Step 3 — Pydantic structured outputs + extract callables  (`src/lero/schemas.py`, `src/lero/llm_client.py`)

**What ships**

- `src/lero/schemas.py`: four Pydantic models — `InnerLLMOutput`,
  `StrategyCard`, `EditorOutput`, `EditorCritique`. All use
  `ConfigDict(extra="forbid")`. `StrategyCard` adds the v3
  `include_signals` + `signal_rationale` fields.
- `LLMClient.generate_structured(messages, schema,
  fallback_parser=...)` tries OpenAI's `response_format={"type":
  "json_schema", ...}` first; if that fails (provider doesn't
  support strict mode, or Pydantic validation fails on a valid
  JSON), falls back to the `fallback_parser` (typically a regex-
  based extractor from v2).
- `src/lero/inner_llm.py::_parse_free_text_to_inner` is the
  free-text fallback parser for `InnerLLMOutput`; it reuses
  `codegen._CODE_BLOCK_RE` + `validate_function` so extraction
  behavior is identical to v2.
- `meta/critique.py::parse_critique` parses `EditorCritique` from
  JSON blocks in LLM output; fills `suggested_signal_change`
  default if the model drops it.

**Known gap**: the legacy `parse_mutation_response` + `parse_strategy_card`
still work on free-form LLM output (regex + YAML). They've been
augmented to populate the new fields (`include_signals`) with safe
defaults but are not yet migrated to pure Pydantic. Full migration
is the v3.1 follow-up; v3 ships both paths so the Critic can run
without forcing the Editor through a new output format.

**Tests**: `tests/test_lero_mp_repro.py` covers all four schemas
(round-trip + invalid-input rejection).

### Step 4 — Temperature=1.0 default  (`src/lero/config.py`)

**What ships**

- `LLMConfig.temperature` default → `1.0` (was `0.8`).
- `MetaPromptConfig.meta_temperature` default → `1.0` (was `0.3`).
- `SEED_META_TEMPERATURE` cycle is **kept** as an override knob for
  ablations but is no longer the default path (caller must still
  pass `meta_temperature_for_seed(seed)` explicitly to use it).

**Why it's safe at T=1.0**

- Structured outputs (Step 3) constrain the shape — at T=1.0 the
  model can only vary *content*, not structure.
- Retry loop (Step 1) handles the rare invalid output.
- Reproducibility (Step 7) uses OpenAI's `seed` parameter, not
  temperature, so `T=1.0` + fixed `seed` still gives stable results.

### Step 5 — Behavioral signals + LLM-gated `include_signals`  (`src/lero/meta/behavioral_summary.py`)

**What ships**

- `behavioral_summary.format_tier1(metrics)` — per-candidate
  scalar line (M1/M2/M3/M4/M6/M8/M9).
- `behavioral_summary.fingerprint_from_csv(path)` — Tier 2: reads
  a BenchMARL scalar CSV (any of the common column-name variants),
  extracts start/peak/end M1, computes a `drop` magnitude, returns
  a `shape=<tag>` suffix.
- `behavioral_summary.classify_learning_curve(trajectory)` —
  Tier 3: standalone curve-shape classifier (no CSV needed).
  Tags: `monotonic_rise`, `plateau_then_collapse`, `oscillating`,
  `flat_zero`, `flat_nonzero`, `reward_hack_shape`.
- `behavioral_summary.outlier_flags(metrics)` — threshold gate:
  returns `["M4_high_collisions", "M9_clustered", …]` based on
  fixed outlier thresholds from plan §5.
- `behavioral_summary.format_behavioral_block(metrics,
  include_signals=..., csv_path=..., trajectory=...)` — renders
  the full block honoring `include_signals`. Unlisted tiers
  produce nothing.

**Integration**

- `StrategyCard.include_signals` field (default `["scalar"]`) drives
  what gets included. The Strategist YAML prompt asks for it
  explicitly with a paragraph of guidance.
- `build_editor_prompt(..., behavioral_block=...)` appends the
  filtered block to the Editor prompt.
- `LeroMpOuterLoop._behavioral_block(top_cands, strategy_card)`
  computes the block from top-1 candidate + the Strategist's
  include_signals choice.

**Known gap**: today the `include_signals` filter applies ONLY to the
Editor prompt. The inner-LLM between-iteration `feedback.txt` is
still the v2 minimal format (M1/M2/M4/M6 scalars). Wiring the same
filter into `codegen.build_feedback` is a v3.1 follow-up (~30 min);
the infrastructure is in place.

**Tests**: `tests/test_lero_mp_signal_gate.py` — 19 tests for
format_tier1, outlier rules, curve classification, CSV fingerprint,
include_signals filtering, and parse_strategy_card default/custom
handling.

### Step 6 — TextGrad Editor Critic  (`src/lero/meta/critique.py`)

**What ships**

- `build_critic_prompt(strategy_card, editor_new_slot,
  fairness_text, prior_slot_versions)` — composes the Critic prompt.
- `parse_critique(text)` — extracts the first `{...}` JSON block,
  validates as `EditorCritique`. Backfills missing optionals on the
  first-try retry path.
- `critique_and_revise(...)` — orchestrates the `Editor → Critic
  → revise → Critic → accept/reject` loop. `max_revisions=2`. On
  `reject` raises `ValueError` which the outer loop catches as a
  graceful stop (same as the old MutationParseError path).

**Integration**

- `propose_new_template` accepts `critic_llm_call: MetaLLMCallable`
  (optional, `None` disables). When passed, it runs the Critic loop
  immediately after the first Editor call. Revision passes rebuild
  the Editor prompt with a `[CRITIC FEEDBACK]` suffix appended.
- `LeroMpOuterLoop` passes `critic_llm_call=self.meta_llm_call` when
  `two_level_meta=True`. Same underlying client, same cache, same
  model.

**Cost**: +1-2 LLM calls per mutation. At gpt-5.4-mini prices ≈ €0.01
extra per mutation. Budget impact negligible.

**Tests**: `tests/test_lero_mp_critique.py` — 7 tests covering
parse, keep path, revise-then-keep, reject, max_revisions cap.

### Step 7 — Reproducibility cache + seed + fingerprint  (`src/lero/llm_cache.py`, `llm_client.py`, `run_lero_mp.py`)

**What ships**

- `LLMCache` class (`llm_cache.py`) with 4 modes: `off`,
  `read_write`, `read_only`, `write_only`. Env override:
  `LERO_LLM_CACHE_MODE`. Default root:
  `~/.cache/lero_llm/`. Cache key =
  `sha256(model, messages, temperature, seed, response_format)`.
- `LLMClient.__init__` accepts an optional `cache=LLMCache(...)`;
  every `_call` checks the cache before network and writes on
  success. Applies to BOTH inner and meta LLM calls.
- `LLMClient._call` accepts per-call `seed` param; forwarded to
  LiteLLM's `completion(..., seed=...)`.
- `LLMClient._call` logs `system_fingerprint` and warns on drift
  between calls — detects silent OpenAI model-version changes.
- `run_lero_mp.py` startup locks `random`, `numpy.random`,
  `torch` (+ `torch.cuda` if available) seeds using `args.seed`.
  Must run *before* any model/env construction.
- `loop.py::_derive_seed(run_id, seed, iteration, cand_idx, level)`
  hashes all dimensions into a 31-bit int. Used as `seed_base` for
  `InnerLLM.generate`; sibling candidates within an iteration get
  `seed_base + 0`, `seed_base + 1`, etc. Same scheme threads through
  `MetaPromptConfig.llm_cache` → `_default_meta_llm_call`.

**How to replay a past run**

```bash
# 1. Run a seed-0 experiment with cache on — builds the cache as a side-effect:
LERO_LLM_CACHE_MODE=read_write python run_lero_mp.py config.yaml --seed 0

# 2. Rerun the same seed with read_only — compares against cached responses:
LERO_LLM_CACHE_MODE=read_only python run_lero_mp.py config.yaml --seed 0
```

`review_dry_run.py` checks the cache directory is populated;
full bit-exact comparison is manual (see
`docs/lero_v3_dryrun_checklist.md` §5).

**Tests**: `tests/test_lero_mp_repro.py` — 13 tests covering cache
key stability + discrimination, cache mode semantics, env override,
schema round-trips.

### Step 8 — Prior-memory activation  (`src/ovh.py`, `meta/outer_loop.py`)

**What ships**

- `submit_training_job(..., history_buckets=[...])` accepts a list
  of `bucket@region[/prefix]` specs. Each is mounted read-only at
  `/workspace/history_<idx>` and exposed via `LERO_HISTORY_PATHS`
  env var (colon-separated mount points).
- `LeroMpOuterLoop.__init__` reads `LERO_HISTORY_PATHS`, recursively
  finds `mutation_log.jsonl` files under each mount, stores in
  `self._history_log_paths`.
- Strategist's `read_recent` call now passes
  `[self._mutation_log_path, *self._history_log_paths]` so prior
  sweeps' verdicts feed into the current run's `avoid` list.

**Configuration**: `max_outer_iters=3` in the 3M dry-run config
gives the Strategist at least one prior slot version to diverge
from on the 2nd mutation.

### Step 9 — Review tooling + integration test  (`scripts/`, `tests/`, `docs/`)

**What ships**

- `scripts/review_dry_run.py` — parses a run directory, emits
  pass/fail per §9 success criterion. Exits 0 if ≥3 of 5 pass.
- `docs/lero_v3_dryrun_checklist.md` — manual grep commands for
  each criterion + ER1 comparability check.
- `tests/test_lero_mp_integration.py` — stub-LLM integration test
  stitching inner retry → include_signals filter → Critic loop.
  Does NOT run real VMAS/BenchMARL; that's covered by the live
  OVH dry-run.

## Test summary

```
tests/test_lero_retry.py              ........  5 tests
tests/test_lero_output_spec.py        ........  5 tests
tests/test_lero_mp_signal_gate.py     .......  19 tests
tests/test_lero_mp_critique.py        ........  7 tests
tests/test_lero_mp_repro.py           .......  13 tests
tests/test_lero_mp_integration.py     ........  2 tests
                                     ─────────────────
                                        Total: 51 new v3 tests

Regression: 182 legacy v2 tests still pass.
```

## Known gaps & follow-ups for v3.1

1. **`codegen.build_feedback` does not honor `include_signals` yet.**
   Inner-LLM next-iteration feedback is still the v2 minimal format.
   Fix: thread `strategy_card` or just `include_signals` into
   `loop.py::build_feedback(...)` call. ~30 min.
2. **Meta-LLM still uses regex parsing** — `parse_mutation_response`
   handles the Editor's free-form output. Full migration to
   `EditorOutput` Pydantic requires changing the Editor prompt to
   emit JSON, which conflicts with slot rendering (the slot body
   needs raw text, not JSON-escaped). v3.1 can solve this with a
   two-field output: `{new_slot_content: "...", rationale: "..."}`
   and strip the slot text out of the JSON before writing.
3. **Cache replay comparison is manual.** `review_dry_run.py` only
   checks the cache exists. Adding an automated bit-exact
   comparator is a v3.1 follow-up.
4. **`suggested_signal_change` from Critic is not yet consumed.**
   The Critic can emit `add_fingerprint` / `drop_curve_shape` but
   nothing reads it today. Wire to the NEXT outer iter's
   `StrategyCard.include_signals` as a carry-over recommendation.

## Running the dry-run

```bash
cd rendezvous_comm
# Local smoke (no OVH, no real training — just stubs):
python -m pytest tests/test_lero_mp_integration.py -v

# OVH dry-run (3 seeds × 3M frames ≈ €22):
python -c "
from src.ovh import submit_training_job
for seed in (0, 1, 2):
    submit_training_job(
        config_yaml='rendezvous_comm/configs/lero_mp/mp_v3_dryrun_3m.yaml',
        runner='run_lero_mp.py',
        extra_cli=f'--seed {seed}',
        exp_id_suffix=f'_s{seed}',
        job_name=f'lero_mp_v3_s{seed}',
        llm_env={'OPENAI_API_KEY': '<key>'},
    )
"

# After all 3 seeds complete:
for seed in 0 1 2; do
    python scripts/review_dry_run.py \
        results/lero_mp/lero_mp_v3_dryrun_3m_s${seed}/*/
done
```

If ≥3 of 5 criteria fire per seed, green-light the full 10M sweep.
Otherwise iterate on prompts (e.g. strengthen the include_signals
guidance, add more specific feature names to the Editor template)
and repeat the dry-run.
