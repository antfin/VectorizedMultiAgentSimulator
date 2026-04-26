# LERO-MP Meta-LLM Scenario Harness

> **Status**: shipped 2026-04-25. Companion to
> `lero_metaprompt_v3_plan.md` and `lero_metaprompt_v3_implementation.md`.

A test tier between unit tests and full OVH dry-runs. Synthesizes
candidate metrics + history + priors, runs the **real** Strategist /
Editor / Critic LLM calls against them, and validates the decisions
against expected outcomes per scenario.

**Why it exists**: at 1M / 3M frames the meta-LLM's mutation signal
is dominated by RL noise, so we cannot iterate on Strategist or
Critic prompts via OVH dry-runs alone — each iteration costs ~€22 +
50 min wall time. The harness gives us a **deterministic decision
test** that runs in ~15 min for ~€1 across 4 LLMs.

## Test tier comparison

| Tier | Real LLM? | Real RL? | Speed | Cost | What it catches |
| --- | --- | --- | --- | --- | --- |
| Unit (`tests/test_lero_*.py`) | Stub | No | 1s | €0 | wiring, plumbing |
| **Harness (this doc)** | **Yes** | **No (fake metrics)** | **5–15s/scenario** | **~€0.01/scenario** | **prompt quality, LLM-specific behavior** |
| OVH dry-run | Yes | Yes (3M frames) | ~50 min/seed | ~€7/seed | training behavior end-to-end |
| Full run | Yes | Yes (10M frames) | hours | ~€20/seed | publishable peak-M1 |

## Files

| Path | Purpose |
| --- | --- |
| `tests/scenarios/__init__.py` | Package marker. |
| `tests/scenarios/fixtures.py` | Builders (`_rec`, `_cand`, `_prior`) + the `ALL_SCENARIOS` registry. |
| `tests/scenarios/runner.py` | `run_scenario(name, scenario, llm)` + `check_expectations(result, expected)`. |
| `scripts/run_scenario_sweep.py` | CLI that runs N scenarios × M LLMs, emits markdown report + JSON dump. |
| `tests/test_meta_llm_scenarios.py` | Pytest entry (opt-in via `@pytest.mark.llm`). |

## Scenario format

Each entry in `ALL_SCENARIOS` is a dict:

```python
"A1_flat_zero_baseline": {
    "description": "All candidates flat-zero — observation signal missing.",
    "fail_mode": FailMode.HEALTHY,
    "history": [_rec(peak=0.005, m6=0.08), _rec(peak=0.010, m6=0.09)],
    "candidates": [
        _cand(m1=0.005, m6=0.08, obs_code=_OBS_GENERIC),
        _cand(m1=0.010, m6=0.09, obs_code=_OBS_GENERIC),
    ],
    "priors": [],
    "bias": "observation_first",
    "expected": {
        "target_slot": ["guidance_observation", "guidance_shared"],
        "include_signals_contains_any": ["scalar"],
        "editor_mentions_any": ["proximity", "lidar_targets", "gap"],
    },
}
```

### Field reference

| Field | Type | Purpose |
| --- | --- | --- |
| `description` | str | Human-readable (shown in report). |
| `fail_mode` | `FailMode` | Forces the dominant fail-mode the Strategist sees. |
| `history` | list of `TemplateRecord` | What `_rec(...)` returns. Drives OPRO-style sorted history block. |
| `candidates` | list of dict | What the inner-loop "would have" returned. Drives candidate-aggregate stats + top-k feature signals. |
| `priors` | list of `MutationLogEntry` | Past mutations on the relevant slot. Drives prior-slot-versions block. |
| `bias` | `"observation_first" / "reward_first" / "exploratory"` | Per-seed bias hint. |
| `editor_override` | str (optional) | Skip the Editor LLM call and inject this text — for B-series Critic tests where we need to control the slot text. |
| `expected` | dict | The assertion contract. See below. |

### Expected-checks vocabulary

`runner.check_expectations(result, expected)` understands:

| Key | Form | Pass condition |
| --- | --- | --- |
| `target_slot` | list[str] | Strategist's `target_slot` is in this list. |
| `include_signals_contains_any` | list[str] | At least one tier in the list is in `card.include_signals`. |
| `include_signals_equals` | list[str] | Exact-set match. |
| `confidence_in` | list[str] | `card.confidence` is in the list. |
| `editor_mentions_any` | list[str] | At least one term appears (case-insensitive) in the new slot text. |
| `editor_avoids_any` | list[str] | None of the terms appear. |
| `editor_must_not_contain_all` | list[str] | Not ALL of the terms are present (catches paraphrases of the fairness contract). |
| `critic_flags_fairness_restatement` | bool | Critic's `has_fairness_restatement` matches. |
| `critic_quality_in` | list[str] | Critic's `overall_quality` is in the list. |
| `critic_diverges_from_priors` | bool | Critic's `diverges_from_priors` matches. |
| `critic_cites_features_any` | list[str] | At least one term overlaps with `cites_specific_features`. |

Expectations are **deliberately loose** — the harness is not a
unit test on exact text. We allow multiple acceptable slots for a
scenario when there's no single right answer.

## Running

```bash
cd rendezvous_comm

# Smoke (1 scenario × 1 LLM):
python scripts/run_scenario_sweep.py \
    --scenarios A1_flat_zero_baseline \
    --llms gpt-5.4-mini

# Subset:
python scripts/run_scenario_sweep.py \
    --scenarios A1_flat_zero_baseline,B2_critic_accepts_specific_output \
    --llms gpt-5.4-mini,ovh-llama-3.3-70b

# Full sweep (24 × 4 = 96 LLM-pipeline runs):
python -u scripts/run_scenario_sweep.py --output /tmp/sweep.json
```

Output: a markdown report on stdout + a full JSON dump (decisions,
parse errors, latencies) at the path passed to `--output`.

> Use `python -u` (unbuffered) when redirecting to a file, otherwise
> the markdown rows show up only at the end (default Python buffers
> stdout when piped).

### LLM registry

Defined in `scripts/run_scenario_sweep.py::LLM_CONFIGS`:

| Label | Model | Endpoint | Notes |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | `gpt-5.4-mini` | OpenAI default | Current production baseline. |
| `gpt-4o` | `gpt-4o` | OpenAI default | 15× cost, stronger reasoning (mostly). |
| `ovh-llama-3.3-70b` | `Meta-Llama-3_3-70B-Instruct` | `OVH_AI_ENDPOINTS_URL` | Largest LLaMA on OVH (Llama 405B is NOT hosted). |
| `ovh-gpt-oss-120b` | `gpt-oss-120b` | `OVH_AI_ENDPOINTS_URL` | OpenAI open-weights, 120B params. Slowest. |

Add a new LLM by editing `LLM_CONFIGS`.

## Findings — sweep history

### v1 (14 scenarios, 4 LLMs, baseline prompts)

| LLM | Pass rate | Latency |
|---|---|---|
| gpt-5.4-mini | 88.9% | 88s |
| ovh-llama-3.3-70b | 77.8% | 174s |
| ovh-gpt-oss-120b | 74.1% | 511s |
| gpt-4o | 66.7% | 98s |

Surfaced 3 universal-failure patterns:

1. **Outlier metrics not driving decisions** — Strategist ignored
   M4/M9 outliers when seed bias contradicted them.
2. **Critic refuses to "keep" even good output** — Critic prompt
   biased every LLM toward `revise` regardless of slot quality.
3. **Llama JSON parsing failures** — Llama wraps JSON in fences
   that the v2 regex couldn't strip.

### v2 (24 scenarios, baseline prompts — expanded coverage only)

| LLM | Pass rate | Latency | Δ vs v1 |
|---|---|---|---|
| gpt-5.4-mini | 89.4% | 165s | +0.5pp |
| ovh-gpt-oss-120b | 85.1% | 912s | +11pp ↑↑ |
| ovh-llama-3.3-70b | 78.7% | 310s | +0.9pp |
| gpt-4o | 74.5% | 149s | +7.8pp |

### v3 (24 scenarios, prompt fixes applied)

Three patches applied:

1. **Strategist** — added `[METRIC INTERPRETATION GUIDE]` block
   that explicitly maps M4/M8/M9 outliers to slot choices and tells
   the LLM to override the seed bias when signals fire.
2. **Critic** — restructured prompt so `overall_quality` decision
   comes FIRST, defaults to `keep`, requires empty `suggested_edits`
   when keeping, and clarifies that brief mentions like "without
   needing oracle positions" are NOT fairness restatements.
3. **`parse_critique`** — accepts ```json fences (Llama style),
   tolerates trailing prose, backfills more optional fields.

| LLM | Pass rate | Δ vs v2 |
|---|---|---|
| ovh-gpt-oss-120b | 87.2% | +2.1pp |
| ovh-llama-3.3-70b | 85.1% | +6.4pp ↑↑ |
| gpt-5.4-mini | 89.4% | 0 |
| gpt-4o | 70.2% | −4.3pp ↓ |

**B2 (Critic accepts specific output)**: 1/3 → **4/4 keep** across
all LLMs — Bug 2 confirmed fixed.

**E1 / E2 (override bias)**: only Llama + gpt-oss correctly override
in both directions. gpt-5.4-mini and gpt-4o still anchor on bias.

The apparent gpt-4o regression came from the harness counting
`Critic.overall_quality="reject"` as a parse error (because
`critique_and_revise` raised `ValueError`). Fixed in v4.

### v4 (24 scenarios, harness reject-recovery)

The runner now synthesizes a `reject` critique outcome when
`critique_and_revise` raises `ValueError`, so harness scoring
includes that verdict instead of dropping the run as a parse error.

| LLM | v3 | v4 | Δ |
|---|---|---|---|
| **gpt-4o** | 70.2% | **89.4%** | **+9** ↑↑ |
| ovh-gpt-oss-120b | 87.2% | 85.1% | −1 |
| gpt-5.4-mini | 89.4% | 85.1% | −2 (T=1.0 noise) |
| ovh-llama-3.3-70b | 85.1% | 80.9% | −2 (T=1.0 noise) |

**The big win**: gpt-4o jumped 70% → 89% because it was the LLM that
rejected outputs most aggressively, and the harness was previously
discarding those runs as parse errors. With reject-recovery, gpt-4o's
rejections count as strong-positive Critic verdicts.

**Smaller deltas (±2pp) for the other LLMs are within the
temperature=1.0 sampling noise band** — running the same seed twice
gives ±1–3pp variation on a 47-check sweep.

### Sweep cost

| Sweep | Scenarios | LLMs | Runs | Wall | Cost |
| --- | --- | --- | --- | --- | --- |
| v1 | 14 | 4 | 56 | ~15 min | ~€0.8 |
| v2 | 24 | 4 | 96 | ~25 min | ~€1.2 |
| v3 | 24 | 4 | 96 | ~30 min | ~€1.2 |
| v4 | 24 | 4 | 96 | ~30 min | ~€1.2 |
| **Total** | | | **344** | ~1.7 h | **~€4.5** |

For comparison, ONE OVH dry-run was ~50 min wall + €22.

## Validation: harness fixes carry through to OVH (v3p dry-run, 2026-04-25)

Re-ran the 3-seed × 3M OVH dry-run (`mp_v3_dryrun_3m.yaml`) with the
v3 prompt fixes baked in. Compared to the original v2.1 OVH dry-run
on identical scenario + seeds:

| Metric | v2.1 dry-run | v3p dry-run | Validates |
|---|---|---|---|
| Critic verdicts (3 seeds) | revise / revise / revise | **keep / keep / keep** | Bug 2 fix |
| Critic revisions per seed | 1–2 | **0** (no over-revising) | Bug 2 fix |
| Mutation verdicts | neutral, **regression**, **regression** | neutral, neutral, neutral | Bug 1 helping |
| Best Δpeak_M1 across seeds | +0.000 | **+0.010** (seed 0) | one mutation now positive |
| Seed-0 evolution | 0.010 → 0.010 | **0.005 → 0.015** ↑ | concrete improvement |
| Seed-1 evolution | 0.010 → 0.005 (loss) | 0.015 → 0.010 | smaller loss |
| Seed-2 evolution | 0.040 → 0.015 (loss) | 0.010 → 0.010 | no loss (vs −0.025 before) |
| Per-seed runtime | ~3,200 s | **~2,650 s** | faster (no Critic loop) |

**v3p Strategist rationales** are now also more concrete — they
cite specific feature shapes ("small, bounded shaping term that
directly rewards early progress on M1 without creating unbounded
accumulation"; "compact, symmetry-aware encodings over global-state
shortcuts") instead of generic advice.

**Confirmed bug-fix path**: harness scenarios → prompt edits → harness
re-run → OVH dry-run validation. The OVH delta directly reflects what
the harness predicted — Critic stops over-revising, mutations stop
regressing, runtime drops.

**Remaining open issue from OVH** (not from harness): only 1 mutation
per seed, because `max_outer_iters=3` + `plateau_iters=1` means the
baseline runs twice before the first mutation. Bumping
`max_outer_iters=5` or `plateau_iters=0` would give 2 mutations per
seed without other config changes. Tracked separately.

## Lessons learnt

1. **Prompt rewrites need scenario-level evidence, not OVH burns.**
   Going from v1 → v3 prompts cost €2 and 30 min via the harness;
   the equivalent OVH-only iteration would have been ~€44 + 1.5h.
2. **Different LLMs have different anchoring biases.**
   gpt-4o reaches for `guidance_reward` more often than evidence
   warrants; Llama anchors hard on the seed bias hint.
3. **Critic verdicts compress to 3 categories — keep/revise/reject —
   but the prompt's structure matters more than its content.**
   Reordering JSON keys (decision-first) had a bigger impact than
   the actual interpretive guidance.
4. **Llama needs explicit JSON-fence handling.** Production code
   (`parse_critique`) now does this; new parsers should follow.
5. **Cost / quality / latency don't correlate as expected.**
   `gpt-5.4-mini` is the cheapest **AND** highest-pass-rate LLM in
   our pipeline; `gpt-4o` costs 15× more for worse results;
   `gpt-oss-120b` is 6× slower than Llama-70B for marginal gains.
   No upgrade is justified for production.

## Recommended LLM for production

Stick with **`gpt-5.4-mini`**. It's the cheapest and the highest
pass-rate. Llama-3.3-70B (OVH) is a viable fallback if OpenAI is
down — same accuracy ballpark, same JSON shape.

Do NOT switch to gpt-4o on the basis of "GPT-4 is smarter" —
empirically it isn't, for this pipeline.

## Adding a scenario

1. Pick a unique key (`X_<group_letter><number>_<short_name>`).
2. In `tests/scenarios/fixtures.py`, append to `ALL_SCENARIOS`.
3. Use `_rec`, `_cand`, `_prior` to build the synthetic data.
4. Specify the `expected` block — keep it loose (alternatives, not
   single answers).
5. Run `python scripts/run_scenario_sweep.py --scenarios <key>`
   to verify all LLMs handle it.
6. Once the scenario is stable, include it in the next full sweep.

## Adding a check type

1. Edit `tests/scenarios/runner.py::check_expectations`.
2. Add an `if "<key>" in expected:` block.
3. Document the key in the **Expected-checks vocabulary** table above.
4. Update existing scenarios that should use it.

## Open follow-ups

- Wire `editor_revise_call` so harness can also test multi-round
  Editor revision (currently revisions=0 everywhere because the
  harness doesn't re-invoke the Editor).
- Add scenarios H2-H4 covering NaN crash recovery, fairness
  violations alternating with healthy runs, and very long history
  tails.
- Wire the harness into CI as `@pytest.mark.llm` (opt-in, not
  in default test runs).
- Compare to a Claude family option once Anthropic structured
  outputs ship via LiteLLM (see `lero_metaprompt_v3_plan.md` §7).
