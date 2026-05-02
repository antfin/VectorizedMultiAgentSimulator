# LERO v8 Plan — Density-First, Bounded Features, Concrete Fewshot

**Status:** Draft, awaiting sign-off before implementation
**Date:** 2026-05-01
**Predecessor:** v7 run 1 (M1=0 with 5/5 feature_stack but 35 features cover-zone-gated)
**Bug-check confirmed:** v5/v6/v7 inner_loop machinery is bug-free. Gap to S3b-local is prompt content.

---

## 1. The premise

v7 closed the **structural** gap to S3b-local (5/5 feature stack, anti-cheat clean) but introduced a new gap: **35 features, most cover-zone-gated, no dense-signal features**. The v7 vs S3b-local comparison (`docs/v7_vs_s3blocal_comparison.md`) showed:

- v7 over-concentrated decision features in the cover zone (zero gradient outside cover_r)
- v7 dropped S3b-local's dense signals (`t_close_mean`, `t_dispersion`, ungated `rendezvous_pressure`, `boundary_dist`)
- v7 produced 35 features vs S3b-local's 19 — likely too many, adding noise rather than signal

v8's hypothesis: **PPO at 1M frames learns better from FEWER, DENSER features than from MORE, COVER-ZONE-GATED features.** Restore S3b-local's density signals; bound the feature count; have the meta-LLM author a multi-feature working fewshot.

---

## 2. The four design changes from v7

### 2.1 Feature count target: configurable, default range [target_min, target_max] = [8, 12], cap = 15

v7 hit 35 features. v8 introduces THREE config variables (knob-tunable in YAML):

```yaml
v8:
  feature_count_target_min: 8        # minimum useful obs size
  feature_count_target_max: 12       # upper end of target range
  feature_count_cap: 15              # HARD CAP — diagnosis fires if exceeded
```

S3b-local has 19 features and works; v7 had 35 and failed. The hypothesis is that simpler is better at 1M-frame budget. Default cap = 15 starts strictly tighter than S3b-local — if smoke shows it's too tight, bump to 19 via config without code change. Could also be set to 10 for an even tighter test.

**Mechanism**:
- Meta system prompt is parameterized — prompt at runtime says: *"Target {target_min}-{target_max} features. Hard cap: {cap}. Quality over quantity."*
- Code-side check: count returned features (AST analyzer's `n_returned_features`) against `feature_count_cap`
- New diagnosis label `too_many_features` fires when `n_returned_features > feature_count_cap`
- New refinement action `trim_features` complementing `add_features`
- Cap behavior is **strict**: a candidate exceeding the cap is still EVALUATED (we want metrics) but the diagnosis flags it and the next iter's slot text forces a trim

### 2.2 Density-first feature framing — also configurable

```yaml
v8:
  gated_feature_cap: 2              # max cover-zone-gated features per candidate
```

v7 over-emphasized cross-source decision features. v8 explicitly biases toward dense signals via the `gated_feature_cap` knob (default 2). Set to 1 if smoke shows even 2 gated features are too many; set to 3 if 2 is too restrictive.

**Mechanism**:
- Meta system prompt instruction parameterized: *"Limit cover-zone-gated features (those that are zero outside cover_r) to at most {gated_feature_cap} per observation. Prefer dense signals: mean of close-target rays, std/variance over rays, count-asymmetry between target and agent rays normalized by their respective totals (NOT gated by cover_r)."*
- Code-side check: scan generated code for `* near_t`, `* (lidar_t < cover_r)`, etc. — count gated features. If `> gated_feature_cap`, diagnose `over_gated`.

### 2.3 Meta-LLM authors a concrete 3-4 feature WORKING fewshot

v7's example was 1-feature trivial; S3b-local's hand-curated example is 4-feature working (`min_dist`, `dir_x`, `dir_y`, `n_close`, `one_hot`). v8 has the meta-LLM write a similar quality working fewshot — anti-cheat-clean because the LLM authors it from task understanding, not from us injecting it.

**Mechanism**:
- Meta system prompt addition: *"Inside `lero_translation_hint`, include a fenced ```python``` code block containing a partial `enhance_observation` function with 3-4 example features and an explicit `return torch.cat(...)`. Choose features that demonstrate the strategy AND include at least one dense signal alongside any cover-zone gate. Do NOT use S3b-local-style winning feature names; pick names that describe the role of the feature (e.g., `nearest_target_dist`, `direction_to_target_x`, `crowd_count`)."*
- Runtime grep redacts forbidden tokens (existing v6/v7 mechanism).

### 2.4 Trimming as a first-class refinement action

v7's reflection had `refine_current_strategy` and `switch_to_next_strategy`. v8 adds `trim_features_in_current_strategy`.

**Mechanism**:
- New diagnosis labels:
  - `too_many_features`: candidates returning >25 features → action `trim_features`
  - `over_gated`: more than 2 cover-zone-gated features per candidate → action `replace_gated_with_dense`
- New `next_action` values: `trim_features`, `replace_gated_with_dense`
- The meta-LLM is given a list of features to remove (by name from AST analyzer's var_sources) when trimming

---

## 3. Hypotheses v8 tests

| H1 | Hypothesis |
|---|---|
| **H1.1** | A 12-19 feature observation with 1-2 gated + many dense features will reach M1>0.02 at 1M-eval (matching S3b-local seed 0 iter 0) |
| **H1.2** | Specifically `t_close_mean` + `t_dispersion` + ungated `rendezvous_pressure` are necessary for the iter-0→iter-1 M6 jump (S3b-local seed 0 iter 0 had M6=0.225, iter 1 winner had M6=0.425) |
| **H1.3** | A meta-LLM-authored 3-4 feature working fewshot is sufficient to make the inner LLM produce candidates with the right density blend |
| **H1.4** | Adding `boundary_dist` reduces wall-hugging and helps M3/M4 metrics meaningfully |

If all four hold, v8 should match S3b-local seed 1 / 2 outcomes (M1=0.82-0.83 after 10M deep-train).

---

## 4. Architecture — same as v7 with three slot additions

```
v8 = v7 + (
  feature count guard,
  density preference instruction,
  meta-authored fewshot in slot text,
  new refinement actions: trim_features / replace_gated_with_dense
)
```

No code changes to:
- inner_loop (v5)
- patched scenario
- bundle (v7's V7Strategy / V7StrategyBundle)
- runtime forbidden-token redaction
- AST analyzer (already has feature count + dense/gated detection)

Code changes to:
- `meta_strategist.py`: extended `_BUNDLE_SYSTEM` (~30 LOC additions for the 4 new instructions)
- `meta_strategist.py`: extended `_REFLECT_SYSTEM` (`trim_features` + `replace_gated_with_dense` actions)
- `diagnosis.py`: add `too_many_features` and `over_gated` labels with detection logic
- `outer_loop.py`: handle the two new next_action values
- `analyzer.py`: add `count_gated_features()` helper

Total estimate: ~150 LOC across the v7 module, + 40 LOC tests/smoke.

---

## 5. Concrete prompt additions — verbatim drafts

### 5.1 Add to `_BUNDLE_SYSTEM` (meta-LLM cold-start)

```
## Density-first feature design (v8 addition)

The inner LLM at 1M frames learns better from few, dense features than from many, gated ones. PPO needs gradient signal everywhere, not just inside the cover zone where the agent rarely visits early.

Concretely, when writing `lero_translation_hint`:

1. **Target 12-19 returned features.** Hard cap: 25. More features = more noise.
2. **Prefer DENSE signals**: features that produce informative values everywhere in state space. Examples (operational, not solution names):
   - mean of LiDAR rays below a threshold (still informative when nothing is in cover zone)
   - std/variance over rays (uncertainty proxy)
   - count-asymmetry between target and agent ray-counts NORMALIZED by their respective totals (a continuous signed value)
   - boundary distance from arena edge
3. **Limit GATED signals to 1-2 per observation**: features that are zero outside the cover zone are useful as a "stay vs go" trigger, but multiplying them adds no information. ONE such feature is enough.
4. **Include a 3-4 feature WORKING fewshot inside `lero_translation_hint`**: a fenced `python` code block with a partial `enhance_observation` showing 3-4 representative features, ending with `return torch.cat(...)`. Make at least one feature DENSE (works outside cover zone). Use neutral variable names that describe the role, NOT S3b-local handles. Example shape (do not copy verbatim):
   ```python
   # snippet — inner LLM should adapt to its own strategy
   def enhance_observation(scenario_state: dict) -> torch.Tensor:
       lidar_t = scenario_state["lidar_targets"]
       cover_r = float(scenario_state["covering_range"])
       nearest_target_dist = lidar_t.min(dim=-1).values        # DENSE
       count_close = (lidar_t < cover_r).float().sum(dim=-1)   # SEMI-DENSE
       std_targets = lidar_t.std(dim=-1)                        # DENSE
       # ... add 1 more feature relevant to this strategy ...
       return torch.cat([nearest_target_dist.unsqueeze(-1),
                         count_close.unsqueeze(-1),
                         std_targets.unsqueeze(-1)], dim=-1)
   ```
```

### 5.2 Add to `_REFLECT_SYSTEM` (per-iter reflection)

```
## v8 refinement actions

In addition to the v7 actions, you may now choose:

  - `trim_features` → the inner candidate produced too many features (>25) and the policy can't differentiate. Identify 5-10 features to remove (by name from the AST analyzer report) and rewrite the slot text requesting fewer features.
  - `replace_gated_with_dense` → the inner candidate has too many cover-zone-gated features (>2). Replace some of them with dense equivalents (mean, std, normalized-count-asymmetry).

The diagnosis report will tell you when these are the appropriate action:
  - too_many_features → trim_features
  - over_gated → replace_gated_with_dense
```

### 5.3 Diagnosis label additions

```python
DiagnosisLabel = Literal[
    "achieved",
    "partial",
    "translation_failure",
    "rl_too_hard",
    "too_many_features",     # NEW — n_features > 25
    "over_gated",            # NEW — n_gated_features > 2
    "too_early",
]
```

Logic order:
```
if M1 ≥ expected_M1 and pattern_present: → "achieved"
elif M6 ≥ expected_M6_min: → "partial"
elif n_features > 25: → "too_many_features"
elif n_gated > 2 and n_dense < 3: → "over_gated"
elif pattern_present: → "rl_too_hard"
else: → "translation_failure"
```

This nudges the meta-LLM toward simpler observations BEFORE it concludes "rl_too_hard" and switches strategy. We get one shot at trimming before giving up on a strategy.

---

## 6. Anti-cheat status

All v8 additions are LLM-authored:
- Meta-LLM authors the 3-4 feature fewshot (not hardcoded)
- Meta-LLM picks which features to trim or replace
- Runtime forbidden-token grep still active (`hold_signal`, etc.)
- Code-side analyzer detects density / gated patterns mechanically — but NEVER injects feature names

The only "hand-typed" content is the META SYSTEM PROMPT (the operational vocabulary including "mean of close rays", "boundary distance", "ungated count-asymmetry"). These are operations, not solutions. Same boundary as v6/v7.

**Anti-cheat audit pre-launch**: grep for forbidden tokens in v8 module + new prompt additions. Should be zero.

---

## 7. Test sequence

### 7.1 Phase 0: extend AST analyzer (1h)
- Add `count_gated_features()` — counts `* near_x`, `* (lidar_x < cover_r)`, etc.
- Add `count_dense_features()` — features whose computation graph stays non-zero everywhere
- Test against S3b-local winner (expected: 1 gated `settle_signal` + 17 dense), v7 best (expected: many gated, fewer dense)

### 7.2 Phase 1: prompt-lab Phase 4 — does v8 produce the right blend? (~$3, 90s)
- Run V0 (v7 baseline) vs V1 (v8 with density preference + count cap) — 3 candidates each
- Verify V1 candidates have ≤25 features, ≤2 gated, ≥3 dense
- Verify V1 includes the meta-authored fewshot in slot text
- Verify the inner LLM uses the fewshot's structure (not just copy)

### 7.3 Phase 2: smoke run on Mac (~10 min)
- 1 outer × 2 inner × 1 cand × 100k frames
- Verify pipeline + the new diagnosis labels fire correctly when triggered
- Verify trimming refinement action correctly reduces feature count

### 7.4 Phase 3: full RL run on Mac (~3h)
- 2 outer × 3 inner × 3 cands × 1M frames
- Same task as v7 (cr=0.25, ms=400)
- **Pass criteria**: best inner candidate's M1 ≥ 0.02 (matches S3b-local seed 0 iter 0) — strictly better than v7's flat 0
- **Stretch**: best M1 ≥ 0.05 (matches S3b-local seed 0 iter 1 winner)

### 7.5 Phase 4: deep-train v8 winner at 10M (~2h)
- Take best inner candidate, train 10M frames
- Compare post-eval M1 to S3b-local 0.845 mean

---

## 8. Risks & mitigations

| risk | severity | mitigation |
|---|---|---|
| Meta-LLM doesn't follow the count cap; produces 35-feature hint anyway | medium | Code-side check + diagnose `too_many_features` → next iter forces trim |
| 12-19 features still too many; needs ~10 | low | If smoke shows it, drop cap to 15 |
| Meta-LLM's authored fewshot is bad code (won't compile) | low | Inner LLM gets the example as INSPIRATION, not literal — it writes its own. Inner LLM has its own 3-attempt retry on AST validation |
| Density-vs-gated detection has false positives | low | Tested in Phase 0 against known-good S3b-local code |
| The whole density hypothesis is wrong; M1 stays 0 anyway | medium | Negative result is informative — would prove the gap is something else (training budget? hyperparams?) |

---

## 9. Open questions to confirm before implementing

1. **Hard cap value (now config)**: defaults `feature_count_cap=15`, `feature_count_target_min/max=8/12`, `gated_feature_cap=2`. Override via YAML if smoke says different.
2. **Should V8 also bound INPUT features** (e.g., disallow stacked top-3 nearest distances)? Or only output count? My default: only output count — let the LLM choose any inputs.
3. **When the meta-LLM trims, does it trim by feature NAME or by COUNT** (just say "remove 5 features")? My default: by name from analyzer report — more precise.
4. **Should v8 deep-train automatically** at the end if `found_good`, or stay inner-only like v6/v7? My default: stay inner-only, deep-train as a separate step.

If you say "go with defaults", I'll build & launch the same evening.

---

## 10. Estimated cost

- Phase 0 (analyzer extension): 1h, $0
- Phase 1 (prompt-lab Phase 4): ~$3 LLM, 90s
- Phase 2 (smoke): 10 min wall, $0
- Phase 3 (full RL run): 3h Mac, ~€1 LLM
- Phase 4 (deep-train if winner): 2h Mac, $0

**Total: 6h Mac time + ~€4 LLM** for a full v8 cycle including deep-train validation.
