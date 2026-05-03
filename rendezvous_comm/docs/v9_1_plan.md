# v9.1 plan — slot-edit validator + runtime caps + production-data simulations

**Date:** 2026-05-03
**Status:** §2.1, §2.2, §2.3, §2.4, §2.6, §2.7 IMPLEMENTED + 76/76 unit tests pass; smoke + full RL pending
**Predecessor:** docs/v9_plan.md (v9 Phase 6 results), docs/v8_vs_s3blocal_prompt_comparison.md

## 0. TL;DR (revised after production-data simulation)

The root cause of v9's drift across outers is **NOT** lost mandatory features per se — it's that the meta-LLM's `refine_current` action emits **prose-only `slot_edits`** that strip the S3b-local-class structured text. Production data:

- Outer 0 `inferable_hints`: 22 lines of bulleted concepts (matches S3b-local).
- Outer 0 `examples`: 159 lines containing 5 fenced ```python``` blocks.
- **Outer 1-4 `inferable_hints`: 1 paragraph of prose** (no bullets, no concepts list).
- **Outer 1-4 `examples`: 1-3 lines of prose ("Example 1: If agent A..." with NO Python code blocks).**

The inner LLM at outer 1+ has nothing concrete to mimic. Every regression downstream (role_one_hot drop, feature explosion, M6 decay) flows from this single root cause.

**Highest-leverage fix:** §2.3 (slot-edit structural validator). Reject any `slot_edits` whose new text doesn't preserve the structural shape of the original.

**Second high-leverage fix added 2026-05-03 after CoT review** (`docs/v9_phase6_cot_full.md`): §2.7 (falsification-gate action override). The CoT review showed the meta-LLM correctly identifies "stuck on this strategy" three outers in a row but the action map forces `refine_current` because the per-outer label is `partial`. §2.7 overrides `refine_current` to `switch_to_next` whenever ≥2 attempts on a strategy all fall below 0.5×expected_M1. This converts already-correct diagnosis into the corresponding action.

## 1. Findings driving v9.1

v9 Phase 6 completed 5/5 outers in 6h 42min (`results/lero_v9/lero_v9_rendezvous_k2_2x3/20260502_1912_s0`):

| | best M1 | best M6 | role one-hot rate (45 cands) | wall |
|---|---|---|---|---|
| v8 Phase 3 | 0.010 | 0.209 | 0% | 2h 42m |
| **v9 Phase 6** | **0.010** | **0.169** | **82%** | 6h 42m |
| S3b-local 1M (3 seeds) | 0.070 | 0.435 | 97% | per-seed ~3h |

v9 closes the structural gap from v8 (0% → 82% role one-hot) but the **`refine_current` action degrades the prompt over outers**. Production data shows:

- Outer 0 (initial): 100% role one-hot, 16.9 avg features (matches S3b-local).
- Outer 4 (after 2 refines + 1 fail-safe switch + 1 refine): 22% role one-hot, 23.8 avg features.

The **soft hint** for `mandatory_features` and the **soft cap** on `feature_budget.hard_cap=20` don't survive multi-iter refinement: the meta-LLM keeps growing complexity and dropping mandatory features under pressure to "improve".

## 2. v9.1 design changes (priority-ordered)

### 2.1 Hard mandatory_features check (HIGH)

**Problem:** task_domain.yaml's `mandatory_features` (role_one_hot, cross_source_signal) are advertised as MUST-HAVE in the meta-prompt, but only as a soft hint. v9 production: 8/45 candidates (18%) lack role_one_hot, all in outer 4.

**Fix:** in `_compute_facts` AND in the inner-loop pre-eval validation, check if mandatory features are present in the candidate code. If not:
- Mark the candidate FAILED before training
- Don't waste 9 min training a candidate that's structurally broken
- Inject the failure reason into the next inner LLM's feedback

Implementation: extend `analyze_inner_code` (already detects role_one_hot) → add `validate_mandatory_features(code, td)` that returns a list of missing mandatory features. Inner loop calls it before submitting to BenchMARL training.

```python
# In src/lero/v5/inner_loop.py before _evaluate_candidate
missing = validate_mandatory_features(cand.obs_source, task_domain)
if missing:
    log.warning("candidate failed: missing mandatory features %s", missing)
    outcome = CandidateOutcome(
        candidate=cand,
        metrics={"M1_success_rate": 0.0, "M6_coverage_progress": 0.0},
        fitness=-99.0,  # mark as worse than any real candidate
        shape="missing_mandatory",
        iter_idx=iter_idx,
        validation_error=f"missing: {missing}",
    )
    # Add to failed registry, continue
```

**Tradeoff:** more aggressive = some good candidates rejected if they implement role_one_hot in an unusual way. Mitigation: keep the AST detector tolerant (already accepts 4 patterns).

### 2.2 Hard runtime feature_budget cap (MEDIUM)

**Problem:** soft hint `hard_cap=20` violated in 32/45 candidates (71%). Outer 2 averaged 28.0 features — well over.

**Fix:** in `_compute_facts`, if `n_features > hard_cap`, mark as `over_budget` and reject. Same flow as 2.1.

**Caveat:** the AST `_estimate_n_features` is best-effort and sometimes returns 0 for valid code. Soften the rejection: only reject if `n_features` is reliably > hard_cap (i.e., AST detected the cat call AND counted > 20). If AST returned 0, skip the check (don't reject what we can't measure).

### 2.3 Refine-action drift mitigation (HIGH)

**Problem:** `refine_current` lets the meta-LLM rewrite slot text freely, and quality drifts across outers. Outer 4 lost role_one_hot from the inferable_hints text entirely.

**Fix:** when applying `slot_edits`, run a structural check on the new slot text BEFORE writing it:
- inferable_hints must mention all 7 task_domain.inferable_concepts
- examples must contain ≥1 fenced ```python``` block with role_one_hot
- examples must contain ≥2 fenced ```python``` blocks total
- text length must be within 0.5x-2x of the original (stop runaway growth)

If any check fails: REJECT the slot_edit, keep the previous slot text, log the rejection. Force the meta-LLM to retry with a more conservative edit on next outer (or fail-safe switch sooner).

### 2.4 Prompt redundancy cleanup (MEDIUM)

**Problem:** task_framing duplicated in `system.txt` AND `task_context.txt` slots. v9 prompts are 2.8× more verbose than S3b-local (1500 vs 542 words). Half is signal, half is duplication.

**Fix:** drop `task_context.txt` slot entirely (move coordination_challenges to a 1-line summary at the bottom of `system.txt`). Cuts ~400 words per prompt with no loss of information.

### 2.5 Bundle exploration: try N strategies before stopping (LOW)

**Problem:** v9 Phase 6 max_outer=5=bundle_size, but only 2/5 strategies tried (pair_and_split 3×, leader_follower_pairing 2×). 3 strategies untested: `crowd_aware_avoidance`, `opportunistic_commit`, `sector_specialization`.

The fail-safe (added mid-run) helped: forced 1 switch. Without it, all 5 outers would have been pair_and_split.

**Fix options:**
- **(A) Tighter fail-safe**: drop the n=2 floor to n=1 — first regression triggers switch. Risk: reactive (one bad sample switches strategy).
- **(B) Hard cap on attempts per strategy**: max 2 outers on any single strategy. After that, force switch regardless of trend.
- **(C) Increase max_outer** beyond bundle_size: allow re-attempting strategies after others fail.

**Recommendation:** (B). Caps each strategy at 2 attempts (1 initial + 1 refine), guaranteed coverage of `min(max_outer, bundle_size)` strategies.

### 2.6 Memory-aware refine (LOW)

**Problem:** the meta-LLM's `memory_recall` field is read but the model doesn't seem to actively use cross-outer memory to prevent drift.

**Fix:** make the system prompt explicit that the LLM must START reflection by quoting the predicted-vs-actual delta from memory, AND that any slot_edits must NOT remove features that were in the memory's prior CoT's `what_is_needed` list. Stronger structural anchor.

### 2.7 Action-map override: convert "stuck" CoT into switch decision (HIGH)

**Problem:** v9 CoT review (`docs/v9_phase6_cot_full.md`) shows the meta-LLM is a competent diagnostician but gets trapped by the action map. Across outers 1-3 it explicitly verbalizes "the recurring pattern is the structural ingredients are present but they don't translate" — yet `next_action` stays `refine_current` because the per-outer label is `partial` (not `rl_too_hard`). Memory is quoted but doesn't drive the decision.

Concrete telemetry from the production CoT:
- Outer 1 says "even lower at M1=0.0/M6=0.1329" → still picks refine.
- Outer 2 says "regressed further" → still picks refine.
- Outer 3 says "outcomes stayed poor across the last three outers" → still picks refine.
- All three outers expected M1=0.18, observed ≤0.01 — an 18× shortfall — but the `success_signature` is treated as immutable.

**Fix (post-LLM override, no prompt change):** in the v9 outer loop, after `reflect_decide_v9` returns, apply a structural override:

```python
if decision.next_action == "refine_current":
    # Read memory + facts and check the falsification gate
    if active.attempts >= 2:
        # Have we ever come close to expected M1?
        recent = mem.read_recent(3)
        same_strategy = [r for r in recent if r["strategy_name"] == active.name]
        actual_M1s = [
            r["actual"].get("M1", 0.0) for r in same_strategy
        ]
        target = 0.5 * active.success_signature.expected_M1_at_1M
        if all(m < target for m in actual_M1s) and len(actual_M1s) >= 2:
            _log.warning(
                "v9 outer %d: OVERRIDE — strategy '%s' has run %d "
                "attempts, all M1=%s below 0.5×expected (%.3f). "
                "Forcing switch_to_next regardless of LLM label.",
                outer_idx, active.name, active.attempts,
                actual_M1s, target,
            )
            decision.next_action = "switch_to_next"
            decision.rationale = (
                "[runtime override] " + decision.rationale
                + f" — falsification gate: M1<{target} for "
                f"{len(actual_M1s)} attempts."
            )
```

Differs from §2.5(B) max-attempts: this is **performance-conditioned**. A strategy that's reaching M1≥0.5×expected stays alive; one consistently below the gate gets demoted regardless of LLM judgment.

**Telemetry:** save the override decision to `_decision.json` so we can post-hoc count how many of the LLM's `refine_current` choices got flipped. If most do, the prompt itself needs strengthening (escalate to §2.6).

**Tradeoff:** more aggressive switching loses some refinement opportunity. Mitigation: 0.5× threshold is generous (matches "rough par" semantics); the strategy can still pass with mediocre learning. Higher threshold = harsher gate.

### 2.8 Plumbing cleanup carried over from v9_plan.md §6.1 (LOW)

These were identified during v9 design but deferred. They don't affect M1 — purely codebase hygiene. Bundle into one cleanup PR after the M1 work lands.

| issue | proposed fix |
|---|---|
| `_v9_checkpoint.pkl` + `_bundle_init.json` + `_bundle_state_before.json` + `_bundle_state_after.json` (4 different snapshots of overlapping state) | Keep `_v9_checkpoint.pkl` (machine-readable resume state) + `_bundle_init.json` (one-time initial bundle for telemetry). Drop `_bundle_state_*.json` per outer — same info lives in `_decision.json` + checkpoint. |
| `_inner_legacy_outdir` directory passed to `LeroLoop` then unused | Remove the parameter; call `v5.inner_loop.run` directly. |
| `LeroLoop` wrapper around `v5.inner_loop` (back-compat shim) | Inline the wrapper into `run_v9_outer_loop`. The `LeroLoop` abstraction was useful for v5/v6 but adds nothing for v9. |
| Inner stagnation pivot (`STAGNATION → pivot prompt`) — was confirmed-dropped in v9_plan §6.2 but the v5 inner-loop code may still emit the warning log | Audit v5 inner-loop, ensure stagnation detection is fully disabled in the v9 path. |
| Per-outer prompt directories grow to 5+ copies of the base template | One symlinked or text-pointer reference instead of `shutil.copytree` for unchanged base files; only slot files are unique per outer. |

None of these block v9.1.7 (the M1 climb test). They become important only if we extend the system to many more strategies / longer runs.

## 2.9 Production-data simulation results (run before any implementation)

Each proposed patch was simulated against the v9 Phase 6 saved artifacts (`results/lero_v9/.../20260502_1912_s0/`). Verdict per patch:

| patch | catches in production data | verdict |
|---|---|---|
| **§2.3 slot-edit validator** | outers 01, 02, 03, 04 ALL fail validation: 0 ```python``` blocks in examples (vs requirement ≥2), inferable_hints concepts hit 0-4/7 (vs requirement ≥5). **Outer 0 passes.** | EXACTLY catches the regression points — highest-leverage fix |
| **§2.1 mandatory_features** | rejects 8/45 candidates (1 in outer 1, 7 in outer 4 — all role_one_hot=False). Saves ~72 min training time on a typical run. | downstream safety net for cases §2.3 misses |
| **§2.2 feature_budget cap** | rejects 28/45 candidates (62%, mostly outers 2-4). Outer 2 had 9/9 candidates over cap (avg 28 features). | rejection rate is high — may need to relax cap to 22 OR pair with §2.3 to fix the upstream cause |
| **§2.5(B) max attempts** | catches 1/5 outers (same as the runtime fail-safe already added) | redundant once §2.3 lands; defer |
| **§2.7 falsification gate** | replays show: outer 2 (`pair_and_split` attempt #2, M1=0.0 ≪ 0.09) and outer 4 (`leader_follower_pairing` attempt #2, M1=0.0 ≪ 0.07) would BOTH have been flipped to `switch_to_next`. Catches the LLM's "stuck-CoT-but-refine-action" pattern visible in `docs/v9_phase6_cot_full.md`. | high-leverage; pairs naturally with §2.3 (one fixes upstream slot quality, the other fixes downstream action selection) |

**Key insight:** §2.3 fixes the upstream cause. §2.1 and §2.2 are downstream safety nets that catch what §2.3 misses. §2.4 (redundancy cleanup) is independent — handles verbosity, not drift. §2.5 becomes redundant.

## 3. Revised implementation order

| phase | what | wall | LLM cost | gating |
|---|---|---|---|---|
| 9.1.0 | Plan review | 30 min | $0 | user sign-off |
| 9.1.1 | **§2.3 slot-edit structural validator** + replay tests (replay outer 1-4 saved slot_edits, confirm rejection). Implement, integrate into outer_loop's refine_current path. | 2 h | $0 | unit tests pass; outer 1-4 production slot_edits would have been rejected |
| 9.1.1b | **§2.7 falsification-gate action override** + replay tests (replay outer 2 and outer 4 — both would flip refine_current → switch_to_next). | 1 h | $0 | unit tests pass; production replay shows ≥2 forced flips |
| 9.1.2 | §2.1 mandatory_features runtime check + replay tests on the 8 production candidates that lacked role_one_hot | 1 h | $0 | unit tests pass |
| 9.1.3 | §2.2 feature_budget hard cap (revisit cap value: 20 may be too tight; consider 22 based on production data) | 1 h | $0 | unit tests pass |
| 9.1.4 | §2.4 redundancy cleanup (drop task_context.txt). Re-run prompt-lab convergence to confirm no regression. | 1 h + 5 min sweep | $3 | role_one_hot ≥80%, hints concepts ≥7/7 still hold; word count drops to <1000 |
| 9.1.5 | §2.6 stronger memory wording in reflect prompt | 30 min | $0 | manual review |
| 9.1.6 | Mac smoke (verify pipeline + new validators work) | 10 min | $1 | smoke runs without ALL slot_edits being rejected (sanity floor) |
| 9.1.7 | Mac full RL (1M, 1 seed) | ~6 h | $8 | M1 best ≥ 0.030 OR clear evidence task hits PPO 1M ceiling |
| 9.1.8 | If 9.1.7 M1 < 0.03: deep-train v9.1 outer 0 winner at 10M | ~3 h | $0 | M1 ≥ 0.30 OR conclude task hits ceiling at 10M too |

Skipped from earlier plan:
- ~~§2.5(B) max attempts/strategy~~ — redundant once §2.3 lands; the runtime fail-safe (already in v9) suffices.

Total: ~14 h dev + ~12 h RL + ~$12 LLM.

## 4. Test plan additions for v9.1

Extend `src/lero/v9/tests/test_v9.py`:

- `TestMandatoryFeaturesValidator` — covers the new `validate_mandatory_features` function: passes/fails on real production candidates (confirms outer 4 cands would have been caught).
- `TestSlotEditValidator` — covers the §2.3 slot-edit structural validator: replays outer 4's actual `slot_edits` from production and confirms they would have been rejected.
- `TestMaxAttemptsPerStrategy` — covers §2.5(B): bundle of 5 strategies, each strategy attempted 2 times, verify forced switch after attempt 2 regardless of trend.
- All replay tests must use real production data from `results/lero_v9/.../_meta_memory.jsonl` and `outer_*/inner/iter_*/candidate_*_obs.py` to demonstrate the patches would have caught the actual production failures.

## 5. Open questions

1. **Hard mandatory_features (§2.1) — escalation if soft fails again?** v9 already escalated soft → strict-via-meta-prompt. v9.1 escalates to runtime AST check. If runtime AST still misses cases, what's next? **Recommend:** if v9.1 production still shows <80% role_one_hot, add an LLM-judge as final gate. Tradeoff: $/cand. Probably not needed.

2. **Slot-edit validator (§2.3) — what if validator rejects every meta-LLM edit?** Could deadlock if the meta-LLM can't satisfy the structural checks. **Mitigation:** rejection limit (3 retries on the same outer), then fail-safe to switch.

3. **Should §2.4 redundancy cleanup happen BEFORE or AFTER the runtime caps land?** If we drop task_context.txt and prompts get shorter, it might INCREASE quality (less noise to drift in). Or it might do nothing (the issue is in the meta-authored slots, not the static ones). **Recommend:** §2.1+§2.2+§2.3 first (root cause), then §2.4 (verbosity polish), then re-test.

## 6. What v9.1 deliberately does NOT change

- The bundle CoT enumeration (it works — outer 0 is high quality).
- The combined diagnose+reflect single-call structure (works).
- task_domain.yaml-driven framing (works, portable).
- MemoryStore + N=3 lookback (works).
- The pathological-refine fail-safe added in §2 (works, tested).
- Reward function (still `evolve_reward: false`).

The hypothesis is that v9.1 caps the drift modes that v9's soft hints couldn't prevent, without redesigning the loop. Phase 9.1.7 falsifies this if M1 still ≤ 0.020.

## TL;DR (revised)

v9.1 = **slot-edit structural validator** (§2.3, upstream fix for prose-only refines) + **falsification-gate action override** (§2.7, downstream fix for stuck-CoT trapped by the action map) + **runtime hard caps for mandatory_features and feature_budget** (§2.1, §2.2 safety nets) + **prompt dedup** (§2.4 verbosity polish). The two HIGH-priority fixes (§2.3 + §2.7) target two different production failures: §2.3 rejects degenerate slot edits before they're written; §2.7 forces strategy switch when 2+ attempts undershoot expected_M1 by 2× regardless of the LLM's per-outer label. Production-data simulation confirms each catches its target failures. ~15 h dev + ~12 h RL + ~$12 LLM.

## Appendix A — full meta-LLM CoT examples from v9 Phase 6

For posterity / debugging reference, see `results/lero_v9/lero_v9_rendezvous_k2_2x3/20260502_1912_s0/`:
  - `_bundle_init.json` — initial bundle of 5 strategies, each with `chain_of_thought = {why_it_works, what_is_needed, failure_modes}`
  - `outer_*/` `_decision.json` — per-outer reflection CoT, `memory_recall`, `diff_vs_predicted`, `reflection_chain_of_thought = {what_went_right, what_went_wrong, remaining_uncertainty}`
  - `_meta_memory.jsonl` — append-only history (5 rows for Phase 6) used as N=3 lookback in subsequent reflections.
