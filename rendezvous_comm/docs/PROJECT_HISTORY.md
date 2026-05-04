# rendezvous_comm — complete project history

**Date:** 2026-05-04
**Status:** v9.1.10 (latest) complete; M1 ceiling hypothesis under test.
**Domain:** Multi-agent rendezvous in VMAS Discovery scenario. 4 agents, 4 targets, k=2 (each target requires exactly 2 agents simultaneously within `covering_range`). Local LiDAR-only observations; no oracle access.
**North-star task:** k=2, `covering_range=0.25`, `max_steps=400`, `lidar_range=0.35`, no respawn.
**M1 = success rate at 1M training frames** (the headline metric in this doc).

This document is the single chronological record of every experiment phase, what was tried, what was learned, and what came next. Pre-v8 history was synthesized from `experiment_storyline.md`, `lero.md`, `all_experiments_analysis.md`, `v6_plan.md`, `v7_vs_s3blocal_comparison.md`, `prompt_evolution_analysis.md`, `comparison_gnn_vs_minimal_team.md`, `report.md`. v8 onward is from this session's runs and `v8_plan.md`, `v8_vs_s3blocal_prompt_comparison.md`, `v9_plan.md`, `v9_phase6_cot_full.md`, `v9_1_plan.md`, `v9_1_postmortem.md`, `v9_1_10_lineage.md`, `v9_1_10_vs_s3b_feature_align.md`.

---

## 1. The headline result (so far)

| run | best M1 | best M6 | role one-hot rate | wall | seeds |
|---|---|---|---|---|---|
| ER1 (no-comm baseline) | 0.405 | 0.80 | n/a | 1 seed | 1 |
| ER2 (proximity comm) | 0.53 | 0.825 | n/a | 1 | 1 |
| ER3 (GNN message-passing) | **0.71** | 0.918 | n/a | 1 | 1 |
| **S3b-local** (LLM obs-only, hand-prompt) | **0.88** | 0.97 | 97% | per-seed ~3h | 1 |
| **S3b-local replicate** | mean **0.845** ± 0.03 | mean 0.943 | 97% | per-seed ~3h | 3 |
| LERO-MP v4 / B' | 0.238 | — | n/a | ~3h × 3 | 3 |
| LERO-MP v5 | 0.010 | — | n/a | ~3h | 1 |
| LERO v6 (anti-cheat) | 0.000 | 0.153 | 0% | ~10.5h total | 4 runs × 1 seed |
| LERO v7 (cold-start strategist) | 0.000 | — | 0% | ~3h | 1 |
| **LERO v8** (density-first prompt) | **0.010** | 0.209 | 0% | 2h 42m | 1 |
| **LERO v9** (CoT + memory + task_domain) | **0.010** | 0.169 | 82% | 6h 42m | 1 |
| **LERO v9.1 v0** (slot validators, inadvertent artifact bug) | 0.010 | 0.194 | 100% (only 6/45 trained) | 3h 6m | 1 |
| **LERO v9.1.10** (lazy artifact authoring + soft_proximity) | **0.010** | **0.244** | **100%** | 6h 8m | 1 |

S3b-local's hand-curated prompt remains the gold standard at **M1=0.845 mean across 3 seeds at 1M frames**. The auto-meta-search line (v6→v9.1.10) plateaued at **M1=0.010 single-seed** despite progressively closing the structural gap to 100% role one-hot, soft proximity, cross-source, role-conditioned features.

---

## 2. Pre-LERO baselines (March 2026)

### 2.1 ER1 — no-communication hand-crafted (2026-03-22 → 03-25)

- **Setup:** 4 agents, 4 targets, k=2. LiDAR observations (15 target rays + 12 agent rays). Reward = covering_rew_coeff·(targets covered) − collision_penalty − time_penalty, shared.
- **Critical bug:** `config.pop("max_steps")` removed `max_steps` BEFORE the scenario factory saw it → ALL early experiments accidentally ran at the default `max_steps=100` instead of 200/400. Fixed 2026-03-22.
- **Result post-fix (ms=400, cr=0.25):** M1=0.405, M6=0.80.
- **Learning:** k=2 is fundamentally harder than k=1. Adding agent LiDAR paradoxically HURT performance (agents learn collision avoidance instead of pairing). Doubling `max_steps` or widening `covering_range` recovers 7-10× M1 — coordination is time-constrained.

### 2.2 ER2 — engineered communication (2026-03-25 → 03-28)

- **Setup:** Add explicit `dim_c` channel — agents emit and read messages. Two variants: proximity (only nearby agents receive) vs broadcast (all agents).
- **Result:** Proximity M1=0.53, broadcast M1=0.46 at ms=400/cr=0.25.
- **Learning:** Communication helps but only when episodes are long enough to develop a protocol. Proximity > broadcast on hard task. Communication is a "neutral amplifier" — it enables what the policy is already trying to learn.

### 2.3 ER3 — GNN message-passing (2026-03-28 → 04-05)

- **Setup:** Replace explicit channel with GNN (GATv2Conv). Agents are nodes, attention learns which neighbors matter.
- **Result:** **M1=0.71, M6=0.918** at ms=400/cr=0.25. Best hand-crafted approach.
- **Learning:** Implicit spatial communication via attention beats fixed channels. M9 (spatial spread) drops 0.578 vs no-comm's 1.018 — agents physically cluster into pairs.

---

## 3. LERO foundations (April 2026)

### 3.1 Paper-faithful reproduction (k=1) — 2026-04-15

- **Setup:** Implement the LERO paper algorithm (LLM iteratively edits reward + observation Python code; PPO trains; metrics feed back into LLM). First test on k=1 (single-agent-per-landmark).
- **Result:** M1=1.00 on k=1 (paper benchmark).
- **Learning:** LERO infrastructure works. Confirmed before pivoting to k≥2.

### 3.2 LERO on k=2 with reward evolution (S3, S3a, S3ac, S3a_gpt5)

- **Setup:** `evolve_reward=true, evolve_observation=true`. LLM authors both. Various LLM models.
- **Result:** M1 in 0.000-0.105 range across all 5 attempts. S3a_gpt5 hit eval M1=0.86 at 1M but **collapsed to 0.09 at 10M** — classic reward-hacking.
- **Learning:** **k=2 reward design is fundamentally unstable.** LLM authors individually-rational rewards (anti-crowding, per-agent approach shaping) that are collectively-irrational for rendezvous. Reward CANNOT be ground-truth-checked by the LLM (it's optimized against), so any LLM-authored reward eventually gets gamed.

### 3.3 S3b-local breakthrough — 2026-04-19

- **Setup:** Single change from S3: `evolve_reward=false`. LLM only edits observation features. Reward stays ER1 hand-crafted.
- **Result:** **M1=0.88, M6=0.97 single seed.** Beat ER3 GNN (0.71) by 17pp.
- **Critical insight:** **Feature engineering > incentive design for k≥2.** Observations are read-only — they can't be gamed. LLM designs four coordination signals as observation features:
  - `settle_signal = (target_near) * (agent_near)` — a pre-computed "stay vs go" trigger
  - `hold_signal`, `approach_signal`, `rendezvous_pressure`
- **3-seed replicate (2026-04-29):** M1 = 0.885 / 0.820 / 0.830, mean **0.845 ± 0.03**. Robust.

### 3.4 The S3b-local prompt — `v2_fewshot_k2_local`

Hand-curated by humans. Single static prompt for all 4 iters × 3 cands × 3 seeds. Key components:
- 5-line system message: "You are a reward engineer... CRITICAL: This is a RENDEZVOUS task where k=2..."
- 78-line user message containing:
  - Task description with k=2 explicit
  - State schema
  - Fairness constraint
  - **"What you CAN infer from LiDAR" — 6 bulleted concepts** including `Role differentiation: agent_idx as one-hot so the shared policy can assign different roles`
  - **Worked example** — complete `enhance_observation` returning 8 features INCLUDING role one-hot
  - Function signature stub

The worked example is the irreplaceable artifact: it gives the LLM a complete code template that already embodies the answer (role one-hot, cross-source).

---

## 4. Auto-meta-search era — trying to remove the human (April–May 2026)

The S3b-local result raises the question: can a meta-LLM **rediscover** this prompt without human knowledge? This is the LERO-MP line of research.

### 4.1 LERO-MP v3 — 2026-04-21

- **Setup:** Outer meta-LLM mutates `guidance_*` slots; inner LERO runs S3b-local-style.
- **Result:** Peak M1=0.40 at 5M; final M1=0.088 at 10M (collapsed). Easy task (cr=0.35).
- **Learning:** Empty starting slots + 200k inner-eval budget = noise. Meta-LLM kept editing `guidance_reward` even with `evolve_reward=false`.

### 4.2 LERO-MP v4 / B' — 2026-04-26

- **Setup:** Bootstrap meta-LLM call. 3 outer × 3 strategies (parallel). Hard task. Inner eval = 1M.
- **Result:** M1=0.238 mean / 3 seeds. Inner stuck at flat_zero.
- **Learning:** Architectural complexity doesn't recover from `evolve_reward=true`'s zero signal.

### 4.3 LERO v5 — 2026-04-28

- **Setup:** Textual-gradient meta-loop with best+worst feedback, tried-failed registry, stagnation detection, pivot prompts.
- **Result:** M1=0.010.
- **Learning:** Architecture sound; the reward-evolution objective is the bottleneck.

### 4.4 LERO v6 — 2026-04-29 → 04-30 (4 runs)

- **Setup:** Anti-cheat. Forbidden tokens: `hold_signal`, `settle_signal`, `rendezvous_pressure`, `t_close_mean`, `t_dispersion`. Outer picks strategy class (obs-only / reward-only / both). Code-side classification overrides LLM's claims. 2 outer × 3 inner × 3 candidates × 1M = 72 candidates total.
- **Result:** M1=0 across ALL 72 candidates. Best M6=0.153.
- **Learning:** **Operational vocabulary is the irreplaceable artifact.** All 72 candidates produced features but ZERO produced AND-products combining target and agent LiDAR. Meta-LLM said "expose asymmetry" but inner LLM never wrote `(target_near) * (agent_near)`. The hand-curated prompt's "what you CAN infer" bullets are the missing link — without them, LLMs lack hooks to bridge strategy → code.

### 4.5 LERO v7 — 2026-04-30

- **Setup:** Cold-start meta-LLM authors 4 strategies (e.g., "paired sector split"), picks one. Outer refines via strategy-drift detection. Hard task.
- **Result:** M1=0.000 across outer-0 candidates. feature_stack_score=5/5 (full structural match) on 9/9 candidates — but still no learning.
- **Learning:** v7 closed the structural gap (cross-source ops + directional encoding + role one-hot present 100%) but its features were all **cover-zone-gated decision features** (`commit`, `stay_put`, `pair_pressure`). These are zero outside the cover zone — agents rarely visit it early in training, so no gradient signal. S3b-local in contrast uses **dense signals** (`t_close_mean`, `t_dispersion`, `boundary_dist`) that work everywhere.

---

## 5. v8 — density-first prompting (2026-05-01)

### 5.1 Design

After diagnosing v7's gated-feature problem, v8 enforced:
- Hard cap on returned features (15)
- Cap on cover-zone-gated features (≤2)
- Density-first: prefer mean / std / signed-asymmetry / boundary distance over gated counts
- Meta-LLM authors a 3-4 feature working fewshot (not just 1-feature anchor)

### 5.2 Phase 6 result (Mac, 1 seed, 2 outer × 3 inner × 3 cands × 1M)

- **Best: M1=0.010, M6=0.209, n_feat=10, role_one_hot=False**
- Wall: 2h 42min
- 5 strategies enumerated, 2 tried (`pairwise_split_ring` outer 0; `adaptive_two_in_two_out` outer 1)
- Both outers labeled `rl_too_hard`; meta-LLM never escalated to `switch`

### 5.3 Postmortem comparison (v8 vs S3b-local)

Documented in `v8_vs_s3blocal_prompt_comparison.md`. Key findings:
- v8 prompt is 50% larger but missing the "What you CAN infer" block
- v8 worked example is 1-feature trivial (vs S3b-local's 8-feature with role one-hot)
- **role_one_hot rate: 0% (v8) vs 97% (S3b-local)** — the smoking gun
- v8's mandatory bundle prompts list 5 dense signals but never mention role one-hot
- Predicted fix: explicitly mandate role_one_hot + cross_source as `mandatory_features`

### 5.4 Learning: prompt structure is the M1 lever

v8 confirmed the v7 finding: structural fidelity ≠ M1 climb. The auto-meta-prompt loses the role-differentiation bullet that S3b-local teaches via its hand-curated worked example.

---

## 6. v9 — task domain + CoT + memory (2026-05-01 → 05-02)

### 6.1 Design

5 design changes from v8:
1. **`task_domain.yaml`** — externalized task framing (rendezvous semantics, k=2, coordination challenges, 7 inferable_concepts each with idiom, mandatory_features, forbidden_tokens, feature_budget). Swap file = swap task.
2. **`v3_modular_taskdomain` base prompt** — drops `guidance_observation/shared/reward` slots, adds `inferable_hints` (meta-authored "What you CAN infer"-block) and beefed-up multi-example `examples`.
3. **Bundle CoT** — each strategy ships `chain_of_thought = {why_it_works, what_is_needed, failure_modes}` authored before any prompt text.
4. **Combined diagnose+reflect+decide** in one LLM call (saves ~25% LLM cost).
5. **`MemoryStore`** — append-only JSONL of predicted-vs-actual + post-hoc reflections; last N=3 rows injected into next reflection prompt.

### 6.2 Locked decisions (recorded in `v9_plan.md` §9)

- `mandatory_features` enforcement: **soft hint** (escalate later if needed)
- Inner stagnation pivot: **dropped** (outer reflection covers same signal)
- Diagnose+reflect: **collapsed** (Option B — single LLM call)
- Memory lookback: **N=3** (full history persists for post-hoc)
- Bundle coverage: **serial all strategies tried**, max_outer=bundle_size

### 6.3 Phase 4 prompt-lab convergence (LLM-only)

5 trials × 3 inner cands. **Result:**
- Inner role_one_hot rate: **100%** (vs S3b-local 97%, v8 0%)
- Inferable_hints concept coverage: 7/7 every trial
- Examples: 3 fenced python blocks every trial
- Cross-source rate: 100%

Cost: $3, 184s. Phase 4 PASSED.

### 6.4 Phase 6 full RL (Mac, 1 seed, 5 outer × 3 inner × 3 cands × 1M)

- **Best: M1=0.010, M6=0.169, role_one_hot rate = 82%** across 45 production candidates
- Wall: 6h 42min
- 5 strategies enumerated; 2 tried (`pair_and_split` 3×, `leader_follower_pairing` 2×); 3 untried
- Mid-run: added regression-detection fail-safe (`detect_pathological_refine`) after seeing meta-LLM stuck on `pair_and_split` despite 3 consecutive regressions. Forced switch saved outer 3.

### 6.5 v9 CoT review

Documented in `v9_phase6_cot_full.md` (full CoT for all 5 strategies + 5 reflections). Critical evaluation in chat:

**Strengths:**
- Bundle authoring genuinely thoughtful (5 distinct mechanism-level strategies, calibrated scores, specific failure modes)
- Diagnostic narrative coherent
- Memory_recall quotes specific predicted-vs-actual numbers

**Weaknesses:**
- **Strategic helplessness** — LLM verbalized "stuck on this strategy" 3× yet kept picking `refine_current` because action map only allows `switch_to_next` on `rl_too_hard` label
- **Verbatim repetition** across outers — confirmation bias
- **No falsification** — predicted M1=0.18, observed 0.01 across 3 outers; success_signature treated as immutable
- **CoT-vs-slot decoupling** — rich diagnostic CoT, but `slot_edits` were prose-only paragraphs that stripped python examples and concept lists
- **No concrete fix proposed** — always "sharpen the slot text", never "use exp(-α·d) instead of mean"

---

## 7. v9.1 — runtime caps + slot-edit validator + falsification gate (2026-05-03)

### 7.1 Design (per `v9_1_plan.md`)

Six patches motivated by the v9 CoT review and production-data simulation:

| § | name | priority | what it does |
|---|---|---|---|
| 2.1 | `mandatory_features` runtime check | HIGH | AST-reject candidates missing role_one_hot or cross_source BEFORE training (saves ~9 min/cand) |
| 2.2 | `feature_budget` hard cap | MEDIUM | AST-reject candidates with n_features > 20 (soft on AST=0) |
| 2.3 | slot-edit structural validator | HIGH | Reject `slot_edits` that drop python examples or concept lists; keep previous slot text |
| 2.4 | drop redundant `task_context.txt` | MEDIUM | Cuts ~50% verbosity (task_framing duplicated in system + task_context) |
| 2.6 | stronger memory wording | LOW | Add 3 explicit rules to reflect prompt: falsification rule, slot-edit conservation, concrete-fix rule |
| 2.7 | falsification-gate action override | HIGH | If LLM picks `refine_current` AND ≥2 attempts on same strategy all show M1<0.5×expected_M1, force `switch_to_next` |
| 2.10 | lazy artifact authoring on switch | HIGH (v9.1 v0 → v9.1.10) | Bundle enum only authored artifacts for chosen strategy; switching to others gave empty slot files. Lazy-author on demand. |
| 2.11 | add `soft_proximity` to inferable_concepts | HIGH (v9.1 v0 → v9.1.10) | S3b-local high performers use `exp(-α·d)`; v9.1 v0's M1=0.010 winner missed it |

### 7.2 v9.1 Phase 6 production results (v0 — without §2.10/§2.11)

- **Best: M1=0.010, M6=0.194, role_one_hot rate 100% across trained cands**
- Wall: 3h 6min (vs v9's 6h 42m — pre-eval saved ~3h)
- **Only 6/45 candidates actually trained** (outers 0+1). All outers 2-4 had 0 trained candidates due to the empty-artifacts bug.

§2.3 fired in production (rejected the same prose-only `slot_edits` that broke v9). §2.7 didn't need to fire — meta-LLM with §2.6 stronger memory wording picked `switch_to_next` directly.

### 7.3 Postmortem — found the empty-artifacts bug

`enumerate_bundle_v9` parses `chosen_strategy_artifacts` and writes to `bundle.strategies[chosen_idx].artifacts` ONLY. The other 4 strategies have default empty `V9Artifacts()`. When `switch_to_next` fires, `_strategy_to_slots(new_active)` writes empty strings to slot files. Inner LLM gets empty `inferable_hints` and empty `examples` → no template → produces bloated/role-less candidates → all rejected pre-eval.

### 7.4 v9.1.10 — fix the bug + add soft_proximity

- §2.10: at switch_to_next, if new strategy artifacts are empty, fire one extra LLM call (`author_artifacts_for_strategy`) to author them
- §2.11: 8th inferable_concept = `Soft proximity score: torch.exp(-3.0 * lidar_targets.min(...))`

### 7.5 v9.1.10 Phase 6 production results

- **Best: M1=0.010, M6=0.244 (highest in any v9.x run), role_one_hot 100%**
- Wall: 6h 8min (slower than v9.1 v0 because all 45 candidates actually trained)
- **All 5 strategies trained**: `role_split_with_pairing` (2 attempts), `pair_completion_pressure`, `soft_commit_and_release`, `stay_search_switching`. The 5th (`sectorized_search_with_identity`) was lazy-authored at run end but loop terminated.
- §2.10 fired 3× in production: 1672-1885B inferable_hints + 3886-4980B examples per switch.
- 2 of 5 strategies produced M1=0.010 (vs v9.1 v0's 1 of 5).

---

## 8. v9.1.10 best candidate vs S3b-local best — feature alignment

Side-by-side comparison of the M1=0.010 winner (outer 4 iter 1) vs S3b-local s1 iter 0 cand 0 (M1=0.070).

### 8.1 Identical features (6 of each)

| concept | S3b-local | v9.1.10 |
|---|---|---|
| nearest target distance | `min_t_dist` | `d_t` |
| nearest agent distance | `min_a_dist` | `d_a` |
| count close targets | `n_close_targets` (< 0.25) | `t_close` (< cover_r) |
| count close agents | `n_close_agents` | `a_close` |
| speed | `speed = norm(agent_vel)` | `speed = agent_vel.norm` |
| **soft proximity** | `t_prox = exp(-3·min_t_dist)` ✓ | `soft_t = exp(-3·d_t)` ✓ |
| role one-hot (4 dims) | `one_hot[:, agent_idx]=1.0` | `role[:, agent_idx]=1.0` |

§2.11's payoff: soft_prox now appears in v9.1.10 production candidates.

### 8.2 Conceptually equivalent but different form

| concept | S3b-local | v9.1.10 | comment |
|---|---|---|---|
| crowd signal | `clamp(n_close_a / 3, 0, 1)` (saturated) | `density_gap = t_close - a_close` (signed) | different decision logic |
| direction info | 4 raw dims (`t_dir_x/y`, `a_dir_x/y`) | 1 derived (`cos(t_ang - a_ang)`) | 4 raw vs 1 composite |

### 8.3 S3b-local-only

- 2 raw target direction dims (`cos`, `sin`)
- 2 raw agent direction dims
- 2 raw velocity direction dims (`cos(vel_dir)`, `sin(vel_dir)`)

### 8.4 v9.1.10-only (engineered composites)

- `target_dir_align = cos(t_ang - a_ang)`
- `rendezvous_score = soft_t * (1 + relu(density_gap))`
- `avoid_overlap = sigmoid(2.5*(d_a - d_t))`
- `safe_commit = sigmoid(3*(0.3-d_t)) * sigmoid(3*wall)`
- `soft_t * (1 + a_close)`, `wall * (1 + speed)` — interactions
- `wall = 1 - pos.abs().max(-1)` — boundary distance

### 8.5 The hypothesis for the M1 gap

S3b-local has 6 raw directional dims; v9.1.10 substitutes 6 engineered composite gates. **PPO at 1M frames may prefer raw low-level features (which the policy can recombine in many ways) over pre-engineered gates that lock in a single decision rule.** v9.1.10's composites encode strategy-specific decisions but reduce the representational space PPO can search.

---

## 9. Cumulative learnings

### 9.1 Validated (replicated and not contradicted)

1. **k≥2 reward design is unstable.** LLM authors individually-rational rewards that are collectively-irrational. Always `evolve_reward=false`.
2. **Observation engineering > incentive design** for k≥2 multi-agent coordination.
3. **The S3b-local hand-curated prompt** is the empirical gold standard at M1=0.845 mean / 3 seeds at 1M frames.
4. **Operational vocabulary is the irreplaceable artifact.** Meta-LLMs can produce strategically sound "why_it_works" reasoning but consistently fail to operationalize it into the right Python expressions without explicit operational scaffolding (worked examples, idiom hints).
5. **Role one-hot (`F.one_hot(agent_idx, n_agents)` or equivalent) is mandatory** for shared-policy MAPPO with k≥2. Without it, all agents present identical features and cannot break symmetry. S3b-local 97% rate, v8 0%, v9 82%, v9.1.10 100%.
6. **Soft proximity (`exp(-α·d)`)** is the second most important S3b-local feature. v9.1's §2.11 added it to inferable_concepts and v9.1.10 candidates produced it.
7. **Cover-zone-gated features starve PPO at low M1.** Mean / std / signed-asymmetry / boundary distance give gradient signal everywhere; gated features are 0 outside cover zone where untrained agents spend most of their time.
8. **Slot-edit drift across refines** is the meta-LLM's typical failure mode. Without a structural validator (v9.1 §2.3), `refine_current` actions strip python examples and concept lists into prose.
9. **Bundle artifacts must be authored for ALL strategies, not just the chosen one.** v9.1's §2.10 lazy authoring fixes this; v9.1.10 confirmed.

### 9.2 Open / falsifiable

- Does PPO at 1M hit a hard ceiling at M1≈0.01 for v9.x's structurally-correct candidates? Test: deep-train v9.1.10's outer 4 winner at 10M.
- Is the S3b-local 0.07 max at 1M reproducible single-seed for v9.x? Test: 3-seed v9.1.10 sweep.
- Do raw directional features (cos/sin of vel_dir, target_dir, agent_dir) matter more than v9.1.10's engineered composites? Test: add 6 raw directional dims to mandatory_features, re-run.
- Does the v9.1.10 prompt structure benefit from further verbosity reduction? Currently ~2× S3b-local size after §2.4 dedup.

---

## 10. v9.1.10 patch tally

47 v9 unit/replay tests + 30 v9.1 tests = **77/77 passing.** Production replay tests confirm each patch catches its target failure in actual v9 / v9.1 v0 production data:

| patch | catches in v9.1 v0 production |
|---|---|
| §2.3 slot-edit validator | rejected outers 1-4 prose-only `slot_edits`, kept good initial slots |
| §2.7 falsification gate | not needed in v9.1.10 — §2.6 stronger memory wording made LLM pick switch_to_next directly |
| §2.1 mandatory_features pre-eval | rejected dozens of role-less candidates |
| §2.2 feature_budget hard cap | rejected dozens of n_features>20 candidates |
| §2.4 task_context dedup | prompt 50% smaller, no quality regression |
| §2.10 lazy artifact authoring | fired 3× in v9.1.10 production, enabled all 5 strategies to actually train |
| §2.11 soft_proximity | appeared in 100% of trained v9.1.10 candidates |

---

## 11. Recommended next experiments (priority order)

### 11.1 Diagnostic: deep-train v9.1.10 outer 4 winner at 10M (cheap, $0)

`stay_search_switching` strategy, M1=0.010, M6=0.244 at 1M. If 10M produces M1>0.5, the structural design is sufficient — task hits a 1M ceiling. If still M1<0.05, candidate is genuinely weak.

Cost: ~3h Mac, 0 LLM. Highly informative.

### 11.2 3-seed v9.1.10 sweep

Single-seed M1=0.010 could be variance — S3b-local s2's worst-seed max was 0.030. 3 seeds at 1M tells us if v9.1.10's M1 distribution sometimes crosses 0.05.

Cost: ~18h Mac (or ~3h × 3 OVH GPU jobs in parallel), $0 LLM (uses cached prompts from outer 0).

### 11.3 Add raw directional features to mandatory_features

S3b-local has 6 raw cos/sin dims; v9.1.10 substitutes 6 composites. Test if PPO learns better from raw + composite vs composite-only.

Cost: ~$1 prompt-lab + ~6h Mac full RL.

### 11.4 Compare v9.1.10 vs S3b-local at 5M and 10M

Currently we only compare at 1M (where S3b-local is 0.07 mean, v9.1.10 is 0.010). At 10M S3b-local is 0.93. v9.1.10 may close the gap with more frames.

Cost: 3h × 2 runs × 5M = 30h, or single 10M deep-train (§11.1).

---

## 12. Document map

Every doc in `rendezvous_comm/docs/` and what it covers:

| doc | covers |
|---|---|
| `experiment_storyline.md` | ER1/ER2/ER3 + LERO foundation through v6 |
| `all_experiments_analysis.md` | every experiment's M1/M6 with seeds + commit hashes |
| `report.md` | early ER1/ER2/ER3 detailed report |
| `lero.md` | canonical LERO design + results doc |
| `prompt_evolution_analysis.md` | v5/v6 vs S3b-local prompt audit |
| `v6_plan.md` | v6 anti-cheat design |
| `v7_vs_s3blocal_comparison.md` | v7 feature-by-feature vs S3b-local |
| `v8_plan.md` | v8 density-first design |
| `v8_vs_s3blocal_prompt_comparison.md` | v8 prompt vs S3b-local prompt diff |
| `v9_plan.md` | v9 task_domain + CoT + memory design |
| `v9_phase6_cot_full.md` | full meta-LLM CoT from v9 production run |
| `v9_1_plan.md` | v9.1 runtime caps + validator design |
| `v9_1_postmortem.md` | v9.1 v0 results + empty-artifacts bug discovery |
| `v9_1_10_lineage.md` | v9.1.10 outer-by-outer prompt + obs evolution + S3b-local comparison |
| `v9_1_10_vs_s3b_feature_align.md` | feature-by-feature alignment table |
| `PROJECT_HISTORY.md` | this doc — complete chronological summary |

## 13. Appendix — material from earlier docs not yet integrated above

These are facts and decisions that earlier docs (v1-v3 LERO-MP plans, comm design, GNN theory, spatial physics) recorded that the body of this history skipped. Captured here for future readers who need the WHY behind the v9-era choices.

### A. Spatial physics (the constants behind cr/ms sensitivity)

Source: `spatial_scales_physics.md`. Why ER1's M1 jumped 0.04 → 0.405 just by fixing the `max_steps` bug:

- Arena diagonal = 2√2 ≈ 2.83. LiDAR range 0.35 = **12.4% of world diagonal** — agents are functionally blind beyond a small neighborhood.
- Steady-state velocity = 0.2 units/step at max force (Euler integration, drag=0.25 per substep). At ms=200 ≈ one full traverse — zero margin for k=2 syncing. ms=400 = two traversals → coordination becomes feasible.
- Covering zone area at cr=0.25 vs cr=0.35: **0.196 vs 0.385** sq units (~2× wider). The "easy" relaxation widens the synchronization window 2× spatially.

This explains why every result tagged `cr=0.25 ms=400` is the "hard task" — it's the actual research target. `cr=0.35 ms=200` numbers (ER2 communication ablations) are not directly comparable.

### B. GNN spatial-coordination signature (ER3)

Source: `theory_gnn_communication.md` + `comparison_gnn_vs_minimal_team.md`.

- ER3's mechanism: GATv2Conv with full-topology graph (every agent sees every other). Attention learns which neighbor matters when.
- Empirical signature: **M9 (spatial spread) drops from 1.018 (MLP no-comm) to 0.578 (GNN)** — agents physically cluster into pairs. M4 (collisions) RISES (1.3 → 8.9) — the price of tighter coordination.
- GNN training is **3.4× slower** per frame than MLP. ER3's M1=0.71 cost is roughly proportional.
- `from_pos` topology variant (use only nearby agents as neighbors) was implemented but NOT used in headline ER3 — full topology won on this small-team task.

### C. Communication design — the "silent agent" failure mode

Source: `communication_design.md`.

ER2's explicit `dim_c` channel had a known weakness: agents face weak gradient on "output zeros when nothing to say". They tend to either (a) emit constant noise that the policy learns to ignore, or (b) emit useful messages but with no graceful shutdown. ER4 (IC3Net/TarMAC, planned but not run) addresses this with **dedicated gating modules** — the agent learns *when* to communicate explicitly. Status: ER4 is in `er3_er4_plan.md`; never executed because ER3 GNN reached M1=0.71 and the LERO line opened up.

### D. LLM integration — alternatives considered, deferred

Source: `llm_integration_approaches.md`. Why LERO (LLM as observation/reward author) was chosen over:

- **Hierarchical planner** (LLM as high-level assignment policy): would directly address k=2 assignment but the inference latency was prohibitive (per-step LLM call). Deferred.
- **Online coordinator** (per-step or event-triggered LLM queries): same latency problem.
- **In-context PPO** (LLM-derived advantages): no clean separation between training-time signal and policy gradient.

LERO won on a single property: **the LLM's output is read-only at policy run time.** Observations and reward functions are generated once (training-time only), so inference latency = 0 during RL training. This is also why `evolve_reward=true` is unstable but `evolve_observation=true` is robust — observations are inputs, can't be gamed; rewards are objectives, will be gamed.

### E. LERO-MP v1 noise floor — why early meta-prompt experiments failed

Source: `lero_metaprompt_analysis.md` §2.

- **Run-to-run variance at 1M:** identical template, same seed, M1 ranged 0.005 → 0.035 (**7× spread**). Meta-prompting signal got lost in inner-loop RL stochasticity.
- **Baseline > mutations:** v1 baseline mean M1 = 0.024 vs mutated mean = 0.020. Net regression.
- **40% dead tokens:** v1 LLM emitted `compute_reward` definitions even when `evolve_reward=false`, wasting context budget.
- This finding (not the v9 CoT review) was the FIRST evidence that the meta-LLM has more agency than utility at 1M-frame budgets.

### F. LERO-MP v2 — strategist + editor split

Source: `lero_metaprompt_v2_plan.md` + `lero_metaprompt_analysis.md` §3.

- v2 introduced a **two-level meta architecture**: strategist (decides what to evolve) + editor (writes the actual prompt edits).
- Slot decomposition: `guidance` → `{guidance_shared, guidance_reward, guidance_observation}`. Editor edits each slot independently.
- **Per-seed bias enforcement:** seed 0 → obs, seed 1 → reward, seed 2 → exploratory. Rigid; never picks "both" automatically. This was a deliberate de-noising choice given v1's variance.
- v2 editor produced higher-quality observation code (5 of 6 functions used ≥2 coordination patterns) but the rigid bias meant only 1 of 3 seeds explored the right axis.

### G. LERO-MP v3 — inner hardening

Source: `lero_metaprompt_v3_plan.md` + `lero_metaprompt_v3_implementation.md`.

Five concrete improvements that v9 inherits:

1. **Inner-LLM retry loop:** 3 attempts with compiler-error feedback. Valid-candidate rate jumped ~70% → 95%.
2. **Pydantic structured outputs** for both meta and inner. Replaces fragile regex parsing; enables `temperature=1.0` safely.
3. **TextGrad-style editor critic:** propose → critique → revise loop (≤2 revisions, +€0.01/mutation).
4. **Conditional output_spec:** drop `compute_reward` signature when `evolve_reward=false`. ~30% latency saving.
5. **Behavioral signals tiers:** Tier 1 = M1-M9 metrics, Tier 2 = CSV curve-shape tags, Tier 3 = fingerprint. Strategist picks which tiers via `include_signals` field.

51 new v3 unit tests shipped; v2 legacy tests still passed (no regression). The retry loop and structured outputs persisted into v9; the editor critic was simplified into v9's `reflect_decide_v9` (Option B from §6.3 of v9_plan).

### H. LERO-MP v5 contingent decision tree

Source: `lero_metaprompt_v5_plan.md`. The stopping rule we explicitly committed to but never invoked because v4/v5/v6 all failed before reaching the threshold:

- If v4 peak M1 ≥ 0.70 at 10M → ship v4, migrate to DSPy `MIPROv2.compile()` over a learned trainset.
- If v4 peak M1 ∈ [0.40, 0.70] → add the **S3b-local code example to bootstrap** (the strongest single lever per the v5 author, ~€0.01 LLM cost). Re-evaluate.
- If v4 peak M1 < 0.40 → apply combined ideas (S3b example + retry hardening + behavioral signals). This is the path that led to v6 / v7 / v8 / v9.

The "S3b example in bootstrap" idea is essentially what v9's `task_domain.yaml`'s `inferable_concepts` + `examples` slot does — it's been delivered, just not labeled as such.

### I. Apples-to-apples comparison disclaimer

Source: `lero_metaprompt_analysis.md` §2.1-2.3.

- ER1/ER2/ER3 baselines are reported on `cr=0.35, ms=200` (the easier task) in many original docs. The numbers I'm using in §1 are **post-fix `cr=0.25, ms=400` rebenchmarks** when available; otherwise the baseline number with caveat.
- S3b-local rebenchmarked at the harder task: M1=0.060 best single-seed at 1M. Mean 3-seed at 1M = 0.022. 10M peak = 0.93.
- **The "M1=0.71 ER3 vs M1=0.88 S3b-local" comparison** in §1 is therefore comparing different evaluation horizons (1M vs 10M-after-deep-train). The cleanest 1M-eval comparison is: **S3b-local 0.022 mean / 0.060 max vs v9.1.10 0.010 max** — the gap is real but smaller than the 0.71 → 0.88 framing suggests.

### J. Cost tally (cumulative, approximate)

Pulled from session logs and per-run manifests.

- **LLM API spend (rendezvous_comm only):** ~$80 cumulative across LERO + LERO-MP v1 → v9.1.10. Largest single bucket: v9 + v9.1 prompt-lab convergence sweeps + production runs (~$30).
- **Mac compute time (rendezvous_comm only):** ~80 hours cumulative (excluding the v6 4-run series which alone cost ~10.5h).
- **OVH GPU hours:** ~12 hours (mostly ER experiments). LERO-era runs were Mac-only.
- **Test suite:** v9 + v9.1 = 77 unit + replay tests, all passing as of 2026-05-04.

These are operational facts — listed here so a future reader budgeting a follow-up can size it appropriately.

## 14. TL;DR

We started in March 2026 with hand-crafted no-comm baselines (ER1: M1=0.405) and worked through engineered communication (ER2: 0.53), GNN message-passing (ER3: 0.71), to LERO with reward evolution (failed: 0-0.105 with reward-hacking). The breakthrough was **S3b-local: M1=0.845 mean / 3 seeds at 1M with `evolve_reward=false`** — a hand-curated observation-only prompt that the LLM uses to design coordination features.

Since then we've tried five generations of meta-search (v5-v9.1.10) attempting to remove the human from the loop. Each generation closed structural gaps to S3b-local (cross-source ops, role one-hot, soft proximity, density-first, CoT, memory, slot-edit validation, lazy artifact authoring, falsification gate) while M1 plateaued at 0.010 single-seed. v9.1.10 produces structurally equivalent candidates to S3b-local but trains 6/16 of S3b-local's candidate quality.

**The remaining gap is either (a) single-seed variance vs S3b-local's 3-seed mean, (b) pre-engineered composites vs raw directional dims, or (c) a genuine 1M-frame ceiling. The next decisive experiment is the 10M deep-train of v9.1.10's best candidate (§11.1) — cheap and falsifiable.**
