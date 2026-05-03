# v9 Phase 6 — full meta-LLM chain-of-thought (strategy + per-outer reflections)

**Date:** 2026-05-03  
**Source:** `results/lero_v9/lero_v9_rendezvous_k2_2x3/20260502_1912_s0/`  
**Run:** seed=0, max_outer=5, bundle_size=5, eval_frames=1M, total wall 6h 42min  
**Model:** gpt-5.4-mini, temperature=0.8

All text below is verbatim from the meta-LLM — `_bundle_init.json`, `outer_*/_decision.json`, and `_meta_memory.jsonl`. No edits.

---

## Part 1 — Initial bundle: 5 strategies authored at outer-0 cold-start

**task_understanding:** This is a cooperative multi-agent rendezvous task with k=2 coverage: agents must form pairs at targets so that exactly two agents are within range simultaneously, causing the target to disappear. Because each agent only sees local LiDAR plus its own state, success depends on implicit coordination, role differentiation under shared policy, and using local crowding cues to avoid overcommitting to the same target.

**chosen_idx:** 0 → `pair_and_split`

### Strategy #0: `pair_and_split` ← CHOSEN

- **lero_codability:** 9
- **rl_trainability:** 8
- **combined_score:** 8.5

**full_solution:**

> Agents infer local target opportunity and local agent congestion, then split into complementary roles: some seek the nearest promising target while others avoid crowded targets and search alternatives. The policy uses identity-conditioned behavior so different agents can specialize without explicit communication, and it uses combined target/agent summary features to decide whether to commit or divert.

**chain_of_thought.why_it_works:**

> Reward comes from exactly two agents simultaneously covering a target, so agents need to sense both target proximity and whether another agent is already near the same area. Local target and agent LiDAR summaries can approximate 'solo vs paired' situations, while agent identity helps the shared policy break symmetry and distribute roles.

**chain_of_thought.what_is_needed:**

- Role differentiation under shared parameters so agents do not all choose the same target
- At least one feature combining target-derived and agent-derived local summaries
- A stable commit heuristic based on distance plus local crowding

**chain_of_thought.failure_modes:**

- If all agents have identical inputs, PPO may converge to collapse-on-nearest-target behavior
- Without a cross-source signal, the policy can see targets and agents but not whether a target is jointly feasible
- If the commit heuristic is too rigid, agents may loiter or miss newly available targets after others leave

**success_signature:**

- ast_pattern_description: Compute scalar target/agent proximity summaries, concatenate with one-hot agent identity, then derive a joint congestion or opportunity scalar before the action head.
- expected_M1_at_1M: 0.18
- expected_M6_at_1M_min: 0.42

---

### Strategy #1: `leader_follower_pairing`

- **lero_codability:** 8
- **rl_trainability:** 7
- **combined_score:** 7.5

**full_solution:**

> A subset of agents behave as explorers that move toward promising targets, while others act as followers that bias toward nearby agent clusters to complete pairs. The shared policy uses agent index to induce consistent but distinct roles across episodes, allowing the team to form two-agent rendezvous units without explicit assignment.

**chain_of_thought.why_it_works:**

> The task rewards coordinated pairing rather than solo occupation, so a leader-follower structure naturally creates one agent that initiates and another that completes coverage. Local agent LiDAR can indicate whether a partner is nearby, and agent identity can stabilize which agents tend to lead or follow.

**chain_of_thought.what_is_needed:**

- Persistent role cues from agent identity
- A feature that measures local agent density to detect possible partners
- A mechanism for switching from exploration to pairing when a partner is detected

**chain_of_thought.failure_modes:**

- Role assignment may drift across episodes if identity cues are weak
- Followers may overattach to crowded regions and waste time
- Shared-policy learning may not discover clean leader/follower specialization without enough training diversity

**success_signature:**

- ast_pattern_description: Branch on identity-conditioned role features; leaders use nearest-target features, followers use nearest-agent or crowding features to finalize coverage.
- expected_M1_at_1M: 0.14
- expected_M6_at_1M_min: 0.35

---

### Strategy #2: `crowd_aware_avoidance`

- **lero_codability:** 7
- **rl_trainability:** 6
- **combined_score:** 6.5

**full_solution:**

> Agents treat nearby agent density as a negative signal and deliberately avoid the most crowded target rays, preferring less-contested targets even if they are slightly farther away. This reduces collapse and increases the chance that two agents arrive together but not all agents pile onto one location.

**chain_of_thought.why_it_works:**

> Exactly-k coverage is harmed by oversubscription, so local crowding is a useful proxy for wasted effort. If an agent can estimate that a target direction is already crowded, it can redirect to another target, improving team-wide coverage efficiency.

**chain_of_thought.what_is_needed:**

- Reliable local agent density features
- A target-choice rule that penalizes crowded directions
- Some identity or stochasticity so multiple agents do not all make the same avoidance choice

**chain_of_thought.failure_modes:**

- Pure avoidance can lead to oscillation if all agents flee the same crowding cue
- Without target-side features, agents may avoid crowds but still fail to coordinate on a specific target
- If reward sparsity is high, the policy may learn to wander without committing

**success_signature:**

- ast_pattern_description: Form crowding counts from lidar_agents and subtract them from target proximity scores before selecting an action.
- expected_M1_at_1M: 0.08
- expected_M6_at_1M_min: 0.22

---

### Strategy #3: `opportunistic_commit`

- **lero_codability:** 8
- **rl_trainability:** 7
- **combined_score:** 7.5

**full_solution:**

> Agents move until they detect a locally promising target that is not overly crowded, then commit by reducing motion and staying within coverage range long enough for a partner to arrive. The policy uses velocity and boundary cues to avoid useless drift, making commitment more reliable in the no-respawn setting.

**chain_of_thought.why_it_works:**

> Coverage requires temporal overlap, so an agent must not only approach but also remain near the target once it is likely to be completed. Self-motion and boundary distance help regulate whether to keep exploring or hold position near a viable target area.

**chain_of_thought.what_is_needed:**

- A state-dependent switch between exploration and low-velocity commitment
- Target proximity plus local crowding to identify feasible rendezvous
- Boundary awareness to prevent getting stuck on arena edges

**chain_of_thought.failure_modes:**

- If commitment triggers too early, agents can freeze at the wrong target
- If commitment triggers too late, pairs miss the overlap window
- Boundary effects may dominate when agents learn to exploit edges instead of targets

**success_signature:**

- ast_pattern_description: Use nearest-target distance, local agent crowding, velocity magnitude, and boundary distance to gate between moving and holding behaviors.
- expected_M1_at_1M: 0.12
- expected_M6_at_1M_min: 0.3

---

### Strategy #4: `sector_specialization`

- **lero_codability:** 9
- **rl_trainability:** 6
- **combined_score:** 7.5

**full_solution:**

> Agents implicitly specialize by angular sectors of the arena, with different agent identities biasing them toward different target regions. This creates a decentralized partition of labor that lowers collision probability and improves the odds that pairs form locally rather than all agents converging globally.

**chain_of_thought.why_it_works:**

> In a symmetric arena, shared-policy agents need a symmetry-breaking prior to avoid all selecting the same target. Identity-conditioned sector preference provides a simple partition that can persist across episodes and still adapt based on local LiDAR cues.

**chain_of_thought.what_is_needed:**

- Agent identity features
- A way to map local rays into coarse directional preferences
- Enough exploration for the policy to discover useful sector partitions

**chain_of_thought.failure_modes:**

- If sectors are too rigid, agents may ignore better nearby opportunities
- Symmetry may still collapse if identities are not used effectively
- Sector specialization can underperform when target layouts are highly clustered

**success_signature:**

- ast_pattern_description: Encode ray-index directional summaries and modulate them by agent one-hot to bias each agent toward a distinct angular sector.
- expected_M1_at_1M: 0.1
- expected_M6_at_1M_min: 0.28

---

## Part 2 — Per-outer reflection chain-of-thought (5 outers)

Each outer: the meta-LLM receives the analyzer facts (`_facts.json`), the bundle state, and the last N=3 memory rows. It outputs a single JSON with `memory_recall`, `current_outcome_reading.{label, diff_vs_predicted}`, `reflection_chain_of_thought.{what_went_right, what_went_wrong, remaining_uncertainty}`, `next_action`, and `slot_edits`.

### Outer 0 — strategy `pair_and_split`

**Analyzer facts (from inner result):**

- M1: 0.010
- M6: 0.169
- n_features: 14
- n_gated: 0, n_dense: 14
- touches_both_lidars: True
- role_one_hot_present: True

**Diagnosis label:** `translation_failure`  
**Next action:** `refine_current`

**memory_recall:**

> No prior outers were recorded, so there is no earlier prediction to compare against. This outer shows a first inner result with strong structural coverage, but performance is far below the expected threshold.

**current_outcome_reading.diff_vs_predicted:**

> The pattern is present: both lidars are touched, role one-hot is present, and there are multiple cross-source ops. However, M1 and M6 are far below expected, which suggests the current slot text is too abstract and not translating the intended pairing/splitting logic into learnable behavior.

**reflection_chain_of_thought.what_went_right:**

- The candidate did include the mandatory structural ingredients: cross-source feature mixing, target and agent summaries, and role identity.
- The design intent matches the task: differentiate agents and use crowding plus opportunity to split behavior.

**reflection_chain_of_thought.what_went_wrong:**

- The learned behavior is not strong enough; M1 is near zero and M6 is also well below the minimum target.
- The slot text likely remains too vague about how to compute and use the joint opportunity/crowding signal.
- The strategy may not specify a concrete enough role split or a sufficiently direct commit/divert rule.

**reflection_chain_of_thought.remaining_uncertainty:**

- Whether the main issue is weak feature phrasing versus an insufficiently explicit role assignment mechanism.
- Whether a more concrete description of the joint scalar should emphasize 'nearest target minus local agent congestion' or a similar direct rule.

**rationale:**

> This is not a pattern-absence case because the AST facts confirm the required cross-source and role-conditioning structure. Since the outcome is poor despite the right skeleton, the best move is to rewrite the current strategy more concretely rather than switching bundles.

**slot_edits keys (which slots were rewritten):** ['inferable_hints', 'examples', 'feedback_template']

**bundle_demote / bundle_add:** demote=[] add=[]

---

### Outer 1 — strategy `pair_and_split`

**Analyzer facts (from inner result):**

- M1: 0.000
- M6: 0.133
- n_features: 18
- n_gated: 3, n_dense: 15
- touches_both_lidars: True
- role_one_hot_present: True

**Diagnosis label:** `partial`  
**Next action:** `refine_current`

**memory_recall:**

> Last outer I predicted the design was structurally sound but too vague, and the actual result again showed a severe performance miss: before it was M1=0.01/M6=0.1686, now it is even lower at M1=0.0/M6=0.1329. The recurring pattern is that cross-source and role identity are present, but the learned policy still fails to convert them into useful behavior.

**current_outcome_reading.diff_vs_predicted:**

> I expected at least moderate M6, but the result stayed below the minimum threshold. However, the mandatory pattern is present, so this is not a translation failure.

**reflection_chain_of_thought.what_went_right:**

- The inner candidate includes the required role identity and cross-source mixing, so the abstract pattern was translated into a valid structural form.
- The features do touch both lidar sources, which means the policy can in principle compare target opportunity against local agent congestion.

**reflection_chain_of_thought.what_went_wrong:**

- Despite the right ingredients, the policy did not learn useful coordination: M1 is zero and M6 is far below the target minimum.
- The current slot text likely does not make the joint decision rule concrete enough for PPO to exploit.
- The role split and commit/divert guidance may still be too open-ended, allowing collapse into ineffective behavior.

**reflection_chain_of_thought.remaining_uncertainty:**

- Whether the main weakness is the exact wording of the joint scalar, the role-specific behavior split, or both.
- Whether sharpening the slot text around a direct 'opportunity versus congestion' comparison will be enough to raise learning.

**rationale:**

> The mandatory pattern is present, so this is not a translation failure. But performance is still well below the expected minimum, which fits partial and calls for sharpening the current strategy’s slot text rather than switching away.

**slot_edits keys (which slots were rewritten):** ['inferable_hints', 'examples', 'feedback_template']

**bundle_demote / bundle_add:** demote=[] add=[]

---

### Outer 2 — strategy `pair_and_split`

**Analyzer facts (from inner result):**

- M1: 0.000
- M6: 0.083
- n_features: 29
- n_gated: 2, n_dense: 27
- touches_both_lidars: True
- role_one_hot_present: True

**Diagnosis label:** `partial`  
**Next action:** `refine_current`

**memory_recall:**

> Last two outer attempts predicted the pair_and_split design would at least partially transfer (expected M1≈0.18, M6≈0.42), but actual performance stayed far below target both times; the first run was labeled translation_failure, the second partial, and this run regressed further (M1=0.0, M6=0.0829). The recurring pattern is: the structural ingredients are present, but they are not translating into usable coordination.

**current_outcome_reading.diff_vs_predicted:**

> Predicted M1/M6 were 0.18/0.42, but actual M1/M6 were 0.0/0.0829. The pattern is present and mandatory role identity exists, yet learning remains far below the minimum useful level.

**reflection_chain_of_thought.what_went_right:**

- The analyzer confirms the intended cross-source structure is still there: both target and agent summaries are used, with multiple cross-source operations.
- Role identity is present, so the policy still has a mechanism for specialization under shared parameters.
- The design is not missing the core ingredients; it is at least structurally aligned with the task.

**reflection_chain_of_thought.what_went_wrong:**

- Despite the correct ingredients, the learned policy is not exploiting them: both M1 and M6 are extremely low.
- The slot text is probably still too abstract about how to convert target proximity and agent congestion into a concrete action choice.
- The role split may be insufficiently explicit, so agents can still collapse into ineffective symmetry or indecisive diversion behavior.

**reflection_chain_of_thought.remaining_uncertainty:**

- Whether the main problem is the exact joint scalar formulation or the lack of a sharper role-conditioned rule.
- Whether a more direct 'commit when opportunity outweighs congestion, otherwise divert' wording will be enough to unlock learning.

**rationale:**

> This is not a translation failure because the pattern is present and role one-hot is present. It is not achieved. The outcome is best treated as partial because the structure exists but performance is far below the minimum. Per the action map, partial means refine the current strategy by sharpening the slot text rather than switching.

**slot_edits keys (which slots were rewritten):** ['inferable_hints', 'examples', 'feedback_template']

**bundle_demote / bundle_add:** demote=[] add=[]

---

### Outer 3 — strategy `leader_follower_pairing`

**Analyzer facts (from inner result):**

- M1: 0.000
- M6: 0.089
- n_features: 30
- n_gated: 0, n_dense: 30
- touches_both_lidars: True
- role_one_hot_present: True

**Diagnosis label:** `partial`  
**Next action:** `refine_current`

**memory_recall:**

> I previously predicted that the leader/follower pairing strategy should be learnable because it already had role identity and cross-source mixing, but the outcomes stayed poor across the last three outers. The pattern is recurring: structural ingredients are present, yet performance remains far below target, suggesting this is not a translation failure but a learning difficulty.

**current_outcome_reading.diff_vs_predicted:**

> Predicted the strategy would be at least moderately learnable, but actual M1 is 0.0 and M6 is 0.0886, far below expected thresholds. The structural pattern is present, so this is not absent-pattern failure; it is a weak-learning outcome.

**reflection_chain_of_thought.what_went_right:**

- The analyzer confirms the mandatory ingredients are present: role one-hot, cross-source operations, and features touching both lidars.
- The strategy still matches the intended decomposition of explorer versus follower behavior.
- The feature stack is structurally rich, so the model is not starved of inputs.

**reflection_chain_of_thought.what_went_wrong:**

- Despite the correct structure, the policy is not learning useful coordination; both metrics are near zero.
- The slot text may still be too abstract, leaving the policy without a crisp decision boundary for when to explore versus when to attach to a cluster.
- The current wording likely allows symmetry collapse or indecisive movement instead of a strong leader/follower split.

**reflection_chain_of_thought.remaining_uncertainty:**

- Whether a sharper role-conditioned rule will be enough, or whether this strategy is simply too hard for PPO at the current budget.
- Whether the strongest fix is to define a more explicit leader trigger and follower trigger using a direct opportunity-versus-crowding comparison.

**rationale:**

> This is partial, not translation_failure: the expected structural pattern is present, but learning is far below target. Per the action map, the right move is to refine the current strategy by sharpening the slot text so the role split and the commit/divert rule are more concrete.

**slot_edits keys (which slots were rewritten):** ['inferable_hints', 'examples', 'feedback_template']

**bundle_demote / bundle_add:** demote=[] add=[]

---

### Outer 4 — strategy `leader_follower_pairing`

**Analyzer facts (from inner result):**

- M1: 0.000
- M6: 0.139
- n_features: 22
- n_gated: 0, n_dense: 22
- touches_both_lidars: True
- role_one_hot_present: False

**Diagnosis label:** `translation_failure`  
**Next action:** `refine_current`

**memory_recall:**

> Across the last 3 outers I predicted the strategy was structurally sound but likely under-specified; that matched the repeated partial outcomes. This outer is different: I would have expected another partial, but the analyzer shows a translation failure because the mandatory role one-hot is missing even though cross-source mixing is present.

**current_outcome_reading.diff_vs_predicted:**

> Predicted a continuation of the partial learning failure pattern, but the actual issue is more basic: the pattern is present, cross-source ops exist, yet role one-hot is absent, so the strategy is missing a mandatory structural feature.

**reflection_chain_of_thought.what_went_right:**

- The design still includes cross-source reasoning and touches both lidars, so the high-level idea was translated into the model.
- The repeated memory shows the earlier variants already had role one-hot and still underperformed, which helped isolate that this outer regressed structurally rather than merely failing to learn.

**reflection_chain_of_thought.what_went_wrong:**

- The mandatory role identity cue is missing, so the shared policy cannot reliably separate leader and follower behavior.
- Because the role signal is absent, the agent-specific specialization described by the strategy cannot be expressed cleanly.
- Even with several cross-source operations, the representation is likely too diffuse without an explicit role split.

**reflection_chain_of_thought.remaining_uncertainty:**

- Whether restoring role one-hot alone will recover the intended specialization, or whether the slot text also needs a sharper leader-versus-follower decision rule.
- Whether this strategy should remain active after repair, given that earlier partial runs already showed weak learning.

**rationale:**

> This is a translation failure, not an RL-hardness case: the analyzer says role_one_hot_present is false, which violates a mandatory feature for this strategy. The right move is to rewrite the current slot text more concretely so the role-conditioned split is explicit and cannot be omitted.

**slot_edits keys (which slots were rewritten):** ['inferable_hints', 'examples', 'feedback_template']

**bundle_demote / bundle_add:** demote=[] add=[]

---

## Part 3 — `_meta_memory.jsonl` rows (predicted vs actual per outer)

Append-only history. Last N=3 rows are injected into the next outer's reflection prompt as the `[MEMORY]` block.

### Outer 0 — `pair_and_split`

- **predicted:** M1=0.18, M6=0.42
  - what_is_needed:
    - Role differentiation under shared parameters so agents do not all choose the same target
    - At least one feature combining target-derived and agent-derived local summaries
    - A stable commit heuristic based on distance plus local crowding
- **actual:** M1=0.01, M6=0.169, label=translation_failure, n_features=14, role_one_hot_present=True
- **delta:** M1=-0.16999999999999998, M6=-0.25142857142857133
- **strategy chain_of_thought (snapshot at this outer):**
  - why_it_works: Reward comes from exactly two agents simultaneously covering a target, so agents need to sense both target proximity and whether another agent is already near the same area. Local target and agent LiDAR summaries can approximate 'solo vs paired' situations, while agent identity helps the shared policy break symmetry and distribute roles.
- **post_hoc_reflection:**
  - what_went_right:
    - The candidate did include the mandatory structural ingredients: cross-source feature mixing, target and agent summaries, and role identity.
    - The design intent matches the task: differentiate agents and use crowding plus opportunity to split behavior.
  - what_went_wrong:
    - The learned behavior is not strong enough; M1 is near zero and M6 is also well below the minimum target.
    - The slot text likely remains too vague about how to compute and use the joint opportunity/crowding signal.
    - The strategy may not specify a concrete enough role split or a sufficiently direct commit/divert rule.
  - remaining_uncertainty:
    - Whether the main issue is weak feature phrasing versus an insufficiently explicit role assignment mechanism.
    - Whether a more concrete description of the joint scalar should emphasize 'nearest target minus local agent congestion' or a similar direct rule.

### Outer 1 — `pair_and_split`

- **predicted:** M1=0.18, M6=0.42
  - what_is_needed:
    - Role differentiation under shared parameters so agents do not all choose the same target
    - At least one feature combining target-derived and agent-derived local summaries
    - A stable commit heuristic based on distance plus local crowding
- **actual:** M1=0.0, M6=0.133, label=partial, n_features=18, role_one_hot_present=True
- **delta:** M1=-0.18, M6=-0.2871428571428571
- **strategy chain_of_thought (snapshot at this outer):**
  - why_it_works: Reward comes from exactly two agents simultaneously covering a target, so agents need to sense both target proximity and whether another agent is already near the same area. Local target and agent LiDAR summaries can approximate 'solo vs paired' situations, while agent identity helps the shared policy break symmetry and distribute roles.
- **post_hoc_reflection:**
  - what_went_right:
    - The inner candidate includes the required role identity and cross-source mixing, so the abstract pattern was translated into a valid structural form.
    - The features do touch both lidar sources, which means the policy can in principle compare target opportunity against local agent congestion.
  - what_went_wrong:
    - Despite the right ingredients, the policy did not learn useful coordination: M1 is zero and M6 is far below the target minimum.
    - The current slot text likely does not make the joint decision rule concrete enough for PPO to exploit.
    - The role split and commit/divert guidance may still be too open-ended, allowing collapse into ineffective behavior.
  - remaining_uncertainty:
    - Whether the main weakness is the exact wording of the joint scalar, the role-specific behavior split, or both.
    - Whether sharpening the slot text around a direct 'opportunity versus congestion' comparison will be enough to raise learning.

### Outer 2 — `pair_and_split`

- **predicted:** M1=0.18, M6=0.42
  - what_is_needed:
    - Role differentiation under shared parameters so agents do not all choose the same target
    - At least one feature combining target-derived and agent-derived local summaries
    - A stable commit heuristic based on distance plus local crowding
- **actual:** M1=0.0, M6=0.083, label=partial, n_features=29, role_one_hot_present=True
- **delta:** M1=-0.18, M6=-0.3371428571428571
- **strategy chain_of_thought (snapshot at this outer):**
  - why_it_works: Reward comes from exactly two agents simultaneously covering a target, so agents need to sense both target proximity and whether another agent is already near the same area. Local target and agent LiDAR summaries can approximate 'solo vs paired' situations, while agent identity helps the shared policy break symmetry and distribute roles.
- **post_hoc_reflection:**
  - what_went_right:
    - The analyzer confirms the intended cross-source structure is still there: both target and agent summaries are used, with multiple cross-source operations.
    - Role identity is present, so the policy still has a mechanism for specialization under shared parameters.
    - The design is not missing the core ingredients; it is at least structurally aligned with the task.
  - what_went_wrong:
    - Despite the correct ingredients, the learned policy is not exploiting them: both M1 and M6 are extremely low.
    - The slot text is probably still too abstract about how to convert target proximity and agent congestion into a concrete action choice.
    - The role split may be insufficiently explicit, so agents can still collapse into ineffective symmetry or indecisive diversion behavior.
  - remaining_uncertainty:
    - Whether the main problem is the exact joint scalar formulation or the lack of a sharper role-conditioned rule.
    - Whether a more direct 'commit when opportunity outweighs congestion, otherwise divert' wording will be enough to unlock learning.

### Outer 3 — `leader_follower_pairing`

- **predicted:** M1=0.14, M6=0.35
  - what_is_needed:
    - Persistent role cues from agent identity
    - A feature that measures local agent density to detect possible partners
    - A mechanism for switching from exploration to pairing when a partner is detected
- **actual:** M1=0.0, M6=0.089, label=partial, n_features=30, role_one_hot_present=True
- **delta:** M1=-0.14, M6=-0.26142857142857134
- **strategy chain_of_thought (snapshot at this outer):**
  - why_it_works: The task rewards coordinated pairing rather than solo occupation, so a leader-follower structure naturally creates one agent that initiates and another that completes coverage. Local agent LiDAR can indicate whether a partner is nearby, and agent identity can stabilize which agents tend to lead or follow.
- **post_hoc_reflection:**
  - what_went_right:
    - The analyzer confirms the mandatory ingredients are present: role one-hot, cross-source operations, and features touching both lidars.
    - The strategy still matches the intended decomposition of explorer versus follower behavior.
    - The feature stack is structurally rich, so the model is not starved of inputs.
  - what_went_wrong:
    - Despite the correct structure, the policy is not learning useful coordination; both metrics are near zero.
    - The slot text may still be too abstract, leaving the policy without a crisp decision boundary for when to explore versus when to attach to a cluster.
    - The current wording likely allows symmetry collapse or indecisive movement instead of a strong leader/follower split.
  - remaining_uncertainty:
    - Whether a sharper role-conditioned rule will be enough, or whether this strategy is simply too hard for PPO at the current budget.
    - Whether the strongest fix is to define a more explicit leader trigger and follower trigger using a direct opportunity-versus-crowding comparison.

### Outer 4 — `leader_follower_pairing`

- **predicted:** M1=0.14, M6=0.35
  - what_is_needed:
    - Persistent role cues from agent identity
    - A feature that measures local agent density to detect possible partners
    - A mechanism for switching from exploration to pairing when a partner is detected
- **actual:** M1=0.0, M6=0.139, label=translation_failure, n_features=22, role_one_hot_present=False
- **delta:** M1=-0.14, M6=-0.21142857142857135
- **strategy chain_of_thought (snapshot at this outer):**
  - why_it_works: The task rewards coordinated pairing rather than solo occupation, so a leader-follower structure naturally creates one agent that initiates and another that completes coverage. Local agent LiDAR can indicate whether a partner is nearby, and agent identity can stabilize which agents tend to lead or follow.
- **post_hoc_reflection:**
  - what_went_right:
    - The design still includes cross-source reasoning and touches both lidars, so the high-level idea was translated into the model.
    - The repeated memory shows the earlier variants already had role one-hot and still underperformed, which helped isolate that this outer regressed structurally rather than merely failing to learn.
  - what_went_wrong:
    - The mandatory role identity cue is missing, so the shared policy cannot reliably separate leader and follower behavior.
    - Because the role signal is absent, the agent-specific specialization described by the strategy cannot be expressed cleanly.
    - Even with several cross-source operations, the representation is likely too diffuse without an explicit role split.
  - remaining_uncertainty:
    - Whether restoring role one-hot alone will recover the intended specialization, or whether the slot text also needs a sharper leader-versus-follower decision rule.
    - Whether this strategy should remain active after repair, given that earlier partial runs already showed weak learning.
