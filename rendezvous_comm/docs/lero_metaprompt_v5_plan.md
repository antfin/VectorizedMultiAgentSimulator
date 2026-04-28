# LERO-MP v5 — Pre-ideas (pending v4 full-run results)

> **Status**: speculative planning doc — 2026-04-26.
> **Depends on**: v4's full-run outcome (`mp_v4_rendezvous_k2.yaml`,
> 3 seeds × 17.8M frames each, ~€30). Until that finishes and we
> see whether v4 reaches S3b-local-class peak_M1 (≥0.7 at 10M),
> none of the items below should be implemented.

The three improvements below are ranked by leverage given current
evidence (v3 scenario sweep, S3b-local mechanism analysis,
LERO-MP-3x2M failure analysis). Each is a candidate v4.x or v5
revision — not committed.

## Decision dependency

| v4 outcome | Suggested action |
|---|---|
| peak_M1 @ 10M ≥ 0.70 (S3b-local class) | Ship v4. Move to **v5 — DSPy compilation** for prompt optimization. |
| peak_M1 @ 10M in [0.40, 0.70] (better than v3, not S3b) | Apply pre-idea #3 (S3b example in bootstrap), retry. |
| peak_M1 @ 10M < 0.40 (matches v3.1 LERO-MP) | Apply pre-ideas #1 + #2 + #3 together. Reconsider whether the meta-prompt loop is the right abstraction. |
| Reward-hacking pattern persists (peak − final > 0.20) | Pre-idea #2 alone may not be enough; consider hard-disable `evolve_reward` for k=2 tasks. |

## Pre-idea #1 — Multiple inner candidates per strategy

**Hypothesis**: at 200k eval frames, single-shot inner-LLM code generation is too noisy. S3b-local generated 3 candidates per inner iteration over 4 iterations = 12 codes total. v4 generates 1 code per strategy × 3 strategies × 3 rounds = 9 codes total. Less search → worse code.

**Change**:
- `LeroMPv4Config.inner_n_candidates: int = 3` (currently fixed at 1).
- v4 outer loop runs N candidates per strategy at eval_frames each, picks best by stability_score, that becomes the strategy's representative.

**Cost impact**:
- Per-round eval: 3 strategies × 3 cands × 200k = 1.8M (vs current 0.6M).
- Per-seed total: 3 × (1.8M + 2M) + 10M = 21.4M (vs 17.8M, +20%).
- 3 seeds parallel: ~€36 total (vs €30, +€6).

**Expected benefit**: better per-strategy code quality at the cost of ~20% more compute. Probably worth it.

## Pre-idea #2 — Disable `evolve_reward` for round 0

**Hypothesis**: round 0 strategists are tempted by reward shaping for novelty, even though obs-only is the proven recipe. Forcing round 0 to be obs-only anchors the search on the safe direction; reward can be re-enabled in later rounds if obs alone underperforms.

**Change**:
- `LeroMPv4Config.evolve_reward_from_round: int = 1` (default).
- Outer loop overrides `evolve_reward=false` when `round_idx < evolve_reward_from_round`.

**Cost impact**: zero. Pure prompt-level constraint.

**Expected benefit**: removes one common round-0 failure mode (reward shaping with peak-collapse), saves a round of evolution effort.

**Risk**: if the obs-only direction is actually wrong for some task, round 0 wastes budget. Mitigated by allowing reward in rounds 1+.

## Pre-idea #3 — S3b-local code example in bootstrap

**Hypothesis**: the bootstrap LLM proposes named features (hold_signal, etc.) but the inner LLM has to invent the implementation. Showing the inner LLM a working code reference grounds it on the proven recipe.

**Change**:
- `v4_bootstrap.py::_build_bootstrap_prompt` — add a section labeled `## REFERENCE — working observation function from a related task` containing the actual `s3b_local/best_obs.py` (the 28-feature winner).
- The bootstrap LLM emits a BootstrapCard that may reference patterns from this example.
- Inner LLM (in round 0+) sees the reference embedded in `guidance_observation.txt` (or a new `examples_local.txt` slot) when it generates code.

**Cost impact**: zero (~10 lines of code; +500 tokens per inner LLM call ≈ +€0.01 per seed).

**Expected benefit**: STRONGEST single lever. The S3b-local code is a known-good 88% solution. Anchoring inner-LLM code generation on that reference should drastically improve round 0's quality.

**Risk**: the LLM might literally copy the reference instead of innovating. Counter-measure: tell the inner LLM "this reference is for a HARDER variant of the task; you may simplify or extend, do not copy verbatim."

## Pre-idea #4 — DSPy migration (the v5 framing)

**Hypothesis**: hand-tuning prompts has diminishing returns. DSPy + MIPROv2 can compile prompts against a learned trainset of (input, expected_output) pairs.

**Change** — three-step migration:

1. **v5.0**: refactor v4_strategist + v4_bootstrap as `dspy.Signature` subclasses. Pydantic schemas are already shape-compatible. Functionality unchanged.
2. **v5.1**: collect a trainset by re-running scenario harness scenarios. Each scenario maps a (BootstrapCard, round_history) → human-rated StrategyBundle. ≥30 examples needed for MIPROv2.
3. **v5.2**: `MIPROv2.compile(strategist_signature, trainset=..., metric=stability_score)` produces an optimized prompt. Replace v4's hand-tuned prompt with the compiled one.

**Cost impact**:
- v5.0: 1 day refactor, no compute change.
- v5.1: ~30 scenarios × ~€0.01 LLM cost = €0.30 (mostly human time labeling).
- v5.2: ~€2 compilation, runtime cost unchanged.

**Expected benefit**: moves from "I tweaked the prompt by hand and it improved 10%" to "the optimizer found a prompt that scores 30% higher on our trainset." Higher ceiling, more rigorous.

**Risk**: trainset construction is the hard part. If our scenarios are biased the compiled prompt overfits. Mitigated by holding out a test scenario set.

## Pre-idea #5 — Per-task description templates (lower priority)

Currently `rendezvous_k2.md` is hand-written for one task. To scale to other VMAS scenarios:

- Template `<task>.md` with placeholders the user fills in (env, state keys, success metric, known patterns).
- A small CLI helper that generates the template from a scenario name.
- Library of "known patterns" the bootstrap LLM can pull from per task family (rendezvous: hold_signal; navigation: target_direction; coverage: voronoi_density).

Defer until v4 is validated on multiple tasks.

## What we'd skip in v5

- **Reasoning-model meta-LLM (o4-mini, gpt-oss)**: harness sweep already ruled this out (gpt-5.4-mini wins on cost + quality).
- **TextGrad library proper**: the pattern (Critic + revise) is in v3.1 / v4 already; the library adds dependency without value for our current pipeline.
- **Multiple seeds for the meta-LLM**: currently we lock one meta-LLM call per round. Adding multi-seed averaging would 3× the meta-LLM cost without clear win.
- **Cross-task transfer**: needs N tasks × M seeds first; out of scope for one PhD experiment.

## How to read this doc after v4 finishes

1. Look at v4's `v4_result.json` files for all 3 seeds.
2. If peak_M1 mean ≥ 0.70 → ship v4, jump straight to v5 DSPy migration.
3. If peak_M1 mean ∈ [0.40, 0.70] → implement pre-idea #3 (1h work, +€0 cost), re-run.
4. If peak_M1 mean < 0.40 → implement pre-ideas #1 + #2 + #3, re-run.
5. Update this doc with what was actually applied + measured outcomes.

---

**Author note**: until v4 finishes, treat all numbers above as expected
ranges, not commitments. Compute estimates assume V100S CPU; GPU rates
would change them.
