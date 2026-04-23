# LERO-MP v2 — Two-Level Meta-Prompting + Evolutionary Memory

> **Status:** Plan for review — 2026-04-23.
> **Author:** afin (with Claude Code).
> **Previous design:** see `lero_metaprompt_plan.md` (v1, single-call meta-LLM editing the whole `guidance` slot).

## 1. Why v1 is not enough

The improved meta-prompt from the 2026-04-23 iteration (cite-evidence + reference-techniques + fairness-restatement guard) did produce a targeted edit on seed 0:

> "add compact local coordination features from lidar: nearest and 2nd-nearest target/agent distances, their gap, a count of rays with distance < covering_range, and a hold_signal = (target_near AND agent_near)"

But even this suffers from three structural problems:

1. **One slot, all concerns mixed.** The same `guidance` text has to carry both reward-specific and observation-specific advice. Reward shaping and observation engineering need very different patterns; mixing them dilutes both.
2. **One LLM call, two jobs.** The meta-LLM is asked to both *decide what to focus on* and *write the content*. It picks the easiest path (generic advice) unless forced with heavy prompt engineering.
3. **No memory across mutations.** Each mutation is blind to prior attempts — it can re-propose an edit that already failed last time, or repeat the same "hold_signal" suggestion three seeds in a row with no diversification.

## 2. Design principles for v2

- **Separate "what to improve" from "how to improve it".** Two LLM calls per mutation: a *Strategist* produces a structured decision; an *Editor* produces the text.
- **Decompose the guidance slot along the axes the inner LLM cares about**: reward, observation, shared. Edits target exactly one sub-slot per mutation.
- **Persist mutation outcomes across runs.** A `mutation_log.jsonl` records every mutation's (strategy, slot, content, peak_M1_before, peak_M1_after, M6 delta). Future meta-LLM calls see this history — "do not re-propose X, it scored −0.05 last time" is now a first-class input.
- **Diversify across seeds.** Different seeds get different strategy *biases* (explore reward vs observation vs combined) so a 3-seed sweep covers more of the edit space. No seed gets a rigid assignment; it's a soft bias the Strategist reads.
- **Keep the inner loop unchanged.** Everything v2 adds is in `src/lero/meta/*`. The fairness contract (frozen slot + runtime whitelist) stays exactly as it is.

## 3. Architecture

```text
  ┌─ OUTER ITER ─────────────────────────────────────────────────────┐
  │                                                                  │
  │   inner LeroLoop.run()  → candidate metrics + top-k code         │
  │                                                                  │
  │   build_template_record() → TemplateRecord                       │
  │                                                                  │
  │   should_meta_iterate(history, …) → TRIGGER fires                │
  │                                                                  │
  │   ┌─ LEVEL 1  STRATEGIST  (meta-LLM call #1) ───────────────┐    │
  │   │  INPUT                                                  │    │
  │   │   - history of TemplateRecords (this run)               │    │
  │   │   - recent mutation_log (cross-run memory)              │    │
  │   │   - candidate-aggregate stats + feature deltas          │    │
  │   │   - seed-level strategy bias (string tag)               │    │
  │   │  OUTPUT (structured YAML)                               │    │
  │   │   StrategyCard:                                         │    │
  │   │     target_domain: reward | observation | shared | both │    │
  │   │     focus: <1-2 concrete ideas, 1-line each>            │    │
  │   │     avoid: <list of patterns that underperformed>       │    │
  │   │     confidence: small | medium | large                  │    │
  │   │     rationale: <why this direction now>                 │    │
  │   └─────────────────────────────────────────────────────────┘    │
  │                                                                  │
  │   ┌─ LEVEL 2  EDITOR  (meta-LLM call #2) ────────────────────┐   │
  │   │  INPUT                                                   │   │
  │   │   - StrategyCard from Level 1                            │   │
  │   │   - current text of the target sub-slot                  │   │
  │   │   - 3 top candidate code snippets (diagnostic only)      │   │
  │   │   - fairness slot (frozen, DO NOT RESTATE)               │   │
  │   │  OUTPUT                                                  │   │
  │   │   - new sub-slot text (targeted, evidence-cited)         │   │
  │   │   - rationale                                            │   │
  │   │   - expected_improvement                                 │   │
  │   └──────────────────────────────────────────────────────────┘   │
  │                                                                  │
  │   materialize_mutation()                                         │
  │     - writes new prompt version with ONLY target sub-slot        │
  │       replaced                                                   │
  │     - other sub-slots copied verbatim from parent                │
  │   mutation_log.jsonl.append()                                    │
  │     - records (strategy, slot, content_hash, pre_peak_M1, ...)   │
  │                                                                  │
  │   current_version := new_version                                 │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

## 4. Sub-slot decomposition

Replace the single `guidance.txt` slot with three sub-slots inside a new `guidance/` sub-directory. The `meta.yaml` declares them just like regular slots so `PromptLoader.render()` concatenates them in the correct order without needing to know about the nesting.

### 4.1 File layout

```text
prompts/v2_fewshot_modular/
  meta.yaml
  system.txt
  task_context.txt
  current_code.txt
  state_schema.txt
  fairness.txt                 # frozen, unchanged
  guidance_shared.txt          # patterns relevant to BOTH reward + obs
  guidance_reward.txt          # reward-shaping specific
  guidance_observation.txt     # observation/feature-engineering specific
  examples.txt
  output_spec.txt
  feedback.txt
```

### 4.2 Content split

- **`guidance_shared.txt`** — numerical safety (NaN/Inf), shape correctness, determinism. Things that apply regardless of what you're computing.
- **`guidance_reward.txt`** — reward-function advice: bounded magnitude, smooth shaping, avoid-hacking patterns, potential-based shaping, progress-delta bonuses vs terminal spikes.
- **`guidance_observation.txt`** — feature-engineering advice: lidar-derived features, nearest/2nd-nearest gaps, proximity counts, coordination signals (hold/approach), intensity features.

At rendering time all three concatenate into the same position the old `guidance.txt` occupied — the inner LLM sees one continuous guidance section. Only mutations target one of the three.

### 4.3 Effect on the inner prompt

Zero observable change for the inner LLM when all three start empty (it's the same prompt). First difference appears after the first mutation, which edits only one sub-slot — the inner LLM then sees guidance focused on that aspect.

## 5. Strategy Card schema

Output of Level 1, consumed by Level 2. Strict YAML so it's parseable:

```yaml
target_domain: observation         # reward | observation | shared | both
target_slot: guidance_observation  # one of the three sub-slots
focus:
  - "Add coordination signals (hold, approach, crowd) from lidar proximity"
  - "Split nearest vs 2nd-nearest distance + gap to disambiguate contested targets"
avoid:
  - "Generic 'use local sensors' restatement (tried 3x, M1 flat)"
  - "Unbounded reward magnitude scaling (caused reward-inflation in v2_fewshot_modular_mp_002)"
confidence: medium
rationale: |
  Current candidates' best-M6 candidate already uses proximity_count
  but none combine target_near with agent_near into a hold flag. The
  mutation_log shows 2 prior attempts to improve reward shaping
  without lifting M1. Shifting to observation feature engineering
  has a higher expected delta per mutation_log history.
```

## 6. Evolutionary memory: `mutation_log.jsonl`

One line per mutation across all runs, appended at the end of each outer iter. Persisted under `$RESULTS_DIR/mutation_log.jsonl` (writable mount on OVH, `results/` locally).

Entry schema:

```json
{
  "ts": "2026-04-23T12:34:56Z",
  "run_id": "lero_mp_quick_k2_cr035_imeta_s0_20260422_1842",
  "seed": 0,
  "outer_iter": 2,
  "parent_version": "v2_fewshot_modular",
  "new_version": "v2_fewshot_modular_mp_003",
  "strategy_card": {
    "target_domain": "observation",
    "target_slot": "guidance_observation",
    "focus": ["add hold_signal + 2nd-nearest gap"],
    "avoid": ["generic fairness paraphrase"],
    "confidence": "medium",
    "rationale": "..."
  },
  "slot_content_sha256": "abc123...",
  "slot_content_excerpt": "When building observations, add compact local...",
  "pre_mutation_peak_M1": 0.010,
  "pre_mutation_best_M6": 0.179,
  "post_mutation_peak_M1": 0.015,
  "post_mutation_best_M6": 0.161,
  "delta_peak_M1": 0.005,
  "delta_M6": -0.018,
  "verdict": "marginal_improvement",
  "fail_modes_during_next_iter": ["reward_magnitude_inflation"]
}
```

At each Level 1 call the Strategist receives a **summary** (not all entries — just the last N = 10 relevant to this task) with deltas and verdicts. This lets it:

- avoid proposing patterns that scored ≤ 0 in prior runs,
- prefer domains (reward vs obs) that historically had positive deltas,
- explicitly reference prior attempts when framing its rationale.

### Verdict classifier (offline, deterministic)

`classify_verdict(delta_peak_M1, delta_M6) -> Literal["strong_improvement", "marginal_improvement", "neutral", "regression", "collapse"]`:

- strong_improvement: delta_peak_M1 ≥ 0.10
- marginal_improvement: 0.01 ≤ delta_peak_M1 < 0.10 AND delta_M6 ≥ 0
- neutral: |delta_peak_M1| < 0.01 AND |delta_M6| < 0.05
- regression: −0.10 < delta_peak_M1 < −0.01 OR delta_M6 < −0.05
- collapse: delta_peak_M1 ≤ −0.10 OR peak_M1 went from non-zero to zero

These thresholds are deliberately permissive early (noise dominates at 1M frames); tighten for 10M.

## 7. Cross-seed strategy diversification

Different seeds get different **strategy biases** passed to Level 1 as a hint. Not a rigid constraint — Strategist can override if evidence strongly points elsewhere.

```python
SEED_STRATEGY_BIAS = {
    0: "observation_first",   # explore feature engineering
    1: "reward_first",         # explore reward shaping
    2: "exploratory",          # pick whichever domain has fewer prior tries
}
```

Level 1's prompt receives `[SEED BIAS] <tag>: <one-line hint>` and is told "this is a soft preference, override only if evidence is unambiguous."

Three seeds → three different trajectories through the mutation space → bigger coverage of what actually helps.

## 8. Code + file changes

### 8.1 New module: `src/lero/meta/strategy.py`

Pure-function Level 1 logic:

```python
@dataclass
class StrategyCard:
    target_domain: Literal["reward", "observation", "shared", "both"]
    target_slot: str           # "guidance_reward" | "guidance_observation" | "guidance_shared"
    focus: List[str]
    avoid: List[str]
    confidence: Literal["small", "medium", "large"]
    rationale: str

def build_strategist_prompt(
    history: Sequence[TemplateRecord],
    mutation_log_summary: List[Dict],
    top_candidates: Sequence[Dict],
    seed_bias: str,
    fail_mode: FailMode,
) -> str: ...

def parse_strategy_card(text: str) -> StrategyCard: ...

def strategize(history, log, candidates, seed_bias, fail_mode, meta_llm_call) -> StrategyCard: ...
```

### 8.2 Refactor `mutation.py`

- `propose_new_template` takes a `StrategyCard` arg.
- `build_meta_prompt` becomes `build_editor_prompt(strategy_card, ...)` — Level 2 only.
- The editor prompt lists:
  - The frozen fairness slot (verbatim, forbidden to restate)
  - The current text of ONLY the target sub-slot
  - The strategy card's `focus` + `avoid`
  - Top-3 candidate excerpts for evidence

### 8.3 New module: `src/lero/meta/mutation_log.py`

```python
def append_entry(path, entry: Dict) -> None: ...
def read_recent(path, n: int = 10, task_id: Optional[str] = None) -> List[Dict]: ...
def classify_verdict(pre_m1, post_m1, pre_m6, post_m6) -> str: ...
```

### 8.4 Update `outer_loop.py`

```python
# After the inner run that produced record K:
strategy = strategize(
    history=history,
    mutation_log_summary=read_recent(log_path, n=10, task_id=spec.exp_id),
    top_candidates=top_cands,
    seed_bias=SEED_STRATEGY_BIAS[seed % 3],
    fail_mode=record.fail_mode,
    meta_llm_call=self.meta_llm_call,
)
mutation = propose_new_template(
    parent_version=current_version,
    strategy_card=strategy,
    history=history,
    top_candidates=top_cands,
    meta_llm_call=self.meta_llm_call,
    outer_iter=outer_iter + 1,
)
# After the NEXT inner run, compute delta and append log entry.
```

### 8.5 Split `guidance.txt` → three sub-slot files

Touch points:

- `v2_fewshot_modular/meta.yaml` — replace `guidance` slot with three entries in the `initial_user_slots` list.
- Write `guidance_shared.txt`, `guidance_reward.txt`, `guidance_observation.txt` initially empty.
- Delete the old `guidance.txt`.
- Migrate any existing `v2_fewshot_modular_mp_*` child versions by running a one-off script that splits their current `guidance.txt` into the three files (heuristic: everything goes to `guidance_shared.txt` by default). Safer: abandon old mutated versions (they're ephemeral anyway) and start fresh.

### 8.6 No Jinja

F-strings plus the existing `string.Template` machinery in `PromptLoader` are enough. Jinja would add a dependency and a second templating dialect for no clear win here.

## 9. Test plan

### 9.1 Unit tests (new)

- `tests/test_lero_mp_strategy.py`
  - `StrategyCard` parsing: well-formed YAML, missing fields, malformed
  - `build_strategist_prompt` contains required sections (seed_bias, history, mutation_log summary)
  - `strategize()` end-to-end with stub meta-LLM
- `tests/test_lero_mp_mutation_log.py`
  - `append_entry` + `read_recent` round-trip
  - `classify_verdict` thresholds for each outcome category
  - Filtering by `task_id`
- `tests/test_lero_mp_mutation.py` (update)
  - Level 2 editor prompt references strategy card
  - Editor output must match `strategy_card.target_slot`
  - Stale guard: if editor's output references a different slot than requested, `MutationParseError`
- `tests/test_lero_mp_outer.py` (update)
  - Outer loop passes strategy through Level 1 → Level 2
  - Mutation log entries written after second inner iter (once post-mutation metrics exist)

### 9.2 Expected test count

~25 new, ~5 updated. After change: ~170 total LERO-MP tests.

### 9.3 Local sanity

Every new function has a pure-Python unit test (no LLM, no GPU). `stub_meta_llm` in outer-loop tests returns a canned StrategyCard YAML or canned editor output depending on which call it's serving.

## 10. Dry-run plan

**Config**: new `configs/lero_mp/mp_quick_v2_k2_cr035.yaml` (copy of `mp_quick_k2_cr035.yaml` with three-subslot `prompt_version: v2_fewshot_modular_v2`).

**Seeds**: 0, 1, 2 — different `SEED_STRATEGY_BIAS` each.

**Budget**: same 1M full-training × 3 outer iters = ~2.5h/seed, 3 seeds in parallel.

**What a successful dry-run looks like**:

1. **Level 1 outputs differ across seeds.** Seed 0 bias="observation_first" produces StrategyCard with `target_domain=observation`. Seed 1 bias="reward_first" → `target_domain=reward`. Seed 2 bias="exploratory" → picks whichever domain is underrepresented in the mutation_log.
2. **Level 2 outputs match the strategy card.** When the card says `target_slot=guidance_observation`, the edit only touches that file. `guidance_reward.txt` and `guidance_shared.txt` are byte-identical to the parent's.
3. **Mutation log grows across outer iters.** After each post-mutation record, a new entry lands in `mutation_log.jsonl` with the pre/post deltas and a verdict.
4. **Restatement guard still fires when warranted.** If Level 2 slips into generic paraphrasing despite Level 1's specific focus, parse error and graceful stop.
5. **The verdict classifier behaves sensibly at 1M frames.** Expect most verdicts to be `neutral` at this budget — that's fine; tightens at 10M.

**What I'll inspect after the dry-run**:

- 3 `strategy_card.yaml` files (one per outer-iter mutation per seed) — do they look like meaningful decisions?
- 3 new `guidance_<domain>.txt` edits per seed — do they target the strategy's domain?
- `mutation_log.jsonl` — one entry per mutation, 3-6 entries total, verdicts computed correctly
- Cross-seed: did the three seeds actually explore different parts of the edit space?

## 11. Rollback

All v1 code paths stay reachable:

- `mutation.propose_new_template(strategy_card=None)` keeps the current single-call behavior as a fallback (tested + documented).
- `outer_loop.LeroMpOuterLoop(two_level_meta=False)` uses the v1 pipeline.
- `SEED_STRATEGY_BIAS` optional — if not set, Level 1 gets "none" bias.

The plan landed as default ON once the dry-run validates. The rollback path is `meta_prompt.two_level_meta: false` in YAML.

## 12. Open questions for review

Please approve / edit before I start coding:

- [ ] Three sub-slots (reward/observation/shared) or just two (reward/observation, merge shared into each)? Three feels cleaner but adds one more file per template version.
- [ ] Strategist picks `target_slot` *directly*, or Level 1 just picks `target_domain` and a fixed map domain→slot? Direct is more flexible; mapped is more predictable.
- [ ] Cross-seed bias: hard-coded 3-way (obs/reward/exploratory) or randomized? Hard-coded gives reproducible sweeps; random gives more coverage.
- [ ] Mutation log path: per-run under `$RESULTS_DIR` or global at `~/.lero/mutation_log.jsonl` so it accumulates across jobs? Per-run + a consolidate step at report time is safer; global keeps things simple.
- [ ] Keep the v1 "restatement guard" heuristic, or retire it once the two-level pipeline is in? I'd keep it — defense in depth.
- [ ] Verdict thresholds: my proposal is lenient at 1M frames (0.01 M1 = marginal_improvement). Should scale for 10M (0.05 as marginal)?
- [ ] Dry-run size: same 1M × 3 outer (3-seed) as before, or bump to 3 outer × 4 candidates to see more mutations?

## 13. Estimated timeline

- Plan review + approve: now
- Implementation: ~4 h (bulk is tests; logic is small)
- Local unit tests: included in implementation
- Dry-run (3 seeds × 2.5 h wall): ~2.5 h wall, ~17 €
- Review dry-run artifacts + possible follow-up: ~1 h
- Full run: ~12 h wall, ~76 €

**Total wall to ready-for-full-run: ~8 h including dry-run.**
