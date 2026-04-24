# LERO-MP v3 — Dry-Run Checklist

Manual verification steps to run after a v3 dry-run lands, before
committing to a full 10M sweep. Paired with
`scripts/review_dry_run.py` which automates the checks.

## Prerequisites

- Run completed: `results/lero_mp/<exp_id>/<YYYYMMDD_HHMM>_s<seed>/`
- Seeds 0, 1, 2 all finished (partial runs produce partial evidence).

## Automated review

```bash
cd rendezvous_comm
for seed in 0 1 2; do
    python scripts/review_dry_run.py \
        results/lero_mp/lero_mp_v3/*_s${seed}
done
```

Exit code 0 = ≥3 of 5 success criteria fired for that seed. Read the
printed report either way.

## Manual checks

### 1. Inner-LLM retries actually salvaged candidates

```bash
grep -r '"attempts": [2-9]' results/lero_mp/.../outer_*/iter_*
```

Expect: at least one seed has a candidate with `"attempts": 2` or `3`
alongside a non-error metrics file. If **all** candidates are
`"attempts": 1`, either the LLM never failed (good but no salvage
signal) or retry wasn't wired in (bug).

### 2. Strategist chose non-default include_signals

```bash
grep -c '"include_signals".*fingerprint\|"include_signals".*curve_shape' \
    results/lero_mp/.../mutation_log.jsonl
```

Expect `≥1`. Inspect the `signal_rationale` field to confirm the
justification references a specific outlier (M4/M8/M9) or curve tag.

### 3. Strategist rationales cite numeric behavioral signals

```bash
grep -oE 'M[1-9]=[0-9.]+' results/lero_mp/.../mutation_log.jsonl | head -20
```

Expect ≥1 match per seed. Generic rationales ("the model failed to
converge") don't count.

### 4. Critic triggered at least one revision

```bash
grep -E "Editor critique:\s+revisions=[1-9]" \
    results/lero_mp/.../*.log
```

Expect ≥1 hit across all seeds. If none — Critic is either agreeing
with every Editor output (possible but suspicious) or silently failing
(check for `Editor critique pass failed:` warnings).

### 5. Bit-exact read_only cache replay

```bash
# After a seed-0 run completes:
cd rendezvous_comm
LERO_LLM_CACHE_MODE=read_only \
    python run_lero_mp.py \
        configs/lero_mp/mp_v3_dryrun_3m.yaml --seed 0 \
        --output-dir /tmp/replay_s0

# Compare strategy cards + mutation logs
diff \
    results/lero_mp/.../<seed-0-run>/mutation_log.jsonl \
    /tmp/replay_s0/mutation_log.jsonl
```

Expect: identical lines for `strategy_card`, `new_slot_content`, and
`critique` fields. Any drift → cache key missing a dimension; check
`_cache_key_for` in `llm_client.py`.

### 6. ER1-comparability

Compare peak-M1 at 1M and 3M between v3 and ER1 baseline:

```bash
python -c "
import json, glob
for p in glob.glob('results/lero_mp/.../iter_0/candidate_*_metrics.json'):
    d = json.load(open(p))
    if '_error' in d: continue
    print(p, d.get('peak_M1'), d.get('M1_success_rate'))
"
```

Expect peak-M1 at 1M within ±0.02 of ER1 peak-M1 at 1M (task
parameters match: n=4, t=4, k=2, cr=0.35, ms=200). Large deviation →
scenario patch is leaking context or RNG seed-lock didn't take.

## Summary scoring

Need **3 of 5** to clear the bar for a full 10M run:

1. Retry salvage ≥ 20% **AND** ≥ 1 candidate salvaged
2. Strategist non-default `include_signals` **AND** rationale cites numeric signal
3. Critic ≥ 1 revision
4. ≥ 1 `marginal_improvement` or `strong_improvement` verdict
5. `LERO_LLM_CACHE_MODE=read_only` replay matches bit-exact

If < 3 fire, iterate on the prompt templates, not on the infra.
