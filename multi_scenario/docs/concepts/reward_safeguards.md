# Reward safeguards

LLM-generated rewards routinely produce values that crash PPO. Three layers protect training:

## Layer A — sanitization

```python
r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
```

Catches LLM-generated divisions by zero / `log(0)` / etc. Always on
when `reward_source` is set.

## Layer B — clipping

```python
r = torch.clamp(r, -reward_clip, +reward_clip)  # default ±50
```

Bounds magnitude. Configurable via `cfg.lero.reward_clip` (set to
`None` to disable — not recommended). Applies to:

- The final reward in `replace` mode.
- The bonus in `bonus` mode (original reward unaffected).

## Layer C — full-training fallback chain

After the LERO inner loop, valid candidates are ranked. Full training
(10M frames) tries rank 0; on `Exception` (typically `AssertionError:
not action.isnan().any()`), falls back to rank 1, then rank 2, etc.

Chain logged in `output/lero/final_summary.json`'s `fallback_chain` field:

```json
[
  {"rank": 0, "iteration": 3, "candidate_idx": 1,
   "outcome": "crashed", "error": "NaN actions at iter 87"},
  {"rank": 1, "iteration": 0, "candidate_idx": 2,
   "outcome": "success", "full_train_metrics": {...}}
]
```

## Why these are needed

The LERO paper used MPE Simple Spread with naturally bounded rewards
`[0, 5]`. VMAS Discovery LLMs produce magnitudes 100–1000× larger
(observed range `[-1192, +896]`). Without clipping, PPO's value-
function gradients explode ~70–90% through 10M-frame training.

Document as an explicit deviation from the paper when citing results.
