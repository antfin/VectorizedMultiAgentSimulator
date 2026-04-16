# LERO Dry-Run Checklist

Short OVH validation run to exercise every LERO code path before launching the P1 + L-series ablations. Config: `configs/e1/lero_dryrun.yaml`.

## Design

- Task: n=3, t=3, k=1 (matches P1 paper-replication target)
- LERO: 2 iterations × 2 candidates, `reward_mode=replace`, `obs_state_mode=global`, prompt v2
- Train: 100k eval per candidate, 200k final (~5 batches each at 20k/batch)
- Expected wall time: ~15–20 min on V100S

## Submit

```python
from dotenv import dotenv_values
from rendezvous_comm.src.ovh import submit_training_job

submit_training_job(
    "rendezvous_comm/configs/e1/lero_dryrun.yaml",
    llm_env=dotenv_values(".env"),
)
```

Then tail logs: `ovhai job logs <job_id>`.

## Log Checklist (from `src/lero/loop.py`)

Watch for these lines in order. If any are missing or out of order, something is wrong.

| Stage | Expected Log Line | Code Ref |
|---|---|---|
| Loop start | `=== LERO START === 2 iterations, 2 candidates/iter` | loop.py:510 |
| Iter 1 open | `--- Iteration 1/2 ---` | loop.py:528 |
| LLM call | `Generating 2 candidates...` | loop.py:534 |
| Extraction | `Extracted N valid candidates from 2 responses` (N ≥ 1) | loop.py:561 |
| Eval start | `  Evaluating candidate 1/N ...` | loop.py:583 |
| Eval result | `    M1=X.XXX  M2=X.XX  M6=X.XXX` (non-NaN) | loop.py:598 |
| Global best | `  NEW GLOBAL BEST (iter 0): M1=..., M2=..., M6=...` | loop.py:653 |
| Iter 2 open | `--- Iteration 2/2 ---` | loop.py:528 |
| Feedback | `iter_1/feedback.txt` exists and mentions iter-0 best | loop.py:700 |
| Full train | `=== FULL TRAINING with best candidate ===` | loop.py:721 |
| Done | `=== LERO COMPLETE ===` | loop.py:742 |

## Failure Signals (must be absent)

| Log Line | Meaning | Code Ref |
|---|---|---|
| `LLM call failed in iteration` | LLM endpoint/key issue | loop.py:540 |
| `No valid candidates extracted` | codegen parse/AST failure on ALL candidates | loop.py:554 |
| `Candidate N failed: ...` | Patched scenario crashed in BenchMARL | loop.py:605 |
| `All candidates failed in iteration` | Full iteration lost | loop.py:613 |
| `No valid candidates found across all iterations` | Run aborted before full training | loop.py:717 |

A single "Candidate N failed" is tolerable (LLM can emit buggy code); **both candidates failing in both iters** is a bug in our patch/eval pipeline.

## Output Files to Verify

Under `results/lero_dryrun/runs/lero/<timestamp>/`:

- [ ] `messages_initial.json` — system prompt is v2, contains "reward engineer" language (not v1 research history)
- [ ] `iter_0/candidate_{0,1}_response.txt` — raw LLM output present
- [ ] `iter_0/candidate_{0,1}_reward.py` — extracted `compute_reward` function, syntactically valid Python
- [ ] `iter_0/candidate_{0,1}_obs.py` — extracted `enhance_observation` function (present because `evolve_observation=true`)
- [ ] `iter_0/candidate_{0,1}_metrics.json` — contains `M1_success_rate`, `M2_avg_return`, `M6_coverage_progress` as numbers (not null)
- [ ] `iter_0/feedback.txt` — contains results table and top-1 code
- [ ] `iter_1/candidate_*_*.py` — second-iteration candidates differ from iter 0 (confirms LLM received feedback)
- [ ] `messages_final.json` — last assistant message is iter-0 best; last user message is feedback (sliding-window confirmed)
- [ ] `evolution_history.json` — 2 entries, best_M1/M2/M6 present
- [ ] `final_metrics.json` — present (full training completed)
- [ ] Policy checkpoint saved (`policy.pt` or similar, see loop.py:879)

## Correctness Checks (beyond "it ran")

1. **Paper-faithful mode wiring** — open any `candidate_N_reward.py`; the reward should be a FULL reward function (not just a bonus delta). The prompt should not mention `bonus_scale` or `tanh`.
2. **Global obs wiring** — `candidate_N_obs.py` should reference global state fields (all agent positions, all target positions), not just local lidar rays.
3. **Patched scenario loaded** — grep logs for `PatchedDiscoveryScenario` or similar class-name traces during BenchMARL init; if absent, patching may be silently skipped.
4. **Metrics non-degenerate** — in the dry run, M1 will probably be 0 (only 100k frames) but M6 should move away from 0 across the run. M2 should be finite (not `-inf` / `NaN`).
5. **Results sync** — confirm `results/lero_dryrun/` is mirrored to S3 under `lero_dryrun/` prefix after job completion.

## If It Fails

| Symptom | First thing to check |
|---|---|
| LLM call fails | `.env` contains correct key; `llm.model` resolvable by LiteLLM; OVH can reach endpoint |
| All candidates fail extraction | Regex in `codegen.py`; LLM may be returning markdown fences not matching parser |
| Candidates evaluate but metrics all NaN | `_evaluate_candidate` + BenchMARL eval loop; check `evaluation_interval` vs `eval_frames` |
| Full training never runs | Fitness ranking returned no valid candidate — check AST validator isn't rejecting everything |
| Job completes but nothing in S3 | `bucket_results` config + per-exp prefix logic in `ovh.py` |

## Go/No-Go Gate for Full Experiments

Proceed to launch P1 + L1 + L4 + L8 + L21 **only if** every box in "Output Files to Verify" is checked AND no entries in "Failure Signals". Otherwise fix before spending 10h × 5 GPUs.
