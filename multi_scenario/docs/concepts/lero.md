# LERO — LLM-driven Evolutionary Reward & Observation

Adapted from [arXiv:2503.21807](https://arxiv.org/abs/2503.21807) for VMAS Discovery.

## What LERO evolves

Two LLM-generated functions splice into the scenario at training time:

- **`compute_reward(scenario_state) → Tensor[batch]`** — replaces the
  hand-crafted reward (`replace` mode) or adds a bonus (`bonus` mode).
- **`enhance_observation(scenario_state) → Tensor[batch, N]`** — extra
  features appended to the policy's observation. `local` mode (CTDE-fair)
  exposes only LiDAR + own pose; `global` mode (oracle) exposes the
  full state.

Per `cfg.lero`, you can evolve either, both, or neither (run as
pure-ER1 baseline).

## The loop

```text
For each iteration in 1..n_iterations:
  1. Compose prompt from history + task knobs (Jinja template).
  2. LLM generates n_candidates code blocks.
  3. extract_candidates() parses + AST-validates each.
  4. For each valid candidate:
       - Splice code into PatchedDiscoveryScenario.
       - Train BenchMARL for cfg.lero.eval_frames_per_candidate (1M).
       - Compute M1–M9 → CandidateResult.
  5. Persist results → next iter's prompt feedback.

After loop:
  6. Rank ALL valid candidates across iterations by (M1, M6, M2).
  7. Fallback chain: full-train rank 0 (10M frames); on crash, try rank 1.
  8. Write final_summary.json with inner + post-full-train metrics.
```

## The 8 ports

The orchestrator (`application/lero_orchestrator.py`) is hex-clean — 8
ports DI'd at construction, no concrete imports:

| Port | Default adapter | Tests substitute with |
|---|---|---|
| `LlmClient` | `CostCapDecorator(DiskCacheDecorator(LiteLlmClient))` | `FakeLlmClient` |
| `PromptComposer` | `InitialAndFeedbackComposer` | `MetaPromptComposer` stub (F9.7.A) |
| `PromptRenderer` | `JinjaPromptRenderer` | — |
| `TraceWriter` | `FilesystemTraceWriter` | `InMemoryTraceWriter` |
| `CandidateEvaluator` | `BenchmarlCandidateEvaluator` | hand-rolled fakes |
| `FullTrainer` | `BenchmarlFullTrainer` | hand-rolled fakes |
| `CostLedger` | `FilesystemCostLedger` | `InMemoryCostLedger` |
| `Logger` | `FileLogger` | `_SilentLogger` |

## Reward safeguards

LLM-generated rewards have wide magnitude variance (we've seen M2
∈ `[-1192, +896]`). Two layers protect PPO:

- **Sanitization**: `nan_to_num(r, nan=0, posinf=0, neginf=0)`.
- **Clipping**: `clamp(r, -reward_clip, +reward_clip)` (default ±50).
- **Fallback chain**: on full-training crash (NaN actions, OOM), try
  the next-ranked candidate.

Detail: [Reward safeguards](reward_safeguards.md).

## Cost cap

The `LiteLlmClient` is wrapped in `CostCapDecorator` enforcing rolling
windows: **€10/day + €100/month** across ALL runs on the host
(persistent ledger at `~/.multi_scenario/cost_ledger.jsonl`). Crossing
either ceiling raises `LlmCostCapExceeded`; the orchestrator gracefully
ranks history-so-far and proceeds to full training.

## Prompt versions shipped

| Version | Use case |
|---|---|
| `v1` | original verbose (bonus + local) |
| `v1_global` | v1 body adapted for replace+global |
| `v2` | paper-faithful minimal |
| `v2_min` | 3-line ultra-minimal |
| `v2_fewshot` | v2 + 2 MPE-style examples |
| `v2_fewshot_k2_local` | **F8.4 default** — local-sensor + k=2 examples |
| `v2_twofn` | asks for agent + global reward decomposition |

Byte-parity vs `rendezvous_comm`'s `string.Template` originals is
pinned by `tests/integration/prompts/test_prompt_byte_parity.py`.

## What this repo's LERO has reproduced

- **rendezvous_comm S3b-local M1 = 0.88** → ours: **M1 = 0.795**
  (single-seed, within plausible variance).
- **LERO at ER1 params M1 = 0.570 vs ER1 baseline 0.405** (+40% relative).

Full comparison: [LERO S3b-local reproduction](../reproducibility/lero_s3b_local_reproduction.md).
