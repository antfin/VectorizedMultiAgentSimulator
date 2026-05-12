# multi_scenario

Cooperative multi-agent RL on VMAS — ER1 baseline + LERO observation/reward
evolution, with a Streamlit UI for submission, inspection, and result browsing.

> **Note**: this folder will be extracted to a standalone repo
> `coopvmas` at F10.6. Internal references to `multi_scenario` still
> appear in code until the F10.7 rename pass.

## Where to start

| If you want to… | Read |
|---|---|
| Install & run your first experiment | [Getting started](getting_started/index.md) |
| Understand the architecture and metrics | [Concepts](concepts/index.md) |
| See the four supported scenarios | [Scenarios](scenarios/index.md) |
| Use the CLI | [CLI](cli/index.md) |
| Use the Streamlit UI | [Frontend](frontend/index.md) |
| Submit to OVH / regenerate videos / debug | [Operations](operations/index.md) |
| Browse results | [Results analysis](results_analysis/index.md) |
| See how to extend with a new adapter | [Ports](ports/index.md) |
| Verify a published result reproduces | [Reproducibility](reproducibility/index.md) |

## Key results so far

- **LERO S3b-local single-seed M1 = 0.795** at `covering_range=0.25`, `max_steps=400` (rendezvous_comm published 0.88; within seed noise).
- **LERO at ER1 params: M1 = 0.570 vs ER1 baseline 0.405** (+40% relative). See [LERO S3b-local reproduction](reproducibility/lero_s3b_local_reproduction.md).

## Project status

- F8.4 (rendezvous_comm reproduction + ER1-params LERO): **complete**.
- Phase 9 (LERO core): **complete** — F9.0–F9.8 all shipped.
- Phase 10 (docs + extraction): **in progress** (this wiki is F10.1).
- Phase 11 (per-scenario campaigns): deferred.
