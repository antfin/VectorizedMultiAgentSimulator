"""v6 prompt-lab — LLM-only sandbox for iterating on v6's meta and inner
prompts without paying the 2.5h-per-run RL training cost.

The 4-run v6 record showed: meta-LLM produces strategically adjacent
guidance, inner LLM never produces cross-source AND-products, all
runs flat_zero. The cost was ~10.5h Mac time per "try a new prompt."

This module short-circuits the training loop:

    1. analyze_inner_code() — AST data-flow analyzer that detects
       cross-source operations (target-LiDAR × agent-LiDAR products,
       boolean masks combining both, etc.) without running the code.

    2. judge_inner_code() — LLM-as-judge scoring for "structural
       similarity to S3b-local's iter-1 winner" on a 0-10 scale.

    3. run_harness() — drives the actual v6 meta_strategist + inner_llm
       with synthetic input distributions (replay of S3b-local data or
       hand-crafted "fake inner result"), returning prompt-quality
       metrics in ~60s instead of 2.5h.

After Phase 2-3 iteration, the winning prompt combination gets ONE
full v6 RL run for validation. Per docs/v6_plan.md this stays inside
the anti-cheat boundary — no winning-feature names allowed in any
prompt the harness writes.
"""
