"""LERO v6 — simplicity-first meta-strategy, inner-only validation.

v6 design changes vs v5:

1. ONE strategy per outer iter (vs v5's bundle of 3 strategies).
   The meta-LLM commits to a single English direction per round; the
   inner LERO loop explores it via 4×3×1M iterative refinement.

2. SIMPLICITY-FIRST + reflection-driven escalation.
   Outer iter 0 must run with the simplest strategy. Complexity only
   escalates when prior round's classification justifies it.

3. META-LLM controls evolve_observation / evolve_reward FLAGS, not
   just slot text. Outer iter 0 forced obs-only; reward unlocks
   only when classification = no_signal_simple or partial_signal.

4. EARLY STOP on found_good. Stop chasing marginal improvements.

5. NO deep-train inside the loop. v6 is a pure inner-validation
   experiment; deep-train is gated to a separate script.

See docs/v6_plan.md for the full rationale.
"""
