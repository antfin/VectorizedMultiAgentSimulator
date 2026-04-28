"""LERO v5 — focused-depth, two-level textual gradient descent.

Architecture:

  OUTER (high-level):  refines guidance_observation/reward/shared.txt
                       via meta-LLM textual gradient with good+bad +
                       cumulative tried-and-failed registry.

  INNER (low-level):   for each metaprompt, runs an S3b-local-style
                       4×3×1M iterative-refinement search where the
                       inner LLM emits Python obs/reward code, sees
                       per-iter feedback (best+worst), maintains a
                       per-feature registry, and triggers a pivot
                       prompt on stagnation.

Decoupled from v4: v4's parallel-strategy fanout doesn't see inner
eval results within a strategy. v5's depth-first design lets every
LLM call (inner OR outer) condition on the previous call's measured
outcome.
"""
