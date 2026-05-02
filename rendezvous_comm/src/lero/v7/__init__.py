"""LERO v7 — full-solution strategy enumeration + grounded reflection.

v7 adds a layer ABOVE v6's operational slot-edit system: at cold-start
the meta-LLM enumerates 3-5 high-level POLICY-LEVEL strategies (e.g.
"pairs commit on shared target", "explore-then-hold"), ranks them by
LERO-codability + RL-trainability, picks the best, then translates
that single strategy into operational guidance for the inner LLM
exactly as v6 does.

Per-iter reflection then GROUNDS the meta-LLM's decision in two
machine-checkable signals:

    1. AST pattern_present — did inner produce code with the
       structural pattern the strategy implies (e.g. cross-source
       AND-product for pairs_commit)?

    2. metrics_signature_match — did the M1/M6/M3 trajectory match
       the strategy's expected success signature?

This separates v6's ambiguous "no_signal_simple" into:

    - translation_failure (pattern absent → refine inner prompt)
    - rl_too_hard       (pattern present + flat metrics → switch strategy)

The strategy bundle persists across outer iters; the meta-LLM updates
it (demote rl_too_hard ones, propose new ones) rather than picking
fresh each round. This is the macro-strategy memory the prior MP
versions never had.
"""
