"""LERO v9 — task-domain framing, CoT meta-prompt, memory, simplification.

See docs/v9_plan.md for the design rationale.

Modules:
  meta_strategist : bundle enumeration with CoT + combined diagnose+reflect
  memory          : append-only JSONL store for cross-outer reasoning
  outer_loop      : v9 outer loop, serial coverage of all bundle strategies
"""
