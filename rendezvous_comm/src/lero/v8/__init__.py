"""LERO v8 — density-first, bounded features, concrete fewshot.

v8 = v7 + (
  configurable feature-count cap,
  density-first preference (gated cap),
  meta-LLM-authored 3-4 feature working fewshot,
  trim_features / replace_gated_with_dense refinement actions
)

See docs/v8_plan.md for the design rationale and the v7→v8 deltas.
"""
