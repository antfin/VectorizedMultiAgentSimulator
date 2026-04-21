"""LERO-MP meta-prompting extension.

Adds an outer loop that evolves prompt templates based on downstream
RL-training feedback. See docs/lero_metaprompt_plan.md for the design.

The inner LERO loop (src/lero/{config,loop,scenario_patch,codegen}.py)
is unchanged; everything here composes on top of it and is a no-op
when MetaPromptConfig.enabled is False.
"""

from .fairness import AllowedKeysDict, FairnessViolation

__all__ = ["AllowedKeysDict", "FairnessViolation"]
