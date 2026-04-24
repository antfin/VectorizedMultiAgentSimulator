"""Pydantic schemas for LLM-structured outputs (LERO-MP v3).

These schemas are the **typed surface** between the inner-LLM candidate
generation, the Strategist, the Editor, and the Critic. They replace
regex parsing of free-form LLM output with OpenAI-compatible
`response_format={"type": "json_schema", ...}` validation.

Design notes:
  - Every schema used as an LLM response_format MUST have `extra="forbid"`
    (OpenAI strict mode rejects unknown keys).
  - Optional fields with None defaults are serialized carefully — strict
    mode requires every field to be `required` in the generated JSON
    schema, so we use `Optional[...] = None` + `exclude_none=True`
    when dumping for tests.
  - The classes also work with regex-extracted payloads (see
    `from_free_text` helpers) so we can migrate incrementally: v3.0
    ships validated models + regex fallback, v3.1 flips structured
    outputs on by default.

v4 migration: `InnerLLMOutput` → `dspy.Signature` with these fields as
`OutputField`. The schema is shape-identical.
"""

from __future__ import annotations

from typing import List, Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Pydantic is required for LERO-MP v3. "
        "Install: pip install 'pydantic>=2.0'"
    ) from e


# ── Level 0: Inner LLM (code generator) ──────────────────────────


class InnerLLMOutput(BaseModel):
    """What the inner LLM returns when asked to generate candidate code.

    The fields reflect which functions the template asked for
    (conditional on ``evolve_reward`` / ``evolve_observation`` —
    see the Step 2 output_spec variants). When a function wasn't
    requested, the corresponding field is None.
    """

    model_config = ConfigDict(extra="forbid")

    obs_code: Optional[str] = Field(
        default=None,
        description=(
            "Complete enhance_observation(scenario_state: dict) -> "
            "torch.Tensor function body. Required when "
            "evolve_observation=True."
        ),
    )
    reward_code: Optional[str] = Field(
        default=None,
        description=(
            "Complete compute_reward(scenario_state: dict) -> "
            "torch.Tensor function body. Required when "
            "evolve_reward=True."
        ),
    )
    rationale: Optional[str] = Field(
        default=None,
        description="1–2 sentence summary of what this candidate encodes.",
    )


# ── Level 1: Strategist ──────────────────────────────────────────


SignalTier = Literal["scalar", "fingerprint", "curve_shape"]


class StrategyCard(BaseModel):
    """Strategist's decision (what slot to edit, with what focus).

    v3 additions: ``include_signals`` + ``signal_rationale`` give the
    Strategist an explicit noise-control knob over Editor + inner-loop
    feedback. Default is ``["scalar"]`` — only Tier-1 scalars are
    forwarded unless the Strategist adds Tier-2 or Tier-3.
    """

    model_config = ConfigDict(extra="forbid")

    target_domain: Literal["reward", "observation", "shared", "both"]
    target_slot: Literal[
        "guidance_shared", "guidance_reward", "guidance_observation",
    ]
    focus: List[str] = Field(
        default_factory=list,
        description="1–2 specific patterns/features to encourage.",
        max_length=3,
    )
    avoid: List[str] = Field(
        default_factory=list,
        description="Patterns that scored regression/collapse previously.",
        max_length=5,
    )
    confidence: Literal["small", "medium", "large"] = "medium"
    rationale: str = Field(
        default="",
        description=(
            "2–4 sentences citing specific evidence (record name, "
            "verdict, feature delta)."
        ),
    )
    include_signals: List[SignalTier] = Field(
        default_factory=lambda: ["scalar"],
        description=(
            "Tiers of behavioral feedback to forward to the Editor and "
            "next-iteration inner-loop feedback. Keep minimal unless "
            "evidence says otherwise."
        ),
    )
    signal_rationale: Optional[str] = Field(
        default=None,
        description=(
            "1 sentence justifying include_signals if it deviates from "
            "['scalar']."
        ),
    )


# ── Level 2: Editor ──────────────────────────────────────────────


class EditorOutput(BaseModel):
    """Editor's proposed rewrite of a single sub-slot."""

    model_config = ConfigDict(extra="forbid")

    new_slot_content: str = Field(
        description=(
            "The complete new text of the target slot. Concise, "
            "specific, non-overlapping with the fairness slot."
        ),
    )
    rationale: str = Field(
        default="",
        description="1–3 sentences mapping the Strategist focus to this edit.",
    )
    expected_improvement: Literal["small", "medium", "large"] = "small"


# ── Level 2.5: Critic (TextGrad-style self-review) ────────────────


class EditorCritique(BaseModel):
    """Second-pass review of the Editor's output.

    Drives a 1–2 round critique-revise loop in
    ``meta/critique.py::critique_and_revise``. When overall_quality
    is ``keep`` the Editor output is accepted; ``revise`` triggers
    re-invocation of the Editor with these notes appended; ``reject``
    raises MutationParseError (outer loop graceful-stops).
    """

    model_config = ConfigDict(extra="forbid")

    addresses_focus: bool
    addresses_focus_reason: str
    cites_specific_features: List[str] = Field(
        default_factory=list,
        description=(
            "Feature identifiers (e.g. 'lidar_targets', 'hold_signal', "
            "'gap', 'proximity_count') explicitly named in the Editor's "
            "new_slot_content."
        ),
    )
    has_fairness_restatement: bool
    has_fairness_restatement_reason: str
    diverges_from_priors: bool
    suggested_edits: List[str] = Field(
        default_factory=list,
        description="Specific text-level changes the Editor should make.",
        max_length=5,
    )
    suggested_signal_change: Literal[
        "keep", "add_fingerprint", "drop_fingerprint",
        "add_curve_shape", "drop_curve_shape",
    ] = "keep"
    overall_quality: Literal["keep", "revise", "reject"]
