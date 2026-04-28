"""Pydantic schemas for LERO-MP v4 — description-driven, multi-strategy,
stability-oriented meta-prompting.

Replaces v3's StrategyCard + EditorOutput + EditorCritique. The v4
loop emits N strategies in parallel per round (one prompt per
candidate) and selects winners by a stability-weighted score that
favors flat 10M-stable trajectories over peak-then-collapse.

DSPy migration path (v5): every class here mirrors a `dspy.Signature`
shape. Replacing the Pydantic base with `dspy.Module` is a one-line
change per class.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _coerce_to_str(v: Any) -> str:
    """Coerce a value to a string. LLMs at temperature=1.0 sometimes
    return dicts or lists where the schema expects a string (e.g.
    {"primary": "...", "secondary": "..."} for a metric description).
    Stringify them rather than fail validation."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _coerce_to_str_list(v: Any) -> list:
    """Coerce a value to a list of strings. Handles the common
    LLM-creative case of a dict-of-items being returned where a list
    was expected."""
    if v is None:
        return []
    if isinstance(v, list):
        return [_coerce_to_str(item) for item in v]
    if isinstance(v, dict):
        # {"item1": "...", "item2": "..."} → ["item1: ...", ...]
        return [
            f"{k}: {_coerce_to_str(val)}" for k, val in v.items()
        ]
    return [_coerce_to_str(v)]

# ── Phase 0: Bootstrap ──────────────────────────────────────────


class BootstrapCard(BaseModel):
    """The meta-LLM's structured interpretation of a problem description.

    Produced once per experiment by reading the user-provided
    description.md. Saved alongside ``bootstrap_thoughts.md`` (free-text
    reasoning) so a human can audit the LLM's understanding before any
    RL training starts.
    """

    model_config = ConfigDict(extra="forbid")

    task_summary: str = Field(
        description="2–3 sentence restatement of the task in the LLM's words.",
    )
    success_metric_understanding: str = Field(
        description="What M1 (or the primary metric) means for THIS task.",
    )
    key_difficulty: str = Field(
        description="What makes this task hard (1–2 sentences).",
    )
    failure_modes_anticipated: List[str] = Field(
        description="3–5 likely failure modes the LLM expects.",
        min_length=2, max_length=8,
    )
    high_level_strategies_considered: List[str] = Field(
        description=(
            "Ranked list of high-level strategies the LLM is considering. "
            "Each one names a coordination concept (not raw code), "
            "e.g. 'pre-compute a hold_signal for arrived agents'."
        ),
        min_length=2, max_length=8,
    )
    proposed_initial_obs_features: List[str] = Field(
        description=(
            "Named features the LLM proposes as the initial observation "
            "augmentation. Names should be concrete identifiers like "
            "'proximity_count' or 'gap_to_nearest_target'."
        ),
        default_factory=list,
    )
    proposed_initial_reward_components: List[str] = Field(
        description=(
            "Named reward components the LLM would consider. Empty if "
            "the LLM thinks the hand-crafted reward is sufficient."
        ),
        default_factory=list,
    )
    fairness_audit: str = Field(
        description=(
            "Confirms which state keys are allowed / forbidden, and "
            "whether the proposed features respect the fairness "
            "constraint."
        ),
    )
    assumptions: List[str] = Field(
        description=(
            "Things the LLM is assuming about the task that the human "
            "should verify (e.g. 'I assume agents are point particles')."
        ),
        default_factory=list,
    )

    # LLMs at temp=1.0 occasionally return dicts/lists where strings
    # are expected (e.g. {"primary": "...", "secondary": "..."}). The
    # validators stringify them instead of erroring out.
    @field_validator(
        "task_summary", "success_metric_understanding", "key_difficulty",
        "fairness_audit", mode="before",
    )
    @classmethod
    def _coerce_str_fields(cls, v):
        return _coerce_to_str(v)

    @field_validator(
        "failure_modes_anticipated", "high_level_strategies_considered",
        "proposed_initial_obs_features", "proposed_initial_reward_components",
        "assumptions", mode="before",
    )
    @classmethod
    def _coerce_list_fields(cls, v):
        return _coerce_to_str_list(v)


# ── Phase 1: Strategy emission ──────────────────────────────────


class StrategyV4(BaseModel):
    """One of N parallel strategies emitted per round.

    Each strategy produces ONE candidate (ONE inner-LLM call generating
    obs and/or reward code) under a prompt composed by overlaying
    ``slot_edits`` onto the bootstrap prompt.

    ``extra="ignore"`` because real LLMs (esp. gpt-5.4-mini at T=1.0)
    occasionally add metadata fields like ``slot_edits_note``. Strict
    rejection breaks otherwise-valid strategies. Required fields still
    enforced.
    """

    model_config = ConfigDict(extra="ignore")

    strategy_id: str = Field(description="'S1', 'S2', 'S3', ...")
    high_level_idea: str = Field(
        description="One-line headline for this strategy.",
    )
    target_domain: Literal["reward", "observation", "both"] = Field(
        description="What this strategy modifies.",
    )
    revert_to_baseline_reward: bool = Field(
        default=False,
        description=(
            "If True, force evolve_reward=false for this candidate "
            "regardless of base config — used when prior rounds with "
            "evolved reward showed peak-collapse or regression."
        ),
    )
    revert_reason: Optional[str] = Field(
        default=None,
        description=(
            "Required when revert_to_baseline_reward=True. Cite the "
            "prior round and the evidence."
        ),
    )
    slot_edits: Dict[str, str] = Field(
        description=(
            "Slot name -> new text. Empty dict means 'inherit base "
            "prompt unchanged'. Keys must be slots in the base "
            "prompt's meta.yaml (typically guidance_observation, "
            "guidance_reward, guidance_shared)."
        ),
        default_factory=dict,
    )
    expected_effect: str = Field(
        description="1–2 sentences: what behavior change this should produce.",
    )
    rationale: str = Field(
        description=(
            "2–3 sentences citing specific prior-round evidence "
            "(empty for round 0)."
        ),
    )

    @field_validator(
        "high_level_idea", "expected_effect", "rationale", mode="before",
    )
    @classmethod
    def _coerce_str_fields(cls, v):
        return _coerce_to_str(v)

    @field_validator("revert_reason", mode="before")
    @classmethod
    def _coerce_optional_str(cls, v):
        if v is None:
            return None
        return _coerce_to_str(v)


class StrategyBundle(BaseModel):
    """N strategies emitted together. The Strategist must justify why
    they cover the search space — not 3 minor variants of one idea."""

    model_config = ConfigDict(extra="ignore")

    round_idx: int = Field(ge=0)
    strategies: List[StrategyV4] = Field(
        min_length=1, max_length=10,
        description="One per inner candidate slot.",
    )
    diversity_rationale: str = Field(
        description=(
            "Why these N strategies cover distinct hypotheses rather "
            "than being minor variants. Cite which dimension each "
            "explores (reward shaping, obs feature space, ...)."
        ),
    )


# ── Phase 1: Per-candidate analysis ─────────────────────────────


ShapeTag = Literal[
    "monotonic_rise",
    "plateau",
    "oscillating",
    "late_ramp",
    "peak_collapse",
    "flat_zero",
    "flat_nonzero",
]


class CandidateAnalysis(BaseModel):
    """Trajectory analysis of ONE candidate run. Computed before
    handing results to the next-round Strategist.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    strategy_id: str
    final_M1: float
    final_M6: float
    peak_M1: float
    peak_at_frame: int
    slope_M6: float = Field(
        description="Linear-regression slope of M6 over frames (per 100k).",
    )
    noise_std_M1: float = Field(
        description="Std of M1 across last 5 evals.",
    )
    shape_tag: ShapeTag
    stability_score: float = Field(
        description=(
            "α·peak + β·final − γ·max(0, peak−final) + δ·M6. "
            "Higher = more stable + higher absolute performance."
        ),
    )
    qualitative_summary: str = Field(
        description="1–2 sentences for the next round's Strategist.",
    )


# ── Round + final results ───────────────────────────────────────


class RoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_idx: int = Field(ge=0)
    bundle: StrategyBundle
    candidates: List[CandidateAnalysis]
    best_strategy_id: str
    best_candidate_2M: CandidateAnalysis = Field(
        description="The best candidate then trained at mid_frames (2M).",
    )
    cross_round_summary: str = Field(
        description=(
            "Cross-strategy comparison text written by the analyzer "
            "for next round's Strategist."
        ),
    )


class V4Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bootstrap: BootstrapCard
    rounds: List[RoundResult]
    global_best_round_idx: int
    final_M1: float
    final_M6: float
    peak_M1: float
    peak_at_frame: int
    final_stability_score: float
    elapsed_seconds: float


# ── Fitness weights ─────────────────────────────────────────────


class FitnessWeights(BaseModel):
    """Configurable weights for stability_score."""

    model_config = ConfigDict(extra="forbid")

    peak: float = 0.3
    final: float = 0.7
    stability_penalty: float = 0.2
    m6_bonus: float = 0.05
