"""LERO configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Uses LiteLLM model naming: "provider/model" or just "model".
    Examples:
        model: "anthropic/claude-sonnet-4-20250514"   # Anthropic
        model: "gpt-4o"                                # OpenAI
        model: "openai/my-model"                       # OpenAI-compatible
          api_base: "https://my-ovh-endpoint.com/v1"   # custom endpoint

    API keys are read from environment variables by default:
        ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
    Or pass api_key explicitly in config for custom endpoints.

    Future DSPy migration: this config maps to dspy.settings(lm=...).
    """

    # Model identifier (LiteLLM format: "provider/model" or just "model")
    model: str = "gpt-5.4-mini"
    # v3 default: 1.0 (was 0.8). Structured outputs + retry loop make
    # temperature-for-diversity the preferred design — content variance
    # comes from *different inputs* (different candidates, different
    # mutation history), not from same-input/different-sampling.
    temperature: float = 1.0
    # Max output tokens. None = use provider default.
    max_tokens: Optional[int] = None

    # Custom endpoint (for OVH or any OpenAI-compatible API)
    # None = use provider's default endpoint
    api_base: Optional[str] = None

    # API key override (None = read from environment variable)
    api_key: Optional[str] = None

    # Context window size in tokens (prompt + completion must fit).
    # None = auto-detect from LiteLLM's model registry.
    # Set explicitly for custom endpoints where auto-detect fails.
    context_window: Optional[int] = None

    # Retry / robustness
    max_retries: int = 3
    retry_delay: float = 2.0

    # Prompt versioning — track which prompts produced which results
    prompt_version: str = "v1"


@dataclass
class LeroConfig:
    """LERO evolutionary loop parameters."""

    # Evolution
    n_iterations: int = 4
    n_candidates: int = 3
    top_k: int = 2

    # Short training budget for candidate evaluation
    eval_frames: int = 1_000_000
    eval_episodes: int = 100

    # Full training budget for the winning candidate
    full_frames: int = 10_000_000

    # What to evolve
    evolve_reward: bool = True
    evolve_observation: bool = True

    # Reward mode: "replace" (paper — LLM designs full reward) or
    # "bonus" (R = R_original + bonus_scale * tanh(LLM_bonus))
    reward_mode: str = "replace"

    # Observation state: "global" (paper — full state including all
    # agent/target positions) or "local" (CTDE — only own sensors)
    obs_state_mode: str = "global"

    # Bonus normalization (only used when reward_mode="bonus"):
    # bonus = bonus_scale * tanh(raw_bonus)
    bonus_scale: float = 0.5

    # Clip LLM reward output to [-reward_clip, +reward_clip] before passing
    # to PPO. NaN/inf are first replaced with 0. Disable by setting to None
    # or a very large number (e.g. 1e9). Default 50.0 prevents PPO gradient
    # explosion observed when LLM produces large-magnitude rewards (M2 in
    # ±200..±1000 range), which causes NaN actions ~70-90% into 10M training.
    # NOTE: This deviates from the LERO paper's "raw rewards" — the paper's
    # MPE Simple Spread had naturally bounded rewards [0,5]; Discovery does
    # not, hence the need for a soft cap.
    reward_clip: Optional[float] = 50.0

    # Fairness guard (LERO-MP §4.3). When True AND obs_state_mode=="local",
    # the state dict passed to enhance_observation is wrapped in an
    # AllowedKeysDict — forbidden-key lookups raise FairnessViolation.
    # No-op in "global" mode. Defaults to False for backward compatibility
    # with existing LERO configs; LERO-MP configs set it True.
    whitelist_strict: bool = False


@dataclass
class MetaPromptTrigger:
    """When the outer (meta-prompt) loop decides to emit a new template."""

    # Plateau: best_peak_M1 hasn't moved by ≥ plateau_delta for N inner iters.
    plateau_iters: int = 2
    plateau_delta: float = 0.03
    # Seed-variance ceiling at the best candidate (Tier-1).
    variance_threshold: float = 0.15
    # Reward-hack guard: Tier-2 peak-M1 minus final-M1 > this → hack.
    peak_vs_final_gap_max: float = 0.20
    # Hard cooldown regardless of triggers.
    cooldown_inner_iters: int = 3


@dataclass
class MetaPromptBudget:
    """Cost ceilings for the outer loop."""

    max_outer_iters: int = 3
    max_total_inner_candidates: int = 200
    # Tier-2 (full training) is run only for templates whose Tier-1 best
    # is within this gap of the current champion.
    tier2_promotion_gap: float = 0.05


@dataclass
class MetaPromptFairness:
    """Fairness policy at the outer-loop level."""

    whitelist_strict: bool = True
    # If set to a non-null string, the run is a diagnostic one that is
    # explicitly allowed to cheat on observation information (e.g. an
    # oracle-comparison baseline). The string is logged into final
    # metrics for provenance.
    waiver: Optional[str] = None


@dataclass
class MetaPromptConfig:
    """Outer loop that evolves prompt templates across inner runs.

    When ``enabled=False`` (default), LERO behaves exactly as before —
    no outer loop, no meta-LLM calls, no fairness whitelist.

    See docs/lero_metaprompt_plan.md for the full design.
    """

    enabled: bool = False

    # Meta-LLM that rewrites prompt slots. May differ from the
    # inner-loop LLMConfig. Kept loose (dict) to avoid tight coupling —
    # the runner instantiates an LLMClient from these fields.
    meta_model: str = "claude-opus-4-7"
    # v3 default: 1.0 (was 0.3). See LLMConfig.temperature for rationale.
    meta_temperature: float = 1.0
    meta_api_base: Optional[str] = None
    # LLM-call cache mode for reproducibility replay (LERO-MP v3 §5.1).
    # "off" | "read_write" | "read_only" | "write_only". Env override:
    # LERO_LLM_CACHE_MODE. Default off so fresh runs don't accidentally
    # replay stale cached responses.
    llm_cache: str = "off"
    llm_cache_dir: Optional[str] = None

    trigger: MetaPromptTrigger = field(default_factory=MetaPromptTrigger)
    budget: MetaPromptBudget = field(default_factory=MetaPromptBudget)
    fairness: MetaPromptFairness = field(default_factory=MetaPromptFairness)

    # Seeds used at Tier-1 (screen) and Tier-2 (confirm). Three seeds
    # is the minimum to detect reward-hacking that is seed-specific.
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Slot-picking policy for the meta-LLM. "failmode_taxonomy" uses
    # the detected dominant failure mode (see meta/failmode.py). Other
    # options: "round_robin", "fixed:<slot_name>".
    slot_policy: str = "failmode_taxonomy"

    # v2 two-level pipeline (Strategist + Editor). When True, the outer
    # loop calls a separate "strategist" LLM to pick (target_domain,
    # target_slot, focus, avoid) before the Editor rewrites the slot.
    # Also turns on the cross-run mutation_log.jsonl memory. See
    # docs/lero_metaprompt_v2_plan.md for the full design. False keeps
    # the v1 single-call behavior as a rollback path.
    two_level_meta: bool = False
