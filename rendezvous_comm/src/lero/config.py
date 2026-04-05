"""LERO configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Optional


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
    temperature: float = 0.8
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
