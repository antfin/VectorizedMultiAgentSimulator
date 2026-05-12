"""ExperimentConfig and its section models — strict YAML/JSON parsing.

This module imports only stdlib + pydantic + yaml — no torch, no vmas, no
benchmarl, no streamlit, no boto3 (enforced by F1.12).

F7.7.D1: every numeric carries an explicit ``gt`` / ``ge`` / ``le`` so
nonsense values (negative seeds, zero ``max_iters``, ``minibatch_size``
larger than ``frames_per_batch``, ``device`` outside ``{cpu, cuda}``,
``experiment.id`` containing path-unfriendly characters) are rejected at
parse time with a clean Pydantic error. The registry-aware cross-field
checks (``scenario.type ∈ available_scenarios()`` etc.) live on the
top-level :class:`ExperimentConfig` so they only fire after Pydantic has
validated each section in isolation.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ._common import STRICT

#: ``experiment.id`` becomes a folder name (``<exp_id>_s<seed>__<timestamp>``)
#: so it must avoid path separators / shell metacharacters. Allowed:
#: alphanumerics + ``_`` + ``-``. Empty string forbidden via ``min_length``.
_EXPERIMENT_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"


class ExperimentSection(BaseModel):
    """Run identity (id, seed) and human-readable metadata (name, description)."""

    model_config = STRICT

    id: str = Field(min_length=1, pattern=_EXPERIMENT_ID_PATTERN)
    seed: int = Field(default=0, ge=0)
    name: str | None = None
    description: str | None = None


class ScenarioSection(BaseModel):
    """Scenario adapter selection (`type`) and its make_env parameters."""

    model_config = STRICT

    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class AlgorithmSection(BaseModel):
    """Algorithm adapter selection (`type`) and its hyperparameters."""

    model_config = STRICT

    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSection(BaseModel):
    """Training-loop orchestration: algorithm-agnostic knobs.

    Includes the universal knobs BenchMARL puts on its ExperimentConfig
    (``lr``, ``gamma``, ``frames_per_batch``, ``minibatch_size``,
    ``n_minibatch_iters``, ``share_policy_params``). Algorithm-specific
    knobs (``lmbda`` for PPO, ``entropy_coef``, ``clip_epsilon``, …) stay
    in ``cfg.algorithm.params``.

    F7.7.D1 cross-field invariant: ``minibatch_size <= frames_per_batch``.
    BenchMARL silently floors this in some code paths; we'd rather fail
    loudly at config parse than have downstream training behave oddly.
    """

    model_config = STRICT

    max_iters: int = Field(gt=0)
    num_envs: int = Field(default=1, gt=0)
    device: Literal["cpu", "cuda"] = "cpu"
    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, ge=0, le=1)
    frames_per_batch: int = Field(default=6000, gt=0)
    minibatch_size: int = Field(default=400, gt=0)
    n_minibatch_iters: int = Field(default=45, gt=0)
    share_policy_params: bool = True
    # F5.7: BenchMARL writes a checkpoint every N iters (sparse). Set to 0 to
    # disable interval checkpoints (final-only via checkpoint_at_end). Smoke
    # runs auto-disable in the adapter; this is the cadence for non-smoke runs.
    checkpoint_interval_iters: int = Field(default=10, ge=0)
    # F8.2.G: how many checkpoint snapshots BenchMARL retains on disk.
    # BenchMARL's default is 3, which silently overwrites earlier snapshots —
    # the ER1 dry-run lost the iter-125 peak (only kept iters 150 / 160 / 167).
    # Set high enough that an entire ~167-iter run keeps everything; F8.5.D's
    # best-checkpoint-policy callback needs the full history to find the peak.
    # Smoke runs ignore this (no checkpoints written).
    keep_checkpoints_num: int = Field(default=1000, gt=0)
    # Phase 5a aftermath: a 167-iter run writes 17 × 107 MiB = 1.83 GiB of
    # checkpoints, and a LERO run also dumps ~2.5 GiB of throwaway inner-loop
    # checkpoints. When the run finishes cleanly, almost none of these are
    # ever needed again (Streamlit replay + eval CLI need only the last one).
    # When this flag is true, post-success cleanup deletes every checkpoint
    # except the final one. False = current behaviour (keep all). LERO YAMLs
    # default to true; ER1-style YAMLs leave it off for back-compat.
    delete_intermediate_checkpoints_on_success: bool = False

    @model_validator(mode="after")
    def _minibatch_fits_in_batch(self) -> "TrainingSection":
        if self.minibatch_size > self.frames_per_batch:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) must be ≤ "
                f"frames_per_batch ({self.frames_per_batch})"
            )
        return self


class EvaluationSection(BaseModel):
    """Evaluation cadence (interval_iters) and scope (episodes per eval)."""

    model_config = STRICT

    interval_iters: int = Field(gt=0)
    episodes: int = Field(gt=0)


class RunnerSection(BaseModel):
    """Runner adapter selection — where the experiment executes.

    ``type`` is the *dispatch* layer (the Python class that wraps the
    submission), not what runs *inside* the host:

    - ``local`` → ``LocalRunner`` — runs the BenchMARL training loop in the
      current Python process. Use when training on the user's machine.
    - ``ovh``   → ``OvhRunner``   — shells out ``ovhai job run`` which boots
      an OVH container that then invokes ``LocalRunner`` *inside*. Use when
      training on OVH GPU nodes.

    The CLI ``--runner`` flag overrides this YAML field at submit time
    (rare; mostly useful for forcing a YAML to run locally for debugging).
    """

    model_config = STRICT

    type: Literal["local", "ovh"] = "local"
    params: dict[str, Any] = Field(default_factory=dict)


class StorageSection(BaseModel):
    """Storage adapter selection — where run outputs are written and read (fs, s3, ...)."""

    model_config = STRICT

    type: str = Field(default="fs", min_length=1)
    path: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RuntimeSection(BaseModel):
    """Optional infrastructure: runner + storage. Defaults are computed by ExperimentService."""

    model_config = STRICT

    runner: RunnerSection = Field(default_factory=RunnerSection)
    storage: StorageSection


class LlmSection(BaseModel):
    """LLM client configuration — broker + model + cost cap (F9.0).

    Required when ``cfg.lero is not None``. Decoupled from :class:`LeroSection`
    so a future feature that calls an LLM outside the LERO loop (e.g., a
    reasoning agent at evaluation time) can reuse the same config block
    without coupling to LERO's evolutionary semantics.
    """

    model_config = STRICT

    #: LiteLLM-style model id (e.g. ``"gpt-4o-mini"``, ``"openai/my-model"``,
    #: ``"claude-sonnet-4-6"``). LiteLLM resolves the provider from the prefix.
    model: str = Field(min_length=1)
    #: Optional API base URL — for OVH-hosted endpoints / Ollama / other
    #: OpenAI-compatible providers. ``None`` means use the default for the
    #: detected provider; the actual API key always flows from env vars
    #: (OPENAI_API_KEY / ANTHROPIC_API_KEY / …) and is never written to the
    #: config to keep YAMLs commit-safe.
    api_base: str | None = None
    #: Sampling temperature. ``0.0`` = deterministic, ``1.0`` = paper-default
    #: for LERO inner-loop exploration.
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    #: Max tokens per completion. Defaults to 4096 — comfortably above
    #: rendezvous_comm's typical 1500-token responses.
    max_tokens: int = Field(default=4096, gt=0)
    #: Optional integer seed for OpenAI-compatible reproducibility. The
    #: LERO orchestrator derives a per-(iteration, candidate) seed from
    #: this base so sibling candidates aren't textually identical.
    seed: int | None = None
    #: Hard € ceiling on rolling 24-hour spend across ALL runs on this
    #: host. Enforced by the cost-cap decorator via a persistent ledger
    #: (default ``~/.multi_scenario/cost_ledger.jsonl``) so the cap holds
    #: across processes / sweeps — not just within one Python invocation.
    #: Locked € (not $) because the user's billing surface is European;
    #: LiteLLM costs are USD-native and converted via :attr:`usd_to_eur_rate`.
    cost_cap_per_day_eur: float = Field(default=10.0, gt=0)
    #: Hard € ceiling on rolling 30-day spend, same enforcement model.
    cost_cap_per_month_eur: float = Field(default=100.0, gt=0)
    #: USD → EUR conversion rate used for the rolling-window cap math.
    #: Default ~0.92 (Apr-2026 ECB reference); set explicitly in the
    #: experiment YAML when rates drift materially. Stored as USD/EUR so
    #: 1 USD ≈ 0.92 EUR.
    usd_to_eur_rate: float = Field(default=0.92, gt=0)
    #: When True, the disk cache is consulted before hitting the LLM and
    #: every successful call is written back. Default off so reproducibility
    #: experiments don't accidentally use stale entries.
    cache_enabled: bool = False


class LeroSection(BaseModel):
    """LERO evolutionary-loop configuration (F9.0).

    Optional on :class:`ExperimentConfig`; when present, ``cfg.llm`` is also
    required (enforced by :meth:`ExperimentConfig._lero_requires_llm`).
    """

    model_config = STRICT

    #: Prompt registry version (``v1``, ``v2``, ``v2_fewshot_k2_local``, …).
    #: Resolves to the templates under ``adapters/prompts/<prompt_version>/``.
    prompt_version: str = Field(default="v2_fewshot_k2_local", min_length=1)
    #: Number of evolutionary iterations (outer loop). Each iteration asks
    #: the LLM for ``n_candidates`` (reward, obs) function pairs.
    n_iterations: int = Field(default=4, gt=0)
    #: Candidates per iteration. rendezvous_comm Phase 4 used 3.
    n_candidates: int = Field(default=3, gt=0)
    #: Whether the LLM evolves the reward function. At least one of
    #: ``evolve_reward`` / ``evolve_observation`` must be True (validated below).
    evolve_reward: bool = True
    #: Whether the LLM evolves the observation enhancement function.
    evolve_observation: bool = True
    #: Reward magnitude clip ±N (post nan_to_num). ``None`` disables clipping
    #: (not recommended — LLM-generated rewards can hit ±1000 magnitude).
    reward_clip: float | None = Field(default=50.0, gt=0)
    #: Frames per LERO inner-loop candidate evaluation (1M is the
    #: rendezvous_comm Phase 4 default; full training at the end uses
    #: ``training.max_iters * training.frames_per_batch``).
    eval_frames_per_candidate: int = Field(default=1_000_000, gt=0)
    #: F9.7.A: ``False`` → InitialAndFeedbackComposer (production today).
    #: ``True`` → MetaPromptComposer (stub today; full Strategist+Editor+Critic
    #: lands post-F10.4 + post-experiments per F9.7.B).
    meta_prompting: bool = False
    #: When True, AllowedKeysDict raises on access to keys outside the
    #: per-mode whitelist (prevents oracle leakage in CTDE-fair runs).
    #: Locked True for reproducibility threads (rendezvous_comm S3b-local
    #: ran with strict-on); set False only for debugging.
    whitelist_strict: bool = True

    @model_validator(mode="after")
    def _at_least_one_evolution_target(self) -> "LeroSection":
        if not (self.evolve_reward or self.evolve_observation):
            raise ValueError(
                "LeroSection: at least one of evolve_reward / "
                "evolve_observation must be True"
            )
        return self


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration. Loaded from YAML; strictly validated."""

    model_config = STRICT

    experiment: ExperimentSection
    scenario: ScenarioSection
    algorithm: AlgorithmSection
    training: TrainingSection
    evaluation: EvaluationSection
    runtime: RuntimeSection | None = None
    lero: LeroSection | None = None
    llm: LlmSection | None = None

    @model_validator(mode="after")
    def _lero_requires_llm(self) -> "ExperimentConfig":
        """When ``cfg.lero`` is set, ``cfg.llm`` must be too — F9.0 invariant."""
        if self.lero is not None and self.llm is None:
            raise ValueError(
                "ExperimentConfig: cfg.lero requires cfg.llm to be set "
                "(LERO needs an LLM client). Add an llm: section to the YAML."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file.

        Schema-only validation. Registry-aware checks (``scenario.type ∈
        available_scenarios()`` etc.) live in
        :func:`multi_scenario.application.config_validation.validate_known_types`
        — they belong in the application layer because they cross the domain
        boundary into ``application.factories``. Entry points that need both
        call ``from_yaml`` then the validator, or use the
        ``application.config_loader.load_experiment_config`` wrapper.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
