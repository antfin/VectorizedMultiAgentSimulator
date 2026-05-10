"""F9.0 — LeroSection / LlmSection contract + ExperimentConfig integration."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.domain.models import ExperimentConfig, LeroSection, LlmSection
from pydantic import ValidationError


# ── LlmSection ────────────────────────────────────────────────────────


def test_llm_section_minimal_construction():
    cfg = LlmSection(model="gpt-4o-mini")
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_base is None  # provider-default
    assert cfg.temperature == 1.0
    assert cfg.max_tokens == 4096
    # Time-window caps are EUR, rolling, persistent across processes.
    assert cfg.cost_cap_per_day_eur == 10.0
    assert cfg.cost_cap_per_month_eur == 100.0
    assert 0.5 < cfg.usd_to_eur_rate < 1.5  # plausible USD→EUR rate
    assert cfg.cache_enabled is False  # locked OFF default for reproducibility


@pytest.mark.parametrize(
    "field,bad",
    [
        ("model", ""),  # min_length
        ("temperature", -0.1),  # ge=0
        ("temperature", 2.1),  # le=2
        ("max_tokens", 0),  # gt=0
        ("cost_cap_per_day_eur", 0),  # gt=0
        ("cost_cap_per_month_eur", -1),  # gt=0
        ("usd_to_eur_rate", 0),  # gt=0
    ],
)
def test_llm_section_rejects_invalid_field_values(field: str, bad):
    payload = {"model": "gpt-4o-mini", field: bad}
    with pytest.raises(ValidationError):
        LlmSection.model_validate(payload)


def test_llm_section_strict_rejects_extra_keys():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LlmSection.model_validate({"model": "gpt-4o-mini", "rogue_key": "x"})


# ── LeroSection ────────────────────────────────────────────────────────


def test_lero_section_default_locked_invariants():
    """Defaults match the locked F9.0 decisions: meta_prompting OFF,
    whitelist_strict ON, reward_clip=±50, both evolution targets ON.
    """
    cfg = LeroSection()
    assert cfg.prompt_version == "v2_fewshot_k2_local"
    assert cfg.n_iterations == 4
    assert cfg.n_candidates == 3
    assert cfg.evolve_reward is True
    assert cfg.evolve_observation is True
    assert cfg.reward_clip == 50.0
    assert cfg.eval_frames_per_candidate == 1_000_000
    assert cfg.meta_prompting is False  # F9.7.A stub off by default
    assert cfg.whitelist_strict is True  # CTDE-fair default


def test_lero_section_requires_at_least_one_evolution_target():
    """With both flags False, the LERO loop has nothing to do."""
    with pytest.raises(ValidationError, match="at least one of"):
        LeroSection(evolve_reward=False, evolve_observation=False)


@pytest.mark.parametrize(
    "evolve_reward,evolve_observation",
    [(True, False), (False, True), (True, True)],
)
def test_lero_section_accepts_any_non_empty_evolution_combo(
    evolve_reward: bool, evolve_observation: bool
):
    cfg = LeroSection(
        evolve_reward=evolve_reward, evolve_observation=evolve_observation
    )
    assert cfg.evolve_reward == evolve_reward
    assert cfg.evolve_observation == evolve_observation


def test_lero_section_reward_clip_can_be_disabled():
    """``None`` disables clipping — documented but discouraged."""
    cfg = LeroSection(reward_clip=None)
    assert cfg.reward_clip is None


@pytest.mark.parametrize("bad", [-0.1, 0])
def test_lero_section_reward_clip_rejects_non_positive(bad):
    with pytest.raises(ValidationError):
        LeroSection(reward_clip=bad)


# ── ExperimentConfig integration: lero+llm cross-validator ─────────────


def _minimal_experiment_dict() -> dict:
    return {
        "experiment": {"id": "demo", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": "cpu"},
        "evaluation": {"interval_iters": 1, "episodes": 1},
    }


def test_experiment_config_without_lero_or_llm_is_valid():
    """Backwards-compat: existing baseline / smoke YAMLs don't carry these."""
    cfg = ExperimentConfig.model_validate(_minimal_experiment_dict())
    assert cfg.lero is None
    assert cfg.llm is None


def test_experiment_config_with_lero_requires_llm():
    payload = _minimal_experiment_dict() | {"lero": LeroSection().model_dump()}
    with pytest.raises(ValidationError, match="cfg.lero requires cfg.llm"):
        ExperimentConfig.model_validate(payload)


def test_experiment_config_with_lero_and_llm_is_valid():
    payload = _minimal_experiment_dict() | {
        "lero": LeroSection().model_dump(),
        "llm": {"model": "gpt-4o-mini"},
    }
    cfg = ExperimentConfig.model_validate(payload)
    assert cfg.lero is not None
    assert cfg.llm is not None
    assert cfg.llm.model == "gpt-4o-mini"


def test_experiment_config_with_only_llm_is_valid():
    """Future extension: an LLM client without LERO (e.g., reasoning agent at eval)."""
    payload = _minimal_experiment_dict() | {"llm": {"model": "gpt-4o-mini"}}
    cfg = ExperimentConfig.model_validate(payload)
    assert cfg.llm is not None
    assert cfg.lero is None
