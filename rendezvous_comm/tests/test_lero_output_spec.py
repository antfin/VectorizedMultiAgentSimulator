"""Tests for conditional output_spec rendering (LERO-MP v3 §3.2)."""

from __future__ import annotations

import pytest

from src.lero.prompts.loader import PromptLoader


@pytest.fixture
def loader():
    return PromptLoader(version="v2_fewshot_modular_v2")


def _render(loader, variant):
    return loader.render(
        "initial_user.txt",
        output_spec_variant=variant,
        n_agents=4, n_targets=4, agents_per_target=2,
        covering_range=0.25, lidar_range=0.35, max_steps=200,
        collision_penalty=-0.01, time_penalty=-0.01,
        n_lidar_rays_entities=15, n_lidar_rays_agents=12,
        agent_lidar_description="", comm_description="",
        reward_description="", coordination_guidance="",
        comm_state_description="", obs_lidar_agents="",
        obs_comm_state="", comm_obs_guidance="",
        scenario_reward_code="", scenario_observation_code="",
        experiment_context="",
    )


def test_both_variant_includes_reward_and_obs(loader):
    text = _render(loader, "both")
    assert "def compute_reward(" in text
    assert "def enhance_observation(" in text


def test_obs_only_variant_drops_reward(loader):
    text = _render(loader, "obs_only")
    # Count the "Generate Functions" header — obs-only variant uses
    # "Generate Function" (singular) instead of the plural.
    assert "## Generate Function" in text
    # Explicit directive
    assert "Do NOT output a" in text
    # Raw output_spec path should not be used
    raw_obs_only = loader.load_raw("output_spec_obs_only.txt")
    assert raw_obs_only in text
    raw_both = loader.load_raw("output_spec_both.txt")
    assert raw_both not in text


def test_reward_only_variant_drops_obs(loader):
    text = _render(loader, "reward_only")
    assert "Do NOT output an" in text
    raw_reward_only = loader.load_raw("output_spec_reward_only.txt")
    assert raw_reward_only in text
    raw_both = loader.load_raw("output_spec_both.txt")
    assert raw_both not in text


def test_default_falls_back_to_output_spec_txt(loader):
    """When variant is None, legacy output_spec.txt is used."""
    text = _render(loader, None)
    # output_spec.txt in v2_fewshot_modular_v2 includes BOTH
    assert "def compute_reward(" in text
    assert "def enhance_observation(" in text


def test_unknown_variant_falls_back_to_default(loader):
    """Unknown variants don't crash; they fall back to legacy."""
    text = _render(loader, "nonsense")
    assert "def compute_reward(" in text
