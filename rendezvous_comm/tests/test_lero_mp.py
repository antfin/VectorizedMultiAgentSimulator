"""Tests for LERO-MP additions.

Covers every symbol added in the meta-prompt extension:
  - PromptLoader slot assembly + frozen-hash guard + introspection
  - meta.fairness.AllowedKeysDict + FairnessViolation
  - scenario_patch._maybe_wrap_obs_state (the integration hook)
  - LeroConfig.whitelist_strict default + propagation
  - MetaPromptConfig + nested trigger/budget/fairness dataclasses
  - src.config.load_experiment parsing of the ``meta_prompt`` section
  - Parity of configs/lero_mp/mp_k2_obsonly_cr035.yaml vs ER1 cr035

These tests are cheap (no GPU, no LLM, no VMAS env) and safe to run in
the existing pytest suite.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from src.config import load_experiment
from src.lero.config import (
    LeroConfig,
    MetaPromptBudget,
    MetaPromptConfig,
    MetaPromptFairness,
    MetaPromptTrigger,
)
from src.lero.meta.fairness import (
    AllowedKeysDict,
    FairnessViolation,
    LOCAL_ALLOWED_KEYS,
    LOCAL_FORBIDDEN_KEYS,
)
from src.lero.prompts.loader import FrozenSlotMismatch, PromptLoader
from src.lero.scenario_patch import _maybe_wrap_obs_state, make_patched_scenario_class


# ── PromptLoader: slot assembly + frozen-hash guard ─────────────


class TestPromptLoaderBackwardCompat:
    """Existing monolithic prompts (v1/v2/v2_fewshot) must still render."""

    def test_v2_fewshot_renders_without_slots(self):
        loader = PromptLoader("v2_fewshot")
        out = loader.render(
            "initial_user.txt",
            n_agents=4,
            n_targets=4,
            agents_per_target=2,
            covering_range=0.35,
            scenario_reward_code="# R",
            scenario_observation_code="# O",
            collision_penalty=-0.01,
            time_penalty=-0.01,
        )
        assert "4 agents must cover 4 targets" in out
        # Monolithic templates have no slots declared.
        assert loader.slot_names() == []
        assert loader.frozen_slot_names() == []


class TestPromptLoaderSlotAssembly:
    """LERO-MP extension: slot-decomposed v2_fewshot_modular."""

    @pytest.fixture
    def loader(self):
        return PromptLoader("v2_fewshot_modular")

    def test_slot_names_in_declared_order(self, loader):
        names = loader.slot_names()
        # Assembly order matters — we want fairness AFTER state_schema
        # (so the LLM sees the available keys first, then the rules).
        assert names == [
            "task_context",
            "current_code",
            "state_schema",
            "fairness",
            "guidance",
            "examples",
            "output_spec",
        ]

    def test_frozen_slot_names_lists_only_fairness(self, loader):
        assert loader.frozen_slot_names() == ["fairness"]

    def test_slot_text_returns_individual_slot(self, loader):
        fairness = loader.slot_text("fairness")
        assert "Fairness constraint" in fairness
        assert "agents_pos" in fairness  # names the forbidden key

    def test_slot_text_unknown_slot_raises(self, loader):
        with pytest.raises(KeyError):
            loader.slot_text("does_not_exist")

    def test_render_assembles_slots(self, loader):
        out = loader.render(
            "initial_user.txt",
            n_agents=4,
            n_targets=4,
            agents_per_target=2,
            covering_range=0.35,
            scenario_reward_code="# R",
            scenario_observation_code="# O",
            collision_penalty=-0.01,
            time_penalty=-0.01,
        )
        # Content from multiple slots must all appear in the output.
        assert "4 agents must cover 4 targets" in out  # task_context
        assert "scenario_state = {" in out  # state_schema
        assert "Fairness constraint" in out  # fairness
        assert "### Example A" in out  # examples
        assert "def compute_reward" in out  # output_spec

    def test_render_differs_from_v2_fewshot_only_in_fairness(self):
        """The slot-decomposed version must be behavior-identical to
        v2_fewshot except for the intentional fairness slot addition.
        """
        ctx = dict(
            n_agents=4,
            n_targets=4,
            agents_per_target=2,
            covering_range=0.35,
            scenario_reward_code="# R",
            scenario_observation_code="# O",
            collision_penalty=-0.01,
            time_penalty=-0.01,
        )
        mono = PromptLoader("v2_fewshot").render("initial_user.txt", **ctx)
        modu = PromptLoader("v2_fewshot_modular").render(
            "initial_user.txt",
            **ctx,
        )
        # The only added block is the fairness section.
        fairness_slot = PromptLoader("v2_fewshot_modular").slot_text("fairness")
        # Strip the added fairness block from modu and compare.
        stripped = modu.replace(fairness_slot, "")
        assert stripped == mono


class TestFrozenSlotHashGuard:
    """Tampering with a frozen slot must be caught before render."""

    SLOT_PATH = (
        Path(__file__).parent.parent
        / "src"
        / "lero"
        / "prompts"
        / "v2_fewshot_modular"
        / "fairness.txt"
    )

    def test_tamper_raises_frozen_slot_mismatch(self):
        original = self.SLOT_PATH.read_text()
        try:
            self.SLOT_PATH.write_text(original + "TAMPER")
            # New loader instance picks up the modified file.
            loader = PromptLoader("v2_fewshot_modular")
            with pytest.raises(FrozenSlotMismatch) as excinfo:
                loader.render(
                    "initial_user.txt",
                    n_agents=4,
                    n_targets=4,
                    agents_per_target=2,
                    covering_range=0.35,
                    scenario_reward_code="# R",
                    scenario_observation_code="# O",
                    collision_penalty=-0.01,
                    time_penalty=-0.01,
                )
            assert "fairness" in str(excinfo.value)
        finally:
            self.SLOT_PATH.write_text(original)

    def test_empty_pinned_hash_does_not_guard(self, tmp_path, monkeypatch):
        """When frozen_hashes is empty for a slot, loader does NOT check."""
        version_dir = tmp_path / "tmpl"
        version_dir.mkdir()
        (version_dir / "slot_a.txt").write_text("hello\n")
        (version_dir / "meta.yaml").write_text(
            yaml.safe_dump(
                {
                    "initial_user_slots": [
                        {"name": "slot_a", "file": "slot_a.txt", "frozen": True},
                    ],
                    "frozen_hashes": {"slot_a": ""},  # empty → not pinned
                }
            )
        )
        # Point the loader's template dir at our temp version.
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_path)
        loader = PromptLoader("tmpl")
        out = loader.render("initial_user.txt")
        assert out.strip() == "hello"  # no raise

    def test_pinned_hash_matching_content_passes(self, tmp_path, monkeypatch):
        version_dir = tmp_path / "tmpl2"
        version_dir.mkdir()
        body = "frozen content\n"
        (version_dir / "slot.txt").write_text(body)
        pinned = hashlib.sha256(body.encode("utf-8")).hexdigest()
        (version_dir / "meta.yaml").write_text(
            yaml.safe_dump(
                {
                    "initial_user_slots": [
                        {"name": "slot", "file": "slot.txt", "frozen": True},
                    ],
                    "frozen_hashes": {"slot": pinned},
                }
            )
        )
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_path)
        loader = PromptLoader("tmpl2")
        assert loader.render("initial_user.txt") == body


# ── meta.fairness: AllowedKeysDict ───────────────────────────────


class TestAllowedKeysDict:
    def _state(self):
        return {
            "agent_pos": "pos",
            "agent_vel": "vel",
            "agent_idx": 0,
            "lidar_targets": "lt",
            "n_agents": 4,
            # Forbidden values present so we can verify they are gated.
            "agents_pos": "ORACLE",
            "covered_targets": "ORACLE",
            "targets_pos": "ORACLE",
        }

    def test_allowed_key_returns_value(self):
        d = AllowedKeysDict(self._state())
        assert d["agent_pos"] == "pos"
        assert d["lidar_targets"] == "lt"

    def test_forbidden_key_raises_fairness_violation(self):
        d = AllowedKeysDict(self._state())
        for forbidden in ["agents_pos", "covered_targets", "targets_pos"]:
            with pytest.raises(FairnessViolation) as excinfo:
                d[forbidden]
            assert forbidden in str(excinfo.value)

    def test_fairness_violation_is_key_error_subclass(self):
        """Existing `except KeyError` patterns still catch violations."""
        assert issubclass(FairnessViolation, KeyError)

    def test_unknown_key_raises_plain_key_error(self):
        d = AllowedKeysDict(self._state())
        with pytest.raises(KeyError) as excinfo:
            d["not_a_real_key"]
        # Must NOT be a FairnessViolation — that's reserved for oracle access.
        assert not isinstance(excinfo.value, FairnessViolation)

    def test_allowed_but_missing_raises_plain_key_error(self):
        """e.g. lidar_agents absent when use_agent_lidar=False."""
        d = AllowedKeysDict({"agent_pos": "p"})  # lidar_agents absent
        with pytest.raises(KeyError) as excinfo:
            d["lidar_agents"]
        assert not isinstance(excinfo.value, FairnessViolation)
        assert "not present" in str(excinfo.value)

    def test_contains(self):
        d = AllowedKeysDict(self._state())
        assert "agent_pos" in d
        assert "agents_pos" not in d  # forbidden
        assert "totally_unknown" not in d

    def test_iter_and_len_only_expose_allowed(self):
        d = AllowedKeysDict(self._state())
        keys = set(iter(d))
        assert keys == {
            "agent_pos",
            "agent_vel",
            "agent_idx",
            "lidar_targets",
            "n_agents",
        }
        assert len(d) == 5

    def test_get_returns_default_on_unknown(self):
        d = AllowedKeysDict(self._state())
        assert d.get("not_a_key", "fallback") == "fallback"

    def test_get_reraises_fairness_violation(self):
        """.get() must NOT swallow a fairness violation — that would be
        the silent-cheating bug we're defending against.
        """
        d = AllowedKeysDict(self._state())
        with pytest.raises(FairnessViolation):
            d.get("targets_pos", "fallback")

    def test_key_sets_are_disjoint(self):
        assert not (LOCAL_ALLOWED_KEYS & LOCAL_FORBIDDEN_KEYS)


# ── scenario_patch integration hook ─────────────────────────────


class TestMaybeWrapObsState:
    """Verifies the wrap-or-passthrough decision without needing VMAS."""

    def test_local_strict_wraps(self):
        state = {"agent_pos": "p", "targets_pos": "ORACLE"}
        out = _maybe_wrap_obs_state(state, mode="local", whitelist_strict=True)
        assert isinstance(out, AllowedKeysDict)
        with pytest.raises(FairnessViolation):
            out["targets_pos"]

    def test_local_non_strict_passthrough(self):
        """Paper-faithful behavior when whitelist_strict=False."""
        state = {"agent_pos": "p", "targets_pos": "ORACLE"}
        out = _maybe_wrap_obs_state(state, mode="local", whitelist_strict=False)
        assert out is state  # identity, no wrap
        assert out["targets_pos"] == "ORACLE"  # cheating allowed

    def test_global_mode_never_wraps(self):
        """Global mode = reward state, which may freely use oracle keys."""
        state = {"agent_pos": "p", "targets_pos": "ORACLE"}
        out = _maybe_wrap_obs_state(state, mode="global", whitelist_strict=True)
        assert out is state
        assert out["targets_pos"] == "ORACLE"


class TestMakePatchedScenarioClassAcceptsNewKwarg:
    """Smoke-test: the new whitelist_strict kwarg is accepted without
    raising, including when no LLM sources are provided."""

    def test_accepts_whitelist_strict_true(self):
        cls = make_patched_scenario_class(whitelist_strict=True)
        assert cls.__name__ == "PatchedDiscoveryScenario"

    def test_accepts_whitelist_strict_false(self):
        cls = make_patched_scenario_class(whitelist_strict=False)
        assert cls.__name__ == "PatchedDiscoveryScenario"


# ── LeroConfig additive field ────────────────────────────────────


class TestLeroConfigWhitelistStrict:
    def test_default_is_false_for_backward_compat(self):
        assert LeroConfig().whitelist_strict is False

    def test_accepts_true(self):
        assert LeroConfig(whitelist_strict=True).whitelist_strict is True

    def test_existing_defaults_unchanged(self):
        """Adding whitelist_strict must not shift any existing default."""
        c = LeroConfig()
        assert c.n_iterations == 4
        assert c.n_candidates == 3
        assert c.reward_clip == 50.0
        assert c.obs_state_mode == "global"  # paper default
        assert c.reward_mode == "replace"


# ── MetaPromptConfig and nested dataclasses ─────────────────────


class TestMetaPromptConfig:
    def test_defaults(self):
        mp = MetaPromptConfig()
        assert mp.enabled is False  # opt-in
        assert mp.meta_model == "claude-opus-4-7"
        assert mp.seeds == [0, 1, 2]  # 3-seed minimum (plan §6.3)
        assert mp.slot_policy == "failmode_taxonomy"

    def test_trigger_defaults(self):
        t = MetaPromptTrigger()
        assert t.plateau_iters == 2
        assert t.plateau_delta == pytest.approx(0.03)
        assert t.variance_threshold == pytest.approx(0.15)
        assert t.peak_vs_final_gap_max == pytest.approx(0.20)

    def test_budget_defaults(self):
        b = MetaPromptBudget()
        assert b.max_outer_iters == 3
        assert b.max_total_inner_candidates == 200
        assert b.tier2_promotion_gap == pytest.approx(0.05)

    def test_fairness_defaults(self):
        f = MetaPromptFairness()
        assert f.whitelist_strict is True  # default ON for meta-prompt
        assert f.waiver is None


# ── YAML loader integration ─────────────────────────────────────

CONFIGS = Path(__file__).parent.parent / "configs"


class TestLoadExperimentMetaPrompt:
    def test_dryrun_config_loads_with_meta_disabled(self):
        spec = load_experiment(CONFIGS / "lero_mp" / "mp_dryrun.yaml")
        assert spec.lero is not None
        assert spec.lero.whitelist_strict is True
        assert spec.lero.obs_state_mode == "local"
        assert spec.llm.prompt_version == "v2_fewshot_modular"
        assert spec.meta_prompt is not None
        assert spec.meta_prompt.enabled is False

    def test_real_config_parses_nested_sections(self):
        spec = load_experiment(CONFIGS / "lero_mp" / "mp_k2_obsonly_cr035.yaml")
        mp = spec.meta_prompt
        assert mp is not None
        assert mp.enabled is True
        assert isinstance(mp.trigger, MetaPromptTrigger)
        assert isinstance(mp.budget, MetaPromptBudget)
        assert isinstance(mp.fairness, MetaPromptFairness)
        assert mp.trigger.plateau_delta == pytest.approx(0.03)
        assert mp.budget.tier2_promotion_gap == pytest.approx(0.05)
        assert mp.fairness.whitelist_strict is True
        assert mp.fairness.waiver is None

    def test_non_lero_config_has_no_meta_prompt(self):
        """Existing ER1 configs must still load with meta_prompt=None."""
        spec = load_experiment(CONFIGS / "er1" / "single_al_lp_sr_cr035.yaml")
        assert spec.lero is None
        assert spec.meta_prompt is None


# ── ER1 parity check ────────────────────────────────────────────


class TestER1Parity:
    """For results to sit in the same comparison table as ER1/ER3 GNN,
    task params must match exactly. This test is the regression guard."""

    @pytest.mark.parametrize(
        "param",
        [
            "n_agents",
            "n_targets",
            "agents_per_target",
            "lidar_range",
            "covering_range",
            "use_agent_lidar",
            "n_lidar_rays_entities",
            "n_lidar_rays_agents",
            "targets_respawn",
            "shared_reward",
            "agent_collision_penalty",
            "covering_rew_coeff",
            "time_penalty",
            "max_steps",
        ],
    )
    def test_param_matches_er1_cr035(self, param):
        mp = load_experiment(CONFIGS / "lero_mp" / "mp_k2_obsonly_cr035.yaml")
        er1 = load_experiment(CONFIGS / "er1" / "single_al_lp_sr_cr035.yaml")
        assert getattr(mp.task, param) == getattr(er1.task, param), (
            f"LERO-MP task.{param} must match ER1 cr035 baseline for " f"comparability"
        )
