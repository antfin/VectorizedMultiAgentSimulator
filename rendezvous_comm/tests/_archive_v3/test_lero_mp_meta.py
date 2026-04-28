"""Tests for LERO-MP meta-loop modules: failmode, trigger, provenance."""

from pathlib import Path

import pytest
import yaml

from src.lero.meta.failmode import (
    SLOT_POLICY_MAP,
    FailMode,
    FailModeThresholds,
    classify_inner_result,
    pick_slot_to_edit,
)
from src.lero.meta.provenance import (
    lineage,
    load_meta,
    materialize_mutation,
    propose_version_name,
    recompute_frozen_hashes,
    sha256_text,
    write_frozen_hashes,
)
from src.lero.meta.trigger import (
    TemplateRecord,
    TriggerConfig,
    TriggerDecision,
    TriggerReason,
    should_meta_iterate,
)
from src.lero.prompts.loader import FrozenSlotMismatch, PromptLoader


# ── failmode ─────────────────────────────────────────────────────

def _ok(**kw):
    """Healthy candidate metrics."""
    base = dict(
        M1_success_rate=0.5, M2_avg_return=10.0,
        M6_coverage_progress=0.8,
    )
    base.update(kw)
    return base


class TestClassifyInnerResult:
    def test_healthy_returns_healthy(self):
        cands = [_ok(), _ok(M1_success_rate=0.6)]
        assert classify_inner_result(cands) == FailMode.HEALTHY

    def test_fairness_violation_dominates_everything(self):
        cands = [
            _ok(),
            {"_error": "oracle access", "_error_type": "FairnessViolation"},
            {"_error": "NaN actions", "_error_type": "NaNAction"},
        ]
        assert classify_inner_result(cands) == FailMode.FAIRNESS_VIOLATION

    def test_nan_crash_from_error_type(self):
        cands = [{"_error": "assert not isnan()", "_error_type": "NaNAction"}]
        assert classify_inner_result(cands) == FailMode.NAN_CRASH

    def test_nan_crash_from_error_string_fallback(self):
        cands = [{"_error": "RuntimeError: NaN detected"}]
        assert classify_inner_result(cands) == FailMode.NAN_CRASH

    def test_dim_mismatch_detected(self):
        cands = [{"_error": "KeyError: 'lidar_targets'"}]
        assert classify_inner_result(cands) == FailMode.DIM_MISMATCH

    def test_reward_hack_from_peak_vs_final(self):
        tier2 = {"peak_M1": 0.86, "final_M1": 0.09}
        assert classify_inner_result([_ok()], tier2_metrics=tier2) == FailMode.REWARD_HACK

    def test_reward_hack_below_threshold_is_healthy(self):
        tier2 = {"peak_M1": 0.60, "final_M1": 0.55}  # gap = 0.05
        assert classify_inner_result([_ok()], tier2_metrics=tier2) == FailMode.HEALTHY

    def test_reward_magnitude_inflation(self):
        history = [
            {"best_M2": 100.0, "best_M1": 0.05},
            {"best_M2": 500.0, "best_M1": 0.05},
        ]
        cands = [_ok(M2_avg_return=6000.0, M1_success_rate=0.05)]
        assert (
            classify_inner_result(cands, template_history=history)
            == FailMode.REWARD_MAGNITUDE_INFLATION
        )

    def test_inflation_not_fired_when_history_baseline_near_zero(self):
        """Regression for 2026-04-21: with history best_M2 ≈ 0 (short
        runs) the old code divided by 1e-6 and always fired a false
        positive. Require a real baseline floor."""
        history = [
            {"best_M2": 0.0, "best_M1": 0.0},
            {"best_M2": 0.0, "best_M1": 0.0},
        ]
        cands = [_ok(M2_avg_return=-1.5, M1_success_rate=0.0)]
        # |M2|=1.5 vs baseline 0 should NOT flag inflation; |M2|=1.5
        # is below the inflation_m2_floor=1.0 baseline requirement.
        assert (
            classify_inner_result(cands, template_history=history)
            != FailMode.REWARD_MAGNITUDE_INFLATION
        )

    def test_inflation_floor_blocks_tiny_baseline(self):
        """Even if the ratio looks big, a tiny absolute baseline must
        not count as real inflation evidence."""
        history = [
            {"best_M2": 0.3, "best_M1": 0.1},   # below floor of 1.0
            {"best_M2": 0.4, "best_M1": 0.1},
        ]
        # 2.0 vs 0.4 = 5× ratio, would fire if not floored
        cands = [_ok(M2_avg_return=2.0, M1_success_rate=0.1)]
        assert (
            classify_inner_result(cands, template_history=history)
            != FailMode.REWARD_MAGNITUDE_INFLATION
        )

    def test_inflation_ignored_when_m1_improved(self):
        """M2 growth is fine if M1 is growing with it."""
        history = [{"best_M2": 100.0, "best_M1": 0.05}]
        cands = [_ok(M2_avg_return=500.0, M1_success_rate=0.9)]
        # Still not HEALTHY necessarily, but not INFLATION.
        assert (
            classify_inner_result(cands, template_history=history)
            != FailMode.REWARD_MAGNITUDE_INFLATION
        )

    def test_stuck_detection(self):
        cands = [_ok(
            M1_success_rate=0.05,
            M1_per_seed=[0.04, 0.05, 0.06],   # σ ≈ 0.008 < 0.05
        )]
        assert classify_inner_result(cands) == FailMode.STUCK

    def test_stuck_not_fired_when_variance_high(self):
        cands = [_ok(
            M1_success_rate=0.05,
            M1_per_seed=[0.0, 0.5, 0.0],       # σ high → not stuck
        )]
        assert classify_inner_result(cands) != FailMode.STUCK

    def test_priority_fairness_over_nan(self):
        """Even if NaN is present, fairness wins."""
        cands = [
            {"_error": "nan", "_error_type": "NaNAction"},
            {"_error": "fairness", "_error_type": "FairnessViolation"},
        ]
        assert classify_inner_result(cands) == FailMode.FAIRNESS_VIOLATION


class TestPickSlotToEdit:
    def test_every_failmode_has_a_slot(self):
        """Must be exhaustive so the outer loop never crashes on lookup."""
        for mode in FailMode:
            assert mode in SLOT_POLICY_MAP
            assert isinstance(SLOT_POLICY_MAP[mode], str)
            assert SLOT_POLICY_MAP[mode] != "fairness"  # never edit frozen

    def test_taxonomy_policy_maps_reward_hack_to_guidance(self):
        assert pick_slot_to_edit(FailMode.REWARD_HACK) == "guidance"

    def test_round_robin_cycles(self):
        order = []
        history: list = []
        for _ in range(6):
            slot = pick_slot_to_edit(
                FailMode.HEALTHY, history=history, policy="round_robin",
            )
            order.append(slot)
            history.append(slot)
        # After cycling through all 5 we should land back on the first.
        assert len(set(order[:5])) == 5
        assert order[5] == order[0]

    def test_fixed_policy(self):
        assert pick_slot_to_edit(
            FailMode.HEALTHY, policy="fixed:examples"
        ) == "examples"

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            pick_slot_to_edit(FailMode.HEALTHY, policy="mystery")


class TestFailModeThresholds:
    def test_thresholds_are_frozen(self):
        t = FailModeThresholds()
        with pytest.raises(Exception):
            t.peak_vs_final_gap = 0.9  # type: ignore[misc]


# ── trigger ──────────────────────────────────────────────────────

def _rec(**kw):
    base = dict(
        template_version="v",
        inner_iter_count=3,
        best_peak_M1=0.5,
        best_final_M1=0.5,
        best_M6=0.8, best_M2=10.0,
        seed_M1_std=0.02,
        fail_mode=FailMode.HEALTHY,
    )
    base.update(kw)
    return TemplateRecord(**base)


class TestShouldMetaIterate:
    cfg = TriggerConfig()

    def test_empty_history_returns_initial(self):
        d = should_meta_iterate([], 0, self.cfg)
        assert d.should_iterate is True
        assert d.should_stop is False
        assert d.reason is TriggerReason.INITIAL

    def test_budget_exceeded_candidates(self):
        d = should_meta_iterate([_rec()], 300, self.cfg)
        assert d.should_stop is True
        assert d.reason is TriggerReason.BUDGET_EXCEEDED

    def test_budget_exceeded_outer_iters(self):
        d = should_meta_iterate(
            [_rec(), _rec(), _rec()], 10, self.cfg,
        )
        assert d.should_stop is True
        assert d.reason is TriggerReason.BUDGET_EXCEEDED

    def test_fairness_repeated_aborts(self):
        hist = [
            _rec(fail_mode=FailMode.FAIRNESS_VIOLATION),
            _rec(fail_mode=FailMode.FAIRNESS_VIOLATION),
        ]
        d = should_meta_iterate(hist, 10, self.cfg)
        assert d.should_stop is True
        assert d.reason is TriggerReason.FAIRNESS_REPEATED

    def test_cycle_detected(self):
        hist = [
            _rec(mutation_target_slot="guidance", mutation_rationale="cap magnitude"),
            _rec(mutation_target_slot="guidance", mutation_rationale="cap magnitude"),
            _rec(mutation_target_slot="guidance", mutation_rationale="cap magnitude"),
        ]
        # Skip budget check so cycle has a chance to fire
        cfg = TriggerConfig(max_outer_iters=100)
        d = should_meta_iterate(hist, 10, cfg)
        assert d.should_stop is True
        assert d.reason is TriggerReason.CYCLE_DETECTED

    def test_converged_stops_loop(self):
        # Four records with flat peak_M1 → converged
        hist = [
            _rec(best_peak_M1=0.80),
            _rec(best_peak_M1=0.805),
            _rec(best_peak_M1=0.81),
            _rec(best_peak_M1=0.81),
        ]
        cfg = TriggerConfig(
            max_outer_iters=10, converged_iters=3, converged_delta=0.02,
        )
        d = should_meta_iterate(hist, 10, cfg, inner_iters_since_last_mutation=99)
        assert d.should_stop is True
        assert d.reason is TriggerReason.CONVERGED

    def test_reward_hack_fires_iterate(self):
        hist = [_rec(best_peak_M1=0.86, best_final_M1=0.09)]
        d = should_meta_iterate(hist, 5, self.cfg, inner_iters_since_last_mutation=5)
        assert d.should_iterate is True
        assert d.should_stop is False
        assert d.reason is TriggerReason.REWARD_HACK

    def test_seed_instability_fires_iterate(self):
        hist = [_rec(seed_M1_std=0.20)]
        d = should_meta_iterate(hist, 5, self.cfg, inner_iters_since_last_mutation=5)
        assert d.should_iterate is True
        assert d.reason is TriggerReason.SEED_INSTABILITY

    def test_plateau_fires_iterate(self):
        hist = [
            _rec(best_peak_M1=0.50),
            _rec(best_peak_M1=0.501),
            _rec(best_peak_M1=0.502),
        ]
        cfg = TriggerConfig(max_outer_iters=100)
        d = should_meta_iterate(hist, 5, cfg, inner_iters_since_last_mutation=5)
        assert d.should_iterate is True
        assert d.reason is TriggerReason.PLATEAU

    def test_cooldown_blocks_even_when_trigger_fires(self):
        hist = [_rec(best_peak_M1=0.86, best_final_M1=0.09)]  # hack
        d = should_meta_iterate(
            hist, 5, self.cfg, inner_iters_since_last_mutation=0,
        )
        assert d.should_iterate is False
        assert d.should_stop is False
        assert d.reason is TriggerReason.COOLDOWN

    def test_fail_mode_cluster(self):
        # Peak-M1 clearly improving so CONVERGED doesn't pre-empt.
        hist = [
            _rec(best_peak_M1=0.1, fail_mode=FailMode.REWARD_HACK),
            _rec(best_peak_M1=0.3, fail_mode=FailMode.REWARD_HACK),
            _rec(best_peak_M1=0.5, fail_mode=FailMode.REWARD_HACK),
            _rec(best_peak_M1=0.7, fail_mode=FailMode.HEALTHY),
        ]
        cfg = TriggerConfig(max_outer_iters=100)
        d = should_meta_iterate(hist, 5, cfg, inner_iters_since_last_mutation=5)
        assert d.should_iterate is True
        assert d.reason is TriggerReason.FAIL_MODE_CLUSTER

    def test_no_trigger_returns_continue(self):
        """Healthy history, cooldown clear, no plateau — do nothing."""
        hist = [
            _rec(best_peak_M1=0.3),
            _rec(best_peak_M1=0.5),  # +0.2 improvement; clear not-plateau
        ]
        d = should_meta_iterate(hist, 5, self.cfg, inner_iters_since_last_mutation=5)
        assert d.should_iterate is False
        assert d.should_stop is False


class TestTriggerDecision:
    def test_keep_going_is_inverse_of_stop(self):
        d1 = TriggerDecision(True, False, TriggerReason.PLATEAU)
        d2 = TriggerDecision(False, True, TriggerReason.BUDGET_EXCEEDED)
        assert d1.keep_going is True
        assert d2.keep_going is False


# ── provenance ───────────────────────────────────────────────────

class TestSha256Text:
    def test_deterministic(self):
        assert sha256_text("hello") == sha256_text("hello")
        assert sha256_text("a") != sha256_text("b")
        # Length sanity
        assert len(sha256_text("x")) == 64


class TestProposeVersionName:
    def test_format(self):
        assert propose_version_name("v2_fewshot_modular", 1) == "v2_fewshot_modular_mp_001"
        assert propose_version_name("foo", 999) == "foo_mp_999"


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    """A fresh prompts directory with a single parent template."""
    parent = tmp_path / "parent_v1"
    parent.mkdir()
    (parent / "system.txt").write_text("system")
    (parent / "initial_user.txt").write_text("whatever")  # unused monolithic
    (parent / "feedback.txt").write_text("fb")
    (parent / "guidance.txt").write_text("guidance original\n")
    (parent / "examples.txt").write_text("examples original\n")
    (parent / "fairness.txt").write_text("FROZEN RULES\n")
    frozen_hash = sha256_text("FROZEN RULES\n")
    (parent / "meta.yaml").write_text(yaml.safe_dump({
        "version": "parent_v1",
        "initial_user_slots": [
            {"name": "guidance", "file": "guidance.txt"},
            {"name": "examples", "file": "examples.txt"},
            {"name": "fairness", "file": "fairness.txt", "frozen": True},
        ],
        "frozen_hashes": {"fairness": frozen_hash},
    }, sort_keys=False))
    return tmp_path


class TestLoadMeta:
    def test_loads_meta_yaml(self, tmp_prompts):
        meta = load_meta("parent_v1", tmp_prompts)
        assert meta["version"] == "parent_v1"
        assert meta["initial_user_slots"][0]["name"] == "guidance"

    def test_missing_returns_empty(self, tmp_prompts):
        assert load_meta("does_not_exist", tmp_prompts) == {}


class TestRecomputeFrozenHashes:
    def test_hashes_only_frozen_slots(self, tmp_prompts):
        hashes = recompute_frozen_hashes("parent_v1", tmp_prompts)
        assert list(hashes.keys()) == ["fairness"]
        assert hashes["fairness"] == sha256_text("FROZEN RULES\n")


class TestWriteFrozenHashes:
    def test_persists_hashes(self, tmp_prompts):
        write_frozen_hashes("parent_v1", {"fairness": "newhash"}, tmp_prompts)
        meta = load_meta("parent_v1", tmp_prompts)
        assert meta["frozen_hashes"]["fairness"] == "newhash"


class TestMaterializeMutation:
    def test_copies_parent_and_applies_edits(self, tmp_prompts):
        new_dir = materialize_mutation(
            parent_version="parent_v1",
            new_version="parent_v1_mp_001",
            slot_edits={"guidance": "NEW guidance content\n"},
            rationale="Add anti-hack constraint.",
            generated_by="test",
            prompts_dir=tmp_prompts,
        )
        assert new_dir.exists()
        # Edited slot updated
        assert (new_dir / "guidance.txt").read_text() == "NEW guidance content\n"
        # Un-edited slot copied verbatim
        assert (new_dir / "examples.txt").read_text() == "examples original\n"
        # Fairness slot preserved byte-for-byte
        assert (new_dir / "fairness.txt").read_text() == "FROZEN RULES\n"
        # meta.yaml records provenance
        meta = load_meta("parent_v1_mp_001", tmp_prompts)
        assert meta["version"] == "parent_v1_mp_001"
        assert meta["parent"] == "parent_v1"
        assert meta["mutation"]["target_slots"] == ["guidance"]
        assert "Add anti-hack" in meta["mutation"]["rationale"]
        assert "generated_at" in meta["provenance"]
        # Frozen hash propagated
        assert meta["frozen_hashes"]["fairness"] == sha256_text("FROZEN RULES\n")

    def test_rejects_edit_of_frozen_slot(self, tmp_prompts):
        with pytest.raises(ValueError) as excinfo:
            materialize_mutation(
                parent_version="parent_v1",
                new_version="parent_v1_mp_002",
                slot_edits={"fairness": "hacked content"},
                rationale="trying to cheat",
                prompts_dir=tmp_prompts,
            )
        assert "frozen" in str(excinfo.value).lower()
        assert not (tmp_prompts / "parent_v1_mp_002").exists()  # no partial write

    def test_rejects_edit_of_undeclared_slot(self, tmp_prompts):
        with pytest.raises(ValueError):
            materialize_mutation(
                parent_version="parent_v1",
                new_version="parent_v1_mp_003",
                slot_edits={"no_such_slot": "x"},
                rationale="typo",
                prompts_dir=tmp_prompts,
            )

    def test_rejects_existing_version_name(self, tmp_prompts):
        materialize_mutation(
            parent_version="parent_v1",
            new_version="parent_v1_mp_004",
            slot_edits={"guidance": "A"},
            rationale="first",
            prompts_dir=tmp_prompts,
        )
        with pytest.raises(FileExistsError):
            materialize_mutation(
                parent_version="parent_v1",
                new_version="parent_v1_mp_004",
                slot_edits={"guidance": "B"},
                rationale="second",
                prompts_dir=tmp_prompts,
            )

    def test_rejects_missing_parent(self, tmp_prompts):
        with pytest.raises(FileNotFoundError):
            materialize_mutation(
                parent_version="nonexistent",
                new_version="child",
                slot_edits={},
                rationale="",
                prompts_dir=tmp_prompts,
            )

    def test_materialized_version_loads_and_renders(self, tmp_prompts, monkeypatch):
        """End-to-end: materialized version is consumable by PromptLoader
        and its frozen-hash guard still works."""
        materialize_mutation(
            parent_version="parent_v1",
            new_version="parent_v1_mp_005",
            slot_edits={"guidance": "tightened guidance\n"},
            rationale="bound magnitudes",
            prompts_dir=tmp_prompts,
        )
        # Point the loader's root at our tmp.
        from src.lero.prompts import loader as loader_mod
        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)
        loader = PromptLoader("parent_v1_mp_005")
        out = loader.render("initial_user.txt")
        assert "tightened guidance" in out
        assert "FROZEN RULES" in out
        assert "examples original" in out
        # Frozen hash still pinned: tamper raises
        (tmp_prompts / "parent_v1_mp_005" / "fairness.txt").write_text("TAMPER")
        loader2 = PromptLoader("parent_v1_mp_005")
        with pytest.raises(FrozenSlotMismatch):
            loader2.render("initial_user.txt")


class TestLineage:
    def test_chain(self, tmp_prompts):
        materialize_mutation(
            "parent_v1", "parent_v1_mp_001",
            {"guidance": "a\n"}, rationale="r1",
            prompts_dir=tmp_prompts,
        )
        materialize_mutation(
            "parent_v1_mp_001", "parent_v1_mp_002",
            {"guidance": "b\n"}, rationale="r2",
            prompts_dir=tmp_prompts,
        )
        chain = lineage("parent_v1_mp_002", tmp_prompts)
        assert chain == [
            "parent_v1_mp_002", "parent_v1_mp_001", "parent_v1",
        ]

    def test_root_template_has_length_one(self, tmp_prompts):
        assert lineage("parent_v1", tmp_prompts) == ["parent_v1"]

    def test_missing_version_returns_empty_or_self(self, tmp_prompts):
        # No meta.yaml → returns [version] because load_meta returns {}.
        chain = lineage("does_not_exist", tmp_prompts)
        assert chain == ["does_not_exist"]
