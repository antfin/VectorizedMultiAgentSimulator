"""Tests for meta/mutation.py — meta-prompt build, parsing, end-to-end.

Uses a stub meta-LLM callable so nothing talks to a real API.
"""

from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from src.lero.meta.failmode import FailMode
from src.lero.meta.mutation import (
    _is_fairness_restatement,
    build_meta_prompt,
    MAX_SLOT_CHARS,
    MIN_SLOT_CHARS,
    MutationParseError,
    MutationResult,
    parse_mutation_response,
    propose_new_template,
    SLOT_BEGIN,
    SLOT_END,
)
from src.lero.meta.provenance import load_meta, sha256_text
from src.lero.meta.trigger import TemplateRecord
from src.lero.prompts.loader import PromptLoader


# ── helpers ──────────────────────────────────────────────────────


def _rec(version, peak_m1, **kw):
    base = dict(
        template_version=version,
        inner_iter_count=3,
        best_peak_M1=peak_m1,
        best_final_M1=peak_m1,
        best_M6=0.8,
        best_M2=10.0,
        seed_M1_std=0.02,
        fail_mode=FailMode.HEALTHY,
    )
    base.update(kw)
    return TemplateRecord(**base)


def _canned_response(
    slot_body: str = "NEW GUIDANCE TEXT\n- Rule one\n- Rule two",
    rationale: str = "Add explicit bound on reward magnitude.",
    expected: str = "medium",
) -> str:
    return (
        f"Rationale: {rationale}\n"
        f"Expected-improvement: {expected}\n"
        f"\n"
        f"{SLOT_BEGIN}\n"
        f"{slot_body}\n"
        f"{SLOT_END}\n"
    )


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    """Parent template that mutation.py can edit (non-frozen guidance)."""
    parent = tmp_path / "p_root"
    parent.mkdir()
    (parent / "system.txt").write_text("sys")
    (parent / "feedback.txt").write_text("fb")
    (parent / "guidance.txt").write_text("original guidance\n")
    (parent / "examples.txt").write_text("original examples\n")
    (parent / "fairness.txt").write_text("FAIRNESS RULES\n")
    h = sha256_text("FAIRNESS RULES\n")
    (parent / "meta.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "p_root",
                "initial_user_slots": [
                    {"name": "guidance", "file": "guidance.txt"},
                    {"name": "examples", "file": "examples.txt"},
                    {"name": "fairness", "file": "fairness.txt", "frozen": True},
                ],
                "frozen_hashes": {"fairness": h},
            },
            sort_keys=False,
        )
    )
    return tmp_path


# ── build_meta_prompt ───────────────────────────────────────────


class TestBuildMetaPrompt:
    def test_contains_required_sections(self):
        hist = [_rec("v1", 0.2), _rec("v2", 0.6)]
        p = build_meta_prompt(
            parent_version="p_root",
            target_slot="guidance",
            history=hist,
            top_candidates=[
                {
                    "M1_success_rate": 0.6,
                    "M2_avg_return": 8.0,
                    "M6_coverage_progress": 0.9,
                    "reward_code": "def compute_reward(s): return s",
                }
            ],
            fail_mode=FailMode.REWARD_HACK,
            loader=None,  # unit-test mode
        )
        for marker in [
            "HARD CONSTRAINTS",
            "frozen",
            "guidance",
            "OBJECTIVE",
            "peak-M1",
            "HISTORY",
            "CONTRASTIVE",
            "CURRENT `guidance` SLOT",
            "TOP CANDIDATES",
            # New sections from the 2026-04-22 redesign
            "DIAGNOSIS",
            "REFERENCE TECHNIQUES",
            "CANDIDATE-AGGREGATE STATS",
            "DO NOT RESTATE",
            SLOT_BEGIN,
            SLOT_END,
            FailMode.REWARD_HACK.value,
        ]:
            assert marker in p, f"missing {marker!r} in prompt"

    def test_history_sorted_ascending_by_peak_m1(self):
        hist = [_rec("bestv", 0.9), _rec("midv", 0.5), _rec("worstv", 0.1)]
        p = build_meta_prompt(
            parent_version="p_root",
            target_slot="guidance",
            history=hist,
            top_candidates=[],
            fail_mode=FailMode.PLATEAU
            if hasattr(FailMode, "PLATEAU")
            else FailMode.HEALTHY,
            loader=None,
        )
        # Worst should appear above best in the emitted block.
        worst_idx = p.index("worstv")
        best_idx = p.index("bestv")
        assert worst_idx < best_idx

    def test_with_real_loader_inlines_current_slot(self, tmp_prompts, monkeypatch):
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)
        loader = PromptLoader("p_root")
        p = build_meta_prompt(
            parent_version="p_root",
            target_slot="guidance",
            history=[],
            top_candidates=[],
            fail_mode=FailMode.HEALTHY,
            loader=loader,
        )
        assert "original guidance" in p  # inlined
        assert sha256_text("FAIRNESS RULES\n")[:12] in p  # fairness-hash prefix shown


# ── parse_mutation_response ─────────────────────────────────────


class TestParseMutationResponse:
    def test_well_formed_response(self):
        body = "do this\nand that"
        resp = _canned_response(slot_body=body, rationale="because", expected="large")
        new_slot, rationale, expected = parse_mutation_response(resp, "guidance")
        assert new_slot == body + "\n"
        assert "because" in rationale
        assert expected == "large"

    def test_missing_delimiters_raises(self):
        bad = "Rationale: something\nExpected-improvement: small\n\nno delimiters here"
        with pytest.raises(MutationParseError) as exc:
            parse_mutation_response(bad, "guidance")
        assert SLOT_BEGIN in str(exc.value)

    def test_too_short_raises(self):
        resp = _canned_response(slot_body="x")  # 1 char
        with pytest.raises(MutationParseError) as exc:
            parse_mutation_response(resp, "guidance")
        assert "suspiciously small" in str(exc.value)

    def test_too_long_raises(self):
        huge = "x" * (MAX_SLOT_CHARS + 1)
        resp = _canned_response(slot_body=huge)
        with pytest.raises(MutationParseError) as exc:
            parse_mutation_response(resp, "guidance")
        assert "size limit" in str(exc.value)

    def test_missing_rationale_is_warning_not_fatal(self):
        resp = f"{SLOT_BEGIN}\n{'ok content here'}\n{SLOT_END}\n"
        new_slot, rationale, expected = parse_mutation_response(resp, "guidance")
        assert "ok content here" in new_slot
        assert "no rationale" in rationale.lower()
        assert expected == "unspecified"

    def test_expected_improvement_case_insensitive(self):
        resp = (
            "Rationale: r\nExpected-improvement: LARGE\n"
            + f"\n{SLOT_BEGIN}\nvalid enough content\n{SLOT_END}\n"
        )
        _, _, expected = parse_mutation_response(resp, "guidance")
        assert expected == "large"

    def test_trailing_newline_normalized(self):
        resp = _canned_response(slot_body="no trailing newline")
        # Response has body+"\n" already, but parse ensures at least one.
        new_slot, _, _ = parse_mutation_response(resp, "guidance")
        assert new_slot.endswith("\n")

    def test_min_chars_boundary(self):
        # Exactly MIN_SLOT_CHARS characters should pass.
        body = "x" * MIN_SLOT_CHARS
        resp = _canned_response(slot_body=body)
        new_slot, _, _ = parse_mutation_response(resp, "guidance")
        assert len(new_slot.rstrip("\n")) == MIN_SLOT_CHARS


class TestFairnessRestatementGuard:
    """Regression for 2026-04-22: 3/3 quick-run seeds produced
    near-identical 'use local sensors, clamp |r|<=50' outputs that
    just paraphrased the fairness slot. Reject those at parse time."""

    def test_generic_fairness_paraphrase_rejected(self):
        # The actual terse output from seed 0 of the 2026-04-22 run.
        paraphrase = (
            "Use only local sensors and received messages in observations; "
            "never reference global/oracle state. Keep tensors shape-stable "
            "across agents and timesteps, and ensure all rewards are finite "
            "and clipped to the configured bound. Prefer simple, monotonic "
            "shaping terms that encourage coverage progress without large "
            "spikes or terminal-only bonuses."
        )
        assert _is_fairness_restatement(paraphrase) is True
        resp = _canned_response(slot_body=paraphrase)
        with pytest.raises(MutationParseError) as exc:
            parse_mutation_response(resp, "guidance")
        assert "paraphrase of the fairness slot" in str(exc.value)

    def test_targeted_feature_edit_passes(self):
        targeted = (
            "Derive per-agent features from lidar_targets: nearest and "
            "second-nearest distances, plus the gap (nearest - 2nd-nearest) "
            "to signal isolated vs contested targets. Add a proximity_count "
            "feature: number of lidar rays with distance below covering_range. "
            "Combine target_near with agent_near into a hold_signal so the "
            "arriving agent waits for a partner instead of overshooting."
        )
        assert _is_fairness_restatement(targeted) is False
        new_slot, _, _ = parse_mutation_response(
            _canned_response(slot_body=targeted),
            "guidance",
        )
        assert "hold_signal" in new_slot

    def test_long_paraphrase_with_specific_feature_passes(self):
        # Real-world compromise: an edit that DOES restate fairness
        # but also introduces ≥1 concrete feature name should pass.
        mixed = (
            "Keep logic local — only lidar_targets, lidar_agents, agent_pos, "
            "agent_vel, and messages are safe at execution time; avoid "
            "oracle state. Clamp |r|<=50. ALSO: add a hold_signal flag "
            "(target_near AND agent_near) so arriving agents wait for "
            "partners."
        )
        assert _is_fairness_restatement(mixed) is False


# ── propose_new_template (end-to-end, stubbed LLM) ──────────────


class TestProposeNewTemplate:
    def test_happy_path_materializes_new_version(self, tmp_prompts, monkeypatch):
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)

        calls: List[List[Dict[str, str]]] = []

        def stub_llm(messages):
            calls.append(messages)
            return _canned_response(
                slot_body="bounded rewards only; no monotonic growth",
                rationale="Parent reward-hacked; add bound.",
                expected="large",
            )

        result = propose_new_template(
            parent_version="p_root",
            target_slot="guidance",
            history=[_rec("p_root", 0.3, fail_mode=FailMode.REWARD_HACK)],
            top_candidates=[],
            fail_mode=FailMode.REWARD_HACK,
            meta_llm_call=stub_llm,
            outer_iter=1,
            prompts_dir=tmp_prompts,
            generated_by="stub",
        )

        assert isinstance(result, MutationResult)
        assert result.new_version == "p_root_mp_001"
        assert result.target_slot == "guidance"
        assert "bounded rewards" in result.new_slot_content
        assert result.expected_improvement == "large"

        # Filesystem side-effects
        new_dir = tmp_prompts / "p_root_mp_001"
        assert new_dir.exists()
        assert (new_dir / "guidance.txt").read_text() == result.new_slot_content
        # Fairness slot preserved verbatim
        assert (new_dir / "fairness.txt").read_text() == "FAIRNESS RULES\n"
        # Provenance recorded
        meta = load_meta("p_root_mp_001", tmp_prompts)
        assert meta["parent"] == "p_root"
        assert meta["mutation"]["target_slots"] == ["guidance"]
        assert "add bound" in meta["mutation"]["rationale"].lower()
        assert meta["provenance"]["generated_by"] == "stub"

        # Meta-LLM was called exactly once with a system + user turn.
        assert len(calls) == 1
        assert len(calls[0]) == 2
        assert calls[0][0]["role"] == "system"
        assert calls[0][1]["role"] == "user"

    def test_refuses_frozen_slot_target(self, tmp_prompts, monkeypatch):
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)

        def stub_llm(_messages):
            return _canned_response(slot_body="this should never be applied")

        with pytest.raises(ValueError) as exc:
            propose_new_template(
                parent_version="p_root",
                target_slot="fairness",  # the FROZEN slot
                history=[],
                top_candidates=[],
                fail_mode=FailMode.FAIRNESS_VIOLATION,
                meta_llm_call=stub_llm,
                outer_iter=1,
                prompts_dir=tmp_prompts,
            )
        assert "frozen" in str(exc.value).lower()
        # No partial write — version dir not created.
        assert not (tmp_prompts / "p_root_mp_001").exists()

    def test_parse_error_propagates(self, tmp_prompts, monkeypatch):
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)

        def stub_llm(_messages):
            return "no delimiters here at all"

        with pytest.raises(MutationParseError):
            propose_new_template(
                parent_version="p_root",
                target_slot="guidance",
                history=[],
                top_candidates=[],
                fail_mode=FailMode.HEALTHY,
                meta_llm_call=stub_llm,
                outer_iter=1,
                prompts_dir=tmp_prompts,
            )

    def test_collision_bumps_outer_iter(self, tmp_prompts, monkeypatch):
        """If p_root_mp_001 exists from a prior run, the next call
        should pick _002 instead of raising FileExistsError."""
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)

        def stub_llm(_messages):
            return _canned_response(slot_body="new content for the slot")

        # First call claims _001.
        r1 = propose_new_template(
            parent_version="p_root",
            target_slot="guidance",
            history=[],
            top_candidates=[],
            fail_mode=FailMode.HEALTHY,
            meta_llm_call=stub_llm,
            outer_iter=1,
            prompts_dir=tmp_prompts,
        )
        assert r1.new_version == "p_root_mp_001"

        # Second call with same outer_iter — must bump, not crash.
        r2 = propose_new_template(
            parent_version="p_root",
            target_slot="guidance",
            history=[],
            top_candidates=[],
            fail_mode=FailMode.HEALTHY,
            meta_llm_call=stub_llm,
            outer_iter=1,
            prompts_dir=tmp_prompts,
        )
        assert r2.new_version == "p_root_mp_002"

    def test_resulting_template_renders(self, tmp_prompts, monkeypatch):
        """End-to-end: mutation output is consumable by PromptLoader."""
        from src.lero.prompts import loader as loader_mod

        monkeypatch.setattr(loader_mod, "_PROMPTS_DIR", tmp_prompts)

        def stub_llm(_messages):
            return _canned_response(slot_body="MUTATED guidance v2\n- clamp rewards")

        result = propose_new_template(
            parent_version="p_root",
            target_slot="guidance",
            history=[],
            top_candidates=[],
            fail_mode=FailMode.HEALTHY,
            meta_llm_call=stub_llm,
            outer_iter=3,
            prompts_dir=tmp_prompts,
        )
        loader = PromptLoader(result.new_version)
        rendered = loader.render("initial_user.txt")
        assert "MUTATED guidance v2" in rendered
        assert "FAIRNESS RULES" in rendered  # frozen slot preserved
        assert "original examples" in rendered  # un-edited slot copied
