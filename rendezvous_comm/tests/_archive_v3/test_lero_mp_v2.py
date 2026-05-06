"""Tests for LERO-MP v2: Strategist, mutation_log, two-level pipeline.

Nothing in this suite touches OVH, BenchMARL, or a real LLM. The
Strategist and Editor LLM calls are stubbed so the unit-test flow is
pure-Python.
"""

import pytest
import yaml

from src.lero.meta.failmode import FailMode
from src.lero.meta.mutation_log import (
    MutationLogEntry,
    append_entry,
    classify_verdict,
    new_entry,
    read_recent,
    summarize_for_prompt,
    update_last_entry_with_post,
)
from src.lero.meta.strategy import (
    SEED_STRATEGY_BIAS,
    StrategyCard,
    StrategyParseError,
    bias_for_seed,
    build_strategist_prompt,
    parse_strategy_card,
    strategize,
)
from src.lero.meta.trigger import TemplateRecord


# ── StrategyCard + parsing ─────────────────────────────────────


def _card_yaml(**overrides) -> str:
    base = {
        "target_domain": "observation",
        "target_slot": "guidance_observation",
        "focus": [
            "Add hold_signal from lidar proximity",
            "Split nearest vs 2nd-nearest gap",
        ],
        "avoid": ["Generic fairness restatement"],
        "confidence": "medium",
        "rationale": "Evidence from record 2 shows top-M6 candidate "
        "uses proximity_count but no hold flag.",
    }
    base.update(overrides)
    return "```yaml\n" + yaml.safe_dump(base) + "\n```"


class TestStrategyCardParsing:
    def test_valid_card(self):
        card = parse_strategy_card(_card_yaml())
        assert card.target_domain == "observation"
        assert card.target_slot == "guidance_observation"
        assert len(card.focus) == 2
        assert "hold_signal" in card.focus[0]
        assert card.confidence == "medium"

    def test_invalid_domain_rejected(self):
        with pytest.raises(StrategyParseError, match="target_domain"):
            parse_strategy_card(_card_yaml(target_domain="xxx"))

    def test_invalid_slot_rejected(self):
        with pytest.raises(StrategyParseError, match="target_slot"):
            parse_strategy_card(_card_yaml(target_slot="guidance_unknown"))

    def test_invalid_confidence_rejected(self):
        with pytest.raises(StrategyParseError, match="confidence"):
            parse_strategy_card(_card_yaml(confidence="huge"))

    def test_non_mapping_rejected(self):
        with pytest.raises(StrategyParseError, match="mapping"):
            parse_strategy_card("```yaml\n- just\n- a\n- list\n```")

    def test_accepts_raw_yaml_without_fence(self):
        """Strategists may forget the ```yaml fence."""
        card = parse_strategy_card(
            yaml.safe_dump(
                {
                    "target_domain": "reward",
                    "target_slot": "guidance_reward",
                    "focus": ["bounded shaping"],
                    "avoid": [],
                    "confidence": "small",
                    "rationale": "test",
                }
            )
        )
        assert card.target_domain == "reward"

    def test_focus_truncated_to_3(self):
        card = parse_strategy_card(_card_yaml(focus=["a", "b", "c", "d", "e"]))
        assert len(card.focus) == 3

    def test_empty_avoid_ok(self):
        card = parse_strategy_card(_card_yaml(avoid=[]))
        assert card.avoid == []


class TestSeedBias:
    def test_three_way_cycle(self):
        assert bias_for_seed(0) == "observation_first"
        assert bias_for_seed(1) == "reward_first"
        assert bias_for_seed(2) == "exploratory"
        assert bias_for_seed(3) == "observation_first"  # cycles

    def test_all_biases_valid(self):
        for b in SEED_STRATEGY_BIAS.values():
            assert isinstance(b, str) and b


# ── build_strategist_prompt content ────────────────────────────


def _rec(version="v", peak=0.01, fail=FailMode.HEALTHY):
    return TemplateRecord(
        template_version=version,
        inner_iter_count=0,
        best_peak_M1=peak,
        best_final_M1=peak,
        best_M6=0.15,
        best_M2=-1.0,
        seed_M1_std=0.0,
        fail_mode=fail,
    )


class TestBuildStrategistPrompt:
    def test_contains_sections(self):
        p = build_strategist_prompt(
            history=[_rec()],
            mutation_log_entries=[],
            top_candidates=[
                {
                    "M1_success_rate": 0.01,
                    "M2_avg_return": -1.0,
                    "M6_coverage_progress": 0.15,
                }
            ],
            seed_bias="observation_first",
            fail_mode=FailMode.HEALTHY,
            fairness_slot_excerpt="Local sensors only.",
        )
        for marker in (
            "SEED BIAS",
            "CURRENT-RUN HISTORY",
            "CROSS-RUN MUTATION LOG",
            "CANDIDATE-AGGREGATE STATS",
            "FAIRNESS CONTRACT",
            "target_domain",
            "target_slot",
            # Bias is surfaced as English hint, not raw tag
            "explore OBSERVATION feature engineering",
            "guidance_shared",
            "guidance_reward",
            "guidance_observation",
        ):
            assert marker in p, f"missing {marker!r}"

    def test_mutation_log_summary_included(self):
        # Build a resolved entry with a verdict
        e = MutationLogEntry(
            ts="2026-04-23T00:00:00Z",
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=1,
            parent_version="v1",
            new_version="v1_mp_001",
            strategy_card={
                "target_domain": "reward",
                "target_slot": "guidance_reward",
                "focus": ["bounded shaping"],
            },
            slot_name="guidance_reward",
            slot_content_sha256="x",
            slot_content_excerpt="x",
            pre_mutation_peak_M1=0.0,
            pre_mutation_best_M6=0.0,
            post_mutation_peak_M1=0.0,
            post_mutation_best_M6=-0.05,
            delta_peak_M1=0.0,
            delta_M6=-0.05,
            verdict="regression",
        )
        p = build_strategist_prompt(
            history=[_rec()],
            mutation_log_entries=[e],
            top_candidates=[],
            seed_bias="exploratory",
            fail_mode=FailMode.HEALTHY,
        )
        assert "v1_mp_001" in p
        assert "regression" in p


class TestStrategizeEndToEnd:
    def test_stub_llm_roundtrip(self):
        calls = []

        def stub(messages):
            calls.append(messages)
            return _card_yaml(
                target_domain="reward",
                target_slot="guidance_reward",
                focus=["Use tanh shaping"],
            )

        card = strategize(
            history=[_rec()],
            mutation_log_entries=[],
            top_candidates=[
                {
                    "M1_success_rate": 0.0,
                    "M2_avg_return": -2.0,
                    "M6_coverage_progress": 0.1,
                }
            ],
            seed_bias="reward_first",
            fail_mode=FailMode.HEALTHY,
            meta_llm_call=stub,
        )
        assert isinstance(card, StrategyCard)
        assert card.target_domain == "reward"
        assert card.target_slot == "guidance_reward"
        assert len(calls) == 1


# ── mutation_log ──────────────────────────────────────────────


class TestClassifyVerdict:
    @pytest.mark.parametrize(
        "pre,post,verdict",
        [
            (0.0, 0.15, "strong_improvement"),
            (0.0, 0.05, "marginal_improvement"),
            (0.5, 0.5, "neutral"),
            (0.5, 0.48, "regression"),
            (0.5, 0.3, "collapse"),
            (0.1, 0.0, "collapse"),  # non-zero → zero ⇒ collapse
        ],
    )
    def test_verdict_thresholds(self, pre, post, verdict):
        assert classify_verdict(pre, post, 0.1, 0.1) == verdict

    def test_m6_drop_triggers_regression(self):
        assert classify_verdict(0.1, 0.1, 0.2, 0.1) == "regression"


class TestEntryRoundtrip:
    def test_new_entry_and_append(self, tmp_path):
        e = new_entry(
            run_id="r1",
            task_id="t1",
            seed=0,
            outer_iter=1,
            parent_version="p",
            new_version="p_mp_001",
            strategy_card={"target_slot": "guidance_reward"},
            slot_name="guidance_reward",
            slot_content="a" * 200,
            pre_peak_M1=0.01,
            pre_M6=0.15,
        )
        assert len(e.slot_content_sha256) == 64
        assert e.slot_content_excerpt == "a" * 200
        path = tmp_path / "log.jsonl"
        append_entry(path, e)
        entries = read_recent(path)
        assert len(entries) == 1
        assert entries[0].new_version == "p_mp_001"

    def test_excerpt_truncated_when_long(self):
        e = new_entry(
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="n",
            strategy_card={},
            slot_name="s",
            slot_content="x" * 1000,
            pre_peak_M1=0.0,
            pre_M6=0.0,
            excerpt_chars=50,
        )
        assert len(e.slot_content_excerpt) <= 50 + 3  # " …" suffix
        assert e.slot_content_excerpt.endswith("…")

    def test_read_recent_filters_by_task(self, tmp_path):
        path = tmp_path / "log.jsonl"
        for i, task in enumerate(["t1", "t2", "t1", "t2", "t1"]):
            e = new_entry(
                run_id=f"r{i}",
                task_id=task,
                seed=0,
                outer_iter=i,
                parent_version="p",
                new_version=f"n{i}",
                strategy_card={},
                slot_name="s",
                slot_content="c",
                pre_peak_M1=0.0,
                pre_M6=0.0,
            )
            append_entry(path, e)
        t1 = read_recent(path, n=10, task_id="t1")
        assert len(t1) == 3
        t2 = read_recent(path, n=10, task_id="t2")
        assert len(t2) == 2

    def test_update_last_entry_resolves_post_fields(self, tmp_path):
        path = tmp_path / "log.jsonl"
        e = new_entry(
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="n",
            strategy_card={},
            slot_name="s",
            slot_content="c",
            pre_peak_M1=0.01,
            pre_M6=0.15,
        )
        append_entry(path, e)
        updated = update_last_entry_with_post(
            path,
            post_peak_M1=0.06,
            post_M6=0.20,
            fail_modes=["healthy"],
        )
        assert updated is not None
        assert updated.post_mutation_peak_M1 == pytest.approx(0.06)
        assert updated.delta_peak_M1 == pytest.approx(0.05)
        assert updated.verdict == "marginal_improvement"
        # Re-read from disk
        again = read_recent(path, n=1)[0]
        assert again.verdict == "marginal_improvement"

    def test_update_is_idempotent(self, tmp_path):
        path = tmp_path / "log.jsonl"
        e = new_entry(
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="n",
            strategy_card={},
            slot_name="s",
            slot_content="c",
            pre_peak_M1=0.0,
            pre_M6=0.0,
        )
        append_entry(path, e)
        # First update: pre=0.0, post=0.05 → marginal_improvement
        update_last_entry_with_post(path, 0.05, 0.1)
        # Second update with post=0.5 would be strong_improvement
        # IF we re-evaluated, but idempotency means we shouldn't.
        u2 = update_last_entry_with_post(path, 0.5, 0.5)
        assert u2.verdict == "marginal_improvement"  # locked by first update
        assert u2.post_mutation_peak_M1 == pytest.approx(0.05)


class TestSummarizeForPrompt:
    def test_empty(self):
        assert summarize_for_prompt([]) == "(no prior mutations on this task)"

    def test_includes_version_and_verdict(self):
        e = MutationLogEntry(
            ts="2026-04-23T00:00:00Z",
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="foo_mp_001",
            strategy_card={"target_domain": "observation", "focus": ["hold signal"]},
            slot_name="guidance_observation",
            slot_content_sha256="x",
            slot_content_excerpt="",
            pre_mutation_peak_M1=0.0,
            pre_mutation_best_M6=0.0,
            post_mutation_peak_M1=0.08,
            post_mutation_best_M6=0.1,
            delta_peak_M1=0.08,
            delta_M6=0.1,
            verdict="marginal_improvement",
        )
        out = summarize_for_prompt([e])
        assert "foo_mp_001" in out
        assert "marginal_improvement" in out
        assert "observation" in out
        assert "hold signal" in out


# ── Level 2 editor prompt uses strategy card ───────────────────


class TestEditorUsesStrategy:
    def test_editor_prompt_references_strategy_fields(self):
        from src.lero.meta.mutation import build_editor_prompt

        card = StrategyCard(
            target_domain="observation",
            target_slot="guidance_observation",
            focus=["Add hold_signal + 2nd-nearest gap from lidar"],
            avoid=["Generic fairness restatement"],
            confidence="medium",
            rationale="Evidence: record 2 top candidate uses proximity_count but no hold flag.",
        )
        p = build_editor_prompt(
            parent_version="anything",
            strategy_card=card,
            top_candidates=[
                {
                    "M1_success_rate": 0.0,
                    "M6_coverage_progress": 0.1,
                    "M2_avg_return": -1.0,
                }
            ],
            loader=None,  # unit-test mode (inline placeholders)
        )
        # All of these should appear in the editor prompt
        for marker in (
            "guidance_observation",
            "observation",
            "hold_signal",
            "STRATEGY CARD",
            "AVOID",
            "FOCUS",
            "FAIRNESS SLOT",
            "DO NOT RESTATE",
        ):
            assert marker in p, f"missing {marker!r}"

    def test_editor_prompt_includes_prior_slot_versions(self):
        """v2.1 B: when prior versions of the same slot exist, the
        Editor must see them with verdicts + excerpts, and be told
        to diverge."""
        from src.lero.meta.mutation import build_editor_prompt
        from src.lero.meta.mutation_log import MutationLogEntry

        prior = MutationLogEntry(
            ts="2026-04-23T00:00:00Z",
            run_id="r1",
            task_id="t",
            seed=0,
            outer_iter=1,
            parent_version="v_root",
            new_version="v_root_mp_001",
            strategy_card={},
            slot_name="guidance_observation",
            slot_content_sha256="x" * 64,
            slot_content_excerpt="Use nearest target distance and simple proximity count.",
            pre_mutation_peak_M1=0.01,
            pre_mutation_best_M6=0.1,
            post_mutation_peak_M1=0.00,
            post_mutation_best_M6=0.08,
            delta_peak_M1=-0.01,
            delta_M6=-0.02,
            verdict="regression",
        )
        card = StrategyCard(
            target_domain="observation",
            target_slot="guidance_observation",
            focus=["Try intensity features"],
            avoid=["simple proximity count"],
            confidence="medium",
            rationale="prior regressed",
        )
        p = build_editor_prompt(
            parent_version="v",
            strategy_card=card,
            top_candidates=[],
            loader=None,
            prior_slot_versions=[prior],
        )
        assert "PRIOR VERSIONS" in p
        assert "v_root_mp_001" in p
        assert "regression" in p
        assert "-0.010" in p  # Δpeak_M1 formatting
        assert "Use nearest target distance" in p  # excerpt included
        assert "SUBSTANTIVELY different" in p

    def test_editor_prompt_omits_prior_block_when_empty(self):
        from src.lero.meta.mutation import build_editor_prompt

        card = StrategyCard(
            target_domain="observation",
            target_slot="guidance_observation",
            focus=["x"],
            avoid=[],
            confidence="medium",
            rationale="",
        )
        p = build_editor_prompt(
            parent_version="v",
            strategy_card=card,
            top_candidates=[],
            loader=None,
            prior_slot_versions=None,
        )
        assert "PRIOR VERSIONS" not in p


# ── v2.1 — additional improvements ─────────────────────────────


class TestVerdictThresholdScaling:
    """v2.1 C: thresholds should scale with full_frames so a delta
    that's meaningful at 1M isn't also 'marginal' at 10M."""

    def test_scale_1_matches_original(self):
        from src.lero.meta.mutation_log import classify_verdict

        assert classify_verdict(0.0, 0.011, scale=1.0) == "marginal_improvement"
        assert classify_verdict(0.0, 0.005, scale=1.0) == "neutral"

    def test_scale_10_tightens_marginal(self):
        """At 10M, a +0.011 move is noise. Scale should demote it to
        neutral while preserving classifier shape."""
        from src.lero.meta.mutation_log import classify_verdict

        assert classify_verdict(0.0, 0.05, scale=10.0) == "neutral"
        assert classify_verdict(0.0, 0.11, scale=10.0) == "marginal_improvement"
        assert classify_verdict(0.0, 1.1, scale=10.0) == "strong_improvement"

    def test_scale_10_tightens_regression(self):
        from src.lero.meta.mutation_log import classify_verdict

        # -0.05 at 1M = regression; at 10M = neutral
        assert classify_verdict(0.1, 0.05, scale=1.0) == "regression"
        # With scale=10 AND M6 dropped significantly → would be regression
        # via M6 path. Test the M6-less path:
        assert (
            classify_verdict(0.1, 0.05, pre_m6=0.5, post_m6=0.5, scale=10.0)
            == "neutral"
        )

    def test_update_last_entry_honors_scale(self, tmp_path):
        """update_last_entry_with_post should pass scale into verdict."""
        from src.lero.meta.mutation_log import (
            append_entry,
            new_entry,
            update_last_entry_with_post,
        )

        path = tmp_path / "log.jsonl"
        e = new_entry(
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="n",
            strategy_card={},
            slot_name="s",
            slot_content="c",
            pre_peak_M1=0.0,
            pre_M6=0.0,
        )
        append_entry(path, e)
        # At scale=10 (10M), +0.05 is only neutral, not marginal.
        updated = update_last_entry_with_post(
            path,
            post_peak_M1=0.05,
            post_M6=0.0,
            verdict_scale=10.0,
        )
        assert updated.verdict == "neutral"


class TestMultiPathReadRecent:
    """v2.1 A-minimal: read_recent accepts list of paths, merges by ts."""

    def test_merge_two_logs_sorted_by_ts(self, tmp_path):
        from src.lero.meta.mutation_log import (
            MutationLogEntry,
            append_entry,
            read_recent,
        )

        p1 = tmp_path / "seed0.jsonl"
        p2 = tmp_path / "seed1.jsonl"
        # Entries with interleaved timestamps
        entries = [
            ("2026-04-23T10:00:00Z", "v1", p1, 0),
            ("2026-04-23T10:05:00Z", "v2", p2, 1),
            ("2026-04-23T10:10:00Z", "v3", p1, 0),
            ("2026-04-23T10:15:00Z", "v4", p2, 1),
        ]
        for ts, ver, pth, seed in entries:
            e = MutationLogEntry(
                ts=ts,
                run_id="r",
                task_id="t",
                seed=seed,
                outer_iter=0,
                parent_version="p",
                new_version=ver,
                strategy_card={},
                slot_name="s",
                slot_content_sha256="x",
                slot_content_excerpt="",
                pre_mutation_peak_M1=0.0,
                pre_mutation_best_M6=0.0,
            )
            append_entry(pth, e)
        merged = read_recent([p1, p2], n=10)
        assert [e.new_version for e in merged] == ["v1", "v2", "v3", "v4"]

    def test_single_path_still_works(self, tmp_path):
        from src.lero.meta.mutation_log import (
            append_entry,
            new_entry,
            read_recent,
        )

        p = tmp_path / "log.jsonl"
        e = new_entry(
            run_id="r",
            task_id="t",
            seed=0,
            outer_iter=0,
            parent_version="p",
            new_version="only",
            strategy_card={},
            slot_name="s",
            slot_content="c",
            pre_peak_M1=0.0,
            pre_M6=0.0,
        )
        append_entry(p, e)
        # Backward compat: passing a single Path, not a list
        got = read_recent(p, n=10)
        assert len(got) == 1


class TestReadPriorSlotVersions:
    """Helper used by outer_loop to feed the Editor's prior-slot block."""

    def test_filters_by_slot_name(self, tmp_path):
        from src.lero.meta.mutation_log import (
            append_entry,
            new_entry,
            read_prior_slot_versions,
        )

        p = tmp_path / "log.jsonl"
        for i, slot in enumerate(
            [
                "guidance_observation",
                "guidance_reward",
                "guidance_observation",
                "guidance_shared",
                "guidance_observation",
            ],
        ):
            e = new_entry(
                run_id="r",
                task_id="t",
                seed=0,
                outer_iter=i,
                parent_version="p",
                new_version=f"v{i}",
                strategy_card={},
                slot_name=slot,
                slot_content=f"text-{i}",
                pre_peak_M1=0.0,
                pre_M6=0.0,
            )
            append_entry(p, e)
        obs = read_prior_slot_versions(
            p,
            slot_name="guidance_observation",
            task_id="t",
            n=10,
        )
        assert [e.new_version for e in obs] == ["v0", "v2", "v4"]

    def test_limit_n(self, tmp_path):
        from src.lero.meta.mutation_log import (
            append_entry,
            new_entry,
            read_prior_slot_versions,
        )

        p = tmp_path / "log.jsonl"
        for i in range(10):
            e = new_entry(
                run_id="r",
                task_id="t",
                seed=0,
                outer_iter=i,
                parent_version="p",
                new_version=f"v{i}",
                strategy_card={},
                slot_name="guidance_observation",
                slot_content="c",
                pre_peak_M1=0.0,
                pre_M6=0.0,
            )
            append_entry(p, e)
        got = read_prior_slot_versions(
            p,
            slot_name="guidance_observation",
            task_id="t",
            n=3,
        )
        assert len(got) == 3
        # Latest 3 (oldest-first)
        assert [e.new_version for e in got] == ["v7", "v8", "v9"]


class TestPerSeedMetaTemperature:
    """v2.1 D: 3-way temperature cycle for meta-LLM diversity."""

    def test_three_way_cycle(self):
        from src.lero.meta.outer_loop import (
            SEED_META_TEMPERATURE,
            meta_temperature_for_seed,
        )

        assert meta_temperature_for_seed(0) == 0.1
        assert meta_temperature_for_seed(1) == 0.3
        assert meta_temperature_for_seed(2) == 0.7
        assert meta_temperature_for_seed(3) == 0.1  # cycles
        # All values in a reasonable range
        for t in SEED_META_TEMPERATURE.values():
            assert 0 < t <= 1.0
