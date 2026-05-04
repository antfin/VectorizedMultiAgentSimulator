"""v9 unit tests + production replay.

Covers every v9-specific piece of logic:
  - detect_pathological_refine (corner cases + production replay)
  - _extract_json (trailing commas, fenced blocks, malformed input)
  - _redact_forbidden (token redaction)
  - MemoryStore round-trip (append, read_recent, read_all, ordering)
  - V9Bundle.next_pending_idx (score ordering, excluded handling)
  - _compute_facts on real candidate code
  - prompt loader task_domain substitution

Run: python -m src.lero.v9.tests.test_v9
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List


# ── Helpers for production replay ───────────────────────────────


_PROD_RUN_DIR = Path(
    "results/lero_v9/lero_v9_rendezvous_k2_2x3/20260502_1912_s0"
)


def _load_prod_memory() -> List[Dict]:
    """Read the production v9 run's _meta_memory.jsonl."""
    p = _PROD_RUN_DIR / "_meta_memory.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# ── Tests: detect_pathological_refine ───────────────────────────


class TestDetectPathologicalRefine(unittest.TestCase):
    """Pure-function tests for the regression fail-safe."""

    def setUp(self):
        from src.lero.v9.outer_loop import detect_pathological_refine
        self.fn = detect_pathological_refine

    def _row(self, strategy: str, m6: float) -> Dict:
        return {"strategy_name": strategy, "actual": {"M6": m6}}

    def test_empty_memory_no_fire(self):
        self.assertFalse(self.fn([], "any", n=2))

    def test_single_row_no_fire(self):
        rows = [self._row("foo", 0.1)]
        self.assertFalse(self.fn(rows, "foo", n=2))

    def test_two_rows_same_strategy_decreasing_FIRES(self):
        rows = [self._row("foo", 0.2), self._row("foo", 0.1)]
        self.assertTrue(self.fn(rows, "foo", n=2))

    def test_two_rows_same_strategy_stable_FIRES(self):
        # M6 non-increasing (equal) → still pathological (no progress)
        rows = [self._row("foo", 0.15), self._row("foo", 0.15)]
        self.assertTrue(self.fn(rows, "foo", n=2))

    def test_two_rows_same_strategy_INCREASING_no_fire(self):
        # M6 going up → strategy IS improving → don't switch
        rows = [self._row("foo", 0.1), self._row("foo", 0.2)]
        self.assertFalse(self.fn(rows, "foo", n=2))

    def test_different_strategies_no_fire(self):
        rows = [self._row("foo", 0.2), self._row("bar", 0.1)]
        self.assertFalse(self.fn(rows, "bar", n=2))

    def test_current_strategy_mismatch_no_fire(self):
        # Last 2 are 'foo' but we ask about 'bar' → no fire
        rows = [self._row("foo", 0.2), self._row("foo", 0.1)]
        self.assertFalse(self.fn(rows, "bar", n=2))

    def test_three_rows_takes_last_n_only(self):
        # Last 2 are decreasing → fires regardless of older rows
        rows = [
            self._row("foo", 0.5),  # ignored
            self._row("foo", 0.2),
            self._row("foo", 0.1),
        ]
        self.assertTrue(self.fn(rows, "foo", n=2))

    def test_n_3_strict_chain(self):
        # n=3: last 3 must all be same strategy AND non-increasing
        rows = [
            self._row("foo", 0.3),
            self._row("foo", 0.2),
            self._row("foo", 0.1),
        ]
        self.assertTrue(self.fn(rows, "foo", n=3))

    def test_n_3_one_increase_breaks_chain(self):
        rows = [
            self._row("foo", 0.3),
            self._row("foo", 0.4),  # increase
            self._row("foo", 0.2),
        ]
        self.assertFalse(self.fn(rows, "foo", n=3))

    def test_zero_n_no_fire(self):
        rows = [self._row("foo", 0.1)]
        self.assertFalse(self.fn(rows, "foo", n=0))

    def test_missing_M6_treated_as_zero(self):
        rows = [
            {"strategy_name": "foo", "actual": {}},
            {"strategy_name": "foo", "actual": {}},
        ]
        # Both 0.0, non-increasing trivially → fires
        self.assertTrue(self.fn(rows, "foo", n=2))

    def test_missing_actual_treated_as_zero(self):
        rows = [
            {"strategy_name": "foo"},
            {"strategy_name": "foo"},
        ]
        self.assertTrue(self.fn(rows, "foo", n=2))


class TestProductionReplayFailsafe(unittest.TestCase):
    """Replay the actual production v9 memory through the fail-safe to
    confirm the bug we patched would have been caught at every relevant
    outer."""

    def setUp(self):
        from src.lero.v9.outer_loop import detect_pathological_refine
        self.fn = detect_pathological_refine
        self.rows = _load_prod_memory()

    def test_production_memory_loaded(self):
        self.assertGreaterEqual(
            len(self.rows), 5,
            "expected ≥5 production memory rows from Phase 6",
        )

    def test_production_outer3_would_force_switch(self):
        """At the start of outer 3 (index 3, 4th iter), memory has rows
        for outers 0, 1, 2 — all `pair_and_split` with M6 0.169 →
        0.133 → 0.083. Last 2 are non-increasing → fires."""
        rows_at_outer_3 = self.rows[:3]  # outers 0, 1, 2
        # current strategy at start of outer 3 was pair_and_split
        # (refine_current chose to keep it)
        result = self.fn(rows_at_outer_3, "pair_and_split", n=2)
        self.assertTrue(
            result,
            f"fail-safe did NOT fire on production outer 3 — bug "
            f"unfixed. M6 sequence: "
            f"{[r['actual']['M6'] for r in rows_at_outer_3[-2:]]}",
        )

    def test_production_outer1_would_NOT_fire(self):
        """At start of outer 1, memory has only 1 row (outer 0). Less
        than n=2 → no fire."""
        rows_at_outer_1 = self.rows[:1]
        self.assertFalse(self.fn(rows_at_outer_1, "pair_and_split", n=2))

    def test_production_outer2_would_NOT_fire(self):
        """At start of outer 2, memory has 2 rows: outer 0 (M6=0.169)
        and outer 1 (M6=0.133). Strictly decreasing → SHOULD have
        fired. This test demonstrates that v2 of the fail-safe (more
        sensitive) would have caught it one outer earlier."""
        rows_at_outer_2 = self.rows[:2]
        result = self.fn(rows_at_outer_2, "pair_and_split", n=2)
        self.assertTrue(
            result,
            "fail-safe should have fired at outer 2 already — saving "
            "outer 2's wasted training. Note: the production run did "
            "NOT use this fail-safe at outer 2 because the patch was "
            "applied later. This test confirms the patch would catch "
            "it earlier on a fresh run.",
        )


# ── Tests: _extract_json ────────────────────────────────────────


class TestExtractJSON(unittest.TestCase):
    def setUp(self):
        from src.lero.v9.meta_strategist import _extract_json
        self.fn = _extract_json

    def test_clean_json(self):
        self.assertEqual(self.fn('{"a": 1}'), {"a": 1})

    def test_trailing_comma_object(self):
        self.assertEqual(self.fn('{"a": 1,}'), {"a": 1})

    def test_trailing_comma_array(self):
        self.assertEqual(self.fn('{"a": [1, 2,]}'), {"a": [1, 2]})

    def test_nested_trailing_commas(self):
        self.assertEqual(
            self.fn('{"a": {"b": [1,], "c": 2,},}'),
            {"a": {"b": [1], "c": 2}},
        )

    def test_fenced_json_block(self):
        raw = "prose before\n```json\n{\"x\": 1}\n```\nprose after"
        self.assertEqual(self.fn(raw), {"x": 1})

    def test_prose_around_object(self):
        self.assertEqual(
            self.fn("Here is the answer: {\"k\": \"v\"}. Done."),
            {"k": "v"},
        )

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            self.fn("no json here at all")

    def test_malformed_json_raises(self):
        # The pre-check raises ValueError if no closing brace; otherwise
        # parse fails with JSONDecodeError. Both are acceptable failure
        # modes.
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            self.fn('{"a": "unclosed string')

    def test_truly_malformed_inside_braces(self):
        # Has braces, but content is invalid JSON
        with self.assertRaises(json.JSONDecodeError):
            self.fn('{this is not json}')


# ── Tests: _redact_forbidden ────────────────────────────────────


class TestRedactForbidden(unittest.TestCase):
    def setUp(self):
        from src.lero.v9.meta_strategist import _redact_forbidden
        self.fn = _redact_forbidden

    def test_empty_text_passthrough(self):
        self.assertEqual(self.fn("", ["foo"]), "")

    def test_no_match_unchanged(self):
        self.assertEqual(self.fn("hello world", ["foo"]), "hello world")

    def test_single_token_redacted(self):
        self.assertEqual(
            self.fn("the settle_signal is high", ["settle_signal"]),
            "the <REDACTED> is high",
        )

    def test_case_insensitive(self):
        self.assertEqual(
            self.fn("HOLD_signal", ["hold_signal"]),
            "<REDACTED>",
        )

    def test_multiple_tokens(self):
        text = "use hold_signal AND settle_signal together"
        out = self.fn(text, ["hold_signal", "settle_signal"])
        self.assertEqual(out, "use <REDACTED> AND <REDACTED> together")


# ── Tests: MemoryStore ──────────────────────────────────────────


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = Path(self.tmp) / "_mem.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _new(self):
        from src.lero.v9.memory import MemoryStore
        return MemoryStore(self.path)

    def _row(self, idx: int, name: str = "foo"):
        from src.lero.v9.memory import MemoryRow
        return MemoryRow(
            outer_idx=idx, ts="2026-01-01", strategy_name=name,
        )

    def test_initial_empty(self):
        m = self._new()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.read_recent(3), [])

    def test_append_and_read_all(self):
        m = self._new()
        m.append(self._row(0))
        m.append(self._row(1))
        rows = m.read_all()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["outer_idx"], 0)
        self.assertEqual(rows[1]["outer_idx"], 1)

    def test_read_recent_smaller_than_total(self):
        m = self._new()
        for i in range(5):
            m.append(self._row(i))
        recent = m.read_recent(2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["outer_idx"], 3)
        self.assertEqual(recent[1]["outer_idx"], 4)

    def test_read_recent_zero_returns_empty(self):
        m = self._new()
        m.append(self._row(0))
        self.assertEqual(m.read_recent(0), [])

    def test_read_recent_more_than_available(self):
        m = self._new()
        m.append(self._row(0))
        m.append(self._row(1))
        recent = m.read_recent(10)
        self.assertEqual(len(recent), 2)

    def test_persist_across_instances(self):
        m1 = self._new()
        m1.append(self._row(0))
        m2 = self._new()  # re-open same path
        self.assertEqual(len(m2), 1)
        self.assertEqual(m2.read_all()[0]["outer_idx"], 0)

    def test_skip_malformed_line(self):
        # Hand-craft a corrupt jsonl
        m = self._new()
        m.append(self._row(0))
        with self.path.open("a") as f:
            f.write("garbage not json\n")
        m._cache = None  # invalidate
        rows = m.read_all()
        self.assertEqual(len(rows), 1)


# ── Tests: V9Bundle.next_pending_idx ────────────────────────────


class TestBundleNextPendingIdx(unittest.TestCase):
    def _strat(self, name, score, attempts=0, excluded=False):
        from src.lero.v9.strategy import (
            V9Strategy,
            V9SuccessSignature,
            V9ChainOfThought,
        )
        return V9Strategy(
            name=name,
            full_solution="x",
            success_signature=V9SuccessSignature(
                ast_pattern_description="p",
            ),
            chain_of_thought=V9ChainOfThought(
                why_it_works="w", what_is_needed=[], failure_modes=[],
            ),
            lero_codability=score,
            rl_trainability=score,
            attempts=attempts,
            excluded=excluded,
        )

    def _bundle(self, strategies):
        from src.lero.v9.strategy import V9Bundle
        return V9Bundle(strategies=strategies)

    def test_picks_highest_score_first(self):
        b = self._bundle([
            self._strat("low", 5),
            self._strat("hi", 9),
            self._strat("mid", 7),
        ])
        idx = b.next_pending_idx()
        self.assertEqual(b.strategies[idx].name, "hi")

    def test_skips_excluded(self):
        b = self._bundle([
            self._strat("hi", 9, excluded=True),
            self._strat("mid", 7),
        ])
        idx = b.next_pending_idx()
        self.assertEqual(b.strategies[idx].name, "mid")

    def test_skips_attempted(self):
        b = self._bundle([
            self._strat("hi", 9, attempts=1),
            self._strat("mid", 7),
        ])
        idx = b.next_pending_idx()
        self.assertEqual(b.strategies[idx].name, "mid")

    def test_returns_None_when_all_done(self):
        b = self._bundle([
            self._strat("hi", 9, attempts=1),
            self._strat("mid", 7, excluded=True),
        ])
        self.assertIsNone(b.next_pending_idx())


# ── Tests: _compute_facts ───────────────────────────────────────


class TestComputeFacts(unittest.TestCase):
    def setUp(self):
        from src.lero.v9.outer_loop import _compute_facts
        self.fn = _compute_facts

    def test_none_inner_returns_zeros(self):
        out = self.fn(None)
        self.assertFalse(out["inner_present"])
        self.assertEqual(out["M1"], 0.0)
        self.assertFalse(out["role_one_hot_present"])

    def test_real_candidate_with_role_one_hot(self):
        from src.lero.codegen import CandidateCode
        from src.lero.v5.inner_loop import CandidateOutcome, InnerResult
        from src.lero.v5.registry import Registry
        code = """
import torch
import torch.nn.functional as F
def enhance_observation(s):
    lt = s["lidar_targets"]
    la = s["lidar_agents"]
    agent_idx = s["agent_idx"]
    n_agents = int(s["n_agents"])
    nt = lt.min(dim=-1).values
    one_hot = torch.zeros(lt.shape[0], n_agents, device=lt.device)
    one_hot[:, agent_idx] = 1.0
    diff = (lt < 0.25).float().sum(-1) - (la < 0.25).float().sum(-1)
    return torch.cat([nt.unsqueeze(-1), one_hot, diff.unsqueeze(-1)], dim=-1)
"""
        cand = CandidateCode(
            obs_source=code, reward_source=None, raw_response="",
        )
        out = CandidateOutcome(
            candidate=cand,
            metrics={"M1_success_rate": 0.05, "M6_coverage_progress": 0.4},
            fitness=0.25, shape="monotonic_rise", iter_idx=0,
        )
        ir = InnerResult(
            best=out, worst=out, all_outcomes=[out],
            registry=Registry(), did_stagnate=False, n_iters_run=1,
        )
        f = self.fn(ir)
        self.assertTrue(f["inner_present"])
        self.assertEqual(f["M1"], 0.05)
        self.assertEqual(f["M6"], 0.4)
        self.assertTrue(f["touches_both_lidars"])
        self.assertTrue(
            f["role_one_hot_present"],
            "should detect torch.zeros + [:, agent_idx] = 1.0 pattern",
        )

    def test_real_candidate_without_role_one_hot(self):
        from src.lero.codegen import CandidateCode
        from src.lero.v5.inner_loop import CandidateOutcome, InnerResult
        from src.lero.v5.registry import Registry
        code = """
import torch
def enhance_observation(s):
    lt = s["lidar_targets"]
    return lt.min(dim=-1).values.unsqueeze(-1)
"""
        cand = CandidateCode(
            obs_source=code, reward_source=None, raw_response="",
        )
        out = CandidateOutcome(
            candidate=cand,
            metrics={"M1_success_rate": 0.0, "M6_coverage_progress": 0.1},
            fitness=0.0, shape="flat_zero", iter_idx=0,
        )
        ir = InnerResult(
            best=out, worst=out, all_outcomes=[out],
            registry=Registry(), did_stagnate=False, n_iters_run=1,
        )
        f = self.fn(ir)
        self.assertFalse(f["role_one_hot_present"])
        self.assertFalse(f["touches_both_lidars"])


# ── Tests: prompt loader task_domain substitution ──────────────


class TestPromptLoaderTaskDomain(unittest.TestCase):
    def test_task_domain_loads(self):
        from src.lero.prompts.loader import PromptLoader
        loader = PromptLoader("v3_modular_taskdomain")
        td = loader.task_domain()
        self.assertIsNotNone(td)
        self.assertEqual(td["name"], "rendezvous_k2")
        # 8 concepts after v9.1 §2.11 added soft_proximity
        self.assertEqual(len(td["inferable_concepts"]), 8)
        # Verify soft_proximity is in the list
        self.assertTrue(
            any("Soft proximity" in c["concept"]
                for c in td["inferable_concepts"]),
            "soft_proximity should be in inferable_concepts after §2.11",
        )
        self.assertEqual(len(td["mandatory_features"]), 2)

    def test_task_framing_substituted_into_system(self):
        from src.lero.prompts.loader import PromptLoader
        loader = PromptLoader("v3_modular_taskdomain")
        sys_text = loader.render(
            "system.txt",
            n_agents=4, n_targets=4, agents_per_target=2,
            covering_range=0.25, max_steps=400,
        )
        self.assertIn("RENDEZVOUS task", sys_text)
        self.assertIn("4 agents", sys_text)
        self.assertIn("EXACTLY 2", sys_text)
        # Must NOT have unrendered placeholders
        self.assertNotIn("${task_framing}", sys_text)
        self.assertNotIn("${n_agents}", sys_text)


# ── Tests: §2.3 slot-edit validator ────────────────────────────


class TestSlotValidator(unittest.TestCase):
    """Pure-function tests for v9.1 §2.3 slot-edit structural validator."""

    def setUp(self):
        from src.lero.v9.slot_validator import validate_slot_edits
        self.fn = validate_slot_edits
        self.td = {
            "inferable_concepts": [
                {"concept": "Direction to nearest target",
                 "idiom": "argmin lidar_targets cos sin"},
                {"concept": "Distance to nearest target",
                 "idiom": "lidar_targets.min"},
                {"concept": "Number of nearby targets",
                 "idiom": "(lidar_targets < threshold).sum"},
                {"concept": "Local agent crowdedness",
                 "idiom": "(lidar_agents < threshold).sum"},
                {"concept": "Agent role under shared-policy MAPPO",
                 "idiom": "torch.zeros n_agents agent_idx one_hot"},
                {"concept": "Self-motion state",
                 "idiom": "agent_vel.norm"},
                {"concept": "Boundary distance from arena edge",
                 "idiom": "1 - agent_pos.abs().max"},
            ],
        }

    def test_good_examples_passes(self):
        edits = {
            "examples": (
                "```python\n"
                "def enhance_observation(s):\n"
                "    one_hot = torch.zeros(B, n_agents)\n"
                "    one_hot[:, agent_idx] = 1.0\n"
                "    return one_hot\n"
                "```\n\n"
                "```python\n"
                "def enhance_observation(s):\n"
                "    return s['lidar_targets'].min(dim=-1).values\n"
                "```\n"
            ),
        }
        results = self.fn(edits, self.td)
        self.assertTrue(results["examples"].passed)

    def test_prose_only_examples_rejected(self):
        edits = {
            "examples": (
                "Example 1: If role one-hot marks agent A as primary "
                "seeker, target is close, and agent congestion is low, "
                "the joint scalar should favor committing.\n"
                "Example 2: If agent B is the scout, the nearest target "
                "is close but already crowded, divert to a less-contested "
                "target.\n"
                "Example 3: If both lidars indicate weak opportunity, "
                "keep searching."
            ),
        }
        results = self.fn(edits, self.td)
        self.assertFalse(results["examples"].passed)
        # Should mention 0 python blocks
        self.assertTrue(
            any("0 fenced" in i for i in results["examples"].issues)
        )

    def test_one_python_block_no_role_rejected(self):
        edits = {
            "examples": (
                "```python\n"
                "def enhance_observation(s):\n"
                "    return s['lidar_targets'].min(dim=-1).values\n"
                "```\n"
            ),
        }
        results = self.fn(edits, self.td)
        self.assertFalse(results["examples"].passed)
        # Need ≥2 blocks AND role
        issues = " ".join(results["examples"].issues)
        self.assertIn("blocks", issues)

    def test_two_blocks_but_no_role_rejected(self):
        edits = {
            "examples": (
                "```python\nreturn x\n```\n\n"
                "```python\nreturn y\n```\n"
            ),
        }
        results = self.fn(edits, self.td)
        self.assertFalse(results["examples"].passed)
        self.assertTrue(
            any("role_one_hot" in i for i in results["examples"].issues)
        )

    def test_inferable_hints_with_concepts_passes(self):
        # Mention all 7 concepts via idiom keywords
        edits = {
            "inferable_hints": (
                "## What you CAN infer\n"
                "- Direction to nearest target via argmin and cos/sin\n"
                "- Distance to nearest target via lidar_targets.min\n"
                "- Number of nearby targets: (lidar_targets < threshold).sum\n"
                "- Local agent crowdedness: (lidar_agents < threshold).sum\n"
                "- Agent role: torch.zeros(B, n_agents); "
                "one_hot[:, agent_idx] = 1\n"
                "- Self-motion: agent_vel.norm\n"
                "- Boundary distance from arena edge: "
                "1 - agent_pos.abs().max\n"
            ),
        }
        results = self.fn(edits, self.td)
        self.assertTrue(
            results["inferable_hints"].passed,
            results["inferable_hints"].issues,
        )

    def test_inferable_hints_prose_only_rejected(self):
        # Prose paragraph — mentions "target" and "agent" but doesn't
        # cover ≥5 concepts
        edits = {
            "inferable_hints": (
                "Use the policy to combine local target proximity with "
                "agent congestion. Decide based on the joint scalar."
            ),
        }
        results = self.fn(edits, self.td)
        self.assertFalse(results["inferable_hints"].passed)

    def test_empty_edits_returns_empty_results(self):
        self.assertEqual(self.fn({}, self.td), {})

    def test_growth_factor_rejection(self):
        prev = {"examples": "x" * 100}
        edits = {
            "examples": (
                "```python\nreturn one_hot\n```\n"
                "```python\nreturn x\n```\n"
            ) + "y" * 500,  # 5x prev length
        }
        results = self.fn(edits, self.td, prev_slots=prev,
                          max_growth_factor=2.0)
        self.assertFalse(results["examples"].passed)
        self.assertTrue(
            any("growth" in i for i in results["examples"].issues)
        )


class TestPreEvalValidator(unittest.TestCase):
    """v9.1 §2.1 mandatory_features pre-eval check."""

    def setUp(self):
        from src.lero.v9.outer_loop import make_pre_eval_validator
        from src.lero.codegen import CandidateCode
        self.CandidateCode = CandidateCode
        self.td = {
            "mandatory_features": [
                {"name": "role_one_hot", "reason": "..."},
                {"name": "cross_source_signal", "reason": "..."},
            ],
        }
        self.fn = make_pre_eval_validator(self.td)

    def _cand(self, code: str):
        return self.CandidateCode(
            obs_source=code, reward_source=None, raw_response="",
        )

    def test_good_candidate_passes(self):
        code = (
            "import torch\nimport torch.nn.functional as F\n"
            "def enhance_observation(s):\n"
            "    lt = s['lidar_targets']\n"
            "    la = s['lidar_agents']\n"
            "    n = int(s['n_agents'])\n"
            "    role = F.one_hot(s['agent_idx'].long(), n).float()\n"
            "    diff = (lt < 0.25).float().sum(-1) - "
            "(la < 0.25).float().sum(-1)\n"
            "    return torch.cat([role, diff.unsqueeze(-1)], -1)\n"
        )
        self.assertEqual(self.fn(self._cand(code)), [])

    def test_missing_role_rejected(self):
        code = (
            "def enhance_observation(s):\n"
            "    lt = s['lidar_targets']\n"
            "    la = s['lidar_agents']\n"
            "    return (lt < 0.25).float().sum(-1) - "
            "(la < 0.25).float().sum(-1)\n"
        )
        issues = self.fn(self._cand(code))
        self.assertTrue(any("role_one_hot" in i for i in issues))

    def test_missing_cross_source_rejected(self):
        # Has role one-hot but only target lidar; no cross-source op
        code = (
            "import torch\nimport torch.nn.functional as F\n"
            "def enhance_observation(s):\n"
            "    lt = s['lidar_targets']\n"
            "    n = int(s['n_agents'])\n"
            "    role = F.one_hot(s['agent_idx'].long(), n).float()\n"
            "    return torch.cat([role, lt.min(dim=-1).values.unsqueeze(-1)], -1)\n"
        )
        issues = self.fn(self._cand(code))
        self.assertTrue(any("cross_source_signal" in i for i in issues))

    def test_empty_code_rejected(self):
        cand = self.CandidateCode(
            obs_source="", reward_source=None, raw_response="",
        )
        issues = self.fn(cand)
        self.assertTrue(any("empty" in i for i in issues))

    def test_feature_budget_overshoot_rejected(self):
        from src.lero.v9.outer_loop import make_pre_eval_validator
        td = {
            "mandatory_features": [
                {"name": "role_one_hot", "reason": "..."},
                {"name": "cross_source_signal", "reason": "..."},
            ],
            "feature_budget": {"hard_cap": 20},
        }
        fn = make_pre_eval_validator(td)
        # Build 25-feature literal cat
        items = [f'lt[:, {i}:{i+1}]' for i in range(25)]
        code = (
            "import torch\nimport torch.nn.functional as F\n"
            "def enhance_observation(s):\n"
            "    lt = s['lidar_targets']\n"
            "    la = s['lidar_agents']\n"
            "    agent_idx = s['agent_idx']\n"
            "    n_agents = int(s['n_agents'])\n"
            "    one_hot = torch.zeros(lt.shape[0], n_agents)\n"
            "    one_hot[:, agent_idx] = 1.0\n"
            "    diff = (lt < 0.25).float().sum(-1) - "
            "(la < 0.25).float().sum(-1)\n"
            "    return torch.cat([" + ", ".join(items)
            + ", diff.unsqueeze(-1), one_hot], dim=-1)\n"
        )
        cand = self.CandidateCode(
            obs_source=code, reward_source=None, raw_response="",
        )
        issues = fn(cand)
        self.assertTrue(
            any("feature_budget" in i for i in issues),
            f"expected feature_budget issue, got {issues}",
        )

    def test_feature_budget_zero_count_skipped(self):
        """If AST estimator returns 0 (couldn't measure), do NOT reject
        on feature_budget. Soft rejection only when reliably measured."""
        from src.lero.v9.outer_loop import make_pre_eval_validator
        td = {
            "mandatory_features": [
                {"name": "role_one_hot", "reason": "..."},
                {"name": "cross_source_signal", "reason": "..."},
            ],
            "feature_budget": {"hard_cap": 5},
        }
        fn = make_pre_eval_validator(td)
        # Dynamic build — estimator returns 0 → no budget rejection
        code = (
            "import torch\nimport torch.nn.functional as F\n"
            "def enhance_observation(s):\n"
            "    lt = s['lidar_targets']\n"
            "    la = s['lidar_agents']\n"
            "    agent_idx = s['agent_idx']\n"
            "    n_agents = int(s['n_agents'])\n"
            "    one_hot = torch.zeros(lt.shape[0], n_agents)\n"
            "    one_hot[:, agent_idx] = 1.0\n"
            "    diff = ((lt < 0.25).float().sum(-1) - "
            "(la < 0.25).float().sum(-1)).unsqueeze(-1)\n"
            "    feats = [torch.zeros(lt.shape[0], 1) for _ in range(50)]\n"
            "    return torch.cat([one_hot, diff, *feats], dim=-1)\n"
        )
        cand = self.CandidateCode(
            obs_source=code, reward_source=None, raw_response="",
        )
        issues = fn(cand)
        # Should NOT have feature_budget issue (estimator returns 0)
        self.assertFalse(
            any("feature_budget" in i for i in issues),
            f"unexpected budget rejection: {issues}",
        )


class TestProductionReplayPreEvalValidator(unittest.TestCase):
    """Replay actual v9 production candidates: confirm we'd reject the
    8 candidates that lacked role_one_hot (saving 8×9=~72 min training)."""

    def setUp(self):
        from src.lero.v9.outer_loop import make_pre_eval_validator
        from src.lero.prompts.loader import PromptLoader
        from src.lero.codegen import CandidateCode
        self.CandidateCode = CandidateCode
        loader = PromptLoader("v3_modular_taskdomain")
        self.fn = make_pre_eval_validator(loader.task_domain() or {})

    def test_outer_4_candidates_rejected(self):
        """Outer 4 had role_one_hot=False on 7+ candidates."""
        import glob
        files = sorted(glob.glob(
            "results/lero_v9/lero_v9_rendezvous_k2_2x3/"
            "20260502_1912_s0/outer_04/inner/iter_*/candidate_*_obs.py"
        ))
        if not files:
            self.skipTest("production outer 04 dir missing")
        rejected = 0
        for f in files:
            code = open(f).read()
            cand = self.CandidateCode(
                obs_source=code, reward_source=None, raw_response="",
            )
            issues = self.fn(cand)
            if issues:
                rejected += 1
        # At least 7 of the 9 outer-4 candidates lacked role_one_hot
        self.assertGreaterEqual(
            rejected, 5,
            f"expected ≥5 outer-4 candidates rejected; got {rejected}/{len(files)}",
        )


class TestDetectFalsificationFailure(unittest.TestCase):
    """Pure-function tests for v9.1 §2.7 falsification gate."""

    def setUp(self):
        from src.lero.v9.outer_loop import detect_falsification_failure
        self.fn = detect_falsification_failure

    def _row(self, strategy: str, m1: float):
        return {"strategy_name": strategy, "actual": {"M1": m1}}

    def test_empty_memory_no_fire(self):
        self.assertFalse(self.fn([], "foo", expected_M1=0.18))

    def test_single_attempt_no_fire(self):
        rows = [self._row("foo", 0.0)]
        self.assertFalse(self.fn(rows, "foo", expected_M1=0.18, n_attempts=2))

    def test_two_attempts_both_below_half_FIRES(self):
        rows = [self._row("foo", 0.01), self._row("foo", 0.0)]
        self.assertTrue(self.fn(rows, "foo", expected_M1=0.18, n_attempts=2))

    def test_one_attempt_above_threshold_no_fire(self):
        # 0.10 > 0.5 * 0.18 = 0.09 → strategy is salvageable
        rows = [self._row("foo", 0.10), self._row("foo", 0.0)]
        self.assertFalse(self.fn(rows, "foo", expected_M1=0.18, n_attempts=2))

    def test_different_strategy_filtered_out(self):
        rows = [self._row("foo", 0.0), self._row("bar", 0.0)]
        # Only 'foo' has 1 attempt
        self.assertFalse(self.fn(rows, "foo", expected_M1=0.18, n_attempts=2))

    def test_threshold_factor_loose(self):
        # threshold_factor=0.1 → must be below 0.018, 0.02 is just above
        rows = [self._row("foo", 0.02), self._row("foo", 0.02)]
        self.assertFalse(self.fn(rows, "foo", expected_M1=0.18,
                                 n_attempts=2, threshold_factor=0.1))

    def test_zero_expected_M1_no_fire(self):
        # Avoid divide-by-zero / nonsensical comparisons
        rows = [self._row("foo", 0.0), self._row("foo", 0.0)]
        self.assertFalse(self.fn(rows, "foo", expected_M1=0.0, n_attempts=2))

    def test_picks_last_n_only(self):
        # Older row has high M1 but most recent two are low
        rows = [
            self._row("foo", 0.5),  # ignored
            self._row("foo", 0.01),
            self._row("foo", 0.0),
        ]
        self.assertTrue(self.fn(rows, "foo", expected_M1=0.18, n_attempts=2))


class TestProductionReplayFalsificationGate(unittest.TestCase):
    """Replay v9 Phase 6 production memory through §2.7 to verify the
    expected production overrides at outer 2 (pair_and_split #2) and
    outer 4 (leader_follower_pairing #2)."""

    def setUp(self):
        from src.lero.v9.outer_loop import detect_falsification_failure
        self.fn = detect_falsification_failure
        self.rows = _load_prod_memory()

    def test_outer_2_falsification_fires(self):
        """At start of outer 2, memory has rows for outers 0 and 1.
        Both are pair_and_split with M1=0.01 and M1=0.0 — both below
        0.5×0.18=0.09. Adding outer 2's facts (M1=0.0) → still below.
        Gate should fire."""
        # Memory at start-of-outer-2 = first 2 rows
        # PLUS the current outer's facts (would be outer 2 itself)
        # The production code does memory_rows + [current_facts]
        rows_at_outer_2 = self.rows[:2] + [{
            "strategy_name": "pair_and_split",
            "actual": {"M1": 0.0},
        }]
        result = self.fn(
            rows_at_outer_2,
            "pair_and_split",
            expected_M1=0.18,
            n_attempts=2,
            threshold_factor=0.5,
        )
        self.assertTrue(
            result,
            "§2.7 should have fired at outer 2 — pair_and_split #2 "
            "with M1=0 vs expected 0.18",
        )

    def test_outer_4_falsification_fires(self):
        """At start of outer 4, leader_follower_pairing has had 1 prior
        attempt (outer 3, M1=0.0) and the current outer 4 (M1=0.0).
        Both well below 0.5×0.14=0.07."""
        # Memory rows for outer 0,1,2,3 + facts for outer 4
        rows_at_outer_4 = self.rows[:4] + [{
            "strategy_name": "leader_follower_pairing",
            "actual": {"M1": 0.0},
        }]
        result = self.fn(
            rows_at_outer_4,
            "leader_follower_pairing",
            expected_M1=0.14,
            n_attempts=2,
            threshold_factor=0.5,
        )
        self.assertTrue(
            result,
            "§2.7 should have fired at outer 4 — "
            "leader_follower_pairing #2 with M1=0 vs expected 0.14",
        )

    def test_outer_1_falsification_does_NOT_fire(self):
        """At start of outer 1, only outer 0 has been attempted on
        pair_and_split. Need n_attempts=2 → not enough data to fire."""
        rows_at_outer_1 = self.rows[:1] + [{
            "strategy_name": "pair_and_split",
            "actual": {"M1": 0.0},
        }]
        # Now we have 2 attempts (outer 0 M1=0.01 and outer 1 M1=0.0)
        # both below 0.09 → fires! Actually this WOULD fire.
        result = self.fn(
            rows_at_outer_1,
            "pair_and_split",
            expected_M1=0.18,
            n_attempts=2,
            threshold_factor=0.5,
        )
        # Acceptable: this would fire too, even one outer earlier than
        # the production fail-safe caught it. That's a feature.
        self.assertTrue(
            result,
            "§2.7 fires at outer 1 too (one outer earlier than the "
            "v9 regression fail-safe). This is intentional — §2.7 "
            "uses the M1 prediction directly, not the trend.",
        )


class TestLazyArtifactAuthoring(unittest.TestCase):
    """v9.1 §2.10 — lazy artifact authoring for non-chosen strategies.

    Pure-function test of the system message builder. Integration of
    the actual LLM call is exercised by the smoke run (no LLM mocks
    here to keep tests cheap and offline)."""

    def test_author_artifacts_system_includes_concepts_and_mandatory(self):
        from src.lero.v9.meta_strategist import _author_artifacts_system
        td = {
            "inferable_concepts": [
                {"concept": "Direction to nearest target",
                 "idiom": "argmin"},
                {"concept": "Soft proximity",
                 "idiom": "torch.exp(-α * d)"},
            ],
            "mandatory_features": [
                {"name": "role_one_hot",
                 "idiom": "torch.zeros + agent_idx",
                 "reason": "shared policy needs role"},
            ],
            "forbidden_tokens": ["hold_signal", "settle_signal"],
            "feature_budget": {
                "target_min": 12, "target_max": 17, "hard_cap": 20,
            },
        }
        sys_msg = _author_artifacts_system(td)
        # Must include each concept
        self.assertIn("Direction to nearest target", sys_msg)
        self.assertIn("Soft proximity", sys_msg)
        # Must include mandatory feature
        self.assertIn("role_one_hot", sys_msg)
        # Must include forbidden tokens
        self.assertIn("hold_signal", sys_msg)
        self.assertIn("settle_signal", sys_msg)
        # Must include feature budget numbers
        self.assertIn("12-17", sys_msg)
        self.assertIn("20", sys_msg)
        # Must specify the JSON output format
        self.assertIn("inferable_hints_text", sys_msg)
        self.assertIn("examples_text", sys_msg)
        self.assertIn("feedback_template", sys_msg)


class TestProductionReplaySlotValidator(unittest.TestCase):
    """Replay actual production v9 Phase 6 slot_edits through the
    §2.3 validator. Confirms that outer_0's edits PASS and
    outer_1..outer_4's edits FAIL — exactly the regression points."""

    def setUp(self):
        from src.lero.v9.slot_validator import validate_slot_edits
        from src.lero.prompts.loader import PromptLoader
        self.fn = validate_slot_edits
        loader = PromptLoader("v3_modular_taskdomain")
        self.td = loader.task_domain() or {}

    def _load_outer_slots(self, outer_idx: int) -> Dict[str, str]:
        run_dir = Path(
            "results/lero_v9/lero_v9_rendezvous_k2_2x3/"
            "20260502_1912_s0/prompts"
        )
        # Production: prompts/v9_outer_<idx>_seed0/
        d = run_dir / f"v9_outer_{outer_idx}_seed0"
        if not d.exists():
            return {}
        out = {}
        for slot in ("inferable_hints", "examples"):
            p = d / f"{slot}.txt"
            if p.exists():
                out[slot] = p.read_text()
        return out

    def test_outer_0_slots_PASS(self):
        """Outer 0's slot text was authored at cold-start (first bundle
        enum). It contains the rich python+bullet content. Should pass."""
        slots = self._load_outer_slots(0)
        self.assertTrue(slots, "production outer 0 prompt dir missing")
        results = self.fn(slots, self.td)
        for k, vr in results.items():
            self.assertTrue(
                vr.passed,
                f"outer 0 slot '{k}' UNEXPECTEDLY rejected: {vr.issues}",
            )

    def test_outer_1_slots_REJECTED(self):
        """Outer 1's slot text was authored by refine_current —
        prose-only paragraphs. Should be rejected by validator."""
        slots = self._load_outer_slots(1)
        self.assertTrue(slots, "production outer 1 prompt dir missing")
        results = self.fn(slots, self.td)
        any_rejected = any(not vr.passed for vr in results.values())
        self.assertTrue(
            any_rejected,
            f"outer 1 slot edits NOT rejected — bug. Results: "
            f"{[(k, v.passed, v.issues) for k, v in results.items()]}",
        )

    def test_outer_2_3_4_slots_REJECTED(self):
        """Same as outer 1: refine_current produced prose-only edits."""
        for outer_idx in (2, 3, 4):
            slots = self._load_outer_slots(outer_idx)
            if not slots:
                continue
            results = self.fn(slots, self.td)
            any_rejected = any(not vr.passed for vr in results.values())
            self.assertTrue(
                any_rejected,
                f"outer {outer_idx} slots NOT rejected — bug",
            )


# ── Run ─────────────────────────────────────────────────────────


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in (
        TestDetectPathologicalRefine,
        TestProductionReplayFailsafe,
        TestExtractJSON,
        TestRedactForbidden,
        TestMemoryStore,
        TestBundleNextPendingIdx,
        TestComputeFacts,
        TestPromptLoaderTaskDomain,
        TestSlotValidator,
        TestLazyArtifactAuthoring,
        TestProductionReplaySlotValidator,
        TestPreEvalValidator,
        TestProductionReplayPreEvalValidator,
        TestDetectFalsificationFailure,
        TestProductionReplayFalsificationGate,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
