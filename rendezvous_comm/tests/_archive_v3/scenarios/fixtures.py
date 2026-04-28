"""Builders for synthetic meta-LLM scenario fixtures.

The harness pipes these into the real Strategist/Editor/Critic so we
can exercise decision logic without running RL.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.lero.meta.failmode import FailMode
from src.lero.meta.mutation_log import MutationLogEntry
from src.lero.meta.trigger import TemplateRecord


# ── TemplateRecord builders ──────────────────────────────────────


def _rec(
    version: str = "v2_fewshot_modular_v2",
    peak: float = 0.05,
    final: Optional[float] = 0.0,
    m6: float = 0.20,
    m2: float = -2.0,
    seed_std: float = 0.02,
    fail_mode: FailMode = FailMode.HEALTHY,
    mutation_slot: Optional[str] = None,
    rationale: Optional[str] = None,
) -> TemplateRecord:
    return TemplateRecord(
        template_version=version,
        inner_iter_count=1,
        best_peak_M1=peak,
        best_final_M1=final,
        best_M6=m6,
        best_M2=m2,
        seed_M1_std=seed_std,
        fail_mode=fail_mode,
        mutation_target_slot=mutation_slot,
        mutation_rationale=rationale,
    )


# ── Fake candidate metrics (what the inner loop would have returned) ──


def _cand(
    m1: float = 0.0,
    m2: float = -2.0,
    m3: float = 180.0,
    m4: float = 3.0,
    m6: float = 0.15,
    m8: float = 0.15,
    m9: float = 0.45,
    peak: Optional[float] = None,
    final: Optional[float] = None,
    obs_code: str = "",
    reward_code: str = "",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """One candidate's metrics + optional code snippet."""
    d: Dict[str, Any] = {
        "M1_success_rate": m1,
        "M2_avg_return": m2,
        "M3_avg_steps": m3,
        "M4_avg_collisions": m4,
        "M6_coverage_progress": m6,
        "M8_agent_utilization_cv": m8,
        "M9_spatial_spread": m9,
    }
    if peak is not None:
        d["peak_M1"] = peak
    if final is not None:
        d["final_M1"] = final
    if obs_code:
        d["obs_code"] = obs_code
    if reward_code:
        d["reward_code"] = reward_code
    if error:
        d["_error"] = error
        d["_error_type"] = "FairnessViolation"
    return d


# ── Prior mutation_log entries ──────────────────────────────────


def _prior(
    slot: str = "guidance_observation",
    verdict: str = "neutral",
    excerpt: str = "",
    delta_peak: Optional[float] = 0.0,
    parent: str = "v2_fewshot_modular_v2",
    new_version: str = "v2_fewshot_modular_v2_mp_001",
) -> MutationLogEntry:
    return MutationLogEntry(
        ts="2026-04-20T10:00:00",
        run_id="prior-run",
        task_id="lero_mp_v3_dryrun_3m",
        seed=0,
        outer_iter=0,
        parent_version=parent,
        new_version=new_version,
        strategy_card={
            "target_slot": slot,
            "focus": ["prior focus text"],
            "rationale": "prior rationale",
        },
        slot_name=slot,
        slot_content_sha256="deadbeef" * 8,
        slot_content_excerpt=excerpt,
        pre_mutation_peak_M1=0.05,
        pre_mutation_best_M6=0.2,
        post_mutation_peak_M1=0.05 + (delta_peak or 0.0),
        post_mutation_best_M6=0.2,
        delta_peak_M1=delta_peak,
        delta_M6=0.0,
        verdict=verdict,
    )


# ── Named synthetic candidate code (for the Editor to cite) ─────


_OBS_GENERIC = """def enhance_observation(scenario_state):
    import torch
    lidar = scenario_state["lidar_targets"]
    return lidar
"""

_OBS_GAP_FEATURE = """def enhance_observation(scenario_state):
    import torch
    lidar = scenario_state["lidar_targets"]
    sorted_d, _ = torch.topk(lidar, 2, largest=False)
    gap = sorted_d[..., 1] - sorted_d[..., 0]
    proximity_count = (lidar < 0.35).sum(dim=-1, keepdim=True).float()
    return torch.cat([sorted_d, gap.unsqueeze(-1), proximity_count], dim=-1)
"""

_OBS_HOLD_APPROACH = """def enhance_observation(scenario_state):
    import torch
    lidar_t = scenario_state["lidar_targets"]
    lidar_a = scenario_state["lidar_agents"]
    near_target = (lidar_t.min(dim=-1, keepdim=True).values < 0.25).float()
    near_agent = (lidar_a.min(dim=-1, keepdim=True).values < 0.4).float()
    hold_signal = near_target * near_agent
    return torch.cat([lidar_t, lidar_a, hold_signal], dim=-1)
"""

_REWARD_UNSTABLE = """def compute_reward(scenario_state):
    import torch
    pos = scenario_state["agent_pos"]
    # Large-magnitude potential, unbounded
    return -(pos * pos).sum(dim=-1) * 100.0
"""

_REWARD_SMOOTH = """def compute_reward(scenario_state):
    import torch
    delta = scenario_state.get("coverage_delta", torch.zeros(1))
    return torch.tanh(delta * 5.0)
"""


# ── Scenario registry ──────────────────────────────────────────


ALL_SCENARIOS: Dict[str, Dict[str, Any]] = {
    # A. Fail-mode targeting
    "A1_flat_zero_baseline": {
        "description": "All candidates flat-zero — observation signal missing.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.005, m6=0.08), _rec(peak=0.010, m6=0.09)],
        "candidates": [
            _cand(m1=0.005, m6=0.08, obs_code=_OBS_GENERIC),
            _cand(m1=0.010, m6=0.09, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "observation_first",
        "expected": {
            "target_slot": ["guidance_observation", "guidance_shared"],
            "include_signals_contains_any": ["scalar"],
            "editor_mentions_any": [
                "proximity", "lidar_targets", "gap", "nearest",
            ],
        },
    },
    "A2_reward_hack_shape": {
        "description": "Peak M1=0.5 then collapses to 0.05 — reward instability.",
        "fail_mode": FailMode.REWARD_HACK,
        "history": [
            _rec(peak=0.50, final=0.05, fail_mode=FailMode.REWARD_HACK),
        ],
        "candidates": [
            _cand(m1=0.05, peak=0.50, final=0.05, reward_code=_REWARD_UNSTABLE),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "target_slot": ["guidance_reward", "guidance_shared"],
            "include_signals_contains_any": ["fingerprint", "curve_shape"],
            "editor_mentions_any": ["bounded", "clamp", "smooth", "tanh", "potential"],
        },
    },
    "A3_high_collisions": {
        "description": "M4=80+ per episode — pileup / crowding.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [
            _cand(m1=0.02, m4=82.0, m9=0.18, m6=0.15, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            "target_slot": ["guidance_observation", "guidance_shared"],
            "include_signals_contains_any": ["fingerprint", "scalar"],
            "editor_mentions_any": [
                "crowd", "proximity_count", "spacing", "lidar_agents", "repuls",
            ],
        },
    },
    "A4_role_imbalance": {
        "description": "M8>0.5 — only some agents contribute.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.05, m6=0.22)],
        "candidates": [
            _cand(m1=0.05, m6=0.22, m8=0.55, m9=0.5, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            "target_slot": [
                "guidance_observation", "guidance_shared", "guidance_reward",
            ],
            "editor_mentions_any": [
                "agent_idx", "assignment", "team", "role", "partner",
            ],
        },
    },
    "A5_all_candidates_errored_fairness": {
        "description": "Every candidate tripped FairnessViolation.",
        "fail_mode": FailMode.FAIRNESS_VIOLATION,
        "history": [
            _rec(peak=0.0, m6=0.0, fail_mode=FailMode.FAIRNESS_VIOLATION),
        ],
        "candidates": [
            _cand(m1=0.0, error="agent_pos forbidden in local mode"),
            _cand(m1=0.0, error="landmark_pos forbidden in local mode"),
        ],
        "priors": [],
        "bias": "observation_first",
        "expected": {
            # Any slot is fine here — the test is that the LLM doesn't
            # try to re-encode the fairness contract itself
            "editor_must_not_contain_all": ["local sensors only", "oracle"],
        },
    },

    # B. Critic behavior
    "B1_critic_catches_fairness_paraphrase": {
        "description": "Editor output reads like a fairness paraphrase.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [],
        "bias": "observation_first",
        "editor_override": (
            "Use only local sensors. Do not leak oracle state into the "
            "observation. Clamp reward magnitudes to [-50, 50]."
        ),
        "expected": {
            "critic_flags_fairness_restatement": True,
            "critic_quality_in": ["revise", "reject"],
        },
    },
    "B2_critic_accepts_specific_output": {
        "description": "Editor names specific features — Critic should keep.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [],
        "bias": "observation_first",
        "editor_override": (
            "Augment observation with lidar_targets top-2 distances plus "
            "their gap (nearest-to-2nd-nearest), and a proximity_count of "
            "how many lidar rays return a distance below covering_range. "
            "These expose target isolation and target crowding without "
            "needing oracle positions. Add a simple hold_signal: 1 when "
            "this agent is within covering_range of a target AND another "
            "agent is within covering_range of the same target."
        ),
        "expected": {
            "critic_flags_fairness_restatement": False,
            "critic_quality_in": ["keep"],
            "critic_cites_features_any": [
                "lidar_targets", "gap", "proximity_count", "hold_signal",
            ],
        },
    },
    "B3_critic_catches_duplicate_of_prior": {
        "description": "Editor output is near-verbatim copy of a prior.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [
            _prior(
                slot="guidance_observation",
                verdict="neutral",
                excerpt=(
                    "Use proximity_count (count of lidar hits within "
                    "covering_range) and the gap between nearest and "
                    "2nd-nearest target distances."
                ),
            ),
        ],
        "bias": "observation_first",
        "editor_override": (
            "Use proximity_count (count of lidar hits within covering_range) "
            "and the gap between nearest and 2nd-nearest target distances."
        ),
        "expected": {
            "critic_diverges_from_priors": False,
            "critic_quality_in": ["revise", "reject"],
        },
    },

    # C. include_signals gating
    "C1_no_outliers_default_scalar_only": {
        "description": "All metrics normal — Strategist should NOT upgrade tiers.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.05, m6=0.25)],
        "candidates": [_cand(m1=0.05, m6=0.25, obs_code=_OBS_GENERIC)],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            "include_signals_equals": ["scalar"],
        },
    },
    "C2_clear_reward_hack_upgrades_tier": {
        "description": "Visible peak-then-collapse — Strategist should add curve_shape or fingerprint.",
        "fail_mode": FailMode.REWARD_HACK,
        "history": [_rec(peak=0.40, final=0.05, fail_mode=FailMode.REWARD_HACK)],
        "candidates": [
            _cand(m1=0.05, peak=0.40, final=0.05, reward_code=_REWARD_UNSTABLE),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "include_signals_contains_any": ["fingerprint", "curve_shape"],
        },
    },

    # D. Prior-slot-version handling
    "D1_avoid_prior_regression_pattern": {
        "description": "Prior tried 'add lidar summary' and got regression.",
        "fail_mode": FailMode.HEALTHY,
        "history": [
            _rec(peak=0.05, mutation_slot="guidance_observation"),
            _rec(peak=0.02, mutation_slot="guidance_observation",
                 rationale="add lidar summary"),
        ],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [
            _prior(
                slot="guidance_observation",
                verdict="regression",
                delta_peak=-0.03,
                excerpt=(
                    "Add a summary of the lidar_targets readings: mean, "
                    "max, and min. This should help the policy understand "
                    "target density."
                ),
            ),
        ],
        "bias": "observation_first",
        "expected": {
            "target_slot": [
                "guidance_observation", "guidance_reward", "guidance_shared",
            ],
            # Editor must NOT duplicate the lidar-summary pattern
            "editor_avoids_any": ["lidar summary", "mean/max/min lidar"],
            "editor_mentions_any": [
                "gap", "proximity_count", "hold", "approach", "partner",
            ],
        },
    },
    "D2_build_on_prior_marginal_improvement": {
        "description": "Prior 'add proximity_count' got marginal improvement.",
        "fail_mode": FailMode.HEALTHY,
        "history": [
            _rec(peak=0.05, mutation_slot="guidance_observation"),
            _rec(peak=0.08, mutation_slot="guidance_observation",
                 rationale="add proximity_count"),
        ],
        "candidates": [_cand(m1=0.08, obs_code=_OBS_GENERIC)],
        "priors": [
            _prior(
                slot="guidance_observation",
                verdict="marginal_improvement",
                delta_peak=+0.03,
                excerpt=(
                    "Add proximity_count: the number of lidar_targets rays "
                    "within covering_range. This tells the policy how "
                    "crowded the local target is."
                ),
            ),
        ],
        "bias": "observation_first",
        "expected": {
            # Either extend proximity_count or add a related coordination signal
            "editor_mentions_any": [
                "proximity_count", "gap", "hold", "approach", "partner", "team",
            ],
        },
    },

    # E. Seed bias
    "E1_override_bias_when_evidence_contradicts": {
        "description": "obs-first bias + reward-hack evidence — Strategist should override bias.",
        "fail_mode": FailMode.REWARD_HACK,
        "history": [_rec(peak=0.40, final=0.05, fail_mode=FailMode.REWARD_HACK)],
        "candidates": [
            _cand(m1=0.05, peak=0.40, final=0.05, reward_code=_REWARD_UNSTABLE),
        ],
        "priors": [],
        "bias": "observation_first",
        "expected": {
            "target_slot": ["guidance_reward", "guidance_shared"],
        },
    },

    # A6-A8: additional fail-mode variants
    "A6_oscillating_M1": {
        "description": "M1 bounces between evals — training instability.",
        "fail_mode": FailMode.HEALTHY,
        "history": [
            _rec(peak=0.20, final=0.05, m6=0.35, seed_std=0.18),
        ],
        "candidates": [
            _cand(m1=0.05, peak=0.20, final=0.05, m6=0.35,
                  reward_code=_REWARD_UNSTABLE),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "target_slot": ["guidance_reward", "guidance_shared"],
            "include_signals_contains_any": ["fingerprint", "curve_shape"],
            "editor_mentions_any": [
                "stability", "smooth", "bounded", "clip", "tanh",
                "variance", "oscillat",
            ],
        },
    },
    "A7_nan_crash_all_candidates": {
        "description": "All candidates crashed with NaN — reward magnitude inflation.",
        "fail_mode": FailMode.NAN_CRASH,
        "history": [
            _rec(peak=0.0, final=0.0, fail_mode=FailMode.NAN_CRASH),
        ],
        "candidates": [
            _cand(m1=0.0, error="RuntimeError: NaN in action logits"),
            _cand(m1=0.0, error="RuntimeError: inf in rewards"),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "target_slot": ["guidance_reward", "guidance_shared"],
            "editor_mentions_any": [
                "bounded", "clip", "clamp", "nan", "magnitude", "tanh",
                "finite", "stability",
            ],
        },
    },
    "A8_low_M2_but_nonzero_M1": {
        "description": "M1 non-zero but M2 very negative — reward shaping misaligned.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.08, final=0.05, m6=0.25, m2=-15.0)],
        "candidates": [
            _cand(m1=0.05, m6=0.25, m2=-15.0, reward_code=_REWARD_UNSTABLE),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "target_slot": ["guidance_reward", "guidance_shared"],
            "editor_mentions_any": [
                "magnitude", "scale", "shaping", "normaliz", "potential",
                "smooth",
            ],
        },
    },

    # B4-B5: Additional Critic edge cases
    "B4_critic_flags_generic_advice": {
        "description": "Editor writes vague non-specific advice.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [],
        "bias": "observation_first",
        "editor_override": (
            "Make the observation more informative for coordination. "
            "Include useful features. Avoid redundant information. "
            "Focus on what helps the agent succeed at the task."
        ),
        "expected": {
            "critic_quality_in": ["revise", "reject"],
            # At least one red flag should fire
        },
    },
    "B5_critic_keeps_concise_specific_edit": {
        "description": "Editor writes short but concretely specific advice.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.15)],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [],
        "bias": "observation_first",
        "editor_override": (
            "Compute proximity_count: number of lidar_targets readings "
            "below covering_range. Also compute the gap between the 1st "
            "and 2nd nearest target distances from lidar_targets."
        ),
        "expected": {
            "critic_quality_in": ["keep", "revise"],  # either acceptable
            "critic_flags_fairness_restatement": False,
            "critic_cites_features_any": [
                "proximity_count", "gap", "lidar_targets",
            ],
        },
    },

    # C3: Multi-outlier — must pick one focus
    "C3_multi_outlier_picks_one": {
        "description": "Both M4=90 AND M9=0.15 — two outliers — must not scattershot.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.20)],
        "candidates": [
            _cand(m1=0.02, m4=90.0, m9=0.15, m6=0.20, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            # Accept either shared or observation as valid picks
            "target_slot": ["guidance_observation", "guidance_shared"],
            # At least ONE focus identifier — but we don't expect 4 focuses
            "editor_mentions_any": [
                "proximity", "spacing", "crowd", "lidar_agents",
            ],
        },
    },

    # D3: Conflicting priors on same slot
    "D3_conflicting_priors_same_slot": {
        "description": "Same slot had both a regression AND a marginal_improvement prior.",
        "fail_mode": FailMode.HEALTHY,
        "history": [
            _rec(peak=0.05, mutation_slot="guidance_observation"),
            _rec(peak=0.08, mutation_slot="guidance_observation"),
            _rec(peak=0.02, mutation_slot="guidance_observation"),
        ],
        "candidates": [_cand(m1=0.02, obs_code=_OBS_GENERIC)],
        "priors": [
            _prior(
                slot="guidance_observation",
                verdict="marginal_improvement",
                delta_peak=+0.03,
                excerpt="Add proximity_count of lidar hits within covering_range.",
                new_version="v2_fewshot_modular_v2_mp_001",
            ),
            _prior(
                slot="guidance_observation",
                verdict="regression",
                delta_peak=-0.06,
                excerpt=(
                    "Add lidar mean, max, min summaries plus raw sensor "
                    "rollouts from all rays."
                ),
                new_version="v2_fewshot_modular_v2_mp_002",
            ),
        ],
        "bias": "observation_first",
        "expected": {
            "target_slot": [
                "guidance_observation", "guidance_reward", "guidance_shared",
            ],
            "editor_avoids_any": ["lidar mean, max, min", "raw sensor rollouts"],
            "editor_mentions_any": [
                "proximity_count", "gap", "hold", "approach", "team",
            ],
        },
    },

    # E2-E3: remaining bias scenarios
    "E2_reward_first_bias_observation_evidence": {
        "description": "reward_first bias + observation evidence (M9 low, crowding) — should override bias.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.02, m6=0.10)],
        "candidates": [
            _cand(m1=0.02, m4=15.0, m9=0.12, m6=0.10, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "reward_first",
        "expected": {
            "target_slot": ["guidance_observation", "guidance_shared"],
        },
    },
    "E3_exploratory_cold_start": {
        "description": "Exploratory bias, minimal prior history — Strategist picks based on fail_mode alone.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.005, m6=0.05)],
        "candidates": [
            _cand(m1=0.005, m6=0.05, obs_code=_OBS_GENERIC),
        ],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            # Any slot OK; just verify it doesn't error
            "target_slot": [
                "guidance_observation", "guidance_reward", "guidance_shared",
            ],
            "confidence_in": ["small", "medium", "large"],
        },
    },

    # H1: repeated fairness violations — should not touch observation at all
    "H1_three_fairness_violations_in_a_row": {
        "description": "Three successive FAIRNESS_VIOLATION runs — candidates can't stop looking at oracle state.",
        "fail_mode": FailMode.FAIRNESS_VIOLATION,
        "history": [
            _rec(peak=0.0, fail_mode=FailMode.FAIRNESS_VIOLATION),
            _rec(peak=0.0, fail_mode=FailMode.FAIRNESS_VIOLATION),
            _rec(peak=0.0, fail_mode=FailMode.FAIRNESS_VIOLATION),
        ],
        "candidates": [
            _cand(m1=0.0, error="KeyError on 'landmark_pos' - AllowedKeysDict"),
            _cand(m1=0.0, error="KeyError on 'agent_pos_others'"),
        ],
        "priors": [],
        "bias": "observation_first",
        "expected": {
            # The Strategist shouldn't try to re-encode the fairness rules
            # but should pick a slot that CAN help (e.g. guidance_shared
            # to remind the LLM what IS allowed)
            "editor_must_not_contain_all": ["local sensors only", "oracle"],
        },
    },

    # F. All-metrics-good — negative control
    "F1_no_mutation_needed": {
        "description": "Healthy + high peak_M1 — LLM should still pick a slot (we can't stop it) but focus should be minimal.",
        "fail_mode": FailMode.HEALTHY,
        "history": [_rec(peak=0.75, final=0.70, m6=0.85, m2=3.0)],
        "candidates": [
            _cand(m1=0.75, m6=0.85, peak=0.75, final=0.70,
                  obs_code=_OBS_HOLD_APPROACH),
        ],
        "priors": [],
        "bias": "exploratory",
        "expected": {
            # Soft expectation — Strategist picks SOMETHING but low confidence
            "confidence_in": ["small", "medium"],
        },
    },
}
