"""F8.2.E — single-source-of-truth registry for the M1–M9 metric explanations.

Used by the Streamlit Run Detail / Comparison pages (tooltips, "Learn more"
deep-links) and by the mkdocs site (metric reference page) so the wiki and
the UI never drift. Keep entries plain-English; this is the surface a
student or external collaborator hits first.

Each entry's ``doc_slug`` is the mkdocs anchor the "Learn more →" link
deep-links to (resolved through ``frontend/doc_links.py`` at F8.2.F).

CommonMetricsBundle (the producer) and this glossary share the same metric
IDs — a unit test asserts both stay in sync, so adding a metric to one
without the other will fail CI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInfo:
    """One metric's user-facing description."""

    label: str  # short human-friendly label (also used as tooltip header)
    description: str  # 1-2 sentence plain-English explanation
    units: str  # "fraction in [0,1]", "steps", "collisions/episode", …
    doc_slug: str  # mkdocs anchor (F8.2.F deep-link target)


_METRICS: dict[str, MetricInfo] = {
    "M1_success_rate": MetricInfo(
        label="M1 — Success Rate",
        description=(
            "Fraction of evaluation episodes where the agents fully solved the "
            "task (covered all targets, reached the goal, etc., per the "
            "scenario's success predicate). Higher is better."
        ),
        units="fraction in [0, 1]",
        doc_slug="results-analysis/metrics#m1-success-rate",
    ),
    "M2_avg_return": MetricInfo(
        label="M2 — Average Return",
        description=(
            "Mean cumulative reward per evaluation episode. Captures the "
            "policy's overall optimisation target — useful for cross-config "
            "comparisons but harder to interpret in absolute terms than M1."
        ),
        units="reward units (scenario-specific)",
        doc_slug="results-analysis/metrics#m2-average-return",
    ),
    "M3_steps": MetricInfo(
        label="M3 — Steps to Termination",
        description=(
            "Mean number of environment steps before each evaluation episode "
            "terminated. Lower is better when M1 is high (= solved faster); "
            "equals max_steps when M1 is low (= didn't solve)."
        ),
        units="steps",
        doc_slug="results-analysis/metrics#m3-steps",
    ),
    "M4_collisions": MetricInfo(
        label="M4 — Collisions per Episode",
        description=(
            "Mean number of agent-agent collisions per evaluation episode. "
            "Lower is better. Often trades off against speed (M3) when the "
            "scenario penalises both."
        ),
        units="collisions / episode",
        doc_slug="results-analysis/metrics#m4-collisions",
    ),
    "M5_tokens": MetricInfo(
        label="M5 — Communication Tokens",
        description=(
            "For comm-enabled scenarios (LERO, learned-comm baselines), the "
            "mean number of unique tokens produced per episode. ``null`` when "
            "the scenario doesn't expose communication."
        ),
        units="tokens / episode (or null)",
        doc_slug="results-analysis/metrics#m5-comm-tokens",
    ),
    "M6_coverage_progress": MetricInfo(
        label="M6 — Coverage Progress",
        description=(
            "Average fraction of targets covered per episode (cumsum-based, "
            "monotone non-decreasing). 0.866 means agents reached 86.6% of "
            "targets on average. Useful for partial-credit scoring when M1 is "
            "near-zero (= rarely full success but consistent partial progress)."
        ),
        units="fraction in [0, 1]",
        doc_slug="results-analysis/metrics#m6-coverage-progress",
    ),
    "M7_sample_efficiency": MetricInfo(
        label="M7 — Sample Efficiency",
        description=(
            "Frames consumed before reaching the run's M1 peak. Smaller values "
            "mean the policy learned faster. Computed at end-of-run from the "
            "eval-M1 trajectory; ``null`` when the curve doesn't peak."
        ),
        units="frames (or null)",
        doc_slug="results-analysis/metrics#m7-sample-efficiency",
    ),
    "M8_agent_utilization": MetricInfo(
        label="M8 — Agent Utilization (CV)",
        description=(
            "Coefficient of variation of per-agent contribution to the team "
            "reward. High CV = some agents do most of the work; CV near zero "
            "= effort is balanced. ``null`` when the scenario only produces a "
            "shared scalar reward (no per-agent decomposition)."
        ),
        units="dimensionless (CV)",
        doc_slug="results-analysis/metrics#m8-agent-utilization",
    ),
    "M9_spatial_spread": MetricInfo(
        label="M9 — Spatial Spread",
        description=(
            "Mean pairwise agent distance over the episode — captures whether "
            "the team disperses (large) or huddles (small). Useful for flocking "
            "and rendezvous tasks where the strategy preference shows up here "
            "before it shows up in M1."
        ),
        units="distance units (scenario-specific)",
        doc_slug="results-analysis/metrics#m9-spatial-spread",
    ),
}


def metric_info(metric_id: str) -> MetricInfo | None:
    """Return the registry entry for ``metric_id`` or None if unknown."""
    return _METRICS.get(metric_id)


def all_metric_ids() -> list[str]:
    """Stable-ordered list of every metric ID covered by this glossary."""
    return list(_METRICS.keys())


def tooltip_text(metric_id: str) -> str:
    """Compose the short tooltip body shown next to a metric tile.

    Returns a Streamlit-friendly multiline string (label + description +
    units). Falls back to the metric_id if the registry doesn't know it
    (defensive — shouldn't happen given the unit-test invariant).

    F8.2.F: when the docs site is reachable, append a ``📖 Learn more →``
    link to the full reference page. Using ``doc_links.doc_link`` keeps
    the URL resolution single-source.
    """
    info = _METRICS.get(metric_id)
    if info is None:
        return metric_id
    # pylint: disable=import-outside-toplevel
    from multi_scenario.frontend.doc_links import doc_link

    learn_more = f"📖 [Learn more →]({doc_link(info.doc_slug)})"
    return (
        f"**{info.label}**\n\n"
        f"{info.description}\n\n"
        f"_Units: {info.units}_\n\n"
        f"{learn_more}"
    )
