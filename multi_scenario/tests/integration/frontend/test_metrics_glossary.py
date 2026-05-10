"""F8.2.E — single-source-of-truth invariant for the metrics glossary.

Every metric ID produced by ``CommonMetricsBundle.compute(...)`` must have
an entry in ``frontend/metrics_glossary.py``. A failing test here means a
new metric landed in the bundle without a user-facing description — fix by
adding a ``MetricInfo`` entry to the glossary.

Same pattern catches the inverse: stale glossary entries for metrics no
longer produced. Both fail loudly so the wiki / Streamlit can never drift
silently from the actual M1–M9 surface.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from multi_scenario.adapters.metrics.common import CommonMetricsBundle
from multi_scenario.frontend.benchmarl_scalars_glossary import (
    all_curated_scalar_names,
    scalar_hint,
)
from multi_scenario.frontend.metrics_glossary import (
    all_metric_ids,
    metric_info,
    tooltip_text,
)


# Reproduce the rollout shape CommonMetricsBundle expects so we can call
# .compute() with a stub scenario and harvest the actual metric IDs.
class _NoOpScenario:
    def has_comm(self) -> bool:
        return False

    def success_predicate(self, rollout):  # noqa: ARG002
        return None

    def coverage_progress(self, rollout):  # noqa: ARG002
        return None

    def utilization_predicate(self, rollout):  # noqa: ARG002
        return None


def _empty_rollout() -> dict:
    return {
        "episode_returns": None,
        "episode_lengths": None,
        "episode_collisions": None,
    }


def _bundle_metric_ids() -> set[str]:
    bundle = CommonMetricsBundle()
    out = bundle.compute(_empty_rollout(), _NoOpScenario())
    return set(out.keys())


def test_glossary_covers_every_metric_in_the_bundle():
    """No M-series metric is allowed to land without a user-facing description."""
    bundle_ids = _bundle_metric_ids()
    glossary_ids = set(all_metric_ids())
    missing = bundle_ids - glossary_ids
    extra = glossary_ids - bundle_ids
    assert not missing, (
        f"glossary is missing entries for: {sorted(missing)} — add to "
        "frontend/metrics_glossary.py"
    )
    assert not extra, (
        f"glossary has stale entries for metrics no longer in the bundle: "
        f"{sorted(extra)} — remove from frontend/metrics_glossary.py"
    )


@pytest.mark.parametrize("metric_id", sorted(_bundle_metric_ids()))
def test_glossary_entry_is_well_formed(metric_id: str):
    """Each entry must have non-empty label / description / units / doc_slug."""
    info = metric_info(metric_id)
    assert info is not None
    assert info.label.strip(), f"{metric_id}: empty label"
    assert info.description.strip(), f"{metric_id}: empty description"
    assert info.units.strip(), f"{metric_id}: empty units"
    assert info.doc_slug.strip(), f"{metric_id}: empty doc_slug"
    # doc_slug should not start with a slash (mkdocs anchors are relative).
    assert not info.doc_slug.startswith(
        "/"
    ), f"{metric_id}: doc_slug should be relative (no leading '/')"


@pytest.mark.parametrize("metric_id", sorted(_bundle_metric_ids()))
def test_tooltip_text_renders_clean_markdown(metric_id: str):
    """``tooltip_text`` returns a non-empty markdown string with all parts."""
    text = tooltip_text(metric_id)
    info = metric_info(metric_id)
    assert text  # non-empty
    assert info.label in text
    assert info.description in text
    assert info.units in text


def test_tooltip_text_falls_back_for_unknown_metric():
    """Defensive: an unknown metric ID returns the raw ID, not a crash."""
    assert tooltip_text("M99_alien") == "M99_alien"


# ── BenchMARL scalar glossary smoke ────────────────────────────────────


def test_curated_scalar_hints_are_nonempty():
    """Every curated entry has a usable hint string."""
    for name in all_curated_scalar_names():
        hint = scalar_hint(name)
        assert hint and hint.strip(), f"empty hint for {name}"


def test_scalar_hint_synthesises_for_unknown_filename():
    """Unknown CSV name → fallback hint generated from the BM convention."""
    hint = scalar_hint("collection_my_custom_signal_mean.csv")
    assert hint and hint.strip()
    # The synthesiser should classify the bucket correctly.
    assert "training" in hint.lower() or "collection" in hint.lower()


def test_scalar_hint_handles_csv_extension_optionally():
    """Stripping ``.csv`` is the caller's job-or-not — both should work."""
    with_ext = scalar_hint("counters_iter.csv")
    without_ext = scalar_hint("counters_iter")
    assert with_ext == without_ext
