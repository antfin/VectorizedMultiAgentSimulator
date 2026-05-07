"""F7.2 tests: filter_runs — multi-axis filter helper for the Experiments Browser."""

# pylint: disable=missing-function-docstring

import pandas as pd

from multi_scenario.frontend.filters import filter_runs


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"run_id": "demo_a_s0", "scenario": "discovery", "algorithm": "mappo", "state": "DONE"},
            {"run_id": "demo_a_s1", "scenario": "discovery", "algorithm": "mappo", "state": "CRASHED"},
            {"run_id": "demo_b_s0", "scenario": "navigation", "algorithm": "ippo", "state": "DONE"},
            {"run_id": "demo_c_s0", "scenario": "transport", "algorithm": "masac", "state": "RUNNING"},
        ]
    )


def test_no_filters_returns_all():
    df = _df()
    out = filter_runs(df)
    assert len(out) == len(df)


def test_scenario_filter():
    out = filter_runs(_df(), scenarios=["discovery"])
    assert len(out) == 2
    assert set(out["run_id"]) == {"demo_a_s0", "demo_a_s1"}


def test_multiple_scenarios_union():
    out = filter_runs(_df(), scenarios=["discovery", "navigation"])
    assert len(out) == 3


def test_algorithm_filter():
    out = filter_runs(_df(), algorithms=["mappo"])
    assert len(out) == 2


def test_state_filter():
    out = filter_runs(_df(), states=["DONE"])
    assert len(out) == 2
    assert set(out["state"]) == {"DONE"}


def test_search_substring_case_insensitive():
    out = filter_runs(_df(), search="DEMO_A")
    assert len(out) == 2
    assert all("demo_a" in r for r in out["run_id"])


def test_filters_compose_with_and():
    """Combining axes intersects (AND), not unions."""
    out = filter_runs(_df(), scenarios=["discovery"], states=["DONE"])
    assert len(out) == 1
    assert out.iloc[0]["run_id"] == "demo_a_s0"


def test_empty_lists_treated_as_no_filter():
    """Passing ``[]`` for an axis should NOT reduce results to zero."""
    out = filter_runs(_df(), scenarios=[], algorithms=[], states=[])
    assert len(out) == len(_df())


def test_missing_columns_skipped_silently():
    """If a column the filter targets doesn't exist on df, the filter no-ops."""
    df = pd.DataFrame([{"run_id": "x"}, {"run_id": "y"}])  # no scenario/algo/state columns
    out = filter_runs(df, scenarios=["discovery"], algorithms=["mappo"], states=["DONE"])
    assert len(out) == 2  # everything kept; columns absent → filter ignored
