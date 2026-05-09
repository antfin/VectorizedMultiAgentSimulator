"""F7.4 tests: aggregate_metric, flatten_config, build_config_diff_table."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from pathlib import Path

import pandas as pd
import pytest

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.frontend.aggregations import (
    aggregate_metric,
    build_config_diff_table,
    flatten_config,
)

# ── aggregate_metric ──────────────────────────────────────────────────


def _df_two_scen_two_algo() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "discovery", "algorithm": "mappo", "M1_success_rate": 0.8},
            {"scenario": "discovery", "algorithm": "mappo", "M1_success_rate": 0.6},
            {"scenario": "discovery", "algorithm": "ippo", "M1_success_rate": 0.5},
            {"scenario": "navigation", "algorithm": "mappo", "M1_success_rate": 0.9},
        ]
    )


def test_aggregate_metric_groups_by_two_axes():
    out = aggregate_metric(
        _df_two_scen_two_algo(), "M1_success_rate", by=["scenario", "algorithm"]
    )
    assert set(out.columns) == {"scenario", "algorithm", "mean", "sem"}
    # 3 groups: (discovery, mappo), (discovery, ippo), (navigation, mappo)
    assert len(out) == 3


def test_aggregate_metric_mean_is_correct():
    out = aggregate_metric(
        _df_two_scen_two_algo(), "M1_success_rate", by=["scenario", "algorithm"]
    )
    discovery_mappo = out[
        (out["scenario"] == "discovery") & (out["algorithm"] == "mappo")
    ]
    assert discovery_mappo["mean"].iloc[0] == pytest.approx(0.7)


def test_aggregate_metric_returns_empty_when_metric_missing():
    out = aggregate_metric(
        _df_two_scen_two_algo(), "M99_does_not_exist", by=["scenario"]
    )
    assert out.empty
    assert list(out.columns) == ["scenario", "mean", "sem"]


def test_aggregate_metric_supports_median():
    out = aggregate_metric(
        _df_two_scen_two_algo(),
        "M1_success_rate",
        by=["scenario", "algorithm"],
        how="median",
    )
    discovery_mappo = out[
        (out["scenario"] == "discovery") & (out["algorithm"] == "mappo")
    ]
    assert discovery_mappo["mean"].iloc[0] == pytest.approx(0.7)


# ── flatten_config ────────────────────────────────────────────────────


def test_flatten_config_dotted_keys():
    payload = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_config(payload)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flatten_config_none_becomes_em_dash():
    assert flatten_config({"x": None}) == {"x": "—"}


def test_flatten_config_keeps_lists_as_is():
    payload = {"seeds": [1, 2, 3]}
    assert flatten_config(payload) == {"seeds": [1, 2, 3]}


# ── build_config_diff_table ──────────────────────────────────────────


def _seed_run(run_dir: Path, seed: int, n_agents: int, algo: str = "mappo") -> None:
    run_dir.mkdir(parents=True)
    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": run_dir.name, "seed": seed},
            "scenario": {"type": "discovery", "params": {"n_agents": n_agents}},
            "algorithm": {"type": algo, "params": {}},
            "training": {"max_iters": 1},
            "evaluation": {"interval_iters": 1, "episodes": 1},
        }
    )
    LocalStorageAdapter().save_config(run_dir, cfg)


def test_build_config_diff_table_empty_for_no_runs(tmp_path: Path):
    table, diffs = build_config_diff_table([])
    assert table.empty
    assert diffs == []


def test_build_config_diff_table_two_runs_diff_seeds(tmp_path: Path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    _seed_run(a, seed=0, n_agents=2)
    _seed_run(b, seed=1, n_agents=2)
    table, diffs = build_config_diff_table([a, b])
    assert {"run_a", "run_b"} <= set(table.columns)
    # ``experiment.seed`` differs (0 vs 1); ``experiment.id`` differs (run_a vs run_b);
    # ``scenario.params.n_agents`` is the same (2). Identical rows must NOT highlight.
    assert "experiment.seed" in diffs
    assert "scenario.params.n_agents" not in diffs


def test_build_config_diff_table_skips_unloadable_runs(tmp_path: Path):
    """A run dir with no config.json is silently skipped."""
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _seed_run(good, seed=0, n_agents=2)
    bad.mkdir()  # empty
    table, _ = build_config_diff_table([good, bad])
    # Only ``good`` survives.
    assert list(table.columns) == ["good"]
