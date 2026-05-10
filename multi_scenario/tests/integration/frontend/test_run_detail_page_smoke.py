"""F7.3 smoke: Run Detail page renders cleanly under streamlit's AppTest harness."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from datetime import datetime, timezone
from pathlib import Path

import pytest

from multi_scenario.adapters.storage.local import LocalStorageAdapter
from multi_scenario.domain.models import (
    ExperimentConfig,
    ExperimentResult,
    MetricRecord,
    RunState,
    RunStateRecord,
)


def _seed_run(experiments_dir: Path) -> Path:
    """Lay down one minimal DONE run so the page has data to load."""
    run_dir = experiments_dir / "demo_s0__t1"
    run_dir.mkdir(parents=True)
    storage = LocalStorageAdapter()
    cfg = ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo", "seed": 0},
            "scenario": {"type": "discovery", "params": {}},
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1},
            "evaluation": {"interval_iters": 1, "episodes": 1},
        }
    )
    storage.save_config(run_dir, cfg)
    storage.save_result(
        run_dir,
        ExperimentResult(
            run_id="demo_s0",
            exp_id="demo",
            scenario="discovery",
            algorithm="mappo",
            seed=0,
            run_timestamp="20260507_1500",
            metrics=[
                MetricRecord(name="M1_success_rate", value=0.5),
                MetricRecord(name="M3_steps", value=12.0),
            ],
            config_snapshot={"n_agents": 2},
            n_envs=1,
            n_eval_episodes=10,
        ),
    )
    ts = datetime(2026, 5, 7, 15, 0, tzinfo=timezone.utc)
    rec = (
        RunStateRecord.initial(ts)
        .transition_to(RunState.RUNNING, ts)
        .transition_to(RunState.DONE, ts)
    )
    storage.save_run_state(run_dir, rec)
    return run_dir


@pytest.mark.slow
def test_run_detail_renders_with_seeded_run(tmp_path: Path) -> None:
    """Seed one run, point the page at it, expect no exception + metrics rendered."""
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY

    # pylint: disable=import-outside-toplevel
    from streamlit.testing.v1 import AppTest

    _seed_run(tmp_path)
    page_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "multi_scenario"
        / "frontend"
        / "pages"
        / "run_detail.py"
    )
    at = AppTest.from_file(str(page_path), default_timeout=10.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(tmp_path)
    at.run()
    assert not at.exception
    # Header + at least one metric tile renders. F8.2.E gave the M1 tile
    # its glossary label ("M1 — Success Rate") instead of the previous
    # auto-titlecased "M1 Success Rate".
    metric_labels = {m.label for m in at.metric}
    assert "M1 — Success Rate" in metric_labels


@pytest.mark.slow
def test_run_detail_empty_state_when_no_runs(tmp_path: Path) -> None:
    """Empty experiments dir → page short-circuits with 'No runs' info, no exception."""
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY

    # pylint: disable=import-outside-toplevel
    from streamlit.testing.v1 import AppTest

    page_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "multi_scenario"
        / "frontend"
        / "pages"
        / "run_detail.py"
    )
    at = AppTest.from_file(str(page_path), default_timeout=10.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(tmp_path)
    at.run()
    assert not at.exception
    info_texts = [(card.value or "") for card in at.info]
    assert any("No runs" in t for t in info_texts)
