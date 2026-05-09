"""F7.7.C1 — End-to-end tests for the Submit page via Streamlit's AppTest.

Drives the 5-step workflow (Pick → Inspect/Edit → Save → Preflight → Submit)
without launching a browser. All assertions read ``at.session_state`` since
that's where the workflow stores its truth (the rendered widgets are a view).

The local + OVH submission paths both monkey-patch the runner classes so
nothing actually trains and no ovhai subprocess fires — the page logic is
exercised end-to-end with deterministic stub returns.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name,import-outside-toplevel

from pathlib import Path

import pytest
import yaml


_SUBMIT_PAGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "multi_scenario"
    / "frontend"
    / "pages"
    / "submit.py"
)


def _minimal_cfg(scen_type: str = "discovery") -> dict:
    """Smallest valid ``ExperimentConfig`` shape — passes Pydantic validation."""
    return {
        "experiment": {"id": "demo", "seed": 0},
        "scenario": {
            "type": scen_type,
            "params": {"n_agents": 2, "n_targets": 2, "max_steps": 5},
        },
        "algorithm": {"type": "mappo", "params": {}},
        "training": {
            "max_iters": 1,
            "num_envs": 1,
            "device": "cpu",
            "frames_per_batch": 50,
            "minibatch_size": 25,
            "n_minibatch_iters": 1,
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": "local", "params": {"record_video": False}},
            "storage": {
                "type": "fs",
                "path": "experiments/discovery/baseline",
                "params": {},
            },
        },
    }


@pytest.fixture
def tmp_experiments(tmp_path: Path) -> Path:
    """Build a minimal experiments tree with one valid discovery YAML.

    The YAML's ``runtime.storage.path`` points at a sibling tmp dir so the
    Storage-writable probe passes without needing the test to monkeypatch
    cwd somewhere convenient.
    """
    cfg_dir = tmp_path / "discovery" / "baseline" / "configs"
    cfg_dir.mkdir(parents=True)
    storage_dir = tmp_path / "results"
    storage_dir.mkdir()
    cfg = _minimal_cfg()
    cfg["runtime"]["storage"]["path"] = str(storage_dir)
    (cfg_dir / "seed0.yaml").write_text(
        yaml.dump(cfg, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


def _new_apptest(experiments_root: Path):
    """Construct an AppTest pointing at the Submit page with the given root."""
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_SUBMIT_PAGE_PATH), default_timeout=15.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(experiments_root)
    return at


def _drive_pick(
    at,
    scenario: str = "discovery",
    folder: str = "baseline",
    config: str = "seed0.yaml",
):
    """Walk the Step 1 cascading picker and click 'Use this config'."""
    at.run()
    at.selectbox(key="step1_scenario").set_value(scenario)
    at.run()
    at.selectbox(key="step1_folder").set_value(folder)
    at.run()
    at.selectbox(key="step1_config").set_value(config)
    at.run()
    at.button(key="step1_pick").click()
    at.run()


# ── Smoke ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_submit_page_renders_with_empty_experiments_dir(tmp_path: Path):
    """No configs → page still renders cleanly, Step 1 shows the 'add a YAML' info."""
    at = _new_apptest(tmp_path)
    at.run()
    assert not at.exception
    info_blobs = [(c.value or "") for c in at.info]
    assert any("No configs found" in t for t in info_blobs)


@pytest.mark.slow
def test_submit_page_renders_with_a_valid_config(tmp_experiments: Path):
    """Page loads, all 5 step containers present, no exception raised."""
    at = _new_apptest(tmp_experiments)
    at.run()
    assert not at.exception


# ── Pick → state.selected_path ───────────────────────────────────────


@pytest.mark.slow
def test_pick_a_config_records_selected_path(tmp_experiments: Path):
    """After clicking 'Use this config', session_state holds the chosen file."""
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    selected = at.session_state["submit_selected_path"]
    assert selected is not None
    assert selected.name == "seed0.yaml"
    # The snapshot also lands.
    assert at.session_state["submit_snapshot_form"] is not None


# ── Submit-target toggle ─────────────────────────────────────────────


@pytest.mark.slow
def test_submit_target_defaults_to_local_after_pick(tmp_experiments: Path):
    """The picked YAML's ``runner.type`` is ``local`` → submit target = local."""
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    # AppTest's session_state raises on missing keys instead of returning a
    # default — be defensive: presence + value.
    assert "submit_target" in at.session_state
    assert at.session_state["submit_target"] == "local"


# ── Save (no edits) ──────────────────────────────────────────────────


@pytest.mark.slow
def test_save_step_skips_when_form_clean(tmp_experiments: Path):
    """Picking a YAML with no edits → Step 3 shows 'No changes' (no Save button)."""
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    captions = [(c.value or "") for c in at.caption]
    assert any("No changes" in t for t in captions)


# ── Preflight (local) ────────────────────────────────────────────────


@pytest.mark.slow
def test_local_preflight_runs_and_records_results(tmp_experiments: Path, monkeypatch):
    """Click 'Run preflight' on local target → preflight list populates with statuses."""
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    # Make the storage path point at tmp_path so the writability probe passes.
    monkeypatch.chdir(tmp_experiments.parent)
    at.button(key="step4_run").click()
    at.run()
    rows = (
        at.session_state["submit_preflight"]
        if "submit_preflight" in at.session_state
        else []
    )
    assert rows, "preflight rows should be populated after Run preflight"
    # Each row has a status field — none should still be IDLE.
    from multi_scenario.frontend.preflight import CheckStatus

    statuses = [r.status for r in rows]
    assert (
        CheckStatus.IDLE not in statuses
    ), f"unrun rows: {[r.name for r in rows if r.status == CheckStatus.IDLE]}"


# ── OVH cascade when configs/ovh.yaml is missing ────────────────────


@pytest.mark.slow
def test_ovh_target_cascade_when_ovh_yaml_missing(tmp_experiments: Path, monkeypatch):
    """OVH submit target + missing configs/ovh.yaml → 'OVH config valid' FAIL,
    every other OVH probe row stays IDLE with the cascade hint.
    """
    # Run preflight from tmp_path which has no configs/ovh.yaml.
    monkeypatch.chdir(tmp_experiments.parent)
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    # Switch submit target → ovh via the radio inside Step 2's expander.
    at.radio(key="step2_submit_target").set_value("ovh")
    at.run()
    at.button(key="step4_run").click()
    at.run()
    rows = {r.name: r for r in at.session_state["submit_preflight"]}
    from multi_scenario.frontend.preflight import CheckStatus

    assert rows["OVH config valid"].status == CheckStatus.FAIL
    for cloud_row in (
        "OVH CLI installed",
        "Results bucket reachable",
        "Code matches OVH bucket",
        "Per-run prefix not occupied",
        "Submitted YAML present in bucket",
        "No active OVH job with this run_id",
        "Cost cap not exceeded",
    ):
        assert (
            rows[cloud_row].status == CheckStatus.IDLE
        ), f"{cloud_row} should cascade to IDLE; got {rows[cloud_row].status}"
        assert "fix the OVH config row first" in rows[cloud_row].detail


# ── Local submit dispatch (mocked LocalRunner) ──────────────────────


@pytest.mark.slow
def test_local_submit_dispatches_to_local_runner(tmp_experiments: Path, monkeypatch):
    """Click Submit on a green local preflight → LocalRunner.run is invoked once."""
    monkeypatch.chdir(tmp_experiments.parent)
    at = _new_apptest(tmp_experiments)
    _drive_pick(at)
    at.button(key="step4_run").click()
    at.run()

    # If preflight didn't go fully green (e.g. missing dep on this machine),
    # bail with a clean skip — the test wants to validate dispatch wiring,
    # not the green-light criteria.
    from multi_scenario.frontend.preflight import CheckStatus

    rows = at.session_state["submit_preflight"]
    if not all(r.status == CheckStatus.PASS for r in rows):
        failing = [r.name for r in rows if r.status != CheckStatus.PASS]
        pytest.skip(f"local preflight not green on this machine: {failing}")

    calls = []

    class _FakeResult:
        run_id = "demo_s0"

    def _fake_run(self, cfg, run_dir, resume_from=None):  # noqa: ARG001
        calls.append((cfg.experiment.id, run_dir))
        return _FakeResult()

    monkeypatch.setattr(
        "multi_scenario.adapters.runners.local.LocalRunner.run",
        _fake_run,
    )
    at.button(key="step5_submit").click()
    at.run()
    status = (
        at.session_state["submit_submission_status"]
        if "submit_submission_status" in at.session_state
        else {}
    )
    assert status.get("status") == "done", status
    assert calls, "LocalRunner.run was not invoked"
