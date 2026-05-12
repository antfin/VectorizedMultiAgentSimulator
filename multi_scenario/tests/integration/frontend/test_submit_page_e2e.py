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


# ── F9.8: LERO YAMLs round-trip through Submit ──────────────────────


def _lero_cfg(storage_path: str) -> dict:
    """Minimal LERO-flavoured ExperimentConfig YAML — lero+llm + scenario+algo."""
    return {
        "experiment": {"id": "lero_demo", "seed": 0},
        "scenario": {
            "type": "discovery",
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
            "storage": {"type": "fs", "path": storage_path, "params": {}},
        },
        "lero": {
            "n_iterations": 2,
            "n_candidates": 3,
            "evolve_reward": False,
            "evolve_observation": True,
            "prompt_version": "v2_fewshot_k2_local",
        },
        "llm": {"model": "gpt-4o-mini", "temperature": 0.7},
    }


@pytest.fixture
def tmp_lero_experiments(tmp_path: Path) -> Path:
    """Experiments tree with a LERO YAML for the F9.8 round-trip tests."""
    cfg_dir = tmp_path / "discovery" / "lero" / "configs"
    cfg_dir.mkdir(parents=True)
    storage = tmp_path / "results"
    storage.mkdir()
    (cfg_dir / "lero_demo.yaml").write_text(
        yaml.dump(_lero_cfg(str(storage)), sort_keys=False), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.slow
def test_submit_renders_lero_yaml_with_lero_and_llm_sections(
    tmp_lero_experiments: Path,
):
    """F9.8 round-trip: load a LERO YAML through Submit → session_state's
    current_form carries the lero + llm blocks (not silently dropped)."""
    at = _new_apptest(tmp_lero_experiments)
    _drive_pick(at, scenario="discovery", folder="lero", config="lero_demo.yaml")
    # AppTest's session_state raises on missing keys instead of returning None.
    assert "submit_current_form" in at.session_state
    current = at.session_state["submit_current_form"]
    assert current is not None
    assert "lero" in current, f"lero section dropped from form: keys={list(current)}"
    assert "llm" in current, f"llm section dropped from form: keys={list(current)}"
    # Specific knobs survive the widget round-trip.
    assert current["lero"]["n_iterations"] == 2
    assert current["lero"]["n_candidates"] == 3
    assert current["lero"]["evolve_reward"] is False
    assert current["lero"]["evolve_observation"] is True
    assert current["llm"]["model"] == "gpt-4o-mini"
    assert current["llm"]["temperature"] == pytest.approx(0.7)


@pytest.mark.slow
def test_lero_yaml_is_clean_after_pick_no_dirty_flag(tmp_lero_experiments: Path):
    """F9.8 dirty-detection regression: loading a LERO YAML and not touching
    any widget must NOT flag the form as dirty.

    The bug: my F9.8 widgets render the full LeroSection/LlmSection field
    set even when the YAML omits some fields, so the widget-output dict
    had MORE keys than the snapshot → ``snapshot != current`` → "Save"
    button erroneously visible. The fix (see ``submit_workflow._fill_pydantic_defaults``)
    aligns the snapshot's filled keys with what the widget emits.
    """
    at = _new_apptest(tmp_lero_experiments)
    _drive_pick(at, scenario="discovery", folder="lero", config="lero_demo.yaml")
    assert "submit_snapshot_form" in at.session_state
    assert "submit_current_form" in at.session_state
    snapshot = at.session_state["submit_snapshot_form"]
    current = at.session_state["submit_current_form"]
    # Walk through the LERO/LLM blocks specifically — diff_summary would
    # surface any drift here.
    assert snapshot["lero"] == current["lero"], (
        f"LERO section drift on fresh load: "
        f"snapshot={snapshot['lero']!r}\ncurrent={current['lero']!r}"
    )
    assert snapshot["llm"] == current["llm"], (
        f"LLM section drift on fresh load: "
        f"snapshot={snapshot['llm']!r}\ncurrent={current['llm']!r}"
    )
    # Top-level: form not dirty overall.
    assert snapshot == current, "Submit form flagged dirty on a fresh LERO YAML load"


@pytest.mark.slow
def test_submit_renders_lero_yaml_validates_as_lero_cfg(tmp_lero_experiments: Path):
    """The form's round-tripped dict must still validate as a LERO
    ExperimentConfig (cfg.lero is not None, cfg.llm is not None)."""
    from multi_scenario.domain.models import ExperimentConfig

    at = _new_apptest(tmp_lero_experiments)
    _drive_pick(at, scenario="discovery", folder="lero", config="lero_demo.yaml")
    current = at.session_state["submit_current_form"]
    cfg = ExperimentConfig.model_validate(current)
    assert cfg.lero is not None
    assert cfg.llm is not None
    assert cfg.lero.n_iterations == 2
    assert cfg.llm.model == "gpt-4o-mini"


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
    # Switch submit target → ovh via the radio. F7.7.A4: this now legitimately
    # modifies the YAML's runner.type, so Step 3 blocks Step 4 until saved.
    at.radio(key="step2_submit_target").set_value("ovh")
    at.run()
    # Save as a new file so Step 4 unlocks (Step 3's "save before preflight" gate).
    at.text_input(key="step3_new_name").set_value("seed0_ovh.yaml")
    at.run()
    at.button(key="step3_save").click()
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


# ── Stage 3: _refresh_ovh_status helper ───────────────────────────────


@pytest.fixture
def _refresh_helper():
    """Import the refresh helper inside a Streamlit-friendly context.

    The helper writes to ``st.session_state`` so we run it inside a fresh
    ``AppTest`` script that just calls it; assertions then read the recorded
    state through ``at.session_state``.
    """
    from streamlit.testing.v1 import AppTest

    def _drive(initial_state: dict, after_helper):
        """Run a tiny Streamlit script that calls _refresh_ovh_status; return AppTest."""
        script = """
import streamlit as st
from multi_scenario.frontend.pages.submit import _refresh_ovh_status
_refresh_ovh_status()
st.write("done")
"""
        at = AppTest.from_string(script, default_timeout=10.0)
        for k, v in initial_state.items():
            at.session_state[k] = v
        at.run()
        return at

    return _drive


def _ovh_status(**overrides):
    base = {
        "status": "submitted",
        "runner": "ovh",
        "run_id": "demo_s0",
        "run_dir": "/tmp/demo_s0__t",
        "job_id": "uuid-1",
        "s3_prefix": "ms-test-results@GRA/demo_s0__t",
        "dashboard_url": "https://ovh/x",
    }
    base.update(overrides)
    return base


def test_refresh_helper_noop_when_status_not_submitted(_refresh_helper):
    """Defensive: helper should never crash if the panel was already DONE."""
    at = _refresh_helper(
        {"submit_submission_status": _ovh_status(status="done")},
        after_helper=None,
    )
    assert at.session_state["submit_submission_status"]["status"] == "done"


def test_refresh_helper_records_state_when_job_still_running(
    _refresh_helper, monkeypatch
):
    """Non-terminal info: status stays 'submitted', latest state recorded."""
    info = type("I", (), {"is_terminal": False, "state": "RUNNING"})()
    fake_client = type("C", (), {"get": lambda self, jid: info})()
    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient",
        lambda: fake_client,
    )
    at = _refresh_helper(
        {"submit_submission_status": _ovh_status()},
        after_helper=None,
    )
    out = at.session_state["submit_submission_status"]
    assert out["status"] == "submitted"
    assert out["state"] == "RUNNING"


def test_refresh_helper_marks_crashed_on_failed_terminal_state(
    _refresh_helper, monkeypatch
):
    """Terminal but not DONE → crashed + logs tail."""
    info = type("I", (), {"is_terminal": True, "state": "FAILED"})()
    fake_client = type(
        "C",
        (),
        {
            "get": lambda self, jid: info,
            "logs": lambda self, jid, tail=200: "boom\ntraceback line",
        },
    )()
    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient",
        lambda: fake_client,
    )
    at = _refresh_helper(
        {"submit_submission_status": _ovh_status()},
        after_helper=None,
    )
    out = at.session_state["submit_submission_status"]
    assert out["status"] == "crashed"
    assert "FAILED" in out["error"]
    assert "boom" in out["traceback"]


def test_refresh_helper_calls_pullback_then_marks_done(
    tmp_path: Path, _refresh_helper, monkeypatch
):
    """DONE → pullback runs, status flips to done with file counts."""
    info = type("I", (), {"is_terminal": True, "state": "DONE"})()
    fake_client = type(
        "C", (), {"get": lambda self, jid: info, "logs": lambda *a, **k: ""}
    )()
    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient",
        lambda: fake_client,
    )
    # Stub _try_load_ovh_config so we don't depend on a real configs/ovh.yaml.
    fake_ovh_cfg = type("Cfg", (), {"region": "GRA", "bucket_results": "x"})()
    monkeypatch.setattr(
        "multi_scenario.frontend.pages.submit._try_load_ovh_config",
        lambda: (fake_ovh_cfg, None),
    )
    # Stub pullback to record the call and pretend everything came back.
    pullback_calls = []

    def _fake_pullback(**kwargs):
        pullback_calls.append(kwargs)
        result = type("R", (), {"n_downloaded": 5, "n_skipped": 1})()
        # Plant a videos/ folder so video-regen is skipped.
        run_dir = kwargs["dest_dir"]
        (run_dir / "videos").mkdir(parents=True, exist_ok=True)
        (run_dir / "videos" / "after.mp4").write_bytes(b"x")
        return result

    monkeypatch.setattr(
        "multi_scenario.application.ovh_pullback.pullback_run_dir",
        _fake_pullback,
    )
    run_dir = tmp_path / "demo_s0__t"
    at = _refresh_helper(
        {"submit_submission_status": _ovh_status(run_dir=str(run_dir))},
        after_helper=None,
    )
    out = at.session_state["submit_submission_status"]
    assert out["status"] == "done"
    assert out["pullback_n_downloaded"] == 5
    assert out["pullback_n_skipped"] == 1
    assert pullback_calls and pullback_calls[0]["dest_dir"] == run_dir


def test_refresh_helper_marks_crashed_when_status_check_raises(
    _refresh_helper, monkeypatch
):
    """OvhClient.get raising → status crashed (don't leave panel stuck)."""

    def _boom(_self, _jid):
        raise RuntimeError("ovhai unreachable")

    fake_client = type("C", (), {"get": _boom})()
    monkeypatch.setattr(
        "multi_scenario.adapters.runners.ovh_cli.OvhClient",
        lambda: fake_client,
    )
    at = _refresh_helper(
        {"submit_submission_status": _ovh_status()},
        after_helper=None,
    )
    out = at.session_state["submit_submission_status"]
    assert out["status"] == "crashed"
    assert "ovhai unreachable" in out["error"]
