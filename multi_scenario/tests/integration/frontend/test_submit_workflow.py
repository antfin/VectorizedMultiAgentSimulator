"""F7.5 Phase A redesign — SubmitState transitions + diff_summary + browser."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest

from multi_scenario.frontend.preflight import CheckStatus, default_checks
from multi_scenario.frontend.submit_workflow import (
    diff_summary,
    list_configs_grouped,
    SubmitState,
)


def _state(**kwargs) -> SubmitState:
    """Construct a SubmitState directly (bypasses st.session_state)."""
    defaults = {
        "selected_path": None,
        "snapshot_form": None,
        "current_form": None,
        "saved_path": None,
        "preflight": [],
    }
    defaults.update(kwargs)
    return SubmitState(**defaults)


# ── derived properties ────────────────────────────────────────────────


def test_active_step_is_1_when_nothing_selected():
    assert _state().active_step == 1


def test_active_step_is_4_after_pick_when_clean():
    s = _state(
        selected_path=Path("/tmp/x.yaml"),
        snapshot_form={"a": 1},
        current_form={"a": 1},
    )
    assert s.is_dirty is False
    assert s.active_step == 4


def test_active_step_is_3_when_dirty_and_unsaved():
    s = _state(
        selected_path=Path("/tmp/x.yaml"),
        snapshot_form={"a": 1},
        current_form={"a": 2},
    )
    assert s.is_dirty is True
    assert s.active_step == 3


def test_active_step_is_4_after_save():
    s = _state(
        selected_path=Path("/tmp/x.yaml"),
        snapshot_form={"a": 1},
        current_form={"a": 1},
        saved_path=Path("/tmp/x_v2.yaml"),
    )
    assert s.active_step == 4


def test_active_step_is_5_when_preflight_passes():
    checks = default_checks()
    for c in checks:
        c.status = CheckStatus.PASS
    s = _state(
        selected_path=Path("/tmp/x.yaml"),
        snapshot_form={"a": 1},
        current_form={"a": 1},
        preflight=checks,
    )
    assert s.has_preflight_passed
    assert s.active_step == 5
    assert s.can_submit


def test_can_submit_blocks_when_dirty_unsaved_even_with_preflight():
    """Edits invalidate preflight pass; can't submit until saved."""
    checks = default_checks()
    for c in checks:
        c.status = CheckStatus.PASS
    s = _state(
        selected_path=Path("/tmp/x.yaml"),
        snapshot_form={"a": 1},
        current_form={"a": 2},  # dirty
        saved_path=None,
        preflight=checks,
    )
    assert not s.can_submit


def test_active_config_path_prefers_saved_over_selected():
    s = _state(
        selected_path=Path("/orig.yaml"),
        saved_path=Path("/edited.yaml"),
    )
    assert s.active_config_path == Path("/edited.yaml")


# ── diff_summary ─────────────────────────────────────────────────────


def test_diff_summary_finds_top_level_change():
    assert diff_summary({"a": 1}, {"a": 2}) == ["a"]


def test_diff_summary_walks_nested_dicts():
    a = {"x": {"y": {"z": 1}}}
    b = {"x": {"y": {"z": 2}}}
    assert diff_summary(a, b) == ["x.y.z"]


def test_diff_summary_handles_added_and_removed_keys():
    assert sorted(diff_summary({"a": 1}, {"b": 2})) == ["a", "b"]


def test_diff_summary_empty_when_equal():
    assert diff_summary({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []


# ── list_configs_grouped ─────────────────────────────────────────────


@pytest.fixture
def synth_experiments(tmp_path: Path) -> Path:
    """Build a fake experiments tree with 2 scenarios × 2 folders × N yamls."""
    for scen in ("discovery", "navigation"):
        for folder in ("baseline", "lero"):
            cfg_dir = tmp_path / scen / folder / "configs"
            cfg_dir.mkdir(parents=True)
            (cfg_dir / "smoke.yaml").write_text("a: 1", encoding="utf-8")
            (cfg_dir / "full.yaml").write_text("b: 2", encoding="utf-8")
    return tmp_path


def test_list_configs_grouped_returns_per_folder_yamls(synth_experiments: Path):
    grouped = list_configs_grouped(synth_experiments, "discovery")
    assert set(grouped) == {"baseline", "lero"}
    assert {p.name for p in grouped["baseline"]} == {"smoke.yaml", "full.yaml"}


def test_list_configs_grouped_empty_for_unknown_scenario(synth_experiments: Path):
    assert list_configs_grouped(synth_experiments, "no_such_scenario") == {}


def test_list_configs_grouped_skips_folders_without_configs_dir(
    synth_experiments: Path,
):
    """A scenario subfolder without ``configs/`` is silently dropped."""
    (synth_experiments / "discovery" / "no_configs_here").mkdir()
    grouped = list_configs_grouped(synth_experiments, "discovery")
    assert "no_configs_here" not in grouped
