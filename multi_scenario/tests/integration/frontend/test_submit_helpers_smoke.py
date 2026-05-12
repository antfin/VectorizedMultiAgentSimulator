"""Phase C — verify the ``_submit_helpers`` module via short focused tests.

Each test exercises ONE helper using the shared smoke YAML factories
from the dispatch-matrix layer. Keeps the helper module honest as a
reusable surface — if any test here fails, every downstream test that
imports the helper will too.
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel

import json
from pathlib import Path

import pytest

from tests.integration.dispatch_matrix._helpers import (
    er1_smoke_cfg,
    lero_smoke_cfg,
    write_smoke_yaml,
)
from tests.integration.frontend._submit_helpers import (
    assert_form_clean,
    assert_form_dirty,
    assert_lero_widgets_rendered,
    assert_section_in_form,
    drive_pick,
    new_apptest,
    session_state_snapshot,
)


# ── Driver / assertion smoke ────────────────────────────────────────


def test_drive_pick_loads_yaml_into_form(tmp_path: Path):
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)
    assert "submit_selected_path" in at.session_state
    assert at.session_state["submit_selected_path"].name == "smoke.yaml"


def test_assert_form_clean_passes_on_fresh_pick(tmp_path: Path):
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)
    assert_form_clean(at)  # no exception


def test_assert_form_dirty_pins_a_specific_path(tmp_path: Path):
    """Edit experiment.id via widget → assert dirty at that dotted path."""
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)
    # Pre-edit: clean.
    assert_form_clean(at)
    # Edit the experiment.id text input.
    at.text_input(key="step2_exp_id").set_value("edited_id")
    at.run()
    assert_form_dirty(at, expecting_path="experiment.id")


def test_assert_lero_widgets_rendered_on_lero_yaml(tmp_path: Path):
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, lero_smoke_cfg(str(storage)), folder="lero")
    at = new_apptest(tmp_path)
    drive_pick(at, folder="lero")
    assert_lero_widgets_rendered(at)
    assert_section_in_form(at, "lero")
    assert_section_in_form(at, "llm")


def test_assert_lero_widgets_rendered_raises_on_non_lero_yaml(tmp_path: Path):
    """Non-LERO YAML → LERO headers shouldn't render → helper raises."""
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)
    with pytest.raises(AssertionError, match="LERO"):
        assert_lero_widgets_rendered(at)


# ── Snapshot tests ──────────────────────────────────────────────────


def test_session_state_snapshot_captures_form(tmp_path: Path):
    """Captured snapshot dict has the requested keys + nothing else."""
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)
    snap = session_state_snapshot(at)
    assert set(snap) == {"submit_current_form", "submit_snapshot_form"}
    # Both populated with the same dict (fresh pick → clean form).
    assert snap["submit_current_form"] == snap["submit_snapshot_form"]


def test_snapshot_diff_caught_after_edit(tmp_path: Path):
    """Snapshot-based regression test pattern: capture, edit, diff.

    Demonstrates the workflow a Phase-C snapshot test would follow —
    capture session_state shape, edit a widget, capture again, diff
    the two. Real "golden" snapshots would commit the JSON shape to
    a fixture file and assert against it; this test stays inline so
    no fixture drift can mask a regression.
    """
    storage = tmp_path / "results"
    storage.mkdir()
    write_smoke_yaml(tmp_path, er1_smoke_cfg(str(storage)))
    at = new_apptest(tmp_path)
    drive_pick(at)

    before = session_state_snapshot(at)
    at.text_input(key="step2_exp_id").set_value("renamed_run")
    at.run()
    after = session_state_snapshot(at)

    # The snapshot field stays pinned to the YAML at load time.
    assert before["submit_snapshot_form"] == after["submit_snapshot_form"]
    # The current form picked up the edit.
    assert before["submit_current_form"] != after["submit_current_form"]
    assert (
        after["submit_current_form"]["experiment"]["id"] == "renamed_run"
    )
    # JSON-serialisable so committing as a golden file would work.
    json.dumps(after, default=str)  # raises if not serialisable
