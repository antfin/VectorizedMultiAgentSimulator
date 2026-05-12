"""Reusable Submit-page driver + assertion helpers.

Each helper does one focused thing so tests stay readable: pick a YAML,
fill the form, run preflight, click Submit. Assertion helpers translate
session_state shape into named conditions ("clean", "dirty", "preflight
green") so tests assert intent, not session_state keys.

Used by ``test_submit_page_e2e.py`` and ``test_dispatch_matrix.py``.
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel

from pathlib import Path
from typing import Any


_SUBMIT_PAGE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "multi_scenario"
    / "frontend"
    / "pages"
    / "submit.py"
)


# ── Drivers ─────────────────────────────────────────────────────────


def new_apptest(experiments_root: Path, *, default_timeout: float = 30.0):
    """Construct an AppTest pointing at the Submit page.

    ``experiments_root`` becomes the sidebar's active experiments dir, so
    the Step 1 picker sees only YAMLs under it.
    """
    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(_SUBMIT_PAGE_PATH), default_timeout=default_timeout)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(experiments_root)
    return at


def drive_pick(
    at,
    scenario: str = "discovery",
    folder: str = "baseline",
    config: str = "smoke.yaml",
) -> None:
    """Walk Step 1's cascading picker — selects + clicks 'Use this config'."""
    at.run()
    at.selectbox(key="step1_scenario").set_value(scenario)
    at.run()
    at.selectbox(key="step1_folder").set_value(folder)
    at.run()
    at.selectbox(key="step1_config").set_value(config)
    at.run()
    at.button(key="step1_pick").click()
    at.run()


def drive_run_preflight(at) -> None:
    """Click 'Run preflight' on the Submit page (Step 4)."""
    at.button(key="step4_run").click()
    at.run()


def drive_submit(at) -> None:
    """Click 'Submit' (Step 5) — only valid when preflight is green."""
    at.button(key="step5_submit").click()
    at.run()


# ── Assertions ──────────────────────────────────────────────────────


def assert_form_clean(at) -> None:
    """No edits since picking the YAML — ``snapshot_form == current_form``."""
    snapshot = at.session_state["submit_snapshot_form"]
    current = at.session_state["submit_current_form"]
    assert snapshot is not None, "no YAML loaded yet (snapshot_form is None)"
    assert current is not None, "form not yet rendered (current_form is None)"
    assert snapshot == current, (
        f"form flagged dirty without user edits — diff:\n"
        f"snapshot keys: {sorted(snapshot)}\ncurrent keys: {sorted(current)}"
    )


def assert_form_dirty(at, *, expecting_path: str | None = None) -> None:
    """Form differs from snapshot — optionally pin a specific dotted-path diff.

    ``expecting_path`` is a dotted key like ``"lero.n_iterations"``; when
    supplied, asserts the value at that path differs between snapshot
    and current. Useful for "I edited X → only X should be dirty" tests.
    """
    snapshot = at.session_state["submit_snapshot_form"]
    current = at.session_state["submit_current_form"]
    assert snapshot != current, "form was expected dirty but matches snapshot"
    if expecting_path is None:
        return
    snap_val = _walk(snapshot, expecting_path)
    curr_val = _walk(current, expecting_path)
    assert snap_val != curr_val, (
        f"expected dirt at {expecting_path}; both sides have {snap_val!r}"
    )


def assert_preflight_pass(at, check_name: str) -> None:
    """A specific preflight row reported PASS."""
    from multi_scenario.frontend.preflight import CheckStatus

    rows = at.session_state["submit_preflight"]
    row = next((r for r in rows if r.name == check_name), None)
    assert row is not None, (
        f"preflight row {check_name!r} not present; available: {[r.name for r in rows]}"
    )
    assert row.status == CheckStatus.PASS, (
        f"expected PASS for {check_name!r}, got {row.status.name}: {row.detail}"
    )


def assert_preflight_fail(at, check_name: str, *, detail_contains: str = "") -> None:
    """A specific preflight row reported FAIL; optionally pin detail substring."""
    from multi_scenario.frontend.preflight import CheckStatus

    rows = at.session_state["submit_preflight"]
    row = next((r for r in rows if r.name == check_name), None)
    assert row is not None, f"preflight row {check_name!r} not present"
    assert row.status == CheckStatus.FAIL, (
        f"expected FAIL for {check_name!r}, got {row.status.name}"
    )
    if detail_contains:
        assert detail_contains in row.detail, (
            f"expected {detail_contains!r} in detail; got: {row.detail!r}"
        )


def assert_lero_widgets_rendered(at) -> None:
    """LERO + LLM section markers are in the rendered markdown.

    The Submit page emits ``**LERO**`` / ``**LLM**`` headers when the
    loaded YAML has the respective blocks — picking up that they
    render confirms the F9.8 widget plumbing is alive.
    """
    headers = " | ".join((m.value or "") for m in at.markdown)
    assert "LERO" in headers, "**LERO** header not rendered — F9.8 widget broken"
    assert "LLM" in headers, "**LLM** header not rendered — F9.8 widget broken"


def assert_section_in_form(at, section: str) -> None:
    """Form's current dict carries this top-level section key."""
    current = at.session_state["submit_current_form"]
    assert section in current, (
        f"form missing section {section!r} — available: {sorted(current)}"
    )


# ── Snapshot helpers ────────────────────────────────────────────────


def session_state_snapshot(
    at, *, keys: tuple[str, ...] = ("submit_current_form", "submit_snapshot_form")
) -> dict[str, Any]:
    """Capture a subset of session_state for golden-file diffing.

    Pass only the keys you want to commit — full session_state has
    widget-internal IDs that change between Streamlit versions and
    would make snapshot tests noisy.
    """
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = at.session_state[k] if k in at.session_state else None
    return out


# ── Internals ───────────────────────────────────────────────────────


def _walk(d: dict[str, Any], path: str) -> Any:
    """Read a dotted path from a nested dict; returns None if missing."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur
