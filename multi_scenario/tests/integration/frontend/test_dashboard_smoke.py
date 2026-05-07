"""F7.1 smoke test: Dashboard.py imports cleanly under streamlit's AppTest harness."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest


@pytest.mark.slow
def test_dashboard_renders_per_scenario_empty_state(tmp_path: Path) -> None:
    """Empty experiments dir → 4 tabs, each with its own scenario-specific info."""
    # streamlit.testing is the official AppTest harness — exercises the
    # script top-to-bottom without spinning up a browser.
    # pylint: disable=import-outside-toplevel
    from streamlit.testing.v1 import AppTest

    dashboard_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "multi_scenario"
        / "frontend"
        / "Dashboard.py"
    )
    at = AppTest.from_file(str(dashboard_path), default_timeout=10.0)
    # First run materialises the widgets; then we point the sidebar at the
    # empty tmp_path and re-run to exercise the no-runs branch.
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path))
    at.run()
    assert not at.exception
    info_texts = [(card.value or "") for card in at.info]
    # Every scenario tab renders its own empty notice; 4 in total.
    assert any("discovery" in t for t in info_texts)
    assert any("navigation" in t for t in info_texts)
    assert any("transport" in t for t in info_texts)
    assert any("flocking" in t for t in info_texts)
