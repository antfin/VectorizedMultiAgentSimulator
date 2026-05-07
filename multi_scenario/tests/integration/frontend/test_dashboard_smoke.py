"""F7.1 smoke: home page renders cleanly under streamlit's AppTest harness."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest


@pytest.mark.slow
def test_home_renders_per_scenario_empty_state(tmp_path: Path) -> None:
    """Empty experiments dir → main shows 'No runs yet for <scenario>'."""
    # streamlit.testing is the official AppTest harness — exercises the
    # script top-to-bottom without spinning up a browser. We point it at the
    # home page directly (not the navigator) to test the page in isolation.
    # pylint: disable=import-outside-toplevel
    from streamlit.testing.v1 import AppTest

    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY

    home_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "multi_scenario"
        / "frontend"
        / "pages"
        / "home.py"
    )
    at = AppTest.from_file(str(home_path), default_timeout=10.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(tmp_path)
    at.run()
    assert not at.exception
    info_texts = [(card.value or "") for card in at.info]
    assert any("discovery" in t and "No runs yet" in t for t in info_texts)
