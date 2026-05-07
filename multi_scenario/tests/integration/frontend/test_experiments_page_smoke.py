"""F7.2 smoke: Experiments Browser page renders cleanly under AppTest."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest


@pytest.mark.slow
def test_experiments_page_empty_state(tmp_path: Path) -> None:
    """Empty experiments dir → 'No runs found' info, no exception."""
    # pylint: disable=import-outside-toplevel
    from streamlit.testing.v1 import AppTest

    from multi_scenario.frontend.sidebar import EXPERIMENTS_ROOT_KEY

    page_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "multi_scenario"
        / "frontend"
        / "pages"
        / "1_Experiments.py"
    )
    at = AppTest.from_file(str(page_path), default_timeout=10.0)
    at.session_state[EXPERIMENTS_ROOT_KEY] = str(tmp_path)
    at.run()
    assert not at.exception
    info_texts = [(card.value or "") for card in at.info]
    assert any("No runs found" in t for t in info_texts)
