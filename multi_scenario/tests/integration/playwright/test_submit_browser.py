"""Phase D — real Chromium browser tests for the Submit page.

These tests drive a real browser and verify what the USER actually
sees — what AppTest can't catch:

- DOM element visibility / CSS rendering
- JS-side interactions (Streamlit auto-rerun on widget change)
- Multi-page navigation through the sidebar
- Browser-side state across user actions

Opt-in via ``pytest -m playwright``. Requires:
    pip install -e '.[playwright]'
    playwright install chromium

Each test is fast (<10s wall) because we use a session-scoped Streamlit
server fixture and rely on Playwright's built-in waiting primitives.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import shutil
from pathlib import Path

import pytest
import yaml

from tests.integration.dispatch_matrix._helpers import (
    er1_smoke_cfg,
    lero_smoke_cfg,
)


pytestmark = pytest.mark.playwright


def _seed_yaml_in_experiments(
    experiments_root: Path,
    cfg: dict,
    *,
    scenario: str = "discovery",
    folder: str = "baseline",
    name: str = "smoke.yaml",
) -> None:
    """Drop a YAML into the experiments tree the Streamlit server is watching."""
    cfg_dir = experiments_root / scenario / folder / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / name).write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")


def test_submit_page_loads_in_browser(
    streamlit_server, chromium_available, page  # noqa: ARG001
):
    """Smoke: the Submit page renders without a JS-side exception."""
    page.goto(f"{streamlit_server['url']}/submit")
    # Streamlit's top-bar title element — present on every page.
    page.wait_for_selector("text=Submit", timeout=10000)
    # No error banners.
    errors = page.locator(".stException, .stAlert--error")
    assert errors.count() == 0


def test_er1_yaml_picker_loads_form_without_lero_widgets(
    streamlit_server, chromium_available, page  # noqa: ARG001
):
    """Pick an ER1 YAML → form loads, LERO section testid NOT present."""
    storage = Path("/tmp/playwright_er1_storage")
    storage.mkdir(exist_ok=True)
    shutil.rmtree(storage, ignore_errors=True)
    storage.mkdir()
    _seed_yaml_in_experiments(
        streamlit_server["experiments_root"], er1_smoke_cfg(str(storage))
    )
    page.goto(f"{streamlit_server['url']}/submit")
    # Streamlit needs a moment to surface the new YAML in the picker.
    page.wait_for_timeout(1000)
    page.get_by_text("smoke.yaml").click()
    # No LERO testid markers on the page after picking ER1.
    assert page.locator('[data-testid="submit-lero-section"]').count() == 0
    assert page.locator('[data-testid="submit-llm-section"]').count() == 0


def test_lero_yaml_picker_shows_lero_section_testid(
    streamlit_server, chromium_available, page  # noqa: ARG001
):
    """Pick a LERO YAML → both data-testid markers appear in the DOM."""
    storage = Path("/tmp/playwright_lero_storage")
    storage.mkdir(exist_ok=True)
    shutil.rmtree(storage, ignore_errors=True)
    storage.mkdir()
    _seed_yaml_in_experiments(
        streamlit_server["experiments_root"],
        lero_smoke_cfg(str(storage)),
        folder="lero",
    )
    page.goto(f"{streamlit_server['url']}/submit")
    page.wait_for_timeout(1000)
    # Pick scenario / folder / config via the cascading picker.
    # Streamlit's selectboxes render with `data-baseweb=select`; we
    # click the visible label text which is robust to internal class
    # changes between Streamlit releases.
    page.get_by_text("smoke.yaml").click()
    # Expect both LERO + LLM section testids to render.
    page.wait_for_selector('[data-testid="submit-lero-section"]', timeout=10000)
    assert page.locator('[data-testid="submit-llm-section"]').count() == 1
