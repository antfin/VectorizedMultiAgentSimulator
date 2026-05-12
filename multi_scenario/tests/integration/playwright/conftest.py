"""Phase D Playwright infrastructure — Streamlit server fixture.

Boots ``streamlit run streamlit_app.py`` once per session on a free
port; tests use ``page`` from pytest-playwright to drive Chromium.

Skipped (collected as ``pytest.skip``) when:
- The ``playwright`` extra isn't installed (``pyproject.toml`` extras
  ``[playwright]``)
- ``playwright install chromium`` hasn't been run (Chromium binary
  missing).

Install:
    pip install -e '.[playwright]'
    playwright install chromium

Run:
    pytest -m playwright
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


_STREAMLIT_APP = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "multi_scenario"
    / "frontend"
    / "streamlit_app.py"
)


def _pick_free_port() -> int:
    """Bind to port 0 → OS picks a free port; close + return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_streamlit(port: int, timeout: float = 30.0) -> None:
    """Poll the port until Streamlit accepts a connection (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"Streamlit did not start within {timeout}s on port {port}")


@pytest.fixture(scope="session")
def _playwright_available():
    """Skip the playwright suite cleanly when the extra isn't installed."""
    pytest.importorskip(
        "playwright",
        reason=(
            "Playwright extra not installed. Run: "
            "pip install -e '.[playwright]' && playwright install chromium"
        ),
    )
    pytest.importorskip(
        "pytest_playwright",
        reason="pytest-playwright not installed",
    )


@pytest.fixture(scope="session")
def streamlit_server(_playwright_available, tmp_path_factory):
    """Boot Streamlit on a free port for the session; tear down on exit.

    Uses a session-scoped temp dir as the experiments root so tests can
    drop YAMLs into it for the picker to discover. Headless server has
    no browser; pytest-playwright provides ``page`` via Chromium.
    """
    port = _pick_free_port()
    experiments_root = tmp_path_factory.mktemp("playwright_experiments")
    env = {**os.environ, "MULTI_SCENARIO_EXPERIMENTS_ROOT": str(experiments_root)}
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(_STREAMLIT_APP),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_streamlit(port)
        yield {"port": port, "url": f"http://127.0.0.1:{port}", "experiments_root": experiments_root}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


@pytest.fixture
def chromium_available(_playwright_available):
    """Skip a test when Chromium binary isn't installed via playwright install."""
    # pytest-playwright surfaces this as a launcher error; we check
    # proactively so the skip reason is actionable instead of cryptic.
    if not shutil.which("chromium-browser") and not _chromium_in_playwright_cache():
        pytest.skip("Chromium not installed. Run: playwright install chromium")


def _chromium_in_playwright_cache() -> bool:
    """Best-effort: detect a chromium binary under playwright's cache."""
    candidates = [
        Path.home() / ".cache" / "ms-playwright",
        Path.home() / "Library" / "Caches" / "ms-playwright",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for child in root.glob("chromium-*"):
            if child.is_dir():
                return True
    return False
