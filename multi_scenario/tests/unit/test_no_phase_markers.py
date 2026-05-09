"""F7.7.E1 — CI guard against stale phase markers in src/.

After F7.7's audit pass the codebase has no remaining "Phase A/B/C" labels
under ``src/`` — those were workflow-tracking artifacts that confused new
readers. This test fails if any reappear so future PRs get caught at
review time instead of accreting fresh stale labels.
"""

# pylint: disable=missing-function-docstring

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def test_no_phase_markers_in_src():
    """grep -rn 'Phase A|Phase B|Phase C' src/ → expects zero hits.

    Excludes ``__pycache__`` (compiled artifacts may carry old strings even
    after the .py is rewritten — they get evicted on next import).
    """
    res = subprocess.run(
        [
            "grep",
            "-rn",
            "-E",
            "--include=*.py",
            "--exclude-dir=__pycache__",
            r"Phase [ABC]\b",
            str(_REPO / "src"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # grep returns 1 when there are no matches — that's the green path.
    assert res.returncode == 1, (
        "Stale 'Phase A/B/C' markers found in src/ — "
        "see F7.7.E1 cleanup pass:\n" + res.stdout
    )
