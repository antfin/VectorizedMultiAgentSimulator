"""F8.2 — ``scripts/compare_to_reference.py`` exercises the threshold logic.

The script reads ``runs.csv`` rows and compares mean-M1-across-seeds to the
hardcoded rendezvous_comm reference (``REFERENCE['er1_cr035']['M1_success_rate']``
= 0.405). PASS = within ±10% absolute AND (when n_seeds ≥ 2) within 1.5σ
of the seed distribution.

These tests synthesise tiny CSVs and invoke the script's internal
``_compare`` helper to pin the gates without spending OVH compute.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compare_to_reference.py"


@pytest.fixture(scope="module")
def script_module():
    """Load the script as a module without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("_ctr_under_test", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ctr_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_runs_csv(tmp_path: Path, m1_values: list[float], exp_id: str = "er1_cr035") -> Path:
    """Synthesise a minimal runs.csv with the columns ``compare_to_reference`` reads."""
    runs_csv = tmp_path / "runs.csv"
    fieldnames = ["record_type", "exp_id", "seed", "M1_success_rate"]
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed, m1 in enumerate(m1_values):
            writer.writerow({
                "record_type": "final", "exp_id": exp_id,
                "seed": str(seed), "M1_success_rate": str(m1),
            })
    return runs_csv


# ── Threshold gates ──────────────────────────────────────────────────


def test_pass_when_mean_within_abs_and_sigma(script_module, tmp_path):
    """3 seeds clustered around the reference (0.405) → PASS."""
    runs_csv = _make_runs_csv(tmp_path, [0.40, 0.41, 0.39])
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.abs_pass is True
    assert res.sigma_pass is True
    assert res.coopvmas_n == 3


def test_fail_when_mean_outside_abs_tolerance(script_module, tmp_path):
    """Mean=0.30 vs reference=0.405 → |Δ|=0.105 > 0.10 → abs FAIL."""
    runs_csv = _make_runs_csv(tmp_path, [0.30, 0.30, 0.30])
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.abs_pass is False


def test_pass_when_single_seed_within_abs(script_module, tmp_path):
    """n=1 → no sigma check; only abs gate runs."""
    runs_csv = _make_runs_csv(tmp_path, [0.40])
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.abs_pass is True
    assert res.sigma_pass is None
    assert res.coopvmas_std is None


def test_fail_on_sigma_when_high_variance_pulls_mean_close_to_reference(
    script_module, tmp_path,
):
    """Mean within abs but spread is wide AND |Δ| > 1.5σ → sigma FAIL.

    With 3 seeds [0.36, 0.37, 0.38] (std=0.01), reference=0.42:
    |Δ|=0.05 ≤ 0.10 (abs PASS); 1.5σ=0.015 → 0.05 > 0.015 (sigma FAIL).
    """
    runs_csv = _make_runs_csv(tmp_path, [0.36, 0.37, 0.38])
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.42)
    assert res.abs_pass is True
    assert res.sigma_pass is False


def test_zero_std_treated_as_sigma_pass(script_module, tmp_path):
    """All seeds identical → std=0; 1.5*0=0 would always FAIL otherwise.

    The script special-cases std=0 → sigma PASS (degenerate case is fine
    when abs gate already passes; only abs matters).
    """
    runs_csv = _make_runs_csv(tmp_path, [0.40, 0.40, 0.40])
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.abs_pass is True
    assert res.sigma_pass is True
    assert res.coopvmas_std == 0.0


# ── Edge cases / data hygiene ────────────────────────────────────────


def test_skips_non_final_rows(script_module, tmp_path):
    """Eval rows / lero_candidate rows / etc. shouldn't enter the comparison."""
    runs_csv = tmp_path / "runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record_type", "exp_id", "seed", "M1_success_rate"],
        )
        writer.writeheader()
        # 1 final row + 2 noise rows that should be ignored
        writer.writerow({"record_type": "final", "exp_id": "er1_cr035",
                         "seed": "0", "M1_success_rate": "0.40"})
        writer.writerow({"record_type": "eval", "exp_id": "er1_cr035",
                         "seed": "0", "M1_success_rate": "999.0"})  # poisoned noise
        writer.writerow({"record_type": "lero_candidate", "exp_id": "er1_cr035",
                         "seed": "0", "M1_success_rate": "999.0"})
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.coopvmas_n == 1, f"only the final row should count, got {res.coopvmas_n}"
    assert res.coopvmas_mean == 0.40


def test_skips_null_metric_values(script_module, tmp_path):
    """A row with empty / null M1 is dropped (not coerced to 0)."""
    runs_csv = tmp_path / "runs.csv"
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record_type", "exp_id", "seed", "M1_success_rate"],
        )
        writer.writeheader()
        writer.writerow({"record_type": "final", "exp_id": "er1_cr035",
                         "seed": "0", "M1_success_rate": "0.40"})
        writer.writerow({"record_type": "final", "exp_id": "er1_cr035",
                         "seed": "1", "M1_success_rate": ""})
        writer.writerow({"record_type": "final", "exp_id": "er1_cr035",
                         "seed": "2", "M1_success_rate": "None"})
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    res = script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)
    assert res.coopvmas_n == 1


def test_raises_when_no_matching_rows(script_module, tmp_path):
    """No rows with the right exp_id → ValueError pointing to the missing data."""
    runs_csv = _make_runs_csv(tmp_path, [0.40], exp_id="some_other_exp")
    rows = list(csv.DictReader(runs_csv.open(encoding="utf-8")))
    with pytest.raises(ValueError, match="er1_cr035"):
        script_module._compare(rows, "er1_cr035", "M1_success_rate", 0.405)


def test_reference_dict_pins_er1_headline_number(script_module):
    """Sanity: the hardcoded reference matches the rendezvous_comm doc value."""
    assert script_module.REFERENCE["er1_cr035"]["M1_success_rate"] == 0.405
