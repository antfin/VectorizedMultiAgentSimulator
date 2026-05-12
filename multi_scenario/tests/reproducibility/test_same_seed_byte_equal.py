"""F10.5 — same config + same seed → identical metrics across runs.

Distinct from :mod:`tests.reproducibility.test_compare_to_reference`
(which tests F8.2's compare-to-reference *threshold logic*). F10.5
asserts the **framework's determinism** as a load-bearing contract:
if a future change introduces non-determinism (e.g. forgets to seed
something, switches to a non-deterministic algorithm, etc.), this
test fails before downstream reproducibility experiments do.

Two runs at the same seed must produce identical:

- ``M2_avg_return`` (the universal eval metric)
- ``M3_steps``
- ``M4_collisions``

Tolerance is 1e-5 to absorb float-precision noise from non-deterministic
ops we can't fully control (CUDA reductions, multi-threaded matmul on
some BLAS implementations). On CPU + single-env this is usually exact;
the tolerance is there to make the test stable on a wider matrix.

Smoke budget: max_iters=1, 5-step episodes, 1 eval episode. Total
wall ~5s per run × 2 runs = ~10s.
"""

# pylint: disable=missing-function-docstring,import-outside-toplevel

import json
from pathlib import Path

import pytest


_METRIC_TOLERANCE = 1e-5


@pytest.mark.slow
def test_same_seed_same_config_produces_identical_metrics(tmp_path: Path):
    """Two identical local runs at seed=0 → metrics agree within 1e-5."""
    # pylint: disable=import-error,no-name-in-module
    from tests.integration.dispatch_matrix._helpers import er1_smoke_cfg

    storage = tmp_path / "results"
    storage.mkdir()
    cfg_dict = er1_smoke_cfg(str(storage))

    metrics_a = _run_once_and_extract_metrics(cfg_dict)
    metrics_b = _run_once_and_extract_metrics(cfg_dict)

    # Compare each universal metric within tolerance.
    for key in ("M2_avg_return", "M3_steps", "M4_collisions"):
        a, b = metrics_a.get(key), metrics_b.get(key)
        assert a is not None, f"metric {key!r} missing from run A"
        assert b is not None, f"metric {key!r} missing from run B"
        assert abs(a - b) < _METRIC_TOLERANCE, (
            f"determinism broken: run A {key}={a!r}, run B {key}={b!r} "
            f"(delta={abs(a - b):.2e}, tolerance={_METRIC_TOLERANCE:.0e}). "
            "Two identical runs with the same seed must produce identical "
            "metrics — a divergence means an unseeded RNG / non-deterministic op "
            "slipped into the pipeline."
        )


def _run_once_and_extract_metrics(cfg_dict: dict) -> dict[str, float]:
    """Fire one local run from the cfg dict, return the metrics.json mapping."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.logging.file_logger import FileLogger
    from multi_scenario.application.submission import build_run_dir, submit_to_local
    from multi_scenario.domain.models import ExperimentConfig

    cfg = ExperimentConfig.model_validate(cfg_dict)
    _, run_dir = build_run_dir(cfg, mkdir=True)
    submit_to_local(
        cfg, run_dir=run_dir, logger=FileLogger(run_dir / "logs" / "run.log")
    )

    metrics_json = run_dir / "output" / "metrics.json"
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    # ``metrics`` on disk is a flat ``{metric_id: value}`` dict (see
    # ``MetricRecord``'s pre-validator that unrolls the list shape).
    return payload["metrics"]
