"""F8.2 — compare ER1 reproducibility runs to the rendezvous_comm reference.

Reads ``experiments/discovery/baseline/runs.csv`` (consolidated by
``multi-scenario consolidate``), pulls every row where ``exp_id`` matches
the canonical baseline (``er1_cr035``), and compares the M1 mean across
seeds to the rendezvous_comm reference number.

**Threshold (locked, F8 plan):** PASS if **|coopvmas_mean − reference| ≤ 10%
absolute** AND (when we have ≥2 seeds) **|coopvmas_mean − reference| ≤ 1.5σ**
of the seed distribution.

The reference dict is hardcoded — it's a small, slowly-changing fact
about the rendezvous_comm baselines that doesn't justify a separate
config file. Edit when rendezvous_comm publishes new headline numbers.

Exit code: 0 on PASS, 1 on FAIL, 2 on missing data (informative; lets
CI gate on PASS only).
"""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

#: Reference numbers from ``rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml``
#: at 10M frames, single seed (seed=0). See implementation_plan.md F8 + the
#: rendezvous_comm/docs/all_experiments_analysis.md doc for context.
REFERENCE: dict[str, dict[str, float]] = {
    "er1_cr035": {
        "M1_success_rate": 0.405,  # the headline number we're reproducing
    },
}

#: Reproducibility threshold knobs (F8 plan, locked 2026-05-09).
ABS_TOLERANCE = 0.10        # ±10% absolute on M1
SIGMA_MULTIPLIER = 1.5      # within 1.5σ of the seed-mean (only when n_seeds ≥ 2)


@dataclass(frozen=True)
class _CompareResult:
    """Full result of one comparison; 8 fields are intentional — each is a
    distinct piece of evidence the user reads on the printout. Splitting
    would just spread the same 8 fields across nested dataclasses.
    """

    # pylint: disable=too-many-instance-attributes

    metric: str
    coopvmas_mean: float
    coopvmas_std: float | None  # None when n_seeds = 1
    coopvmas_n: int
    reference: float
    abs_delta: float
    abs_pass: bool
    sigma_pass: bool | None  # None when std is None


def _load_runs_csv(runs_csv: Path) -> list[dict[str, str]]:
    if not runs_csv.is_file():
        raise FileNotFoundError(
            f"runs.csv not found at {runs_csv}. Run "
            f"'multi-scenario consolidate {runs_csv.parent}' first."
        )
    with runs_csv.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compare(
    rows: list[dict[str, str]], exp_id: str, metric: str, reference: float,
) -> _CompareResult:
    matching = [
        float(r[metric])
        for r in rows
        if r["record_type"] == "final"
        and r["exp_id"] == exp_id
        and r[metric] not in ("", "None", "null")
    ]
    if not matching:
        raise ValueError(
            f"no rows in runs.csv with exp_id={exp_id!r} and a non-null {metric}; "
            "did the runs complete + consolidate?"
        )
    mean = statistics.mean(matching)
    std = statistics.stdev(matching) if len(matching) >= 2 else None
    abs_delta = abs(mean - reference)
    abs_pass = abs_delta <= ABS_TOLERANCE
    sigma_pass: bool | None = None
    if std is not None and std > 0:
        sigma_pass = abs_delta <= SIGMA_MULTIPLIER * std
    elif std is not None:  # std == 0 → all seeds identical, no sigma check needed
        sigma_pass = True
    return _CompareResult(
        metric=metric,
        coopvmas_mean=mean,
        coopvmas_std=std,
        coopvmas_n=len(matching),
        reference=reference,
        abs_delta=abs_delta,
        abs_pass=abs_pass,
        sigma_pass=sigma_pass,
    )


def _print_result(result: _CompareResult) -> bool:
    """Pretty-print + return True if all gates pass."""
    typer.echo(f"  metric:           {result.metric}")
    typer.echo(f"  reference (rcom): {result.reference:.4f}")
    if result.coopvmas_std is not None:
        typer.echo(
            f"  coopvmas:         {result.coopvmas_mean:.4f} ± {result.coopvmas_std:.4f}"
            f" (n={result.coopvmas_n})"
        )
    else:
        typer.echo(
            f"  coopvmas:         {result.coopvmas_mean:.4f} (n={result.coopvmas_n})"
        )
    typer.echo(f"  |Δ|:              {result.abs_delta:.4f}")
    abs_marker = "✓" if result.abs_pass else "✗"
    typer.echo(
        f"  abs ≤ {ABS_TOLERANCE:.2f}:       {abs_marker} "
        f"({'PASS' if result.abs_pass else 'FAIL'})"
    )
    if result.sigma_pass is None:
        typer.echo("  sigma check:      n/a (need n≥2)")
        return result.abs_pass
    sigma_marker = "✓" if result.sigma_pass else "✗"
    typer.echo(
        f"  |Δ| ≤ {SIGMA_MULTIPLIER}σ:       {sigma_marker} "
        f"({'PASS' if result.sigma_pass else 'FAIL'})"
    )
    return result.abs_pass and result.sigma_pass


_app = typer.Typer(add_completion=False, help=__doc__.splitlines()[0])


@_app.command()
def main(
    exp_type_dir: Path = typer.Option(
        Path("experiments/discovery/baseline"),
        "--exp-type-dir",
        help="Directory containing runs.csv (default: experiments/discovery/baseline).",
    ),
    exp_id: str = typer.Option(
        "er1_cr035",
        "--exp-id",
        help="Which exp_id rows to compare (default: er1_cr035 = baseline.yaml).",
    ),
) -> None:
    """Compare coopvmas runs to the rendezvous_comm reference."""
    runs_csv = exp_type_dir / "runs.csv"
    try:
        rows = _load_runs_csv(runs_csv)
    except FileNotFoundError as exc:
        typer.echo(f"✗ {exc}", err=True)
        raise typer.Exit(2) from exc

    ref = REFERENCE.get(exp_id)
    if ref is None:
        typer.echo(
            f"✗ no reference for exp_id={exp_id!r}; known: {sorted(REFERENCE)}",
            err=True,
        )
        raise typer.Exit(2)

    typer.echo(f"F8.2 — coopvmas vs rendezvous_comm reproducibility ({exp_id})")
    typer.echo("")
    all_pass = True
    for metric, reference in ref.items():
        try:
            result = _compare(rows, exp_id, metric, reference)
        except ValueError as exc:
            typer.echo(f"✗ {exc}", err=True)
            raise typer.Exit(2) from exc
        passed = _print_result(result)
        all_pass = all_pass and passed
        typer.echo("")

    if all_pass:
        typer.echo("PASS — coopvmas reproduces rendezvous_comm within threshold.")
        raise typer.Exit(0)
    typer.echo("FAIL — see |Δ| above; threshold breach noted.")
    raise typer.Exit(1)


if __name__ == "__main__":
    sys.exit(_app() or 0)
