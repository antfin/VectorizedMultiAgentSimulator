"""F8.2 — fire ER1 ×3 seeds on OVH for reproducibility validation.

Thin convenience wrapper over ``multi-scenario sweep`` so the canonical
ER1 command lives next to its reference numbers and parity tests.

Usage::

    # Preview only (no jobs submitted):
    python scripts/run_er1_reproducibility.py

    # Actually fire 3 OVH jobs (~€19, ~3-4h on V100S parallel):
    python scripts/run_er1_reproducibility.py --execute

After the jobs land, run::

    python scripts/compare_to_reference.py

to see PASS/FAIL against the rendezvous_comm reference.
"""

# pylint: disable=missing-function-docstring

import subprocess
import sys
from pathlib import Path

import typer

_BASELINE_YAML = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "discovery" / "baseline" / "configs" / "baseline.yaml"
)
_OVH_CFG = Path(__file__).resolve().parents[1] / "configs" / "ovh.yaml"
_SEEDS = "0,1,2"

_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help=__doc__.splitlines()[0],
)


@_app.command()
def main(
    execute: bool = typer.Option(
        False,
        "--execute",
        help=(
            "Actually submit the OVH jobs (default: preview only). "
            "Costs ~€19 (3 seeds × ~3-4h V100S parallel)."
        ),
    ),
    follow: bool = typer.Option(
        True,
        "--follow / --no-follow",
        help="Block until all 3 jobs reach a terminal state + sync results back.",
    ),
) -> None:
    """Submit ER1 reproducibility sweep (3 seeds, OVH GPU, ~€19)."""
    if not _BASELINE_YAML.is_file():
        typer.echo(f"✗ baseline YAML not at {_BASELINE_YAML}", err=True)
        raise typer.Exit(1)
    if not _OVH_CFG.is_file():
        typer.echo(f"✗ OVH config not at {_OVH_CFG}", err=True)
        raise typer.Exit(1)

    cmd = [
        "multi-scenario", "sweep",
        "--seeds", _SEEDS,
        "--runner", "ovh",
        "--ovh-config", str(_OVH_CFG),
        "--yes",  # F6.7 cost-cap bypass — script runs unattended
    ]
    if follow:
        cmd.append("--follow")
    cmd.append(str(_BASELINE_YAML))

    typer.echo("F8.2 — ER1 reproducibility sweep")
    typer.echo(f"  baseline: {_BASELINE_YAML.name}")
    typer.echo(f"  seeds:    {_SEEDS}")
    typer.echo(f"  runner:   ovh ({_OVH_CFG.name})")
    typer.echo("")
    typer.echo("$ " + " ".join(cmd))
    typer.echo("")

    if not execute:
        typer.echo("(preview only — pass --execute to fire)")
        raise typer.Exit(0)

    typer.echo("Submitting 3 jobs to OVH …")
    rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
    if rc != 0:
        typer.echo(f"✗ sweep exited with rc={rc}", err=True)
        raise typer.Exit(rc)
    typer.echo("")
    typer.echo("Sweep complete. Next: python scripts/compare_to_reference.py")


if __name__ == "__main__":
    sys.exit(_app() or 0)
