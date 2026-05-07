"""Shared helpers used by more than one CLI subcommand."""

from pathlib import Path


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Locate the most-recent ``*.pt`` under ``run_dir/output/benchmarl/.../checkpoints/``.

    Used by ``resume``, ``eval``, and ``regenerate-videos`` — three commands
    that all need to point BenchMARL at the trained policy on disk.
    """
    bm_root = run_dir / "output" / "benchmarl"
    if not bm_root.is_dir():
        return None
    pts = list(bm_root.rglob("checkpoints/*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: p.stat().st_mtime)
