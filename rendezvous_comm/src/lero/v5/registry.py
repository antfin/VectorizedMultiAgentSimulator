"""Tried-and-failed registries for v5 inner + outer loops.

InnerRegistry  : tracks code-feature outcomes across inner iterations
OuterRegistry  : tracks metaprompt-framing outcomes across outer iters

Both share the same shape: a list of (handle, summary, fitness, shape)
entries plus a stagnation detector based on best-fitness trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RegistryEntry:
    iter_idx: int  # which iter this came from
    handle: str  # short identifier (e.g. "lidar_dump_28d")
    summary: str  # 1-line description of what was tried
    fitness: float  # weighted_fitness score
    M1: float  # raw M1 for sorting / filtering
    shape: str  # shape_tag
    code_excerpt: Optional[str] = None  # first ~30 lines for context

    def is_failure(self, threshold: float = 0.05) -> bool:
        return self.M1 < threshold and self.shape in ("flat_zero", "flat_nonzero")


@dataclass
class Registry:
    entries: List[RegistryEntry] = field(default_factory=list)
    fitness_trajectory: List[float] = field(default_factory=list)

    def add(self, entry: RegistryEntry) -> None:
        self.entries.append(entry)

    def record_iter_best_fitness(self, fitness: float) -> None:
        self.fitness_trajectory.append(fitness)

    def successes(self, k: int = 3) -> List[RegistryEntry]:
        return sorted(
            [e for e in self.entries if not e.is_failure()],
            key=lambda e: e.fitness,
            reverse=True,
        )[:k]

    def failures(self) -> List[RegistryEntry]:
        return [e for e in self.entries if e.is_failure()]

    def stagnated(self, window: int = 2, eps: float = 0.05) -> bool:
        """True if best-fitness hasn't improved by more than `eps`
        over the last `window` iterations."""
        traj = self.fitness_trajectory
        if len(traj) < window + 1:
            return False
        ref = traj[-(window + 1)]
        recent_best = max(traj[-window:])
        return (recent_best - ref) < eps

    def all_flat_zero(self) -> bool:
        """True if every recorded entry was sub-threshold (flat_zero)."""
        if not self.entries:
            return False
        return all(e.is_failure() for e in self.entries)

    def format_for_prompt(self, max_failures: int = 8) -> str:
        lines = []
        succ = self.successes(k=3)
        fail = self.failures()[-max_failures:]
        if succ:
            lines.append("✅ TRIED AND WORKED (kept):")
            for e in succ:
                lines.append(
                    f"  - iter {e.iter_idx}: {e.handle} → "
                    f"M1={e.M1:.3f} shape={e.shape} fitness={e.fitness:+.3f}"
                )
                lines.append(f"    why: {e.summary}")
        if fail:
            lines.append("")
            lines.append("❌ TRIED AND FAILED (do NOT regenerate):")
            for e in fail:
                lines.append(
                    f"  - iter {e.iter_idx}: {e.handle} → "
                    f"M1={e.M1:.3f} shape={e.shape}"
                )
                lines.append(f"    why: {e.summary}")
        if not lines:
            lines.append("(registry empty — no prior iterations yet)")
        return "\n".join(lines)


def InnerRegistry() -> Registry:
    return Registry()


def OuterRegistry() -> Registry:
    return Registry()
