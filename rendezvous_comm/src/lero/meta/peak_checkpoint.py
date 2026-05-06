"""Peak-M1 checkpointing during Tier-2 full training.

Why: LERO's Phase-5 k=2 results had templates like S3a_gpt5 reach
peak M1=0.86 at ~1M frames and then collapse to 0.09 at 10M as the
policy discovered exploits in the LLM-designed reward. Without
peak checkpointing we lose the good intermediate policy forever.
See lero.md §2.5 and §5.1.

This module implements the side-quest the LERO doc flags as
"highest-priority code change for future experiments".

Shape:

  - ``PeakM1Tracker`` — pure data holder (peak value, frame, policy
    state_dict). Unit-testable without BenchMARL.
  - ``PeakM1Callback`` — thin BenchMARL hook that inspects an
    ``_EvalMetricsCallback``'s latest ``m1_history`` entry on every
    ``on_evaluation_end`` and updates the tracker. Does not compute M1
    itself — reuses what the existing callback already provides, so
    there is one definition of M1 per run.

Output on disk (written at end of training by the caller):
  - ``best_policy_peak.pt`` (state_dict saved at peak-M1 frame)
  - peak fields merged into final_metrics.json
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

# BenchMARL callbacks are pickled per worker. A closure-local subclass
# can't be resolved by stdlib pickle (which BenchMARL uses in parts of
# its data-collection pipeline), so the bound callback MUST live at
# module scope. We lazy-import the base class with a fallback to
# ``object`` — mirrors the same pattern used in ``src/runner.py``.
try:
    from benchmarl.experiment import Callback as _BenchCallbackBase  # type: ignore
except Exception:  # pragma: no cover — only hit when BenchMARL is absent
    _BenchCallbackBase = object  # type: ignore[misc, assignment]


@dataclass
class PeakM1Tracker:
    """Append-only record of per-eval M1 with best-so-far state cached."""

    peak_M1: float = -1.0  # start below any real M1
    peak_at_frame: int = -1  # -1 = no eval seen yet
    peak_iter: int = -1  # BenchMARL iteration index
    peak_state_dict: Optional[Dict[str, Any]] = None
    # Full trajectory — helpful for post-hoc analysis + regression tests.
    m1_over_time: List[Dict[str, float]] = field(default_factory=list)

    def record(
        self,
        iteration: int,
        frame: int,
        m1: float,
        state_dict_factory: Optional[Any] = None,
    ) -> None:
        """Record an M1 measurement; snapshot state when it's a new peak.

        Args:
            iteration: BenchMARL iter counter (integer).
            frame: total frames collected so far.
            m1: M1 value from the most recent evaluation.
            state_dict_factory: callable returning a NEW dict of CPU
                tensors (cloned so gradient steps don't mutate our
                checkpoint). Only called when we beat the current peak,
                so cheap M1 updates don't pay the copy cost.
        """
        self.m1_over_time.append(
            {
                "iteration": iteration,
                "frame": frame,
                "M1": m1,
            }
        )
        if m1 > self.peak_M1:
            self.peak_M1 = float(m1)
            self.peak_at_frame = int(frame)
            self.peak_iter = int(iteration)
            if state_dict_factory is not None:
                self.peak_state_dict = state_dict_factory()

    def save_peak_policy(self, path: Path) -> bool:
        """Write the best-seen state_dict to ``path``. Returns True on save."""
        if self.peak_state_dict is None:
            return False
        if torch is None:
            raise RuntimeError("torch is required to save peak policy")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.peak_state_dict, path)
        return True

    def summary(self) -> Dict[str, Any]:
        """JSON-safe rollup for final_metrics.json."""
        return {
            "peak_M1": self.peak_M1 if self.peak_at_frame >= 0 else None,
            "peak_at_frame": self.peak_at_frame if self.peak_at_frame >= 0 else None,
            "peak_iter": self.peak_iter if self.peak_at_frame >= 0 else None,
            "peak_m1_trajectory": list(self.m1_over_time),
        }


class PeakM1Callback:
    """BenchMARL callback that drives a ``PeakM1Tracker``.

    Intentionally does NOT subclass any BenchMARL type at import time —
    the repo wraps it in ``_BenchMARLCallback`` via a simple factory so
    we can unit-test this class without importing BenchMARL.

    Wired in by ``make_peak_m1_callback`` below (the factory that
    takes the repo's ``_BenchMARLCallback`` base so this module stays
    import-clean).
    """

    def __init__(
        self,
        tracker: PeakM1Tracker,
        eval_source: Any,  # _EvalMetricsCallback-ish
        frames_per_iteration: int = 1,
    ) -> None:
        self.tracker = tracker
        self.eval_source = eval_source
        self.frames_per_iteration = frames_per_iteration
        self._iter = 0

    # Callback hooks — duck-typed; the real subclass (built by the
    # factory below) delegates into these.

    def on_batch_collected(self, batch) -> None:
        self._iter += 1

    def on_evaluation_end(self, rollouts) -> None:
        m1_history = getattr(self.eval_source, "m1_history", None)
        if not m1_history:
            return
        latest_iter, latest_m1 = m1_history[-1]
        # ``self.experiment`` is set by the BenchMARL subclass below.
        state_factory = None
        exp = getattr(self, "experiment", None)
        policy = getattr(exp, "policy", None) if exp is not None else None
        if policy is not None and torch is not None:

            def state_factory():  # noqa: F811 — intentional shadow
                # BenchMARL policy state_dicts can contain non-tensor
                # values (e.g. torch.Size for shape metadata). Only
                # detach/clone real tensors; pass through other
                # picklable objects unchanged.
                out: Dict[str, Any] = {}
                for k, v in policy.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        out[k] = v.detach().cpu().clone()
                    else:
                        out[k] = v
                return out

        self.tracker.record(
            iteration=latest_iter,
            frame=latest_iter * self.frames_per_iteration,
            m1=latest_m1,
            state_dict_factory=state_factory,
        )


class _BoundPeakCallback(_BenchCallbackBase):
    """BenchMARL callback that delegates to a PeakM1Callback.

    Defined at module scope (not inside a factory) so stdlib pickle
    can locate the class by name — BenchMARL pickles callbacks when
    spawning data-collection workers, and closure-local classes are
    not picklable.

    Instances are assigned ``_impl`` by ``make_peak_m1_callback``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._impl: Optional[PeakM1Callback] = None

    def __getstate__(self):
        return {"_dummy": True}

    def __setstate__(self, state):
        # State is intentionally not restored: a pickled worker copy
        # does not need to track peaks (only the main process does).
        self._impl = None

    def on_batch_collected(self, batch):
        if self._impl is not None:
            self._impl.on_batch_collected(batch)

    def on_evaluation_end(self, rollouts):
        if self._impl is None:
            return
        # BenchMARL sets ``self.experiment`` before firing hooks.
        self._impl.experiment = getattr(self, "experiment", None)
        self._impl.on_evaluation_end(rollouts)


def make_peak_m1_callback(
    tracker: PeakM1Tracker,
    eval_source: Any,
    frames_per_iteration: int,
    bench_callback_base: Any = None,
):
    """Return a BenchMARL-subclass instance wrapping PeakM1Callback.

    ``bench_callback_base`` is kept as a parameter for unit tests that
    want to inject a stub base (e.g. ``object``). In production the
    module-level ``_BoundPeakCallback`` is used directly, which is
    necessary for pickle-based multi-process data collection.
    """
    if bench_callback_base is None or bench_callback_base is _BenchCallbackBase:
        cb = _BoundPeakCallback()
    else:
        # Test path: dynamically extend the stub base.
        class _TestBoundCallback(bench_callback_base):  # type: ignore[misc, valid-type]
            def __init__(self):
                super().__init__()
                self._impl: Optional[PeakM1Callback] = None

            def __getstate__(self):
                return {"_dummy": True}

            def __setstate__(self, state):
                self._impl = None

            def on_batch_collected(self, batch):
                if self._impl is not None:
                    self._impl.on_batch_collected(batch)

            def on_evaluation_end(self, rollouts):
                if self._impl is None:
                    return
                self._impl.experiment = getattr(self, "experiment", None)
                self._impl.on_evaluation_end(rollouts)

        cb = _TestBoundCallback()

    cb._impl = PeakM1Callback(
        tracker=tracker,
        eval_source=eval_source,
        frames_per_iteration=frames_per_iteration,
    )
    return cb
