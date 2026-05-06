"""Tests for peak-M1 checkpointing (meta.peak_checkpoint).

Uses a stub _BenchMARLCallback base so BenchMARL doesn't need to be
fully initialized — we're testing our tracking logic, not TorchRL.
"""

from typing import List

import pytest
import torch

from src.lero.meta.peak_checkpoint import (
    PeakM1Callback,
    PeakM1Tracker,
    make_peak_m1_callback,
)


# ── PeakM1Tracker: pure data ────────────────────────────────────


class TestPeakM1Tracker:
    def test_initial_state(self):
        t = PeakM1Tracker()
        assert t.peak_M1 == -1.0
        assert t.peak_at_frame == -1
        assert t.peak_state_dict is None
        assert t.m1_over_time == []

    def test_records_every_m1(self):
        t = PeakM1Tracker()
        for i, m in enumerate([0.1, 0.3, 0.2, 0.4]):
            t.record(iteration=i, frame=i * 100, m1=m)
        assert len(t.m1_over_time) == 4
        assert t.peak_M1 == pytest.approx(0.4)
        assert t.peak_at_frame == 300
        assert t.peak_iter == 3

    def test_snapshots_state_only_on_new_peak(self):
        calls: List[int] = []
        t = PeakM1Tracker()

        def factory():
            calls.append(1)
            return {"w": torch.zeros(3)}

        t.record(0, 100, 0.2, state_dict_factory=factory)  # new peak
        t.record(1, 200, 0.15, state_dict_factory=factory)  # lower, no snap
        t.record(2, 300, 0.5, state_dict_factory=factory)  # new peak
        t.record(3, 400, 0.3, state_dict_factory=factory)  # lower, no snap

        assert len(calls) == 2  # factory only called on new peaks
        assert t.peak_M1 == pytest.approx(0.5)
        assert t.peak_at_frame == 300

    def test_ties_do_not_resnapshot(self):
        t = PeakM1Tracker()
        calls: List[int] = []

        def factory():
            calls.append(1)
            return {"w": torch.zeros(1)}

        t.record(0, 100, 0.5, state_dict_factory=factory)
        t.record(1, 200, 0.5, state_dict_factory=factory)  # tie
        assert len(calls) == 1  # strict > comparison

    def test_save_peak_policy_writes_file(self, tmp_path):
        t = PeakM1Tracker()
        t.record(0, 100, 0.5, state_dict_factory=lambda: {"w": torch.arange(3.0)})
        path = tmp_path / "peak.pt"
        assert t.save_peak_policy(path) is True
        assert path.exists()
        reloaded = torch.load(path, weights_only=False)
        assert torch.equal(reloaded["w"], torch.arange(3.0))

    def test_save_peak_policy_no_snapshot_returns_false(self, tmp_path):
        t = PeakM1Tracker()
        assert t.save_peak_policy(tmp_path / "never.pt") is False
        assert not (tmp_path / "never.pt").exists()

    def test_summary_before_any_record(self):
        t = PeakM1Tracker()
        s = t.summary()
        assert s["peak_M1"] is None
        assert s["peak_at_frame"] is None
        assert s["peak_iter"] is None
        assert s["peak_m1_trajectory"] == []

    def test_summary_after_records(self):
        t = PeakM1Tracker()
        t.record(0, 100, 0.2)
        t.record(1, 200, 0.6)
        s = t.summary()
        assert s["peak_M1"] == pytest.approx(0.6)
        assert s["peak_at_frame"] == 200
        assert s["peak_iter"] == 1
        assert len(s["peak_m1_trajectory"]) == 2
        # Serializable as JSON (no torch / numpy)
        import json

        assert json.loads(json.dumps(s))["peak_M1"] == pytest.approx(0.6)


# ── PeakM1Callback: hook behavior ───────────────────────────────


class _FakeEvalCb:
    """Minimal stand-in for _EvalMetricsCallback."""

    def __init__(self):
        self.m1_history: List[tuple] = []

    def record(self, it: int, m1: float) -> None:
        self.m1_history.append((it, m1))


class _FakePolicy:
    def __init__(self, value=1.0):
        self._v = value

    def state_dict(self):
        return {"weight": torch.tensor([self._v, self._v])}


class _FakeExperiment:
    def __init__(self, policy):
        self.policy = policy


class TestPeakM1Callback:
    def test_records_latest_m1_on_eval_end(self):
        eval_cb = _FakeEvalCb()
        eval_cb.record(0, 0.3)
        eval_cb.record(1, 0.7)

        tracker = PeakM1Tracker()
        cb = PeakM1Callback(
            tracker=tracker,
            eval_source=eval_cb,
            frames_per_iteration=100,
        )
        cb.experiment = _FakeExperiment(_FakePolicy(1.0))
        cb.on_evaluation_end(rollouts=None)

        # Latest M1 = 0.7 at iter 1 → frame = 100.
        assert tracker.peak_M1 == pytest.approx(0.7)
        assert tracker.peak_at_frame == 100
        assert tracker.peak_state_dict is not None
        assert torch.equal(
            tracker.peak_state_dict["weight"],
            torch.tensor([1.0, 1.0]),
        )

    def test_no_history_is_no_op(self):
        """Evaluation callback hasn't fired yet — don't crash, don't record."""
        eval_cb = _FakeEvalCb()  # empty
        tracker = PeakM1Tracker()
        cb = PeakM1Callback(tracker, eval_cb, 100)
        cb.experiment = _FakeExperiment(_FakePolicy())
        cb.on_evaluation_end(None)
        assert tracker.peak_at_frame == -1

    def test_state_dict_is_cloned(self):
        """Subsequent gradient steps must not mutate the snapshot."""
        eval_cb = _FakeEvalCb()
        eval_cb.record(0, 0.5)

        policy = _FakePolicy(value=1.0)
        tracker = PeakM1Tracker()
        cb = PeakM1Callback(tracker, eval_cb, 100)
        cb.experiment = _FakeExperiment(policy)
        cb.on_evaluation_end(None)

        # Simulate gradient update mutating the live policy.
        policy._v = 999.0
        # Snapshot should still reflect the original value.
        assert torch.equal(
            tracker.peak_state_dict["weight"],
            torch.tensor([1.0, 1.0]),
        )

    def test_on_batch_collected_tracks_iter(self):
        cb = PeakM1Callback(PeakM1Tracker(), _FakeEvalCb(), 1)
        for _ in range(5):
            cb.on_batch_collected(batch=None)
        assert cb._iter == 5

    def test_state_dict_with_non_tensor_values(self):
        """BenchMARL policy state_dicts can hold torch.Size and other
        non-Tensor objects. The factory must not call ``.detach()`` on
        them — only real tensors. Regression for 2026-04-21 bug where
        ``'torch.Size' object has no attribute 'detach'`` killed full
        training.
        """

        class MixedPolicy:
            def state_dict(self):
                return {
                    "weight": torch.tensor([1.0, 2.0]),
                    "bias": torch.tensor([0.5]),
                    "input_shape": torch.Size([4, 2]),  # non-tensor
                    "metadata_str": "relu",  # non-tensor
                }

        eval_cb = _FakeEvalCb()
        eval_cb.record(0, 0.5)
        tracker = PeakM1Tracker()
        cb = PeakM1Callback(tracker, eval_cb, 100)
        cb.experiment = _FakeExperiment(MixedPolicy())
        # Must not raise.
        cb.on_evaluation_end(rollouts=None)

        snap = tracker.peak_state_dict
        assert snap is not None
        assert torch.equal(snap["weight"], torch.tensor([1.0, 2.0]))
        assert snap["input_shape"] == torch.Size([4, 2])
        assert snap["metadata_str"] == "relu"


# ── make_peak_m1_callback factory ───────────────────────────────


class _StubBench:
    """Stand-in for runner._BenchMARLCallback — just needs to be a class
    with a no-arg __init__."""

    def __init__(self):
        self.experiment = None


class TestMakePeakM1Callback:
    def test_subclass_of_given_base(self):
        tracker = PeakM1Tracker()
        eval_cb = _FakeEvalCb()
        bound = make_peak_m1_callback(
            tracker=tracker,
            eval_source=eval_cb,
            frames_per_iteration=100,
            bench_callback_base=_StubBench,
        )
        assert isinstance(bound, _StubBench)

    def test_getstate_returns_dummy(self):
        """BenchMARL multi-process training expects __getstate__ to return
        a minimal dict so the live reference graph (tracker, policy,
        experiment) doesn't get pickled into every worker.

        Matches the convention used by _EvalMetricsCallback.
        """
        tracker = PeakM1Tracker()
        bound = make_peak_m1_callback(
            tracker,
            _FakeEvalCb(),
            100,
            _StubBench,
        )
        assert bound.__getstate__() == {"_dummy": True}
        # __setstate__ must not crash when given the dummy back.
        bound.__setstate__({"_dummy": True})

    def test_on_evaluation_end_delegates(self):
        """The factory-bound callback forwards hooks into PeakM1Callback."""
        tracker = PeakM1Tracker()
        eval_cb = _FakeEvalCb()
        eval_cb.record(0, 0.42)

        bound = make_peak_m1_callback(
            tracker,
            eval_cb,
            50,
            _StubBench,
        )
        bound.experiment = _FakeExperiment(_FakePolicy(2.0))
        bound.on_evaluation_end(rollouts=None)

        assert tracker.peak_M1 == pytest.approx(0.42)
        assert tracker.peak_at_frame == 0  # iter 0 × 50 frames
        assert tracker.peak_state_dict is not None
