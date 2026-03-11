"""Tests for _EvalMetricsCallback and _TqdmProgressCallback.

Verifies:
  1. _EvalMetricsCallback correctly computes M1/M4 from mock rollouts
  2. CSV files are saved into the correct scalars directory
  3. Saved CSVs are loadable by RunStorage.load_benchmarl_scalars()
  4. Pickle support for BenchMARL name hashing
  5. _TqdmProgressCallback pickle support and iteration counting
  6. _suppress_noise context manager behavior
"""
import csv
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from src.runner import (
    _EvalMetricsCallback,
    _TqdmProgressCallback,
    _suppress_noise,
)
from src.storage import RunStorage


# ── Helpers ──────────────────────────────────────────────────────


def _make_rollout(done=False, n_collisions=0, n_steps=10):
    """Create a mock evaluation rollout TensorDict."""
    data = {}
    # Done signal
    done_tensor = torch.zeros(n_steps, 1, dtype=torch.bool)
    if done:
        done_tensor[-1] = True
    data[("next", "done")] = done_tensor

    # Collision reward: negative when collision happens
    coll_rew = torch.zeros(n_steps, 1)
    for i in range(min(n_collisions, n_steps)):
        coll_rew[i] = -0.1
    data[("next", "agents", "info", "collision_rew")] = coll_rew

    td = TensorDict(data, batch_size=[n_steps])
    return td


def _make_callback_with_experiment(group_name="agents"):
    """Create an _EvalMetricsCallback with a mock experiment."""
    cb = _EvalMetricsCallback()
    cb.experiment = MagicMock()
    cb.experiment.group_map = {group_name: ["agent_0", "agent_1"]}
    return cb


# ── _EvalMetricsCallback unit tests ─────────────────────────────


class TestEvalMetricsCallbackInit:
    """Test initialization state."""

    def test_initial_state(self):
        cb = _EvalMetricsCallback()
        assert cb._iter == 0
        assert cb.m1_history == []
        assert cb.m4_history == []

    def test_has_experiment_attr(self):
        cb = _EvalMetricsCallback()
        assert hasattr(cb, "experiment")


class TestEvalMetricsCallbackIterCounting:
    """Test iteration counting via on_batch_collected."""

    def test_increments_iter(self):
        cb = _EvalMetricsCallback()
        cb.on_batch_collected(None)
        cb.on_batch_collected(None)
        cb.on_batch_collected(None)
        assert cb._iter == 3


class TestEvalMetricsCallbackM1:
    """Test M1 (success rate) computation."""

    def test_all_episodes_done(self):
        cb = _make_callback_with_experiment()
        cb._iter = 10
        rollouts = [_make_rollout(done=True) for _ in range(5)]
        cb.on_evaluation_end(rollouts)
        assert len(cb.m1_history) == 1
        assert cb.m1_history[0] == (10, 1.0)

    def test_no_episodes_done(self):
        cb = _make_callback_with_experiment()
        cb._iter = 5
        rollouts = [_make_rollout(done=False) for _ in range(4)]
        cb.on_evaluation_end(rollouts)
        assert cb.m1_history[0] == (5, 0.0)

    def test_partial_success(self):
        cb = _make_callback_with_experiment()
        cb._iter = 20
        rollouts = [
            _make_rollout(done=True),
            _make_rollout(done=False),
            _make_rollout(done=True),
            _make_rollout(done=False),
        ]
        cb.on_evaluation_end(rollouts)
        assert cb.m1_history[0] == (20, 0.5)

    def test_single_episode(self):
        cb = _make_callback_with_experiment()
        cb._iter = 1
        cb.on_evaluation_end([_make_rollout(done=True)])
        assert cb.m1_history[0] == (1, 1.0)


class TestEvalMetricsCallbackM4:
    """Test M4 (collision count) computation."""

    def test_no_collisions(self):
        cb = _make_callback_with_experiment()
        cb._iter = 10
        rollouts = [_make_rollout(n_collisions=0) for _ in range(3)]
        cb.on_evaluation_end(rollouts)
        assert cb.m4_history[0] == (10, 0.0)

    def test_collisions_averaged(self):
        cb = _make_callback_with_experiment()
        cb._iter = 10
        rollouts = [
            _make_rollout(n_collisions=4),
            _make_rollout(n_collisions=2),
        ]
        cb.on_evaluation_end(rollouts)
        # 4 + 2 = 6 collisions across 2 episodes = 3.0 avg
        assert cb.m4_history[0] == (10, 3.0)

    def test_single_episode_collisions(self):
        cb = _make_callback_with_experiment()
        cb._iter = 5
        cb.on_evaluation_end([_make_rollout(n_collisions=7)])
        assert cb.m4_history[0] == (5, 7.0)


class TestEvalMetricsCallbackMultipleEvals:
    """Test accumulation across multiple evaluation checkpoints."""

    def test_history_grows(self):
        cb = _make_callback_with_experiment()

        cb._iter = 10
        cb.on_evaluation_end([_make_rollout(done=True, n_collisions=2)])
        cb._iter = 20
        cb.on_evaluation_end([_make_rollout(done=False, n_collisions=5)])
        cb._iter = 30
        cb.on_evaluation_end([_make_rollout(done=True, n_collisions=0)])

        assert len(cb.m1_history) == 3
        assert len(cb.m4_history) == 3
        assert cb.m1_history[0][0] == 10
        assert cb.m1_history[1][0] == 20
        assert cb.m1_history[2][0] == 30

    def test_m1_and_m4_recorded_together(self):
        cb = _make_callback_with_experiment()
        cb._iter = 42
        cb.on_evaluation_end([_make_rollout(done=True, n_collisions=3)])
        assert len(cb.m1_history) == 1
        assert len(cb.m4_history) == 1
        assert cb.m1_history[0][0] == cb.m4_history[0][0] == 42


class TestEvalMetricsCallbackEmptyRollouts:
    """Edge cases with empty rollout lists."""

    def test_empty_rollouts(self):
        cb = _make_callback_with_experiment()
        cb._iter = 1
        cb.on_evaluation_end([])
        assert cb.m1_history[0] == (1, 0.0)
        assert cb.m4_history[0] == (1, 0.0)


# ── CSV save and storage integration ─────────────────────────────


class TestEvalMetricsCallbackSaveCsvs:
    """Test CSV saving to correct paths."""

    def test_saves_to_existing_scalars_dir(self, tmp_path):
        scalars = tmp_path / "hash123" / "scalars"
        scalars.mkdir(parents=True)

        cb = _EvalMetricsCallback()
        cb.m1_history = [(10, 0.5), (20, 0.8)]
        cb.m4_history = [(10, 3.0), (20, 1.5)]
        cb.save_csvs(tmp_path)

        m1_csv = scalars / "eval_M1_success_rate.csv"
        m4_csv = scalars / "eval_M4_avg_collisions.csv"
        assert m1_csv.exists()
        assert m4_csv.exists()

        with open(m1_csv) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2
        assert rows[0] == ["10", "0.5"]
        assert rows[1] == ["20", "0.8"]

    def test_creates_scalars_dir_if_missing(self, tmp_path):
        cb = _EvalMetricsCallback()
        cb.m1_history = [(1, 0.2)]
        cb.m4_history = [(1, 5.0)]
        cb.save_csvs(tmp_path)

        created = tmp_path / "eval_metrics" / "scalars"
        assert created.is_dir()
        assert (created / "eval_M1_success_rate.csv").exists()
        assert (created / "eval_M4_avg_collisions.csv").exists()

    def test_skips_empty_history(self, tmp_path):
        scalars = tmp_path / "hash" / "scalars"
        scalars.mkdir(parents=True)

        cb = _EvalMetricsCallback()
        cb.m1_history = []
        cb.m4_history = [(5, 2.0)]
        cb.save_csvs(tmp_path)

        assert not (scalars / "eval_M1_success_rate.csv").exists()
        assert (scalars / "eval_M4_avg_collisions.csv").exists()

    def test_loadable_by_run_storage(self, tmp_path):
        """Saved CSVs should be picked up by load_benchmarl_scalars."""
        rs = RunStorage(tmp_path / "run", "test_run")
        scalars_dir = rs.benchmarl_dir / "hash" / "scalars"
        scalars_dir.mkdir(parents=True)

        cb = _EvalMetricsCallback()
        cb.m1_history = [(10, 0.3), (20, 0.6), (30, 0.9)]
        cb.m4_history = [(10, 5.0), (20, 2.0), (30, 1.0)]
        cb.save_csvs(rs.benchmarl_dir)

        loaded = rs.load_benchmarl_scalars()
        assert "eval_M1_success_rate" in loaded
        assert "eval_M4_avg_collisions" in loaded
        assert loaded["eval_M1_success_rate"] == [
            (10, 0.3), (20, 0.6), (30, 0.9),
        ]
        assert loaded["eval_M4_avg_collisions"] == [
            (10, 5.0), (20, 2.0), (30, 1.0),
        ]

    def test_csvs_in_correct_experiment_folder(self, tmp_path):
        """Verify CSVs end up inside the run's benchmarl dir."""
        rs = RunStorage(tmp_path / "run", "test_run")
        scalars_dir = rs.benchmarl_dir / "exp_hash" / "scalars"
        scalars_dir.mkdir(parents=True)

        cb = _EvalMetricsCallback()
        cb.m1_history = [(1, 0.5)]
        cb.m4_history = [(1, 3.0)]
        cb.save_csvs(rs.benchmarl_dir)

        # Files must be INSIDE the run's benchmarl directory
        m1_path = scalars_dir / "eval_M1_success_rate.csv"
        assert m1_path.exists()
        assert str(rs.benchmarl_dir) in str(m1_path)
        # Must NOT be in some global or parent location
        assert str(m1_path).startswith(str(rs.run_dir))


# ── Pickle support ───────────────────────────────────────────────


class TestEvalMetricsCallbackPickle:
    """BenchMARL pickles callbacks for experiment name hashing."""

    def test_pickle_roundtrip(self):
        cb = _EvalMetricsCallback()
        cb.m1_history = [(10, 0.5)]
        cb._iter = 42
        data = pickle.dumps(cb)
        restored = pickle.loads(data)
        assert isinstance(restored, _EvalMetricsCallback)

    def test_pickle_excludes_state(self):
        cb = _EvalMetricsCallback()
        cb.m1_history = [(10, 0.5)]
        state = cb.__getstate__()
        assert "m1_history" not in state


class TestTqdmProgressCallbackPickle:
    """Verify tqdm callback survives pickle."""

    def test_pickle_roundtrip(self):
        cb = _TqdmProgressCallback(60000, 6000, "test")
        data = pickle.dumps(cb)
        restored = pickle.loads(data)
        assert isinstance(restored, _TqdmProgressCallback)
        cb.close()

    def test_iteration_counting(self):
        cb = _TqdmProgressCallback(60000, 6000, "test")
        cb.on_batch_collected(None)
        cb.on_batch_collected(None)
        # Verify no crash; tqdm updated
        cb.close()


# ── _suppress_noise ──────────────────────────────────────────────


class TestSuppressNoise:
    """Test that _suppress_noise redirects stdout."""

    def test_suppresses_stdout(self, capsys):
        import sys
        with _suppress_noise():
            print("this should be suppressed")
        captured = capsys.readouterr()
        assert "suppressed" not in captured.out

    def test_restores_stdout(self, capsys):
        import sys
        with _suppress_noise():
            pass
        print("visible")
        captured = capsys.readouterr()
        assert "visible" in captured.out

    def test_suppresses_warnings(self):
        import warnings
        caught = []
        old_showwarning = warnings.showwarning

        def catcher(*args, **kwargs):
            caught.append(args)

        warnings.showwarning = catcher
        try:
            with _suppress_noise():
                warnings.warn("test warning")
            # Warning should be suppressed inside context
            assert len(caught) == 0
        finally:
            warnings.showwarning = old_showwarning
