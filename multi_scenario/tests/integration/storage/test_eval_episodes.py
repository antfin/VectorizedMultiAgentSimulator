"""F2.10.1 tests: LocalStorageAdapter.save_eval_episodes — JSON round-trip."""

import json
from pathlib import Path

import torch

from multi_scenario.adapters.storage.local import LocalStorageAdapter


def test_save_eval_episodes_universal_keys(tmp_path: Path) -> None:
    """Universal tensors (returns / lengths / collisions) round-trip as lists."""
    rollout = {
        "episode_returns": torch.tensor([1.5, 2.5, 3.5]),
        "episode_lengths": torch.tensor([10, 20, 30]),
        "episode_collisions": torch.tensor([0.0, 1.0, 2.0]),
    }

    LocalStorageAdapter().save_eval_episodes(tmp_path, rollout)

    on_disk = json.loads((tmp_path / "output" / "eval_episodes.json").read_text(encoding="utf-8"))
    assert on_disk["episode_returns"] == [1.5, 2.5, 3.5]
    assert on_disk["episode_lengths"] == [10, 20, 30]
    assert on_disk["episode_collisions"] == [0.0, 1.0, 2.0]


def test_save_eval_episodes_with_discovery_keys(tmp_path: Path) -> None:
    """Discovery rollouts include `targets_covered` (2D) + `n_targets` (int)."""
    rollout = {
        "episode_returns": torch.tensor([1.0]),
        "episode_lengths": torch.tensor([5]),
        "episode_collisions": torch.tensor([0.0]),
        "targets_covered": torch.tensor([[0, 0, 1, 1, 2]]),
        "n_targets": 7,
    }

    LocalStorageAdapter().save_eval_episodes(tmp_path, rollout)

    on_disk = json.loads((tmp_path / "output" / "eval_episodes.json").read_text(encoding="utf-8"))
    assert on_disk["targets_covered"] == [[0, 0, 1, 1, 2]]
    assert on_disk["n_targets"] == 7


def test_save_eval_episodes_skips_unknown_keys(tmp_path: Path) -> None:
    """Only documented schema keys are serialised; unknown keys are dropped silently."""
    rollout = {
        "episode_returns": torch.tensor([1.0]),
        "episode_lengths": torch.tensor([5]),
        "episode_collisions": torch.tensor([0.0]),
        "future_token_field": torch.tensor([42]),  # not in schema yet
        "internal_state": "anything",
    }

    LocalStorageAdapter().save_eval_episodes(tmp_path, rollout)

    on_disk = json.loads((tmp_path / "output" / "eval_episodes.json").read_text(encoding="utf-8"))
    assert "future_token_field" not in on_disk
    assert "internal_state" not in on_disk
    assert "episode_returns" in on_disk


def test_save_eval_episodes_creates_output_dir(tmp_path: Path) -> None:
    """Parent dir is auto-created (consistent with the other LocalStorage writers)."""
    rollout = {"episode_returns": torch.tensor([1.0])}
    LocalStorageAdapter().save_eval_episodes(tmp_path, rollout)
    assert (tmp_path / "output" / "eval_episodes.json").is_file()
