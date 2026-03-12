"""Tests for the provenance module (config+code hashing, freshness detection).

Config freshness is based on non-sweep parameters (task + train dicts),
NOT on the source YAML file. Two different YAML files that produce the
same task+train params are treated as equivalent.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from src.provenance import (
    Freshness,
    Provenance,
    check_freshness,
    compute_code_hash,
    compute_config_hash,
    load_provenance,
    save_provenance,
)


# ── Helpers ───────────────────────────────────────────────────────


def _sample_task_dict():
    return {
        "n_agents": 4, "n_targets": 7, "agents_per_target": 2,
        "lidar_range": 0.35, "covering_range": 0.25,
        "max_steps": 200,
    }


def _sample_train_dict():
    return {
        "algorithm": "mappo", "max_n_frames": 6_000_000,
        "gamma": 0.99, "lr": 5e-5,
    }


def _make_spec(task_dict=None, train_dict=None):
    """Create a mock ExperimentSpec with task.to_dict() and train.__dict__."""
    spec = MagicMock()
    td = task_dict or _sample_task_dict()
    tr = train_dict or _sample_train_dict()
    spec.task.to_dict.return_value = td
    spec.train.__dict__ = tr
    return spec


# ── compute_config_hash ──────────────────────────────────────────


class TestComputeConfigHash:
    """Verify config hashing behaviour."""

    def test_returns_sha256_prefix(self):
        h = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        assert h.startswith("sha256:")

    def test_hash_length(self):
        h = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        # "sha256:" (7 chars) + 16 hex chars = 23
        assert len(h) == 23

    def test_same_content_same_hash(self):
        h1 = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        h2 = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_config_hash(
            _sample_task_dict(), _sample_train_dict(),
        )
        different_train = _sample_train_dict()
        different_train["lr"] = 1e-3
        h2 = compute_config_hash(
            _sample_task_dict(), different_train,
        )
        assert h1 != h2

    def test_key_order_independent(self):
        """Dict order should not matter (YAML normalization)."""
        task_a = {"z_key": 1, "a_key": 2}
        task_b = {"a_key": 2, "z_key": 1}
        train = {"x": 1}
        assert compute_config_hash(task_a, train) == compute_config_hash(
            task_b, train,
        )

    def test_deterministic(self):
        h1 = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        h2 = compute_config_hash(_sample_task_dict(), _sample_train_dict())
        assert h1 == h2


# ── compute_code_hash ────────────────────────────────────────────


class TestComputeCodeHash:
    """Verify code hashing behaviour."""

    def test_returns_sha256_prefix(self):
        h = compute_code_hash()
        assert h.startswith("sha256:")

    def test_hash_length(self):
        h = compute_code_hash()
        assert len(h) == 23

    def test_deterministic(self):
        h1 = compute_code_hash()
        h2 = compute_code_hash()
        assert h1 == h2


# ── save_provenance + load_provenance ────────────────────────────


class TestSaveLoadProvenance:
    """Round-trip save/load and edge cases."""

    def test_round_trip(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        task = _sample_task_dict()
        train = _sample_train_dict()

        save_provenance(run_dir, task, train)
        prov = load_provenance(run_dir)

        assert isinstance(prov, Provenance)
        assert prov.config_hash == compute_config_hash(task, train)
        assert prov.code_hash == compute_code_hash()

    def test_creates_input_dir(self, tmp_path):
        run_dir = tmp_path / "run_no_input"
        # Deliberately do NOT create run_dir/input/
        save_provenance(run_dir, _sample_task_dict(), _sample_train_dict())
        assert (run_dir / "input" / "provenance.json").exists()

    def test_all_fields_populated(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(run_dir, _sample_task_dict(), _sample_train_dict())
        prov = load_provenance(run_dir)

        assert prov.config_hash
        assert prov.code_hash
        assert prov.git_commit  # at least "unknown"
        assert isinstance(prov.git_dirty, bool)
        assert prov.created_at  # ISO timestamp string
        assert isinstance(prov.hashed_source_files, list)
        assert len(prov.hashed_source_files) > 0

    def test_hashed_source_files_sorted(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(run_dir, _sample_task_dict(), _sample_train_dict())
        prov = load_provenance(run_dir)

        assert prov.hashed_source_files == sorted(prov.hashed_source_files)

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_provenance(tmp_path) is None

    def test_provenance_json_is_valid_json(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(run_dir, _sample_task_dict(), _sample_train_dict())

        prov_path = run_dir / "input" / "provenance.json"
        with open(prov_path) as f:
            data = json.load(f)
        assert "config_hash" in data
        assert "code_hash" in data


# ── check_freshness ──────────────────────────────────────────────


class TestCheckFreshness:
    """Freshness detection with real and mocked hashes."""

    def test_no_provenance(self, tmp_path):
        spec = _make_spec()
        assert check_freshness(tmp_path, spec) == Freshness.NO_PROVENANCE

    def test_valid_when_unchanged(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        task = _sample_task_dict()
        train = _sample_train_dict()

        save_provenance(run_dir, task, train)
        spec = _make_spec(task, train)
        assert check_freshness(run_dir, spec) == Freshness.VALID

    def test_config_changed(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(
            run_dir, _sample_task_dict(), _sample_train_dict(),
        )

        # Check with different task params
        different_task = _sample_task_dict()
        different_task["n_agents"] = 8
        spec = _make_spec(different_task, _sample_train_dict())
        assert check_freshness(run_dir, spec) == Freshness.CONFIG_CHANGED

    def test_code_changed(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        task = _sample_task_dict()
        train = _sample_train_dict()

        save_provenance(run_dir, task, train)

        # Fake a different code hash
        monkeypatch.setattr(
            "src.provenance.compute_code_hash",
            lambda: "sha256:aaaaaaaaaaaaaaaa",
        )
        spec = _make_spec(task, train)
        assert check_freshness(run_dir, spec) == Freshness.CODE_CHANGED

    def test_both_changed(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(
            run_dir, _sample_task_dict(), _sample_train_dict(),
        )

        # Different task params
        different_task = _sample_task_dict()
        different_task["max_steps"] = 500
        spec = _make_spec(different_task, _sample_train_dict())
        # Fake a different code hash
        monkeypatch.setattr(
            "src.provenance.compute_code_hash",
            lambda: "sha256:bbbbbbbbbbbbbbbb",
        )
        assert check_freshness(run_dir, spec) == Freshness.BOTH_CHANGED

    def test_valid_with_same_params_different_source(self, tmp_path):
        """Two configs with identical task+train params → VALID."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        task = _sample_task_dict()
        train = _sample_train_dict()

        save_provenance(run_dir, task, train)

        # "Different YAML" but same params → still valid
        spec = _make_spec(dict(task), dict(train))
        assert check_freshness(run_dir, spec) == Freshness.VALID

    def test_train_param_change_detected(self, tmp_path):
        """Changing a train param (e.g. lr) triggers CONFIG_CHANGED."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        save_provenance(
            run_dir, _sample_task_dict(), _sample_train_dict(),
        )

        different_train = _sample_train_dict()
        different_train["lr"] = 1e-3
        spec = _make_spec(_sample_task_dict(), different_train)
        assert check_freshness(run_dir, spec) == Freshness.CONFIG_CHANGED


# ── Freshness enum ───────────────────────────────────────────────


class TestFreshnessEnum:
    """Verify the Freshness enum members."""

    def test_all_values_are_strings(self):
        for member in Freshness:
            assert isinstance(member.value, str)

    def test_expected_members_exist(self):
        expected = {
            "VALID",
            "CONFIG_CHANGED",
            "CODE_CHANGED",
            "BOTH_CHANGED",
            "NO_PROVENANCE",
        }
        actual = {m.name for m in Freshness}
        assert actual == expected

    def test_member_count(self):
        assert len(Freshness) == 5
