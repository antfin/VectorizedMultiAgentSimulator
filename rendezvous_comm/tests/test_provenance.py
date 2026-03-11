"""Tests for the provenance module (config+code hashing, freshness detection)."""
import json
from pathlib import Path

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


def _write_yaml(path: Path, data: dict):
    """Dump a dict to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f)


def _sample_config():
    return {"algorithm": "mappo", "n_agents": 4, "lidar_range": 0.35}


# ── compute_config_hash ──────────────────────────────────────────


class TestComputeConfigHash:
    """Verify config hashing behaviour."""

    def test_returns_sha256_prefix(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())
        h = compute_config_hash(cfg)
        assert h.startswith("sha256:")

    def test_hash_length(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())
        h = compute_config_hash(cfg)
        # "sha256:" (7 chars) + 16 hex chars = 23
        assert len(h) == 23

    def test_same_content_same_hash(self, tmp_path):
        cfg_a = tmp_path / "a.yaml"
        cfg_b = tmp_path / "b.yaml"
        data = _sample_config()
        _write_yaml(cfg_a, data)
        _write_yaml(cfg_b, data)
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)

    def test_different_content_different_hash(self, tmp_path):
        cfg_a = tmp_path / "a.yaml"
        cfg_b = tmp_path / "b.yaml"
        _write_yaml(cfg_a, {"x": 1})
        _write_yaml(cfg_b, {"x": 2})
        assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)

    def test_key_order_independent(self, tmp_path):
        """YAML normalization means different key order produces same hash."""
        cfg_a = tmp_path / "a.yaml"
        cfg_b = tmp_path / "b.yaml"
        # Write raw strings with deliberately different key order
        cfg_a.write_text("z_key: 1\na_key: 2\nm_key: 3\n")
        cfg_b.write_text("a_key: 2\nm_key: 3\nz_key: 1\n")
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)

    def test_deterministic(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())
        h1 = compute_config_hash(cfg)
        h2 = compute_config_hash(cfg)
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
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)
        prov = load_provenance(run_dir)

        assert isinstance(prov, Provenance)
        assert prov.config_hash == compute_config_hash(cfg)
        assert prov.code_hash == compute_code_hash()
        assert prov.source_config_path == str(cfg)

    def test_creates_input_dir(self, tmp_path):
        run_dir = tmp_path / "run_no_input"
        # Deliberately do NOT create run_dir/input/
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)

        assert (run_dir / "input" / "provenance.json").exists()

    def test_all_fields_populated(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)
        prov = load_provenance(run_dir)

        assert prov.config_hash
        assert prov.code_hash
        assert prov.git_commit  # at least "unknown"
        assert isinstance(prov.git_dirty, bool)
        assert prov.created_at  # ISO timestamp string
        assert prov.source_config_path
        assert isinstance(prov.hashed_source_files, list)
        assert len(prov.hashed_source_files) > 0

    def test_hashed_source_files_sorted(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)
        prov = load_provenance(run_dir)

        assert prov.hashed_source_files == sorted(prov.hashed_source_files)

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_provenance(tmp_path) is None

    def test_provenance_json_is_valid_json(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)

        prov_path = run_dir / "input" / "provenance.json"
        with open(prov_path) as f:
            data = json.load(f)
        assert "config_hash" in data
        assert "code_hash" in data


# ── check_freshness ──────────────────────────────────────────────


class TestCheckFreshness:
    """Freshness detection with real and mocked hashes."""

    def test_no_provenance(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())
        assert check_freshness(tmp_path, cfg) == Freshness.NO_PROVENANCE

    def test_valid_when_unchanged(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)
        assert check_freshness(run_dir, cfg) == Freshness.VALID

    def test_config_changed(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)

        # Modify the config file
        _write_yaml(cfg, {"algorithm": "ippo", "n_agents": 6})
        assert check_freshness(run_dir, cfg) == Freshness.CONFIG_CHANGED

    def test_code_changed(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)

        # Fake a different code hash
        monkeypatch.setattr(
            "src.provenance.compute_code_hash",
            lambda: "sha256:aaaaaaaaaaaaaaaa",
        )
        assert check_freshness(run_dir, cfg) == Freshness.CODE_CHANGED

    def test_both_changed(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        _write_yaml(cfg, _sample_config())

        save_provenance(run_dir, cfg)

        # Modify config
        _write_yaml(cfg, {"algorithm": "ippo"})
        # Fake a different code hash
        monkeypatch.setattr(
            "src.provenance.compute_code_hash",
            lambda: "sha256:bbbbbbbbbbbbbbbb",
        )
        assert check_freshness(run_dir, cfg) == Freshness.BOTH_CHANGED

    def test_valid_after_identical_rewrite(self, tmp_path):
        """Re-writing identical YAML content still counts as VALID."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = tmp_path / "cfg.yaml"
        data = _sample_config()
        _write_yaml(cfg, data)

        save_provenance(run_dir, cfg)

        # Re-write the same data
        _write_yaml(cfg, data)
        assert check_freshness(run_dir, cfg) == Freshness.VALID


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
