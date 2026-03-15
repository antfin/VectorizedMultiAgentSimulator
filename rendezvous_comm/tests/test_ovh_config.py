"""Tests for OVH config loading from configs/ovh.yaml."""
import pytest
import yaml

from src.ovh import (
    GPU_MODELS,
    load_ovh_config,
    default_region,
    default_bucket_code,
    default_bucket_results,
    default_gpu,
    default_n_gpu,
    default_image,
    default_mount_code,
    default_mount_results,
    estimate_cost,
    reload_ovh_config,
    _OVH_CONFIG_PATH,
)


class TestOvhConfigFileExists:
    """configs/ovh.yaml should exist and be valid YAML."""

    def test_config_file_exists(self):
        assert _OVH_CONFIG_PATH.exists(), (
            f"OVH config not found at {_OVH_CONFIG_PATH}"
        )

    def test_config_is_valid_yaml(self):
        with open(_OVH_CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_config_has_required_sections(self):
        data = load_ovh_config()
        assert "storage" in data
        assert "training" in data
        assert "gpu_models" in data
        assert "mounts" in data


class TestDefaultsFromConfig:
    """Default functions return values from ovh.yaml."""

    def test_default_region(self):
        assert isinstance(default_region(), str)
        assert len(default_region()) >= 2

    def test_default_bucket_code(self):
        assert isinstance(default_bucket_code(), str)
        assert len(default_bucket_code()) > 0

    def test_default_bucket_results(self):
        assert isinstance(default_bucket_results(), str)
        assert len(default_bucket_results()) > 0

    def test_default_gpu(self):
        gpu = default_gpu()
        assert gpu in GPU_MODELS

    def test_default_n_gpu(self):
        assert isinstance(default_n_gpu(), int)
        assert default_n_gpu() >= 1

    def test_default_image(self):
        img = default_image()
        assert "pytorch" in img or "cuda" in img

    def test_default_mount_code(self):
        assert default_mount_code().startswith("/")

    def test_default_mount_results(self):
        assert default_mount_results().startswith("/")


class TestGpuModels:
    """GPU_MODELS populated from config."""

    def test_has_entries(self):
        assert len(GPU_MODELS) >= 1

    def test_each_has_vram_and_price(self):
        for name, info in GPU_MODELS.items():
            assert "vram_gb" in info, f"{name} missing vram_gb"
            assert "eur_per_hr" in info, f"{name} missing eur_per_hr"
            assert info["vram_gb"] > 0
            assert info["eur_per_hr"] > 0

    def test_estimate_cost_uses_config_gpu(self):
        gpu = list(GPU_MODELS.keys())[0]
        cost = estimate_cost(gpu, 1, 60, 1.0)
        expected_gpu_cost = GPU_MODELS[gpu]["eur_per_hr"]
        assert cost["gpu_cost_eur"] == pytest.approx(
            expected_gpu_cost, abs=0.01,
        )


class TestConfigNoSecrets:
    """Config file must not contain sensitive data as YAML keys."""

    def test_no_secret_keys(self):
        data = load_ovh_config()
        # Flatten all keys recursively
        keys = set()

        def _collect(d, prefix=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    keys.add(k.lower())
                    _collect(v, f"{prefix}.{k}")

        _collect(data)
        secret_keys = {
            "password", "token", "secret", "api_key", "apikey",
            "access_key", "private_key", "credentials",
        }
        found = keys & secret_keys
        assert not found, (
            f"ovh.yaml should not have secret keys: {found}"
        )
