"""Tests for policy export/import via RunStorage.

Verifies that:
  1. state_dict round-trips through save_policy / load_policy_state_dict
  2. A loaded state_dict restores identical model weights
  3. A restored model produces identical outputs for the same input
  4. BenchMARL experiment policy survives export/import (integration)
"""
import torch
import torch.nn as nn
import pytest

from src.storage import RunStorage


# ── Helpers ──────────────────────────────────────────────────────


class _SimplePolicy(nn.Module):
    """Minimal policy network for testing."""

    def __init__(self, obs_dim=10, act_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class _LargerPolicy(nn.Module):
    """Multi-layer policy with different param types."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Basic round-trip ─────────────────────────────────────────────


class TestPolicySaveLoadRoundTrip:
    """Verify state_dict survives disk round-trip."""

    def test_save_and_load_returns_dict(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        policy = _SimplePolicy()
        rs.save_policy(policy)

        loaded = rs.load_policy_state_dict()
        assert loaded is not None
        assert isinstance(loaded, dict)

    def test_load_returns_none_when_missing(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        assert rs.load_policy_state_dict() is None

    def test_state_dict_keys_match(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        policy = _SimplePolicy()
        original_keys = set(policy.state_dict().keys())

        rs.save_policy(policy)
        loaded = rs.load_policy_state_dict()
        assert set(loaded.keys()) == original_keys

    def test_state_dict_values_match(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        policy = _SimplePolicy()

        rs.save_policy(policy)
        loaded = rs.load_policy_state_dict()

        for key in policy.state_dict():
            assert torch.equal(
                policy.state_dict()[key],
                loaded[key],
            ), f"Mismatch in {key}"

    def test_file_exists_after_save(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        rs.save_policy(_SimplePolicy())

        assert (rs.output_dir / "policy.pt").exists()
        assert rs.has_policy()

    def test_file_size_reasonable(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        rs.save_policy(_SimplePolicy())

        size = (rs.output_dir / "policy.pt").stat().st_size
        # A small MLP should be a few KB, not empty or huge
        assert 100 < size < 1_000_000


# ── Weight restoration ───────────────────────────────────────────


class TestPolicyWeightRestoration:
    """Verify loaded weights restore an identical model."""

    def test_load_state_dict_restores_weights(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")

        # Train-like: create and save
        original = _SimplePolicy()
        # Modify weights to non-default values
        with torch.no_grad():
            for p in original.parameters():
                p.fill_(42.0)
        rs.save_policy(original)

        # Reload into a fresh model
        fresh = _SimplePolicy()
        loaded_sd = rs.load_policy_state_dict()
        fresh.load_state_dict(loaded_sd)

        for key in original.state_dict():
            assert torch.equal(
                original.state_dict()[key],
                fresh.state_dict()[key],
            ), f"Weight mismatch in {key}"

    def test_restored_model_produces_same_output(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")

        original = _SimplePolicy(obs_dim=10, act_dim=2)
        rs.save_policy(original)

        # Rebuild and load
        restored = _SimplePolicy(obs_dim=10, act_dim=2)
        restored.load_state_dict(rs.load_policy_state_dict())

        # Same input → same output
        test_input = torch.randn(5, 10)
        original.eval()
        restored.eval()
        with torch.no_grad():
            out_orig = original(test_input)
            out_rest = restored(test_input)

        assert torch.equal(out_orig, out_rest)

    def test_larger_policy_round_trip(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")

        original = _LargerPolicy()
        rs.save_policy(original)

        restored = _LargerPolicy()
        restored.load_state_dict(rs.load_policy_state_dict())

        test_input = torch.randn(8, 20)
        original.eval()
        restored.eval()
        with torch.no_grad():
            assert torch.equal(original(test_input), restored(test_input))

    def test_mismatched_architecture_raises(self, tmp_path):
        """Loading into a differently-shaped model should fail."""
        rs = RunStorage(tmp_path / "run", "test_run")

        original = _SimplePolicy(obs_dim=10, act_dim=2)
        rs.save_policy(original)

        wrong_shape = _SimplePolicy(obs_dim=10, act_dim=5)
        with pytest.raises(RuntimeError):
            wrong_shape.load_state_dict(rs.load_policy_state_dict())


# ── Trained weights differ from random init ──────────────────────


class TestPolicyNotRandom:
    """Verify saved policy is not just random initialization."""

    def test_saved_weights_differ_from_fresh_init(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")

        # Simulate "trained" policy (modified weights)
        trained = _SimplePolicy()
        with torch.no_grad():
            for p in trained.parameters():
                p.add_(10.0)  # shift all weights
        rs.save_policy(trained)

        # Fresh init should differ
        fresh = _SimplePolicy()
        loaded_sd = rs.load_policy_state_dict()

        any_different = False
        for key in loaded_sd:
            if not torch.equal(loaded_sd[key], fresh.state_dict()[key]):
                any_different = True
                break
        assert any_different, "Loaded weights should differ from fresh init"


# ── Device mapping ───────────────────────────────────────────────


class TestPolicyDeviceMapping:
    """Verify load_policy_state_dict maps to CPU."""

    def test_loaded_tensors_on_cpu(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")
        rs.save_policy(_SimplePolicy())

        loaded = rs.load_policy_state_dict()
        for key, tensor in loaded.items():
            assert tensor.device == torch.device("cpu"), (
                f"{key} not on CPU: {tensor.device}"
            )


# ── Multiple saves (overwrite) ───────────────────────────────────


class TestPolicyOverwrite:
    """Verify that re-saving overwrites the previous policy."""

    def test_second_save_overwrites_first(self, tmp_path):
        rs = RunStorage(tmp_path / "run", "test_run")

        # Save first policy
        policy1 = _SimplePolicy()
        with torch.no_grad():
            for p in policy1.parameters():
                p.fill_(1.0)
        rs.save_policy(policy1)

        # Save different policy
        policy2 = _SimplePolicy()
        with torch.no_grad():
            for p in policy2.parameters():
                p.fill_(99.0)
        rs.save_policy(policy2)

        # Load should get policy2
        loaded = rs.load_policy_state_dict()
        for key in loaded:
            assert torch.allclose(
                loaded[key], policy2.state_dict()[key],
            ), "Should load the latest saved policy"


# ── BenchMARL integration (skip if not installed) ────────────────


def _has_benchmarl():
    try:
        import benchmarl  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_benchmarl(), reason="BenchMARL not installed",
)
class TestBenchMARLPolicyExport:
    """Test export/import with a real BenchMARL experiment policy."""

    def test_benchmarl_policy_round_trip(self, tmp_path):
        """Build a BenchMARL experiment, save its policy, reload, verify."""
        from src.config import TaskConfig, TrainConfig
        from src.runner import build_experiment

        rs = RunStorage(tmp_path / "run", "test_benchmarl_policy")

        experiment = build_experiment(
            task_config=TaskConfig(n_agents=3, n_targets=2, max_steps=10),
            train_config=TrainConfig(
                max_n_frames=60_000,
                on_policy_collected_frames_per_batch=6000,
                on_policy_n_envs_per_worker=10,
            ),
            algorithm="mappo",
            seed=0,
            save_folder=str(rs.benchmarl_dir),
        )

        policy = experiment.policy
        # BenchMARL state_dict may contain non-tensor values (torch.Size);
        # only clone actual tensors for comparison
        original_sd = {}
        for k, v in policy.state_dict().items():
            original_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v

        # Save
        rs.save_policy(policy)
        assert rs.has_policy()

        # Load
        loaded_sd = rs.load_policy_state_dict()
        assert loaded_sd is not None

        # Keys match
        assert set(loaded_sd.keys()) == set(original_sd.keys())

        # Tensor values match
        for key in original_sd:
            orig = original_sd[key]
            loaded = loaded_sd[key]
            if isinstance(orig, torch.Tensor):
                assert torch.equal(orig, loaded), (
                    f"BenchMARL policy mismatch in {key}"
                )
            else:
                assert orig == loaded, (
                    f"Non-tensor mismatch in {key}: {orig} vs {loaded}"
                )

    def test_benchmarl_policy_reload_produces_same_actions(self, tmp_path):
        """Verify reloaded policy produces identical actions."""
        from src.config import TaskConfig, TrainConfig
        from src.runner import build_experiment

        task_config = TaskConfig(n_agents=3, n_targets=2, max_steps=10)
        train_config = TrainConfig(
            max_n_frames=60_000,
            on_policy_collected_frames_per_batch=6000,
            on_policy_n_envs_per_worker=10,
        )

        rs = RunStorage(tmp_path / "run", "test_benchmarl_actions")

        # Build, save policy
        exp1 = build_experiment(
            task_config, train_config, "mappo", seed=0,
            save_folder=str(rs.benchmarl_dir),
        )
        rs.save_policy(exp1.policy)

        # Get a test observation from the env
        test_env = exp1.test_env
        td = test_env.reset()

        # Get actions from original policy
        with torch.no_grad():
            td_orig = exp1.policy(td.clone())

        # Load state_dict into a rebuilt experiment
        # BenchMARL requires save_folder to pre-exist
        run2_benchmarl = tmp_path / "run2" / "benchmarl"
        run2_benchmarl.mkdir(parents=True)
        exp2 = build_experiment(
            task_config, train_config, "mappo", seed=0,
            save_folder=str(run2_benchmarl),
        )
        loaded_sd = rs.load_policy_state_dict()
        exp2.policy.load_state_dict(loaded_sd)

        # Get actions from restored policy
        with torch.no_grad():
            td_rest = exp2.policy(td.clone())

        # Compare action tensors across all groups
        for group in exp1.group_map.keys():
            action_key = (group, "action")
            if action_key in td_orig.keys(True):
                assert torch.equal(
                    td_orig[action_key], td_rest[action_key],
                ), f"Action mismatch for group {group}"
