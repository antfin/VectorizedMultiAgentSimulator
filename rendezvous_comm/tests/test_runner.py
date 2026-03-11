"""Unit tests for runner helper functions."""
import pytest

from src.runner import _in_notebook, _fmt_elapsed, make_heuristic_policy_fn


def _has_benchmarl():
    try:
        import benchmarl  # noqa: F401
        return True
    except ImportError:
        return False


class TestInNotebook:
    """Test _in_notebook() detection."""

    def test_returns_false_in_test_env(self):
        assert _in_notebook() is False

    def test_returns_bool(self):
        assert isinstance(_in_notebook(), bool)


class TestFmtElapsed:
    """Test _fmt_elapsed() time formatting."""

    def test_seconds_only(self):
        assert _fmt_elapsed(45) == "45s"

    def test_minutes_and_seconds(self):
        assert _fmt_elapsed(125) == "2m05s"

    def test_hours_and_minutes(self):
        assert _fmt_elapsed(3725) == "1h02m"

    def test_zero(self):
        assert _fmt_elapsed(0) == "0s"

    def test_large_value(self):
        assert _fmt_elapsed(7200) == "2h00m"

    def test_exactly_one_minute(self):
        assert _fmt_elapsed(60) == "1m00s"

    def test_exactly_one_hour(self):
        assert _fmt_elapsed(3600) == "1h00m"

    def test_float_input_truncates(self):
        assert _fmt_elapsed(45.9) == "45s"


class TestMakeHeuristicPolicyFn:
    """Test make_heuristic_policy_fn() returns a usable callable."""

    def test_returns_callable(self):
        fn = make_heuristic_policy_fn()
        assert callable(fn)

    def test_callable_accepts_obs_and_env(self):
        """Verify the returned function works with a real VMAS env."""
        from vmas import make_env

        env = make_env(
            scenario="discovery",
            num_envs=2,
            device="cpu",
            continuous_actions=True,
            n_agents=3,
            n_targets=3,
            agents_per_target=1,
        )
        obs = env.reset()
        fn = make_heuristic_policy_fn()
        actions = fn(obs, env)

        assert isinstance(actions, list)
        assert len(actions) == 3  # one per agent


class TestHeuristicOutperformsRandom:
    """Heuristic policy should generally outperform random on coverage."""

    def test_heuristic_coverage_at_least_half_of_random(self):
        from src.config import TaskConfig
        from src.runner import evaluate_with_vmas

        tc = TaskConfig(
            n_agents=3, n_targets=3, agents_per_target=1,
            max_steps=20,
        )
        heur = evaluate_with_vmas(
            tc, policy_fn=make_heuristic_policy_fn(),
            n_eval_episodes=4, n_envs=4,
        )
        rand = evaluate_with_vmas(
            tc, policy_fn=None,
            n_eval_episodes=4, n_envs=4,
        )
        assert heur["M6_coverage_progress"] >= (
            rand["M6_coverage_progress"] * 0.5
        )


@pytest.mark.skipif(
    not _has_benchmarl(), reason="BenchMARL not installed"
)
class TestGetAlgorithmConfig:
    """Test get_algorithm_config with BenchMARL present."""

    def test_mappo(self):
        from src.runner import get_algorithm_config
        cfg = get_algorithm_config("mappo")
        assert cfg is not None

    def test_ippo(self):
        from src.runner import get_algorithm_config
        cfg = get_algorithm_config("ippo")
        assert cfg is not None

    def test_qmix(self):
        from src.runner import get_algorithm_config
        cfg = get_algorithm_config("qmix")
        assert cfg is not None

    def test_maddpg(self):
        from src.runner import get_algorithm_config
        cfg = get_algorithm_config("maddpg")
        assert cfg is not None

    def test_unknown_algorithm_raises(self):
        from src.runner import get_algorithm_config
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm_config("nonexistent_algo")
