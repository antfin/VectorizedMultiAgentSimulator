"""MADDPG adapter — wraps BenchMARL's MaddpgConfig (off-policy)."""

from benchmarl.algorithms import MaddpgConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class MaddpgAdapter(BenchmarlBaseAdapter):
    """Multi-Agent DDPG via BenchMARL — centralised critics, off-policy."""

    # pylint: disable=too-few-public-methods

    name = "maddpg"
    _config_class = MaddpgConfig
