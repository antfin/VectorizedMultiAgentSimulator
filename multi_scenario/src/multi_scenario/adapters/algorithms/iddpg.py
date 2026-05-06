"""IDDPG adapter — wraps BenchMARL's IddpgConfig (off-policy)."""

from benchmarl.algorithms import IddpgConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class IddpgAdapter(BenchmarlBaseAdapter):
    """Independent DDPG via BenchMARL — per-agent independent critics."""

    # pylint: disable=too-few-public-methods

    name = "iddpg"
    _config_class = IddpgConfig
