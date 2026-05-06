"""IPPO adapter — wraps BenchMARL's IppoConfig."""

from benchmarl.algorithms import IppoConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class IppoAdapter(BenchmarlBaseAdapter):
    """Independent PPO via BenchMARL — per-agent independent critics."""

    # pylint: disable=too-few-public-methods

    name = "ippo"
    _config_class = IppoConfig
