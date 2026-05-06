"""MAPPO adapter — wraps BenchMARL's MappoConfig."""

from benchmarl.algorithms import MappoConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class MappoAdapter(BenchmarlBaseAdapter):
    """Multi-Agent PPO via BenchMARL — centralised critic, on-policy."""

    # pylint: disable=too-few-public-methods

    name = "mappo"
    _config_class = MappoConfig
