"""MASAC adapter — wraps BenchMARL's MasacConfig (off-policy)."""

from benchmarl.algorithms import MasacConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class MasacAdapter(BenchmarlBaseAdapter):
    """Multi-Agent SAC via BenchMARL — centralised critics, max-entropy."""

    # pylint: disable=too-few-public-methods

    name = "masac"
    _config_class = MasacConfig
