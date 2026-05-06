"""ISAC adapter — wraps BenchMARL's IsacConfig (off-policy)."""

from benchmarl.algorithms import IsacConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter


class IsacAdapter(BenchmarlBaseAdapter):
    """Independent SAC via BenchMARL — per-agent independent critics, max-entropy."""

    # pylint: disable=too-few-public-methods

    name = "isac"
    _config_class = IsacConfig
