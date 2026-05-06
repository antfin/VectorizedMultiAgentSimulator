"""MAPPO adapter — wraps BenchMARL's MappoConfig."""

from typing import Any

from benchmarl.algorithms import MappoConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.domain.models import ExperimentConfig


class MappoAdapter(BenchmarlBaseAdapter):
    """Multi-Agent PPO via BenchMARL."""

    # pylint: disable=too-few-public-methods

    name = "mappo"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        bm = MappoConfig.get_from_yaml()
        # cfg.algorithm.params overrides on the MappoConfig dataclass; only set
        # fields that exist there to fail loudly on typos.
        for key, value in cfg.algorithm.params.items():
            if not hasattr(bm, key):
                raise ValueError(f"unknown MappoConfig field: {key}")
            setattr(bm, key, value)
        return bm
