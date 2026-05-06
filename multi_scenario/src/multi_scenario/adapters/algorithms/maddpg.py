"""MADDPG adapter — wraps BenchMARL's MaddpgConfig (off-policy)."""

from typing import Any

from benchmarl.algorithms import MaddpgConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.domain.models import ExperimentConfig


class MaddpgAdapter(BenchmarlBaseAdapter):
    """Multi-Agent DDPG via BenchMARL — first off-policy algorithm in the framework."""

    # pylint: disable=too-few-public-methods

    name = "maddpg"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        bm = MaddpgConfig.get_from_yaml()
        # cfg.algorithm.params overrides on the MaddpgConfig dataclass; only
        # set fields that exist there to fail loudly on typos.
        for key, value in cfg.algorithm.params.items():
            if not hasattr(bm, key):
                raise ValueError(f"unknown MaddpgConfig field: {key}")
            setattr(bm, key, value)
        return bm
