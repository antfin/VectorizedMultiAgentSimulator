"""IDDPG adapter — wraps BenchMARL's IddpgConfig (off-policy)."""

from typing import Any

from benchmarl.algorithms import IddpgConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.domain.models import ExperimentConfig


class IddpgAdapter(BenchmarlBaseAdapter):
    """Independent DDPG via BenchMARL — per-agent independent critics."""

    # pylint: disable=too-few-public-methods

    name = "iddpg"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        bm = IddpgConfig.get_from_yaml()
        # cfg.algorithm.params overrides on the IddpgConfig dataclass; only
        # set fields that exist there to fail loudly on typos.
        for key, value in cfg.algorithm.params.items():
            if not hasattr(bm, key):
                raise ValueError(f"unknown IddpgConfig field: {key}")
            setattr(bm, key, value)
        return bm
