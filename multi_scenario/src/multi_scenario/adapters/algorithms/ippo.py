"""IPPO adapter — wraps BenchMARL's IppoConfig."""

from typing import Any

from benchmarl.algorithms import IppoConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.domain.models import ExperimentConfig


class IppoAdapter(BenchmarlBaseAdapter):
    """Independent PPO via BenchMARL — per-agent independent critics."""

    # pylint: disable=too-few-public-methods

    name = "ippo"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        bm = IppoConfig.get_from_yaml()
        # cfg.algorithm.params overrides on the IppoConfig dataclass; only set
        # fields that exist there to fail loudly on typos.
        for key, value in cfg.algorithm.params.items():
            if not hasattr(bm, key):
                raise ValueError(f"unknown IppoConfig field: {key}")
            setattr(bm, key, value)
        return bm
