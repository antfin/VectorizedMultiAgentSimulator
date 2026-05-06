"""MASAC adapter — wraps BenchMARL's MasacConfig (off-policy)."""

from typing import Any

from benchmarl.algorithms import MasacConfig

from multi_scenario.adapters.algorithms.benchmarl_base import BenchmarlBaseAdapter
from multi_scenario.domain.models import ExperimentConfig


class MasacAdapter(BenchmarlBaseAdapter):
    """Multi-Agent SAC via BenchMARL — centralised critics, max-entropy."""

    # pylint: disable=too-few-public-methods

    name = "masac"

    def _algorithm_config(self, cfg: ExperimentConfig) -> Any:
        bm = MasacConfig.get_from_yaml()
        # cfg.algorithm.params overrides on the MasacConfig dataclass; only
        # set fields that exist there to fail loudly on typos.
        for key, value in cfg.algorithm.params.items():
            if not hasattr(bm, key):
                raise ValueError(f"unknown MasacConfig field: {key}")
            setattr(bm, key, value)
        return bm
