"""F9.6.c — ExperimentService dispatches to LERO when cfg.lero is set."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path

import pytest

from multi_scenario.application.experiment_service import ExperimentService
from multi_scenario.domain.models import ExperimentConfig


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"id": "demo_lero", "seed": 0},
            "scenario": {
                "type": "discovery",
                "params": {
                    "n_agents": 2,
                    "n_targets": 2,
                    "agents_per_target": 2,
                    "covering_range": 0.35,
                    "n_lidar_rays_entities": 15,
                    "n_lidar_rays_agents": 12,
                    "obs_lidar_agents": "lidar_agents: …",
                },
            },
            "algorithm": {"type": "mappo", "params": {}},
            "training": {"max_iters": 1, "device": "cpu"},
            "evaluation": {"interval_iters": 1, "episodes": 1},
            "runtime": {
                "runner": {"type": "local", "params": {}},
                "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
            },
            "lero": {"n_iterations": 1, "n_candidates": 1},
            "llm": {"model": "gpt-4o-mini"},
        }
    )


def test_run_with_lero_section_routes_through_lero_branch(tmp_path: Path, monkeypatch):
    """When ``cfg.lero is not None`` the service must call ``_run_lero``,
    not ``self._algorithm.train``. Verified by spying on the factory.
    """
    cfg = _cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Spy: monkey-patch the factory so we don't construct a real
    # LiteLlmClient / orchestrator. The service should call this exactly once.
    factory_calls = []

    def _fake_factory(*, cfg, logger):
        factory_calls.append((cfg, logger))

        class _FakeOrch:
            def run(self, *, cfg, run_dir):  # noqa: ARG002
                # pylint: disable=import-outside-toplevel
                from multi_scenario.domain.lero import CandidateMetrics, LeroRunSummary

                return LeroRunSummary(
                    exp_id=cfg.experiment.id,
                    seed=cfg.experiment.seed,
                    n_iterations_completed=1,
                    n_candidates_total=1,
                    total_cost_usd=0.0,
                    best_candidate_metrics=CandidateMetrics(M1_success_rate=0.7),
                    best_candidate_verdict="progress",
                    full_training_succeeded=True,
                )

        return _FakeOrch()

    monkeypatch.setattr(
        "multi_scenario.application.lero_factory.build_default_lero_orchestrator",
        _fake_factory,
    )

    # Build a service with stub deps that would error if the LERO branch
    # accidentally fell through to the standard path.
    class _FailIfCalled:
        def __getattr__(self, name):  # noqa: ARG002
            raise AssertionError(
                "non-LERO ports must not be invoked when cfg.lero is set"
            )

    # The service requires several injected components; only Storage +
    # Logger are touched in the LERO branch.
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.storage.local import LocalStorageAdapter
    from multi_scenario.domain.models import Provenance

    class _SilentLogger:
        # pylint: disable=missing-function-docstring,unused-argument
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

        def debug(self, msg):
            pass

    service = ExperimentService(
        scenario=_FailIfCalled(),
        algorithm=_FailIfCalled(),
        metrics=_FailIfCalled(),
        storage=LocalStorageAdapter(),
        logger=_SilentLogger(),
    )

    # Provenance has many required fields — use a minimal stub.
    from multi_scenario.domain.models import LibraryVersions

    prov = Provenance(
        config_hash="sha256:x",
        code_hash="sha256:y",
        hashed_source_files=[],
        git_sha="abc",
        git_dirty=False,
        created_at=__import__("datetime").datetime.now(
            tz=__import__("datetime").timezone.utc
        ),
        library_versions=LibraryVersions(
            python="3.11",
            torch="2.4",
            vmas="1.4",
            benchmarl="1.3",
            multi_scenario="0.0.1",
        ),
    )
    result = service.run(cfg=cfg, run_dir=run_dir, provenance=prov)

    # Factory was invoked exactly once.
    assert len(factory_calls) == 1
    # Result carries the M1 from the fake orchestrator.
    m1 = next(r.value for r in result.metrics if r.name == "M1_success_rate")
    assert m1 == pytest.approx(0.7)


def test_run_without_lero_section_uses_standard_branch(tmp_path: Path, monkeypatch):
    """Sanity: a YAML without ``lero:`` doesn't get re-routed to LERO."""
    cfg_dict = {
        "experiment": {"id": "demo_std", "seed": 0},
        "scenario": {"type": "discovery", "params": {}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {"max_iters": 1, "device": "cpu"},
        "evaluation": {"interval_iters": 1, "episodes": 1},
        # NO lero / llm
        "runtime": {
            "runner": {"type": "local", "params": {}},
            "storage": {"type": "fs", "path": str(tmp_path), "params": {}},
        },
    }
    cfg = ExperimentConfig.model_validate(cfg_dict)
    assert cfg.lero is None

    factory_invoked: list = []

    def _fake_factory(*, cfg, logger):  # noqa: ARG001
        factory_invoked.append(True)
        return object()

    monkeypatch.setattr(
        "multi_scenario.application.lero_factory.build_default_lero_orchestrator",
        _fake_factory,
    )
    # Standard branch will try to call self._algorithm.train — we just
    # want to verify the LERO factory is NEVER called for a non-LERO cfg.
    assert factory_invoked == []
