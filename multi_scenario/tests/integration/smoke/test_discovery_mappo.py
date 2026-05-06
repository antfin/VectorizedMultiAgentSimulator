"""F2.9 — Phase 2 milestone: discovery + MAPPO smoke through ``LocalRunner``.

Loads the real ``experiments/discovery/baseline/configs/mappo_smoke.yaml`` to
prove the on-disk config drives the runner, then asserts the §3.5.2 layout is
fully populated with a DONE state, real provenance, and non-stub metric values
(M1 / M2 / M3) — confirming F2.3 (discovery DI primitives) + F2.4.3 (rollout
aggregation) are wired through end-to-end.
"""

import json
from pathlib import Path

import pytest

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.adapters.runners.local import LocalRunner
from multi_scenario.domain.models import ExperimentConfig, RunId, RunReport

# Sibling to this file's repo: the canonical smoke config under experiments/.
_SMOKE_YAML = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "discovery"
    / "baseline"
    / "configs"
    / "mappo_smoke.yaml"
)


@pytest.mark.slow
def test_discovery_mappo_smoke_completes(tmp_path: Path) -> None:
    """Phase 2 milestone: full §3.5.2 layout produced with real metrics + provenance."""
    # The test legitimately inspects many distinct §3.5.2 artefacts (state /
    # config / provenance / metrics / eval_episodes / report / benchmarl /
    # log); named path variables keep the assertions readable. Splitting this
    # into multiple tests would re-run the slow training each time.
    # pylint: disable=too-many-locals,too-many-statements
    cfg = ExperimentConfig.from_yaml(_SMOKE_YAML)
    # Redirect storage so the test never pollutes the real experiments folder.
    assert cfg.runtime is not None
    cfg.runtime.storage.path = str(tmp_path)

    run_id = RunId(exp_id=cfg.experiment.id, seed=cfg.experiment.seed)
    run_dir = tmp_path / f"{run_id}__20260506_0000"
    run_dir.mkdir(parents=True)

    logger = FileLogger(run_dir / "logs" / "run.log")
    runner = LocalRunner(logger=logger)  # default ProvenanceWriter()
    result = runner.run(cfg, run_dir=run_dir)

    # Identity sanity — the result reflects the loaded YAML.
    assert result.run_id == "mappo_smoke_s0"
    assert result.scenario == "discovery"
    assert result.algorithm == "mappo"

    # Lifecycle: run_state.json exists and ended in DONE.
    state_path = run_dir / "run_state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["state"] == "DONE"

    # Config round-trips from disk (proves the JSON snapshot is structurally valid).
    config_path = run_dir / "input" / "config.json"
    assert config_path.is_file()
    ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))

    # Provenance is real (not the F2.6-style stub): git_sha + library_versions populated.
    prov_path = run_dir / "input" / "provenance.json"
    assert prov_path.is_file()
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["git_sha"]  # non-empty string
    assert prov["library_versions"]["multi_scenario"]
    assert prov["library_versions"]["torch"]
    assert prov["library_versions"]["vmas"]
    assert prov["library_versions"]["benchmarl"]

    # Metrics: M1/M2/M3 are real floats (not None) — proves F2.3 + F2.4.3 wired through.
    # On disk, ExperimentResult.metrics is serialised as a flat {name: value} dict
    # (see _serialise_metrics_as_dict in domain/models/result.py).
    metrics_path = run_dir / "output" / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric_dict = metrics["metrics"]
    assert isinstance(metric_dict["M1_success_rate"], float)
    assert isinstance(metric_dict["M2_avg_return"], float)
    assert isinstance(metric_dict["M3_steps"], float)

    # BenchMARL native output landed under the run folder (not the parent).
    bm_dir = run_dir / "output" / "benchmarl"
    assert bm_dir.is_dir()
    assert any(bm_dir.iterdir()), "BenchMARL output dir is empty"

    # Logger wrote at least the train/evaluate/done banners.
    log_path = run_dir / "logs" / "run.log"
    assert log_path.is_file()
    assert log_path.stat().st_size > 0

    # F2.10.1: eval_episodes.json with the universal rollout fields, sized to
    # cfg.evaluation.episodes; discovery rollouts also carry targets_covered + n_targets.
    eval_path = run_dir / "output" / "eval_episodes.json"
    assert eval_path.is_file()
    eval_episodes = json.loads(eval_path.read_text(encoding="utf-8"))
    n_eps = cfg.evaluation.episodes
    assert len(eval_episodes["episode_returns"]) == n_eps
    assert len(eval_episodes["episode_lengths"]) == n_eps
    assert len(eval_episodes["episode_collisions"]) == n_eps
    assert eval_episodes["n_targets"] == cfg.scenario.params["n_targets"]
    assert len(eval_episodes["targets_covered"]) == n_eps

    # F2.10: report.json manifest with status=DONE; every non-None link resolves.
    report_path = run_dir / "output" / "report.json"
    assert report_path.is_file()
    report = RunReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.status == "DONE"
    assert report.duration_seconds >= 0
    for rel in (
        report.links.config,
        report.links.provenance,
        report.links.log,
        report.links.metrics,
        report.links.eval_episodes,  # populated now (F2.10.1)
        report.links.benchmarl_dir,
        report.links.benchmarl_scalars,
    ):
        assert rel is not None
        assert (run_dir / rel).exists()
