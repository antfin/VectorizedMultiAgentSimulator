"""F8.1 — ER1 config parity vs rendezvous_comm reference.

The coopvmas baseline at ``experiments/discovery/baseline/configs/baseline.yaml``
must encode exactly the same hyperparameters as
``rendezvous_comm/configs/er1/single_al_lp_sr_cr035.yaml`` — silent drift
breaks reproducibility before any run completes.

These tests parse both YAMLs and compare per-field. When a coopvmas YAML
schema doesn't have a 1:1 slot for a rendezvous_comm field (e.g.
``max_n_frames`` becomes ``max_iters * frames_per_batch``), the test
asserts the *derived* coopvmas value matches the reference within the
documented rounding rule.

The reference YAML is read at test-time from the rendezvous_comm sibling
folder; if it moves, this test breaks loudly (intentional — it's a
reproducibility canary).
"""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest
import yaml

from multi_scenario.application.config_validation import validate_known_types
from multi_scenario.domain.models import ExperimentConfig

_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "discovery" / "baseline" / "configs" / "baseline.yaml"
)

# Located via path-relative climb: tests → multi_scenario → VMAS root → rendezvous_comm.
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "rendezvous_comm" / "configs" / "er1" / "single_al_lp_sr_cr035.yaml"
)


def _load_reference() -> dict:
    """Load the rendezvous_comm reference YAML; skip the test module if absent."""
    if not _REFERENCE_PATH.is_file():
        pytest.skip(
            f"rendezvous_comm reference not at {_REFERENCE_PATH} — "
            "F8.1 parity tests can only run alongside the rendezvous_comm sibling repo."
        )
    return yaml.safe_load(_REFERENCE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference() -> dict:
    return _load_reference()


@pytest.fixture(scope="module")
def baseline() -> ExperimentConfig:
    """The coopvmas baseline config, schema- and registry-validated."""
    cfg = ExperimentConfig.from_yaml(_BASELINE_PATH)
    validate_known_types(cfg)
    return cfg


def test_baseline_yaml_parses_and_resolves_known_types(baseline):
    """Sanity gate — the YAML loads via the strict schema + registry."""
    assert baseline.experiment.id == "er1_cr035"
    assert baseline.scenario.type == "discovery"
    assert baseline.algorithm.type == "mappo"


# ── Scenario / task params ───────────────────────────────────────────


@pytest.mark.parametrize(
    "field",
    [
        "n_agents",
        "n_targets",
        "agents_per_target",
        "lidar_range",
        "covering_range",
        "use_agent_lidar",
        "n_lidar_rays_entities",
        "n_lidar_rays_agents",
        "targets_respawn",
        "shared_reward",
        "agent_collision_penalty",
        "covering_rew_coeff",
        "time_penalty",
        "max_steps",
    ],
)
def test_scenario_param_matches_reference(baseline, reference, field):
    """Every rendezvous_comm `task.<field>` reproduces in `scenario.params.<field>`."""
    assert baseline.scenario.params[field] == reference["task"][field], (
        f"scenario.params.{field} drifted: "
        f"coopvmas={baseline.scenario.params[field]!r} "
        f"vs rendezvous_comm task.{field}={reference['task'][field]!r}"
    )


# ── Algorithm + training ─────────────────────────────────────────────


def test_algorithm_type_matches(baseline, reference):
    assert baseline.algorithm.type == reference["train"]["algorithm"]


def test_lmbda_carried_in_algorithm_params(baseline, reference):
    """``lmbda`` is PPO-specific — coopvmas puts it in algorithm.params."""
    assert baseline.algorithm.params["lmbda"] == reference["train"]["lmbda"]


@pytest.mark.parametrize(
    "coopvmas_path,reference_train_field",
    [
        ("training.lr", "lr"),
        ("training.gamma", "gamma"),
        ("training.frames_per_batch", "on_policy_collected_frames_per_batch"),
        ("training.minibatch_size", "on_policy_minibatch_size"),
        ("training.n_minibatch_iters", "on_policy_n_minibatch_iters"),
        ("training.num_envs", "on_policy_n_envs_per_worker"),
        ("training.share_policy_params", "share_policy_params"),
        ("training.device", "train_device"),
        ("evaluation.episodes", "evaluation_episodes"),
    ],
)
def test_training_field_matches_reference(baseline, reference, coopvmas_path, reference_train_field):
    """Per-field 1:1 mappings between coopvmas TrainingSection and rendezvous_comm `train.*`."""
    cur = baseline
    for part in coopvmas_path.split("."):
        cur = getattr(cur, part)
    assert cur == reference["train"][reference_train_field], (
        f"{coopvmas_path} drifted: coopvmas={cur!r} vs "
        f"rendezvous_comm train.{reference_train_field}={reference['train'][reference_train_field]!r}"
    )


def test_total_frames_matches_reference_within_one_batch(baseline, reference):
    """coopvmas uses ``max_iters``; rendezvous_comm uses ``max_n_frames``.

    Convert: ``max_iters * frames_per_batch`` should fall within one batch
    of ``max_n_frames`` (closest integer iter count covering the target).
    """
    coopvmas_total = baseline.training.max_iters * baseline.training.frames_per_batch
    reference_total = reference["train"]["max_n_frames"]
    delta = coopvmas_total - reference_total
    # We always round UP (cover at least max_n_frames), so delta ≥ 0 and
    # < frames_per_batch.
    assert 0 <= delta < baseline.training.frames_per_batch, (
        f"total frames drifted by {delta}: "
        f"coopvmas={coopvmas_total} vs rendezvous_comm={reference_total}, "
        f"frames_per_batch={baseline.training.frames_per_batch}"
    )


def test_evaluation_interval_iters_matches_reference_frames(baseline, reference):
    """coopvmas's ``evaluation.interval_iters`` × ``frames_per_batch``
    must equal rendezvous_comm's ``train.evaluation_interval`` (frames).
    """
    coopvmas_frames = baseline.evaluation.interval_iters * baseline.training.frames_per_batch
    reference_frames = reference["train"]["evaluation_interval"]
    assert coopvmas_frames == reference_frames, (
        f"eval cadence drifted: coopvmas={coopvmas_frames} frames vs "
        f"rendezvous_comm={reference_frames} frames"
    )


# ── Identity / metadata ──────────────────────────────────────────────


def test_seed_matches_reference_first_seed(baseline, reference):
    """Default baseline.yaml is seed 0 (the headline run); F8.2 produces _s1/_s2 variants."""
    assert baseline.experiment.seed == reference["sweep"]["seeds"][0]


def test_baseline_runtime_storage_path_under_baseline_folder(baseline):
    """``runtime.storage.path`` lands run dirs under ``experiments/discovery/baseline/``
    so F8.5.B's reproducibility Streamlit page can find them by convention.
    """
    assert baseline.runtime is not None
    assert baseline.runtime.storage.path == "experiments/discovery/baseline"
