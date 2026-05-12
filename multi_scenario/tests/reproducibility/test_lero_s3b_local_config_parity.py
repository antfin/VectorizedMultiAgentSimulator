"""F8.4 Phase 4 — parity test for the ported S3b-local YAML.

The ported YAML at ``experiments/discovery/lero/configs/lero_s3b_local.yaml``
must encode the same numerical experiment as rendezvous_comm's
``configs/lero/s3b_local.yaml``. Drift in any of the scenario / training
/ LERO / LLM knobs would explain a reproducibility miss; pinning the
parity here catches the drift at CI time instead of post-€6 OVH run.

What this test checks (load-bearing):

- ``scenario.params`` matches ``task.*`` field-for-field.
- ``training.max_iters * training.frames_per_batch == train.max_n_frames``.
- ``training.frames_per_batch == train.on_policy_collected_frames_per_batch``.
- ``evaluation.interval_iters * training.frames_per_batch == train.evaluation_interval``.
- ``evaluation.episodes == train.evaluation_episodes``.
- ``algorithm.type == train.algorithm``; ``algorithm.params.lmbda == train.lmbda``.
- ``lero.eval_frames_per_candidate == lero.eval_frames`` and friends.
- ``lero.prompt_version == llm.prompt_version`` (we store it under lero).
- ``llm.model``, ``llm.temperature`` match.

What this test does NOT check (out of scope):

- The rendezvous_comm YAML's ``sweep:`` section — that's a separate
  surface that multi_scenario handles via the CLI's ``--seeds`` flag
  rather than embedding in the YAML.
- ``llm.max_retries`` — multi_scenario doesn't expose it (LiteLLM has
  its own retry policy). If the user wants tighter control they can
  set ``LITELLM_RETRIES``.
- ``lero.reward_mode``, ``lero.obs_state_mode``, ``lero.bonus_scale``,
  ``lero.top_k`` — multi_scenario derives these from ``prompt_version``
  + ``evolve_*`` flags (the F9.6.e wiring) so they're not stored
  as separate cfg fields. The parity test asserts the prompt_version
  matches, which implicitly pins obs_state_mode.
"""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest
import yaml

from multi_scenario.domain.models import ExperimentConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUR_YAML = (
    _REPO_ROOT
    / "multi_scenario"
    / "experiments"
    / "discovery"
    / "lero"
    / "configs"
    / "lero_s3b_local.yaml"
)
_THEIR_YAML = _REPO_ROOT / "rendezvous_comm" / "configs" / "lero" / "s3b_local.yaml"


@pytest.fixture(scope="module")
def our_cfg() -> ExperimentConfig:
    return ExperimentConfig.from_yaml(_OUR_YAML)


@pytest.fixture(scope="module")
def their_cfg() -> dict:
    if not _THEIR_YAML.is_file():
        pytest.skip(f"rendezvous_comm source missing: {_THEIR_YAML}")
    return yaml.safe_load(_THEIR_YAML.read_text(encoding="utf-8"))


# ── experiment.id / name ─────────────────────────────────────────────


def test_experiment_id_matches(our_cfg, their_cfg):
    assert our_cfg.experiment.id == their_cfg["exp_id"]


# ── scenario.params == task.* ────────────────────────────────────────


@pytest.mark.parametrize(
    "key",
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
def test_scenario_params_match_rendezvous_task(key, our_cfg, their_cfg):
    """Each task.* field maps byte-equal to scenario.params.*."""
    assert key in their_cfg["task"], f"rendezvous_comm task missing {key!r}"
    ours = our_cfg.scenario.params.get(key)
    theirs = their_cfg["task"][key]
    assert (
        ours == theirs
    ), f"scenario.params.{key} = {ours!r} but rendezvous_comm task.{key} = {theirs!r}"


# ── training/evaluation derived from train.* ─────────────────────────


def test_total_frames_match(our_cfg, their_cfg):
    """training.max_iters covers train.max_n_frames with at most one extra batch.

    ER1's baseline.yaml established the "smallest integer iters ≥ target
    frames" convention: BenchMARL bounds by iters, not frames, so the
    closest exact match is ``ceil(target_frames / frames_per_batch)``.
    For S3b-local: ceil(10M / 60k) = 167 iters → 10.02M frames, 0.2%
    above target. Anything more than one batch over is a drift.
    """
    fpb = our_cfg.training.frames_per_batch
    our_total = our_cfg.training.max_iters * fpb
    their_total = their_cfg["train"]["max_n_frames"]
    # Same frames_per_batch (pinned by another test); the iter count
    # must round-up to cover the rendezvous_comm target without going
    # more than one batch over.
    assert our_total >= their_total, (
        f"frames undershoot: ours={our_total}, theirs={their_total}. "
        "max_iters too small to reach rendezvous_comm's training budget."
    )
    assert our_total - their_total < fpb, (
        f"frames overshoot: ours={our_total}, theirs={their_total}, "
        f"delta={our_total - their_total} > frames_per_batch={fpb}. "
        "max_iters drifted high."
    )


def test_frames_per_batch_matches(our_cfg, their_cfg):
    assert (
        our_cfg.training.frames_per_batch
        == their_cfg["train"]["on_policy_collected_frames_per_batch"]
    )


def test_num_envs_matches(our_cfg, their_cfg):
    assert (
        our_cfg.training.num_envs == their_cfg["train"]["on_policy_n_envs_per_worker"]
    )


def test_n_minibatch_iters_matches(our_cfg, their_cfg):
    assert (
        our_cfg.training.n_minibatch_iters
        == their_cfg["train"]["on_policy_n_minibatch_iters"]
    )


def test_minibatch_size_matches(our_cfg, their_cfg):
    assert (
        our_cfg.training.minibatch_size
        == their_cfg["train"]["on_policy_minibatch_size"]
    )


def test_lr_matches(our_cfg, their_cfg):
    assert our_cfg.training.lr == their_cfg["train"]["lr"]


def test_gamma_matches(our_cfg, their_cfg):
    assert our_cfg.training.gamma == their_cfg["train"]["gamma"]


def test_share_policy_params_matches(our_cfg, their_cfg):
    assert (
        our_cfg.training.share_policy_params
        == their_cfg["train"]["share_policy_params"]
    )


def test_evaluation_interval_matches(our_cfg, their_cfg):
    """evaluation.interval_iters × frames_per_batch should equal their evaluation_interval."""
    our_interval_frames = (
        our_cfg.evaluation.interval_iters * our_cfg.training.frames_per_batch
    )
    assert our_interval_frames == their_cfg["train"]["evaluation_interval"]


def test_evaluation_episodes_matches(our_cfg, their_cfg):
    assert our_cfg.evaluation.episodes == their_cfg["train"]["evaluation_episodes"]


# ── algorithm.* ──────────────────────────────────────────────────────


def test_algorithm_type_matches(our_cfg, their_cfg):
    assert our_cfg.algorithm.type == their_cfg["train"]["algorithm"]


def test_algorithm_lmbda_matches(our_cfg, their_cfg):
    """PPO-specific GAE λ lives under algorithm.params for us, train.lmbda for them."""
    assert our_cfg.algorithm.params.get("lmbda") == their_cfg["train"]["lmbda"]


# ── lero.* ───────────────────────────────────────────────────────────


def test_lero_n_iterations_matches(our_cfg, their_cfg):
    assert our_cfg.lero is not None
    assert our_cfg.lero.n_iterations == their_cfg["lero"]["n_iterations"]


def test_lero_n_candidates_matches(our_cfg, their_cfg):
    assert our_cfg.lero is not None
    assert our_cfg.lero.n_candidates == their_cfg["lero"]["n_candidates"]


def test_lero_eval_frames_per_candidate_matches(our_cfg, their_cfg):
    """Our field is named ``eval_frames_per_candidate``; theirs is ``eval_frames``."""
    assert our_cfg.lero is not None
    assert our_cfg.lero.eval_frames_per_candidate == their_cfg["lero"]["eval_frames"]


def test_lero_evolve_flags_match(our_cfg, their_cfg):
    assert our_cfg.lero is not None
    assert our_cfg.lero.evolve_reward == their_cfg["lero"]["evolve_reward"]
    assert our_cfg.lero.evolve_observation == their_cfg["lero"]["evolve_observation"]


def test_lero_reward_clip_matches(our_cfg, their_cfg):
    assert our_cfg.lero is not None
    assert our_cfg.lero.reward_clip == their_cfg["lero"]["reward_clip"]


def test_lero_prompt_version_implies_local_obs_mode(our_cfg, their_cfg):
    """``obs_state_mode=local`` in theirs → our prompt_version ends with ``_local``.

    The F9.6.e patched-experiment factory uses the prompt-version suffix
    to pick obs_state_mode at runtime (no separate cfg field), so this
    is the structural way to pin the mapping.
    """
    assert our_cfg.lero is not None
    assert their_cfg["lero"]["obs_state_mode"] == "local"
    assert our_cfg.lero.prompt_version.endswith("_local"), (
        f"prompt_version {our_cfg.lero.prompt_version!r} doesn't end in "
        "'_local' but rendezvous_comm declared obs_state_mode=local"
    )


def test_lero_whitelist_strict_is_on_for_local_mode(our_cfg):
    """CTDE fairness check: we never want to ship an S3b-local run with
    whitelist_strict=False — that would silently let the LLM cheat via
    global state lookups, invalidating the comparison to ER1/ER2/ER3."""
    assert our_cfg.lero is not None
    assert our_cfg.lero.whitelist_strict is True


# ── llm.* ────────────────────────────────────────────────────────────


def test_llm_model_matches(our_cfg, their_cfg):
    assert our_cfg.llm is not None
    assert our_cfg.llm.model == their_cfg["llm"]["model"]


def test_llm_temperature_matches(our_cfg, their_cfg):
    assert our_cfg.llm is not None
    assert our_cfg.llm.temperature == their_cfg["llm"]["temperature"]


def test_llm_prompt_version_matches(our_cfg, their_cfg):
    """Theirs stores ``prompt_version`` under llm; we store under lero
    (the orchestrator is the consumer, not the LLM client). Same value."""
    assert our_cfg.lero is not None
    assert our_cfg.lero.prompt_version == their_cfg["llm"]["prompt_version"]
