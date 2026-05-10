"""F7.7.D1 — parametrised matrix of valid + invalid ExperimentConfig variants.

Field-level (gt/ge/le, Literal, regex) + cross-field (minibatch_size ≤
frames_per_batch) checks live on :class:`ExperimentConfig` itself.
Registry-aware checks (``scenario.type ∈ available_scenarios()`` etc.)
live in :mod:`multi_scenario.application.config_validation` so the domain
model stays free of an inverted dependency on the application layer
(F7.7.A2 hex-compliance lesson). Tests cover both layers.
"""

# pylint: disable=missing-function-docstring

import copy
from typing import Any

import pytest

from multi_scenario.domain.models import ExperimentConfig
from pydantic import ValidationError


def _kitchen_sink_valid() -> dict[str, Any]:
    """One known-good config the parametrised tests mutate via dotted-path."""
    return {
        "experiment": {"id": "demo", "seed": 0, "name": "n", "description": "d"},
        "scenario": {"type": "discovery", "params": {"n_agents": 2, "n_targets": 2}},
        "algorithm": {"type": "mappo", "params": {}},
        "training": {
            "max_iters": 10,
            "num_envs": 1,
            "device": "cpu",
            "lr": 3e-4,
            "gamma": 0.99,
            "frames_per_batch": 100,
            "minibatch_size": 50,
            "n_minibatch_iters": 1,
        },
        "evaluation": {"interval_iters": 1, "episodes": 1},
        "runtime": {
            "runner": {"type": "local", "params": {}},
            "storage": {"type": "fs", "path": "experiments/x", "params": {}},
        },
    }


def _set_dotted(d: dict, path: str, value: Any) -> dict:
    """Mutate ``d`` in place, setting the dotted ``path`` to ``value``."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    cur[keys[-1]] = value
    return d


def test_kitchen_sink_valid_config_parses():
    """Sanity: the canonical valid config doesn't trip any validator."""
    ExperimentConfig.model_validate(_kitchen_sink_valid())


@pytest.mark.parametrize(
    "path,value,msg_substring",
    [
        # ── numeric range checks ─────────────────────────────────
        ("training.max_iters", 0, "greater than 0"),
        ("training.num_envs", -1, "greater than 0"),
        ("training.frames_per_batch", 0, "greater than 0"),
        ("training.lr", 0.0, "greater than 0"),
        ("training.gamma", 1.5, "less than or equal to 1"),
        ("training.gamma", -0.1, "greater than or equal to 0"),
        ("training.checkpoint_interval_iters", -1, "greater than or equal to 0"),
        ("evaluation.episodes", 0, "greater than 0"),
        ("evaluation.interval_iters", 0, "greater than 0"),
        ("experiment.seed", -1, "greater than or equal to 0"),
        # ── Literal / pattern / min_length ───────────────────────
        ("training.device", "tpu", "Input should be 'cpu' or 'cuda'"),
        ("experiment.id", "", "at least 1 character"),
        ("experiment.id", "has spaces", "match pattern"),
        ("experiment.id", "uses/slashes", "match pattern"),
        ("runtime.storage.path", "", "at least 1 character"),
    ],
)
def test_invalid_field_rejected_with_clear_message(path, value, msg_substring):
    """Each invalid value lands in a Pydantic error whose message contains a hint."""
    cfg = _kitchen_sink_valid()
    _set_dotted(cfg, path, value)
    with pytest.raises(ValidationError) as exc:
        ExperimentConfig.model_validate(cfg)
    assert msg_substring in str(
        exc.value
    ), f"expected '{msg_substring}' in:\n{exc.value}"


# ── Registry-aware checks live in the application layer (hex compliance) ──


@pytest.mark.parametrize(
    "path,value,msg_substring",
    [
        # ``runtime.runner.type`` is constrained by ``Literal["local", "ovh"]``
        # at the schema layer (F7.7.A4), so unknown values fail with a
        # Pydantic ValidationError earlier than registry-validation. That's
        # tested in ``test_invalid_field_rejected_with_clear_message``.
        # The remaining ``*.type`` fields stay ``str`` so registry validation
        # is still the gate that catches typos.
        ("scenario.type", "unknown_scenario", "scenario.type"),
        ("algorithm.type", "unknown_algo", "algorithm.type"),
        ("runtime.storage.type", "unknown_storage", "storage.type"),
    ],
)
def test_validate_known_types_rejects_unknown_adapter(path, value, msg_substring):
    """``application.config_validation.validate_known_types`` catches typos in
    ``cfg.*.type`` fields once schema validation has succeeded.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.config_validation import validate_known_types

    cfg_dict = _kitchen_sink_valid()
    _set_dotted(cfg_dict, path, value)
    cfg = ExperimentConfig.model_validate(cfg_dict)  # schema-OK, registry not yet
    with pytest.raises(ValueError) as exc:
        validate_known_types(cfg)
    assert msg_substring in str(exc.value)


def test_runner_type_unknown_rejected_at_schema_layer():
    """F7.7.A4 — ``runtime.runner.type: Literal["local", "ovh"]`` fails fast
    at Pydantic validation rather than waiting for registry validation.
    """
    cfg_dict = _kitchen_sink_valid()
    _set_dotted(cfg_dict, "runtime.runner.type", "unknown_runner")
    with pytest.raises(ValidationError, match="Input should be 'local' or 'ovh'"):
        ExperimentConfig.model_validate(cfg_dict)


def test_validate_known_types_rejects_device_outside_runner_capabilities():
    """F7.7.A4 — when ``training.device`` isn't in
    ``runner_spec(runtime.runner.type).supported_devices``, the cross-field
    check inside ``validate_known_types`` raises a clear ValueError.

    This makes adding a new runner with restricted device support trivial:
    declare it in factories.py and configs targeting an unsupported device
    fail at parse time rather than minutes into a run.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application import factories
    from multi_scenario.application.config_validation import validate_known_types

    # Inject a fake runner that only supports cpu — no code edits, just
    # registry mutation in the test.
    fake_spec = factories.RunnerSpec(
        name="fake_cpu_only",
        supported_devices=frozenset({"cpu"}),
        default_device="cpu",
    )
    factories._RUNNERS["fake_cpu_only"] = fake_spec  # noqa: SLF001
    try:
        cfg_dict = _kitchen_sink_valid()
        # Bypass the schema's Literal["local", "ovh"] by mutating the model
        # post-validation: validate_known_types is the gate we're testing.
        # (Real users will get this error after extending Literal[...] to
        # include the new runner first.)
        cfg = ExperimentConfig.model_validate(cfg_dict)
        cfg.runtime.runner.type = "fake_cpu_only"  # type: ignore[assignment]
        cfg.training.device = "cuda"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="not supported by"):
            validate_known_types(cfg)
    finally:
        factories._RUNNERS.pop("fake_cpu_only", None)


def test_validate_known_types_accepts_kitchen_sink():
    """Sanity: every registered type passes the registry check."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.config_validation import validate_known_types

    validate_known_types(ExperimentConfig.model_validate(_kitchen_sink_valid()))


def test_validate_known_types_skips_runtime_when_absent():
    """``runtime: None`` → only scenario + algorithm types checked."""
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.config_validation import validate_known_types

    cfg_dict = _kitchen_sink_valid()
    cfg_dict.pop("runtime")
    validate_known_types(ExperimentConfig.model_validate(cfg_dict))


def test_minibatch_larger_than_frames_per_batch_rejected():
    """Cross-field invariant on TrainingSection."""
    cfg = _kitchen_sink_valid()
    cfg["training"]["frames_per_batch"] = 50
    cfg["training"]["minibatch_size"] = 9999
    with pytest.raises(ValidationError, match="minibatch_size"):
        ExperimentConfig.model_validate(cfg)


def test_minibatch_equal_to_frames_per_batch_accepted():
    """Boundary: equality is allowed (≤, not <)."""
    cfg = _kitchen_sink_valid()
    cfg["training"]["frames_per_batch"] = 100
    cfg["training"]["minibatch_size"] = 100
    ExperimentConfig.model_validate(cfg)


def test_runtime_omitted_skips_runner_storage_validation():
    """``runtime: None`` is valid → ExperimentService picks defaults later."""
    cfg = _kitchen_sink_valid()
    cfg.pop("runtime")
    ExperimentConfig.model_validate(cfg)


def test_extra_keys_in_top_level_rejected():
    """STRICT model: unknown top-level keys still fail loudly."""
    cfg = _kitchen_sink_valid()
    cfg["unknown_section"] = {"x": 1}
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(cfg)


def test_validation_messages_useful_for_multiple_field_errors():
    """When several fields are bad, the error mentions each — no early-exit."""
    cfg = _kitchen_sink_valid()
    _set_dotted(cfg, "training.max_iters", 0)
    _set_dotted(cfg, "evaluation.episodes", 0)
    with pytest.raises(ValidationError) as exc:
        ExperimentConfig.model_validate(cfg)
    msg = str(exc.value)
    assert "max_iters" in msg
    assert "episodes" in msg


def test_storage_section_default_path_required():
    """``runtime.storage.path`` is required (no default) — surfaces if YAML omits."""
    cfg = _kitchen_sink_valid()
    cfg["runtime"]["storage"].pop("path")
    with pytest.raises(ValidationError, match="path"):
        ExperimentConfig.model_validate(cfg)


def test_deep_copy_independence():
    """Ensure helper doesn't leak state across parametrise calls (defensive)."""
    base = _kitchen_sink_valid()
    mutated = copy.deepcopy(base)
    _set_dotted(mutated, "experiment.seed", 99)
    assert base["experiment"]["seed"] == 0
    assert mutated["experiment"]["seed"] == 99
