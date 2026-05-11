"""F9.2 — load-bearing byte-parity test.

For each ported prompt version, render the Jinja ``.j2`` and the
rendezvous_comm ``string.Template`` against the SAME context, and
assert the two byte sequences are equal. If this drifts, F8.4's
S3b-local replication will silently change too — that's the failure
mode this test catches early.

When this test fails, two scenarios:

1. We deliberately modified our Jinja template (e.g., a new prompt
   version layered on the ported ones). In that case the `parametrize`
   list should drop the no-longer-byte-parity version.
2. The rendezvous_comm prompt was edited and we forgot to re-run the
   port script. Run ``python -m scripts.port_lero_prompts`` from the
   repo root and the test should pass again.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import string
from pathlib import Path

import pytest

from multi_scenario.adapters.prompts import JinjaPromptRenderer


_TARGET_VERSIONS = (
    "v1",
    "v1_global",
    "v2",
    "v2_min",
    "v2_fewshot",
    "v2_twofn",
    "v2_fewshot_k2_local",
)
_TEMPLATES = ("system", "initial_user", "feedback")

_REPO_ROOT = Path(__file__).resolve().parents[3].parent
_RENDEZVOUS_PROMPTS_ROOT = _REPO_ROOT / "rendezvous_comm" / "src" / "lero" / "prompts"


def _full_context() -> dict[str, object]:
    """Generous context covering every variable any ported prompt uses.

    Keys missing from a particular template are silently ignored by both
    Jinja's ``StrictUndefined`` (template doesn't reference them, so no
    error) and ``string.Template.safe_substitute``. Extra keys are fine.

    Values chosen to mimic a real rendezvous_comm S3b-local invocation
    (k=2, n=4 agents, t=4 targets) but are arbitrary as long as both
    renderers see the same dict.
    """
    return {
        "n_agents": 4,
        "n_targets": 4,
        "agents_per_target": 2,
        "covering_range": 0.35,
        "max_steps": 200,
        "n_lidar_rays_entities": 15,
        "n_lidar_rays_agents": 12,
        "lidar_range": 0.35,
        "collision_penalty": -0.01,
        "time_penalty": -0.01,
        "experiment_context": "ER1+LERO S3b-local replication",
        "agent_lidar_description": "Agent LiDAR is ENABLED — agents can detect nearby agents.",
        "comm_description": "This is a NO-COMMUNICATION experiment.",
        "comm_state_description": "",
        "comm_metrics": "",
        "obs_lidar_agents": '"lidar_agents":     # [batch, 12] — distance to nearest AGENT in each ray direction',
        "obs_comm_state": "",
        "comm_obs_guidance": "",
        "reward_description": "Original reward: covering_reward + agent_collision_penalty + time_penalty.",
        "coordination_guidance": "Encourage rendezvous-style coordination.",
        "scenario_reward_code": "# (truncated reward source from VMAS Discovery)",
        "scenario_observation_code": "# (truncated observation source)",
        "candidates_results": "[stub: 3 candidates summarised here]",
        "n_candidates": 3,
        "best_idx": 0,
    }


@pytest.fixture(scope="module")
def jinja_renderer() -> JinjaPromptRenderer:
    return JinjaPromptRenderer()


def _render_string_template(version: str, template: str, ctx: dict) -> str:
    src_path = _RENDEZVOUS_PROMPTS_ROOT / version / f"{template}.txt"
    raw = src_path.read_text(encoding="utf-8")
    # Match rendezvous_comm's PromptLoader: two-pass safe_substitute so
    # indirected vars (like ``$obs_lidar_agents`` containing other ``$x``)
    # resolve. Our prompts here only need one pass, but two is harmless.
    out = string.Template(raw).safe_substitute(**ctx)
    return string.Template(out).safe_substitute(**ctx)


@pytest.mark.parametrize("version", _TARGET_VERSIONS)
@pytest.mark.parametrize("template", _TEMPLATES)
def test_byte_parity_jinja_vs_string_template(
    version: str, template: str, jinja_renderer: JinjaPromptRenderer
):
    """Jinja-rendered .j2 == rendezvous_comm string.Template-rendered .txt."""
    src_path = _RENDEZVOUS_PROMPTS_ROOT / version / f"{template}.txt"
    if not src_path.is_file():
        pytest.skip(f"rendezvous_comm source missing: {src_path}")

    ctx = _full_context()
    expected = _render_string_template(version, template, ctx)
    actual = jinja_renderer.render(version=version, template=template, context=ctx)

    if actual != expected:
        # Show a useful diff in the failure: line-by-line first diverging.
        exp_lines = expected.splitlines()
        act_lines = actual.splitlines()
        for i, (e, a) in enumerate(zip(exp_lines, act_lines)):
            if e != a:
                pytest.fail(
                    f"{version}/{template}.j2 diverges at line {i + 1}:\n"
                    f"  expected: {e!r}\n  actual:   {a!r}\n"
                    f"(re-run python -m scripts.port_lero_prompts if the "
                    f"rendezvous_comm template changed)"
                )
        if len(exp_lines) != len(act_lines):
            pytest.fail(
                f"{version}/{template}.j2 line count differs: "
                f"expected {len(exp_lines)}, got {len(act_lines)}"
            )
        pytest.fail(f"{version}/{template}.j2 bytes differ but lines match")


def test_every_target_version_has_all_three_templates():
    """Sanity: the registry isn't half-ported."""
    repo_root = _REPO_ROOT
    adapter_root = (
        repo_root / "multi_scenario" / "src" / "multi_scenario" / "adapters" / "prompts"
    )
    for version in _TARGET_VERSIONS:
        for template in _TEMPLATES:
            path = adapter_root / version / f"{template}.j2"
            assert path.is_file(), f"missing port: {path}"


def test_meta_yaml_is_copied_verbatim():
    """meta.yaml ports as-is; not Jinja-translated."""
    repo_root = _REPO_ROOT
    adapter_root = (
        repo_root / "multi_scenario" / "src" / "multi_scenario" / "adapters" / "prompts"
    )
    for version in _TARGET_VERSIONS:
        src = _RENDEZVOUS_PROMPTS_ROOT / version / "meta.yaml"
        dst = adapter_root / version / "meta.yaml"
        if not src.is_file():
            continue
        assert dst.is_file(), f"meta.yaml missing for {version}"
        assert src.read_bytes() == dst.read_bytes(), (
            f"{version}/meta.yaml diverges from rendezvous_comm — "
            "re-run python -m scripts.port_lero_prompts"
        )
