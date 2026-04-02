"""Tests for the LERO module — config, prompts, codegen, scenario_patch."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from src.config import ExperimentSpec, load_experiment
from src.lero.codegen import (
    CandidateCode,
    build_feedback,
    extract_candidates,
    validate_function,
)
from src.lero.config import LeroConfig, LLMConfig
from src.lero.llm_client import LLMClient
from src.lero.prompts.loader import PromptLoader
from src.lero.scenario_patch import (
    _build_scenario_state,
    _compile_function,
    patch_scenario,
    unpatch_scenario,
)


# ── Helpers ──────────────────────────────────────────────────────


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _make_lero_yaml(**overrides):
    """Minimal YAML data for a LERO experiment."""
    base = {
        "exp_id": "e1",
        "name": "Test LERO",
        "description": "Test",
        "task": {"n_agents": 4, "n_targets": 4, "agents_per_target": 2},
        "lero": {
            "n_iterations": 2,
            "n_candidates": 2,
            "eval_frames": 100_000,
        },
        "llm": {"model": "anthropic/test-model"},
    }
    base.update(overrides)
    return base


VALID_REWARD_SRC = """\
import torch

def compute_reward(scenario_state):
    collision = scenario_state["collision_rew"]
    time_pen = torch.full_like(collision, scenario_state["time_penalty"])
    agent_pos = scenario_state["agent_pos"]
    targets_pos = scenario_state["targets_pos"]
    dists = torch.cdist(agent_pos.unsqueeze(1), targets_pos).squeeze(1)
    min_dist = dists.min(dim=-1).values
    return -min_dist + collision + time_pen
"""

VALID_OBS_SRC = """\
import torch

def enhance_observation(scenario_state):
    agent_pos = scenario_state["agent_pos"]
    targets_pos = scenario_state["targets_pos"]
    dists = torch.cdist(agent_pos.unsqueeze(1), targets_pos).squeeze(1)
    min_dist = dists.min(dim=-1).values.unsqueeze(-1)
    n_covered = scenario_state["covered_targets"].float().sum(dim=-1, keepdim=True)
    progress = n_covered / scenario_state["n_targets"]
    return torch.cat([min_dist, progress], dim=-1)
"""

FAKE_LLM_RESPONSE = f"""\
Here are the improved functions:

```python
{VALID_REWARD_SRC}
```

```python
{VALID_OBS_SRC}
```
"""


# ── LeroConfig / LLMConfig ──────────────────────────────────────


class TestLeroConfig:
    def test_defaults(self):
        cfg = LeroConfig()
        assert cfg.n_iterations == 4
        assert cfg.n_candidates == 3
        assert cfg.top_k == 2
        assert cfg.eval_frames == 1_000_000
        assert cfg.full_frames == 10_000_000
        assert cfg.evolve_reward is True
        assert cfg.evolve_observation is True

    def test_custom_values(self):
        cfg = LeroConfig(n_iterations=2, n_candidates=5, top_k=3)
        assert cfg.n_iterations == 2
        assert cfg.n_candidates == 5
        assert cfg.top_k == 3


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert "claude" in cfg.model
        assert cfg.temperature == 0.8
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 3
        assert cfg.prompt_version == "v1"
        assert cfg.api_base is None
        assert cfg.api_key is None

    def test_custom_model(self):
        cfg = LLMConfig(model="gpt-4o")
        assert cfg.model == "gpt-4o"

    def test_custom_endpoint(self):
        cfg = LLMConfig(
            model="openai/llama-3-70b",
            api_base="https://ovh-endpoint.com/v1",
            api_key="sk-ovh-key",
        )
        assert cfg.api_base == "https://ovh-endpoint.com/v1"
        assert cfg.api_key == "sk-ovh-key"


# ── Config loading with LERO sections ────────────────────────────


class TestConfigWithLero:
    def test_load_lero_yaml(self, tmp_path):
        data = _make_lero_yaml()
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)

        assert spec.lero is not None
        assert spec.llm is not None
        assert spec.lero.n_iterations == 2
        assert spec.lero.n_candidates == 2
        assert spec.llm.model == "anthropic/test-model"

    def test_load_yaml_without_lero(self, tmp_path):
        data = {
            "exp_id": "er1",
            "name": "No LERO",
            "description": "",
        }
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)
        assert spec.lero is None
        assert spec.llm is None

    def test_load_lero_without_llm_gets_defaults(self, tmp_path):
        data = _make_lero_yaml()
        del data["llm"]
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        spec = load_experiment(path)
        assert spec.lero is not None
        assert spec.llm is not None
        assert "claude" in spec.llm.model

    def test_load_real_e1_config(self):
        cfg_path = (
            Path(__file__).parent.parent
            / "configs" / "e1" / "single_lero_al_lp_sr_ms400.yaml"
        )
        spec = load_experiment(cfg_path)
        assert spec.exp_id == "e1_al_lp_sr_ms400"
        assert spec.lero is not None
        assert spec.task.dim_c == 0
        assert spec.task.shared_reward is True
        assert spec.task.agent_collision_penalty == -0.01
        assert spec.task.max_steps == 400

    def test_load_real_e1_n2_k1_config(self):
        cfg_path = (
            Path(__file__).parent.parent
            / "configs" / "e1" / "single_lero_al_lp_sr_ms400_n2_k1.yaml"
        )
        spec = load_experiment(cfg_path)
        assert spec.exp_id == "e1_al_lp_sr_ms400_n2_k1"
        assert spec.lero is not None
        assert spec.task.n_agents == 2
        assert spec.task.agents_per_target == 1
        assert spec.task.n_targets == 4

    def test_load_real_e2_config(self):
        cfg_path = (
            Path(__file__).parent.parent
            / "configs" / "e2" / "single_lero_al_lp_sr_prox_dc8_ms400.yaml"
        )
        spec = load_experiment(cfg_path)
        assert spec.exp_id == "e2_al_lp_sr_prox_dc8_ms400"
        assert spec.lero is not None
        assert spec.task.dim_c == 8
        assert spec.task.comm_proximity is True
        assert spec.task.shared_reward is True

    def test_e1_matches_er1_task_params(self):
        """E1 task params must match ER1 baseline for fair comparison."""
        e1_path = (
            Path(__file__).parent.parent
            / "configs" / "e1" / "single_lero_al_lp_sr_ms400.yaml"
        )
        er1_path = (
            Path(__file__).parent.parent
            / "configs" / "er1" / "single_al_lp_sr_ms400.yaml"
        )
        e1 = load_experiment(e1_path)
        er1 = load_experiment(er1_path)
        # All task params must match exactly
        for key in [
            "n_agents", "n_targets", "agents_per_target", "lidar_range",
            "covering_range", "use_agent_lidar", "targets_respawn",
            "shared_reward", "agent_collision_penalty", "covering_rew_coeff",
            "time_penalty", "max_steps",
        ]:
            assert getattr(e1.task, key) == getattr(er1.task, key), (
                f"E1 vs ER1 mismatch on {key}: "
                f"{getattr(e1.task, key)} != {getattr(er1.task, key)}"
            )

    def test_e2_matches_er2_task_params(self):
        """E2 task params must match ER2 baseline for fair comparison."""
        e2_path = (
            Path(__file__).parent.parent
            / "configs" / "e2" / "single_lero_al_lp_sr_prox_dc8_ms400.yaml"
        )
        er2_path = (
            Path(__file__).parent.parent
            / "configs" / "er2" / "single_al_lp_sr_prox_dc8_ms400.yaml"
        )
        e2 = load_experiment(e2_path)
        er2 = load_experiment(er2_path)
        for key in [
            "n_agents", "n_targets", "agents_per_target", "lidar_range",
            "covering_range", "use_agent_lidar", "targets_respawn",
            "shared_reward", "agent_collision_penalty", "covering_rew_coeff",
            "time_penalty", "max_steps", "dim_c", "comm_proximity",
        ]:
            assert getattr(e2.task, key) == getattr(er2.task, key), (
                f"E2 vs ER2 mismatch on {key}: "
                f"{getattr(e2.task, key)} != {getattr(er2.task, key)}"
            )


# ── PromptLoader ─────────────────────────────────────────────────


class TestPromptLoader:
    def test_load_v1_system(self):
        loader = PromptLoader("v1")
        text = loader.render(
            "system.txt",
            experiment_context=(
                "**E1 LERO** (e1_test)\n\n"
                "No-communication experiment. 4 agents, 4 targets, k=2."
            ),
        )
        assert "reward engineer" in text
        assert "E1 LERO" in text
        assert "ER1" in text  # Research context mentions baselines
        assert "ER2" in text

    def test_load_v1_initial_user(self):
        loader = PromptLoader("v1")
        text = loader.render(
            "initial_user.txt",
            n_agents=4, n_targets=4, agents_per_target=2,
            covering_range=0.25, lidar_range=0.35,
            max_steps=400, collision_penalty=-0.01,
            time_penalty=-0.01,
            n_lidar_rays_entities=15, n_lidar_rays_agents=12,
            agent_lidar_description="Agent LiDAR is ENABLED.",
            comm_description="- **Communication**: NONE.",
            reward_description="- Covering reward: +1.0 per target.",
            comm_state_description="# no comm",
            comm_obs_guidance="",
            scenario_reward_code="def reward(): pass",
            scenario_observation_code="def observation(): pass",
        )
        assert "compute_reward" in text
        assert "enhance_observation" in text
        assert "400" in text  # max_steps
        assert "NONE" in text  # no comm

    def test_load_v1_feedback(self):
        loader = PromptLoader("v1")
        text = loader.render(
            "feedback.txt",
            n_candidates=3, n_agents=4, n_targets=4,
            agents_per_target=2,
            candidates_results="Candidate #1: M1=0.5",
            best_idx=1, comm_metrics="",
        )
        assert "M1" in text
        assert "Candidate #1" in text

    def test_nonexistent_version_raises(self):
        with pytest.raises(FileNotFoundError):
            PromptLoader("v99")

    def test_nonexistent_template_raises(self):
        loader = PromptLoader("v1")
        with pytest.raises(FileNotFoundError):
            loader.render("nonexistent.txt")

    def test_load_raw(self):
        loader = PromptLoader("v1")
        raw = loader.load_raw("system.txt")
        # Raw should contain $variable placeholders
        assert "$experiment_context" in raw

    def test_safe_substitute_missing_vars(self):
        loader = PromptLoader("v1")
        # Render with missing variables — should not raise
        text = loader.render("system.txt", n_agents=4)
        assert "4" in text


# ── Codegen: validate_function ───────────────────────────────────


class TestValidateFunction:
    def test_valid_reward_function(self):
        assert validate_function(
            VALID_REWARD_SRC.strip(),
            "compute_reward", ["scenario_state"],
        )

    def test_valid_obs_function(self):
        assert validate_function(
            VALID_OBS_SRC.strip(),
            "enhance_observation", ["scenario_state"],
        )

    def test_wrong_function_name(self):
        src = "def wrong_name(scenario_state):\n    return 0"
        assert not validate_function(src, "compute_reward", ["scenario_state"])

    def test_wrong_arguments(self):
        src = "def compute_reward(x, y):\n    return 0"
        assert not validate_function(
            src, "compute_reward", ["scenario_state"],
        )

    def test_syntax_error(self):
        src = "def compute_reward(scenario_state)\n    return 0"  # missing :
        assert not validate_function(
            src, "compute_reward", ["scenario_state"],
        )

    def test_disallowed_import(self):
        src = "import os\ndef compute_reward(scenario_state):\n    return 0"
        assert not validate_function(
            src, "compute_reward", ["scenario_state"],
        )

    def test_allowed_imports(self):
        src = (
            "import torch\nimport math\nimport numpy\n"
            "def compute_reward(scenario_state):\n    return 0"
        )
        assert validate_function(
            src, "compute_reward", ["scenario_state"],
        )

    def test_disallowed_from_import(self):
        src = (
            "from subprocess import call\n"
            "def compute_reward(scenario_state):\n    return 0"
        )
        assert not validate_function(
            src, "compute_reward", ["scenario_state"],
        )


# ── Codegen: extract_candidates ──────────────────────────────────


class TestExtractCandidates:
    def test_extract_both_functions(self):
        candidates = extract_candidates([FAKE_LLM_RESPONSE])
        assert len(candidates) == 1
        c = candidates[0]
        assert c.reward_source is not None
        assert c.obs_source is not None
        assert "compute_reward" in c.reward_source
        assert "enhance_observation" in c.obs_source

    def test_extract_reward_only(self):
        resp = f"```python\n{VALID_REWARD_SRC}\n```"
        candidates = extract_candidates(
            [resp], evolve_reward=True, evolve_observation=False,
        )
        assert len(candidates) == 1
        assert candidates[0].reward_source is not None
        assert candidates[0].obs_source is None

    def test_extract_obs_only(self):
        resp = f"```python\n{VALID_OBS_SRC}\n```"
        candidates = extract_candidates(
            [resp], evolve_reward=False, evolve_observation=True,
        )
        assert len(candidates) == 1
        assert candidates[0].obs_source is not None
        assert candidates[0].reward_source is None

    def test_no_code_blocks_returns_empty(self):
        candidates = extract_candidates(["No code here"])
        assert len(candidates) == 0

    def test_invalid_code_skipped(self):
        resp = "```python\ndef compute_reward(x, y):\n    return 0\n```"
        candidates = extract_candidates([resp])
        # Invalid args, should be skipped
        assert len(candidates) == 0

    def test_multiple_responses(self):
        candidates = extract_candidates(
            [FAKE_LLM_RESPONSE, FAKE_LLM_RESPONSE],
        )
        assert len(candidates) == 2

    def test_candidate_source_property(self):
        candidates = extract_candidates([FAKE_LLM_RESPONSE])
        c = candidates[0]
        source = c.source
        assert "compute_reward" in source
        assert "enhance_observation" in source

    def test_raw_response_preserved(self):
        candidates = extract_candidates([FAKE_LLM_RESPONSE])
        assert candidates[0].raw_response == FAKE_LLM_RESPONSE


# ── Codegen: build_feedback ──────────────────────────────────────


class TestBuildFeedback:
    def test_basic_feedback(self):
        loader = PromptLoader("v1")
        candidates = [
            CandidateCode(VALID_REWARD_SRC, VALID_OBS_SRC, "resp1"),
            CandidateCode(VALID_REWARD_SRC, VALID_OBS_SRC, "resp2"),
        ]
        metrics = [
            {"M1_success_rate": 0.5, "M2_avg_return": 10.0,
             "M4_avg_collisions": 1.0, "M6_coverage_progress": 0.8},
            {"M1_success_rate": 0.3, "M2_avg_return": 5.0,
             "M4_avg_collisions": 2.0, "M6_coverage_progress": 0.6},
        ]
        feedback = build_feedback(
            candidates, metrics, top_k=1,
            prompt_loader=loader,
            task_params={"n_agents": 4, "n_targets": 4, "agents_per_target": 2},
        )
        assert "M1" in feedback
        assert "0.500" in feedback  # best M1
        assert "rank 1" in feedback

    def test_feedback_with_comm_metrics(self):
        loader = PromptLoader("v1")
        candidates = [
            CandidateCode(VALID_REWARD_SRC, None, "resp"),
        ]
        metrics = [
            {"M1_success_rate": 0.5, "M2_avg_return": 10.0,
             "M4_avg_collisions": 1.0, "M6_coverage_progress": 0.8,
             "M5_avg_tokens": 32.0},
        ]
        feedback = build_feedback(
            candidates, metrics, top_k=1,
            prompt_loader=loader,
            task_params={"n_agents": 4, "n_targets": 4, "agents_per_target": 2},
        )
        assert "M5" in feedback


# ── scenario_patch: _compile_function ────────────────────────────


class TestCompileFunction:
    def test_compile_reward(self):
        fn = _compile_function(VALID_REWARD_SRC.strip(), "compute_reward")
        assert callable(fn)

    def test_compile_obs(self):
        fn = _compile_function(VALID_OBS_SRC.strip(), "enhance_observation")
        assert callable(fn)

    def test_missing_function_raises(self):
        src = "def wrong_name(x):\n    return 0"
        with pytest.raises(ValueError, match="not found"):
            _compile_function(src, "compute_reward")

    def test_compiled_reward_runs(self):
        fn = _compile_function(VALID_REWARD_SRC.strip(), "compute_reward")
        batch = 8
        state = {
            "collision_rew": torch.zeros(batch),
            "time_penalty": -0.01,
            "agent_pos": torch.randn(batch, 2),
            "targets_pos": torch.randn(batch, 4, 2),
        }
        result = fn(state)
        assert result.shape == (batch,)

    def test_compiled_obs_runs(self):
        fn = _compile_function(VALID_OBS_SRC.strip(), "enhance_observation")
        batch = 8
        state = {
            "agent_pos": torch.randn(batch, 2),
            "targets_pos": torch.randn(batch, 4, 2),
            "covered_targets": torch.zeros(batch, 4, dtype=torch.bool),
            "n_targets": 4,
        }
        result = fn(state)
        assert result.shape == (batch, 2)  # min_dist + progress


# ── scenario_patch: patch/unpatch ────────────────────────────────


class TestScenarioPatch:
    """Test patching with a mock scenario object."""

    @pytest.fixture
    def mock_scenario(self):
        """Create a minimal mock that mimics Discovery scenario."""
        batch = 4
        n_agents = 2
        n_targets = 2
        device = torch.device("cpu")

        scenario = MagicMock()
        scenario.world.batch_dim = batch
        scenario.world.device = device
        scenario.n_targets = n_targets
        scenario._covering_range = 0.25
        scenario._agents_per_target = 1
        scenario.agent_collision_penalty = -0.1
        scenario.time_penalty = -0.01
        scenario.dim_c = 0

        # Agents
        agents = []
        for i in range(n_agents):
            agent = MagicMock()
            agent.state.pos = torch.randn(batch, 2)
            agent.state.vel = torch.randn(batch, 2)
            agent.collision_rew = torch.zeros(batch)
            agent.covering_reward = torch.zeros(batch)
            agents.append(agent)
        scenario.world.agents = agents

        # Targets
        targets = []
        for i in range(n_targets):
            t = MagicMock()
            t.state.pos = torch.randn(batch, 2)
            targets.append(t)
        scenario._targets = targets

        # Computed state (normally set by reward preamble)
        scenario.agents_pos = torch.stack(
            [a.state.pos for a in agents], dim=1,
        )
        scenario.targets_pos = torch.stack(
            [t.state.pos for t in targets], dim=1,
        )
        scenario.agents_targets_dists = torch.cdist(
            scenario.agents_pos, scenario.targets_pos,
        )
        scenario.agents_per_target = torch.sum(
            (scenario.agents_targets_dists < 0.25).int(), dim=1,
        )
        scenario.covered_targets = scenario.agents_per_target >= 1
        scenario.all_time_covered_targets = torch.zeros(
            batch, n_targets, dtype=torch.bool,
        )

        # Original methods
        def orig_reward(agent):
            return torch.ones(batch)

        def orig_observation(agent):
            return torch.randn(batch, 10)

        scenario.reward = orig_reward
        scenario.observation = orig_observation

        return scenario

    def test_build_scenario_state(self, mock_scenario):
        agent = mock_scenario.world.agents[0]
        state = _build_scenario_state(mock_scenario, agent, 0)

        assert "agents_pos" in state
        assert "targets_pos" in state
        assert "agent_pos" in state
        assert "agent_vel" in state
        assert "agent_idx" in state
        assert state["agent_idx"] == 0
        assert state["n_agents"] == 2
        assert state["n_targets"] == 2

    def test_build_scenario_state_no_comm(self, mock_scenario):
        agent = mock_scenario.world.agents[0]
        state = _build_scenario_state(mock_scenario, agent, 0)
        assert "messages" not in state

    def test_build_scenario_state_with_comm(self, mock_scenario):
        mock_scenario.dim_c = 4
        for a in mock_scenario.world.agents:
            a.state.c = torch.randn(4, 4)
        agent = mock_scenario.world.agents[0]
        state = _build_scenario_state(mock_scenario, agent, 0)
        assert "messages" in state
        assert state["messages"].shape[-1] == 4  # dim_c

    def test_patch_class_and_unpatch(self):
        """Test class-level patching and restoration."""
        from vmas.scenarios.discovery import Scenario
        from src.lero.scenario_patch import (
            patch_scenario_class, unpatch_scenario_class,
        )

        orig_reward = Scenario.reward
        orig_obs = Scenario.observation

        originals = patch_scenario_class(
            obs_source=VALID_OBS_SRC.strip(),
        )
        # After patch, observation method is different
        assert Scenario.observation is not orig_obs
        # Reward not patched — still original
        assert Scenario.reward is orig_reward

        unpatch_scenario_class(originals)
        # Restored
        assert Scenario.observation is orig_obs
        assert Scenario.reward is orig_reward

    def test_patch_class_reward_only(self):
        from vmas.scenarios.discovery import Scenario
        from src.lero.scenario_patch import (
            patch_scenario_class, unpatch_scenario_class,
        )

        orig_obs = Scenario.observation
        originals = patch_scenario_class(
            reward_source=VALID_REWARD_SRC.strip(),
        )
        # Obs not touched
        assert Scenario.observation is orig_obs
        unpatch_scenario_class(originals)


# ── LLMClient ────────────────────────────────────────────────────


class TestLLMClient:
    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_generate_calls_n_times(self):
        cfg = LLMConfig(model="anthropic/test")
        client = LLMClient(cfg)

        with patch.object(client, "_call", return_value="response text"):
            results = client.generate(
                [{"role": "user", "content": "test"}], n=3,
            )
        assert len(results) == 3
        assert all(r == "response text" for r in results)

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_retry_on_failure(self):
        cfg = LLMConfig(model="anthropic/test", max_retries=3, retry_delay=0.01)
        client = LLMClient(cfg)

        call_count = 0

        def flaky_call(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("API error")
            return "success"

        with patch.object(client, "_call", side_effect=flaky_call):
            results = client.generate(
                [{"role": "user", "content": "test"}], n=1,
            )
        assert results == ["success"]
        assert call_count == 3

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_exhausted_retries_raises(self):
        cfg = LLMConfig(model="anthropic/test", max_retries=2, retry_delay=0.01)
        client = LLMClient(cfg)

        with patch.object(
            client, "_call", side_effect=RuntimeError("fail"),
        ):
            with pytest.raises(RuntimeError, match="failed after 2"):
                client.generate(
                    [{"role": "user", "content": "test"}], n=1,
                )

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_call_passes_api_base_and_key(self):
        cfg = LLMConfig(
            model="openai/llama-3-70b",
            api_base="https://ovh.example.com/v1",
            api_key="sk-test-key",
        )
        client = LLMClient(cfg)
        mock_completion = MagicMock()
        mock_completion.return_value.choices = [
            MagicMock(message=MagicMock(content="response"))
        ]
        client._litellm.completion = mock_completion

        result = client._call([{"role": "user", "content": "hi"}])

        assert result == "response"
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "https://ovh.example.com/v1"
        assert call_kwargs["api_key"] == "sk-test-key"
        assert call_kwargs["model"] == "openai/llama-3-70b"

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_call_omits_api_base_when_none(self):
        cfg = LLMConfig(model="gpt-4o")
        client = LLMClient(cfg)
        mock_completion = MagicMock()
        mock_completion.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"))
        ]
        client._litellm.completion = mock_completion

        client._call([{"role": "user", "content": "hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert "api_base" not in call_kwargs
        assert "api_key" not in call_kwargs
