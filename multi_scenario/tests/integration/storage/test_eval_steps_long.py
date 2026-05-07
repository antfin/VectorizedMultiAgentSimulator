"""F5.4 tests: LocalStorageAdapter.save_eval_steps_long — long-format CSV."""

from pathlib import Path

import pandas as pd
import torch
from tensordict import TensorDict

from multi_scenario.adapters.storage.local import LocalStorageAdapter


def _fake_rollout_td(num_envs: int = 2, t_steps: int = 3, n_agents: int = 2, action_dim: int = 2):
    """Build a minimal rollout-shaped TensorDict matching BenchMARL's keys.

    Mirrors the shapes confirmed via probe: reward [E,T,A,1], action [E,T,A,D],
    done/terminated [E,T,1] (env-wide, not per-agent).
    """
    bs = (num_envs, t_steps)
    # Distinct values per (env, step, agent) so we can assert ordering downstream.
    e_idx = torch.arange(num_envs).view(num_envs, 1, 1).expand(num_envs, t_steps, n_agents)
    t_idx = torch.arange(t_steps).view(1, t_steps, 1).expand(num_envs, t_steps, n_agents)
    a_idx = torch.arange(n_agents).view(1, 1, n_agents).expand(num_envs, t_steps, n_agents)
    reward = (e_idx * 100 + t_idx * 10 + a_idx).float().unsqueeze(-1)  # [E,T,A,1]
    action = e_idx.float().unsqueeze(-1).expand(num_envs, t_steps, n_agents, action_dim).clone()
    done = torch.zeros(num_envs, t_steps, 1, dtype=torch.bool)
    done[:, -1, 0] = True  # last step is "done"
    terminated = done.clone()

    return TensorDict(
        {
            "agents": TensorDict(
                {"action": action},
                batch_size=bs,
            ),
            "next": TensorDict(
                {
                    "agents": TensorDict(
                        {"reward": reward},
                        batch_size=bs,
                    ),
                    "done": done,
                    "terminated": terminated,
                },
                batch_size=bs,
            ),
        },
        batch_size=bs,
    )


def test_save_eval_steps_long_row_count(tmp_path: Path) -> None:
    """Row count = num_envs × T × n_agents."""
    td = _fake_rollout_td(num_envs=2, t_steps=3, n_agents=2)

    LocalStorageAdapter().save_eval_steps_long(
        tmp_path, td, group_map={"agents": ["agent_0", "agent_1"]}
    )

    df = pd.read_csv(tmp_path / "output" / "eval_steps.csv")
    assert len(df) == 2 * 3 * 2  # 12 rows


def test_save_eval_steps_long_universal_columns(tmp_path: Path) -> None:
    """Schema: env_idx, step, agent, reward, done, terminated + action_d{i}."""
    td = _fake_rollout_td(num_envs=1, t_steps=2, n_agents=2, action_dim=2)

    LocalStorageAdapter().save_eval_steps_long(
        tmp_path, td, group_map={"agents": ["agent_0", "agent_1"]}
    )

    df = pd.read_csv(tmp_path / "output" / "eval_steps.csv")
    assert {"env_idx", "step", "agent", "reward", "done", "terminated"} <= set(df.columns)
    # One column per action dim discovered at runtime.
    assert "action_d0" in df.columns
    assert "action_d1" in df.columns


def test_save_eval_steps_long_reward_values_correct(tmp_path: Path) -> None:
    """Per-(env, step, agent) reward values land in the right rows."""
    td = _fake_rollout_td(num_envs=2, t_steps=3, n_agents=2)

    LocalStorageAdapter().save_eval_steps_long(
        tmp_path, td, group_map={"agents": ["agent_0", "agent_1"]}
    )

    df = pd.read_csv(tmp_path / "output" / "eval_steps.csv")
    # env 1, step 2, agent 0 → reward = 1*100 + 2*10 + 0 = 120
    row = df[(df["env_idx"] == 1) & (df["step"] == 2) & (df["agent"] == "agents:agent_0")]
    assert len(row) == 1
    assert row.iloc[0]["reward"] == 120.0


def test_save_eval_steps_long_uses_group_map_names(tmp_path: Path) -> None:
    """The agent column uses ``<group>:<agent_name>`` from group_map."""
    td = _fake_rollout_td(num_envs=1, t_steps=1, n_agents=2)

    LocalStorageAdapter().save_eval_steps_long(
        tmp_path, td, group_map={"agents": ["agent_0", "agent_1"]}
    )

    df = pd.read_csv(tmp_path / "output" / "eval_steps.csv")
    assert set(df["agent"]) == {"agents:agent_0", "agents:agent_1"}


def test_save_eval_steps_long_done_terminated_propagate(tmp_path: Path) -> None:
    """``done`` / ``terminated`` get broadcast to per-agent rows of the same step."""
    td = _fake_rollout_td(num_envs=1, t_steps=2, n_agents=2)
    # _fake_rollout_td sets done=True only at last step.

    LocalStorageAdapter().save_eval_steps_long(
        tmp_path, td, group_map={"agents": ["agent_0", "agent_1"]}
    )

    df = pd.read_csv(tmp_path / "output" / "eval_steps.csv")
    last_step = df[df["step"] == 1]
    first_step = df[df["step"] == 0]
    assert (last_step["done"]).all()
    assert (last_step["terminated"]).all()
    assert (~first_step["done"]).all()


def test_save_eval_steps_long_creates_output_dir(tmp_path: Path) -> None:
    """Parent dir is auto-created (consistent with other LocalStorage writers)."""
    td = _fake_rollout_td(num_envs=1, t_steps=1, n_agents=1)
    LocalStorageAdapter().save_eval_steps_long(tmp_path, td, group_map={"agents": ["agent_0"]})
    assert (tmp_path / "output" / "eval_steps.csv").is_file()
