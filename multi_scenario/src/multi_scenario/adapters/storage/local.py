"""LocalStorageAdapter — filesystem implementation of the Storage port.

Inherits the eight Protocol methods (save/load × {config, provenance, result,
run_state}) from :class:`StorageAdapterBase`; only the I/O primitives
(``_write_text`` / ``_read_text``) are implemented here.

Optional artefacts (eval_episodes, report, videos, log, eval_steps_long,
eval_runs) and the cross-run ``runs.csv`` / ``runs.json`` are not in the
Storage Protocol surface and live as concrete-adapter extras here when each
writer feature lands (F2.7 / F2.10 / F2.10.1 / F2.11 / F5.2 / F5.3 / F5.4 /
F5.8).
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from multi_scenario.domain.models import EvalRunRecord, RunReport

from ._base import StorageAdapterBase

# Documented schema for `output/eval_episodes.json`. Only these keys are
# serialised from the rollout dict — anything else (algorithm-internal state,
# future M5 token fields, etc.) is silently dropped to keep the schema stable.
_EVAL_EPISODES_SCHEMA = (
    "episode_returns",
    "episode_lengths",
    "episode_collisions",
    "episode_terminated",
    "targets_covered",
    "n_targets",
)


class LocalStorageAdapter(StorageAdapterBase):
    """Persists per-run JSON files to the local filesystem under ``run_dir``."""

    name = "fs"

    # ── Off-Protocol extras (F1.9 minimalism) ──────────────────────────

    def save_report(self, run_dir: Path, report: RunReport) -> None:
        """Write the run-end manifest to ``output/report.json`` (F2.10)."""
        self._write_text(run_dir, "output/report.json", report.model_dump_json(indent=2))

    def save_eval_episodes(self, run_dir: Path, rollout: dict[str, Any]) -> None:
        """Write per-episode raw eval data to ``output/eval_episodes.json`` (F2.10.1).

        Coerces tensor values to plain JSON via ``.tolist()``. Only keys in the
        documented schema (universal: ``episode_returns / episode_lengths /
        episode_collisions``; discovery: ``targets_covered / n_targets``) are
        serialised — unknown keys are silently dropped to keep the file's
        schema stable as new scenarios add fields.
        """
        payload: dict[str, Any] = {}
        for key in _EVAL_EPISODES_SCHEMA:
            if key not in rollout:
                continue
            value = rollout[key]
            payload[key] = value.tolist() if hasattr(value, "tolist") else value
        self._write_text(run_dir, "output/eval_episodes.json", json.dumps(payload, indent=2))

    def save_eval_run(self, run_dir: Path, record: EvalRunRecord) -> None:
        """Write a re-evaluation record to ``output/eval_runs/<eval_id>.json`` (F5.8)."""
        self._write_text(
            run_dir,
            f"output/eval_runs/{record.eval_id}.json",
            record.model_dump_json(indent=2),
        )

    def save_eval_steps_long(
        self,
        run_dir: Path,
        rollout_td: Any,
        group_map: dict[str, list[str]],
    ) -> None:
        """Write per-(env, step, agent) rows to ``output/eval_steps.csv`` (F5.4).

        Long-format CSV — row count = ``num_envs × T × sum(len(g) for g in group_map)``.
        Off the ``Storage`` Protocol surface (F1.9 minimalism) and gated upstream
        by ``cfg.runtime.storage.params['long_format']``. Schema is universal:
        ``env_idx, step, agent, reward, done, terminated, action_d{i}``.
        """
        rows = list(_iter_long_rows(rollout_td, group_map))
        df = pd.DataFrame(rows)
        target = run_dir / "output" / "eval_steps.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)

    # ── I/O primitives (subclass hook for StorageAdapterBase) ──────────

    def _write_text(self, run_dir: Path, rel: str, body: str) -> None:
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def _read_text(self, run_dir: Path, rel: str) -> str:
        return (run_dir / rel).read_text(encoding="utf-8")


def _iter_long_rows(rollout_td: Any, group_map: dict[str, list[str]]):
    """Yield one dict per (env_idx, step, group:agent) tuple from a rollout TD.

    Pulls reward from ``("next", group, "reward")``, action from ``(group, "action")``,
    and broadcasts env-wide done/terminated to per-agent rows.
    """
    # The triple-nested expansion is what makes this a long format; the locals
    # tracked are all the indices and field views needed per (env, step, agent).
    # pylint: disable=too-many-locals
    bs = rollout_td.batch_size
    num_envs, t_steps = bs[0], bs[1]
    done = rollout_td["next", "done"].squeeze(-1)  # [E, T]
    terminated = rollout_td["next", "terminated"].squeeze(-1)  # [E, T]

    for group, agent_names in group_map.items():
        reward = rollout_td["next", group, "reward"].squeeze(-1)  # [E, T, A]
        action = rollout_td[group, "action"]  # [E, T, A, D]
        action_dim = action.shape[-1]
        for env_idx in range(num_envs):
            for step in range(t_steps):
                for agent_idx, agent_name in enumerate(agent_names):
                    row: dict[str, Any] = {
                        "env_idx": env_idx,
                        "step": step,
                        "agent": f"{group}:{agent_name}",
                        "reward": float(reward[env_idx, step, agent_idx]),
                        "done": bool(done[env_idx, step]),
                        "terminated": bool(terminated[env_idx, step]),
                    }
                    for d in range(action_dim):
                        row[f"action_d{d}"] = float(action[env_idx, step, agent_idx, d])
                    yield row
