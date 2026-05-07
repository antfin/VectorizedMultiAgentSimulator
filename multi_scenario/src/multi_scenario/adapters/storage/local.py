"""LocalStorageAdapter — filesystem implementation of the Storage port.

Writes the §3.5.2 layout to disk via JSON for everything we own. Pydantic's
``model_dump_json`` / ``model_validate_json`` handles datetime fields
(Provenance, RunStateRecord) cleanly via built-in ISO-8601 (de)serialization.

Optional artefacts (eval_episodes, report, videos, log, eval_steps_long,
eval_runs) and the cross-run ``runs.csv`` / ``runs.json`` are not in the
Storage Protocol surface and are added on later concrete adapters when each
writer feature lands (F2.7 / F2.10 / F2.10.1 / F2.11 / F5.2 / F5.3 / F5.4 /
F5.8).
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from multi_scenario.domain.models import (
    EvalRunRecord,
    ExperimentConfig,
    ExperimentResult,
    Provenance,
    RunReport,
    RunStateRecord,
)

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


class LocalStorageAdapter:
    """Persists per-run JSON files to the local filesystem under ``run_dir``."""

    name = "fs"

    def save_config(self, run_dir: Path, config: ExperimentConfig) -> None:
        """Write the resolved config to ``input/config.json``."""
        self._write(run_dir / "input" / "config.json", config.model_dump_json(indent=2))

    def save_provenance(self, run_dir: Path, provenance: Provenance) -> None:
        """Write provenance to ``input/provenance.json``."""
        self._write(
            run_dir / "input" / "provenance.json",
            provenance.model_dump_json(indent=2),
        )

    def save_result(self, run_dir: Path, result: ExperimentResult) -> None:
        """Write the final aggregate to ``output/metrics.json``."""
        self._write(run_dir / "output" / "metrics.json", result.model_dump_json(indent=2))

    def save_run_state(self, run_dir: Path, state: RunStateRecord) -> None:
        """Write the run-state record to ``run_state.json`` at the run-folder root."""
        self._write(run_dir / "run_state.json", state.model_dump_json(indent=2))

    def save_report(self, run_dir: Path, report: RunReport) -> None:
        """Write the run-end manifest to ``output/report.json`` (F2.10).

        Off the ``Storage`` Protocol surface on purpose — optional artefact
        per F1.9. Callers (``LocalRunner``) wire this directly against the
        concrete adapter after ``ExperimentService.run`` returns.
        """
        self._write(run_dir / "output" / "report.json", report.model_dump_json(indent=2))

    def save_eval_episodes(self, run_dir: Path, rollout: dict[str, Any]) -> None:
        """Write per-episode raw eval data to ``output/eval_episodes.json`` (F2.10.1).

        Coerces tensor values to plain JSON via ``.tolist()``. Only keys in the
        documented schema (universal: ``episode_returns / episode_lengths /
        episode_collisions``; discovery: ``targets_covered / n_targets``) are
        serialised — unknown keys are silently dropped to keep the file's
        schema stable as new scenarios add fields.

        Off the ``Storage`` Protocol surface (same minimalism rule as
        ``save_report``); ``ExperimentService`` calls it via the optional
        ``eval_episodes_writer`` constructor arg, not through the Protocol.
        """
        payload: dict[str, Any] = {}
        for key in _EVAL_EPISODES_SCHEMA:
            if key not in rollout:
                continue
            value = rollout[key]
            payload[key] = value.tolist() if hasattr(value, "tolist") else value
        self._write(run_dir / "output" / "eval_episodes.json", json.dumps(payload, indent=2))

    def save_eval_run(self, run_dir: Path, record: EvalRunRecord) -> None:
        """Write a re-evaluation record to ``output/eval_runs/<eval_id>.json`` (F5.8).

        Off the ``Storage`` Protocol surface (F1.9 minimalism). The CLI
        ``multi-scenario eval`` is the canonical caller. Multiple eval runs
        coexist as separate files keyed by ``eval_id``.
        """
        self._write(
            run_dir / "output" / "eval_runs" / f"{record.eval_id}.json",
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

    def load_config(self, run_dir: Path) -> ExperimentConfig:
        """Read back ``input/config.json`` as ``ExperimentConfig``."""
        return ExperimentConfig.model_validate_json(
            (run_dir / "input" / "config.json").read_text(encoding="utf-8")
        )

    def load_provenance(self, run_dir: Path) -> Provenance:
        """Read back ``input/provenance.json`` as ``Provenance``."""
        return Provenance.model_validate_json(
            (run_dir / "input" / "provenance.json").read_text(encoding="utf-8")
        )

    def load_result(self, run_dir: Path) -> ExperimentResult:
        """Read back ``output/metrics.json`` as ``ExperimentResult``."""
        return ExperimentResult.model_validate_json(
            (run_dir / "output" / "metrics.json").read_text(encoding="utf-8")
        )

    def load_run_state(self, run_dir: Path) -> RunStateRecord:
        """Read back ``run_state.json`` as ``RunStateRecord``."""
        return RunStateRecord.model_validate_json(
            (run_dir / "run_state.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


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
