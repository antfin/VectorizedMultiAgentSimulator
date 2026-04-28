"""Per-strategy prompt composition (LERO-MP v4).

Each StrategyV4 maps to a unique prompt directory built by overlaying
``strategy.slot_edits`` onto the bootstrap prompt directory. The
composed prompt becomes the inner LLM's prompt template for the one
candidate generated under that strategy.

When ``revert_to_baseline_reward=True``, ``guidance_reward.txt`` is
forcibly emptied so the inner LLM is not nudged to invent reward code
even if ``evolve_reward=true`` at the LeroLoop level.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from .v4_schemas import StrategyV4

_log = logging.getLogger("rendezvous.lero.mp.v4.composer")


# Slot files we accept edits to. Anything else is rejected.
_VALID_SLOT_FILES = {
    "guidance_observation",
    "guidance_reward",
    "guidance_shared",
}


def compose_prompt_for_strategy(
    base_prompt_dir: Path,
    strategy: StrategyV4,
    output_root: Path,
    candidate_id: str,
    prior_winner_code: Optional[str] = None,
) -> Path:
    """Materialize a strategy-specific prompt directory.

    Steps:
      1. Copy ``base_prompt_dir`` → ``output_root/<candidate_id>/``
      2. For each (slot_name, text) in strategy.slot_edits, overwrite
         the corresponding file (e.g. guidance_observation.txt).
      3. If ``strategy.revert_to_baseline_reward`` is True, force
         guidance_reward.txt to empty.
      4. v4.1 Change E — if ``prior_winner_code`` is provided, append
         a "[PRIOR ROUND WINNER]" block to ``guidance_shared.txt``.
         The inner LLM sees this as cross-round context — it can
         build on what worked instead of redeveloping from scratch.

    Returns the path to the materialized directory. The caller can
    instantiate ``PromptLoader(version=str(returned_path.name))`` if
    the parent dir is on the prompts root, or pass the directory
    directly to a custom loader.
    """
    base_prompt_dir = Path(base_prompt_dir)
    output_root = Path(output_root)
    if not base_prompt_dir.exists():
        raise FileNotFoundError(
            f"base_prompt_dir does not exist: {base_prompt_dir}"
        )

    target = output_root / candidate_id
    if target.exists():
        shutil.rmtree(target)
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_prompt_dir, target)

    for slot_name, new_text in (strategy.slot_edits or {}).items():
        if slot_name not in _VALID_SLOT_FILES:
            raise ValueError(
                f"Strategy {strategy.strategy_id} edited unknown slot "
                f"'{slot_name}'. Allowed: {sorted(_VALID_SLOT_FILES)}."
            )
        slot_path = target / f"{slot_name}.txt"
        normalised = (new_text or "").rstrip() + "\n"
        slot_path.write_text(normalised)

    if strategy.revert_to_baseline_reward:
        # Force-empty regardless of whether slot_edits also touched it.
        (target / "guidance_reward.txt").write_text("")
        _log.info(
            "Strategy %s reverted to baseline reward "
            "(guidance_reward.txt cleared).",
            strategy.strategy_id,
        )

    # v4.1 Change E — append prior-round winning code to guidance_shared
    # so the inner LLM sees cross-round context.
    if prior_winner_code:
        shared_path = target / "guidance_shared.txt"
        existing = shared_path.read_text() if shared_path.exists() else ""
        appended = (
            existing.rstrip()
            + "\n\n[PRIOR ROUND WINNER — cross-round reference]\n"
            + prior_winner_code.rstrip()
            + "\n"
        )
        shared_path.write_text(appended)
        _log.info(
            "Strategy %s prompt augmented with prior-round winner code "
            "(%d chars).",
            strategy.strategy_id, len(prior_winner_code),
        )

    _log.info(
        "Composed prompt for %s at %s (%d slot edits, revert_reward=%s)",
        strategy.strategy_id, target,
        len(strategy.slot_edits or {}),
        strategy.revert_to_baseline_reward,
    )
    return target


def applied_slot_summary(strategy: StrategyV4) -> List[str]:
    """Human-readable summary of what this strategy modifies — used
    in logs and round-summary text."""
    lines = []
    if strategy.target_domain:
        lines.append(f"target_domain={strategy.target_domain}")
    if strategy.revert_to_baseline_reward:
        lines.append("revert_to_baseline_reward=True")
    if strategy.slot_edits:
        slots = ", ".join(strategy.slot_edits.keys())
        lines.append(f"slot_edits={{{slots}}}")
    return lines
