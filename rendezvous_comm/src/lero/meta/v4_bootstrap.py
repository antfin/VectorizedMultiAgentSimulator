"""LERO-MP v4 Phase 0 — bootstrap from a problem description.md.

The user writes a ~1-page problem description; the meta-LLM reads it
and emits:

  1. ``BootstrapCard`` — typed structured output of the LLM's task
     understanding (assumptions, anticipated failure modes, proposed
     features).
  2. ``bootstrap_thoughts.md`` — free-text reasoning the human can
     audit BEFORE any RL training starts.
  3. A composable initial prompt directory (``prompts/v4_bootstrap_<hash>/``)
     that the inner LLM will consume — pre-filled with the high-level
     framing the LLM derived.

Caching: keyed by sha256(description + meta_model + temperature),
stored per-experiment under ``<output_dir>/bootstrap_cache/``. A run
reuses a cached bootstrap iff the description hash matches; any change
to the description triggers a fresh meta-LLM call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from ..llm_client import LLMClient
from .v4_schemas import BootstrapCard

_log = logging.getLogger("rendezvous.lero.mp.v4.bootstrap")


# ── Cache key ───────────────────────────────────────────────────


def _cache_key(description: str, meta_model: str, meta_temp: float) -> str:
    payload = json.dumps(
        {
            "description": description,
            "meta_model": meta_model,
            "meta_temperature": meta_temp,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ── Prompt builder ──────────────────────────────────────────────


_BOOTSTRAP_SYSTEM = (
    "You are an expert RL researcher. You read task descriptions and "
    "produce: (a) a high-level decomposition of why the task is hard, "
    "(b) named coordination features the policy could use, and (c) a "
    "structured BootstrapCard. Always think at the abstraction level of "
    "named decisions rather than raw arithmetic. Prefer simple, named "
    "features over complex feature ensembles."
)


def _build_bootstrap_prompt(description: str) -> str:
    return f"""[TASK DESCRIPTION FROM HUMAN]
{description.strip()}

[YOUR JOB]
Read the description carefully. Then think out loud about:
  1. What makes this task hard? (Consider multi-agent coordination,
     reward sparsity, observability constraints.)
  2. What failure modes is the policy likely to encounter at 10M
     frames of training?
  3. At the abstraction level of NAMED features, what should the
     policy be able to perceive? Think about what DECISIONS the
     policy needs to make, then propose features that pre-compute
     those decisions from local sensors. Avoid raw sensor dumps —
     prefer features that combine sensors into actionable signals.
     Name each feature with a short identifier.
  4. What reward components would be safe (non-exploitable)? If the
     description provides a hand-crafted reward that already covers
     the success metric, you can recommend keeping it.
  5. Does the proposed feature set respect the fairness constraint
     (only locally observable / allowed keys)?

After thinking, output a structured JSON BootstrapCard with these fields:
  - task_summary
  - success_metric_understanding
  - key_difficulty
  - failure_modes_anticipated      (list, 3–5 items)
  - high_level_strategies_considered  (list, 2–8 items, ranked best-first)
  - proposed_initial_obs_features  (list of NAMED features)
  - proposed_initial_reward_components  (list, may be empty)
  - fairness_audit                 (string)
  - assumptions                    (list of items the human should verify)

[OUTPUT FORMAT]
Reply with EXACTLY two sections, in order:

### THOUGHTS
<free-text reasoning, 200–600 words>

### BOOTSTRAP_CARD
```json
{{
  ...the JSON object above...
}}
```
"""


# ── Result container ────────────────────────────────────────────


class BootstrapResult:
    """In-memory bundle returned by ``bootstrap_from_description``."""

    def __init__(
        self,
        card: BootstrapCard,
        thoughts: str,
        bootstrap_dir: Path,
        cache_hit: bool,
        elapsed_seconds: float,
    ):
        self.card = card
        self.thoughts = thoughts
        self.bootstrap_dir = bootstrap_dir
        self.cache_hit = cache_hit
        self.elapsed_seconds = elapsed_seconds


# ── Output materialization ──────────────────────────────────────


def _materialize_bootstrap_prompt(
    template_dir: Path,  # base v2_fewshot_modular_v2 to copy structure from
    output_dir: Path,
    card: BootstrapCard,
) -> None:
    """Copy the base modular template structure to output_dir, then
    populate guidance_observation / guidance_shared with the LLM's
    proposed features. Reward stays empty by default; the strategist
    can fill it later."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(template_dir, output_dir)

    obs_lines: list[str] = []
    if card.proposed_initial_obs_features:
        obs_lines.append(
            "Augment the local observation with the following NAMED features. "
            "Implement them in pure tensor ops over the ALLOWED state keys "
            "(no oracle access)."
        )
        obs_lines.append("")
        for feat in card.proposed_initial_obs_features:
            obs_lines.append(f"- {feat.strip()}")
        obs_lines.append("")
        obs_lines.append(
            "When implementing, prefer correctness and simplicity over "
            "feature count. Stable end-of-training M1 matters more than "
            "intermediate peaks."
        )
    obs_text = "\n".join(obs_lines).strip() + "\n" if obs_lines else ""
    (output_dir / "guidance_observation.txt").write_text(obs_text)

    rew_lines: list[str] = []
    if card.proposed_initial_reward_components:
        rew_lines.append(
            "Reward components the LLM proposed during bootstrap (these "
            "are SUGGESTIONS — the strategist may opt to keep the "
            "hand-crafted reward instead):"
        )
        rew_lines.append("")
        for comp in card.proposed_initial_reward_components:
            rew_lines.append(f"- {comp.strip()}")
        rew_lines.append("")
        rew_lines.append(
            "Any reward component MUST be bounded and non-accumulating. "
            "Long training will exploit any unbounded shaping."
        )
    rew_text = "\n".join(rew_lines).strip() + "\n" if rew_lines else ""
    (output_dir / "guidance_reward.txt").write_text(rew_text)

    sh_lines = [
        "TASK PRIORITY: produce STABLE end-of-training behavior at 10M frames.",
        "Intermediate peaks at 2-5M that collapse later are NOT the goal.",
        "Prefer simple, robust features over complex ensembles.",
        "",
    ]
    if card.failure_modes_anticipated:
        sh_lines.append("Anticipated failure modes — design defensively against:")
        for fm in card.failure_modes_anticipated:
            sh_lines.append(f"- {fm.strip()}")
    (output_dir / "guidance_shared.txt").write_text("\n".join(sh_lines).strip() + "\n")


# ── Output parsing ──────────────────────────────────────────────


def _parse_response(raw: str) -> tuple[str, BootstrapCard]:
    """Split ``### THOUGHTS`` and ``### BOOTSTRAP_CARD`` sections."""
    if "### THOUGHTS" not in raw or "### BOOTSTRAP_CARD" not in raw:
        raise ValueError(
            "Bootstrap response missing required section headers "
            "'### THOUGHTS' / '### BOOTSTRAP_CARD'."
        )
    thoughts_part, card_part = raw.split("### BOOTSTRAP_CARD", 1)
    thoughts_part = thoughts_part.split("### THOUGHTS", 1)[1].strip()

    # Find JSON block in card_part
    json_start = card_part.find("{")
    json_end = card_part.rfind("}")
    if json_start < 0 or json_end < json_start:
        raise ValueError(
            "Bootstrap response BOOTSTRAP_CARD section did not contain "
            "valid JSON ({...})."
        )
    json_text = card_part[json_start : json_end + 1]
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"BootstrapCard JSON parse failed: {e}. Body: {json_text[:200]!r}"
        )
    return thoughts_part, BootstrapCard.model_validate(data)


# ── Main entry point ────────────────────────────────────────────


def bootstrap_from_description(
    description_path: Path,
    meta_llm: LLMClient,
    output_dir: Path,
    base_prompt_dir: Path,
    cache_dir: Optional[Path] = None,
) -> BootstrapResult:
    """Read description.md, call meta-LLM, materialize bootstrap prompt.

    Args:
        description_path: ``configs/lero_mp/v4/<task>.md``
        meta_llm: pre-built LLMClient for the meta-LLM
        output_dir: per-run directory; bootstrap artifacts go in
            ``<output_dir>/bootstrap/``
        base_prompt_dir: the modular template to copy structure from
            (typically ``src/lero/prompts/v2_fewshot_modular_v2/``)
        cache_dir: per-experiment cache root. Defaults to
            ``<output_dir>/bootstrap_cache/``.

    Returns:
        BootstrapResult with the parsed card, thoughts, and the path
        to the materialized prompt directory.

    Idempotent on identical (description content, meta_model,
    meta_temperature). Any change re-invokes the LLM.
    """
    description_path = Path(description_path)
    description = description_path.read_text(encoding="utf-8")
    meta_temp = meta_llm.config.temperature
    meta_model = meta_llm.config.model
    key = _cache_key(description, meta_model, meta_temp)

    cache_dir = cache_dir or (output_dir / "bootstrap_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_card_path = cache_dir / f"{key}_card.json"
    cache_thoughts_path = cache_dir / f"{key}_thoughts.md"

    bootstrap_dir = output_dir / "bootstrap"
    bootstrap_prompt_dir = bootstrap_dir / "prompts" / f"v4_bootstrap_{key}"

    t0 = time.monotonic()

    if cache_card_path.exists() and cache_thoughts_path.exists():
        _log.info(
            "v4 bootstrap CACHE HIT (key=%s key=%s) — skipping meta-LLM call",
            key,
            meta_model,
        )
        card = BootstrapCard.model_validate_json(
            cache_card_path.read_text(),
        )
        thoughts = cache_thoughts_path.read_text()
        cache_hit = True
    else:
        _log.info(
            "v4 bootstrap CACHE MISS — calling meta-LLM (model=%s)",
            meta_model,
        )
        prompt = _build_bootstrap_prompt(description)
        raw = meta_llm.generate(
            [
                {"role": "system", "content": _BOOTSTRAP_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        thoughts, card = _parse_response(raw)
        cache_card_path.write_text(card.model_dump_json(indent=2))
        cache_thoughts_path.write_text(thoughts)
        cache_hit = False

    # Always (re)materialize the bootstrap prompt directory and the
    # human-readable thoughts file in the run output dir so the run
    # is self-contained.
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    (bootstrap_dir / "bootstrap_card.json").write_text(
        card.model_dump_json(indent=2),
    )
    (bootstrap_dir / "bootstrap_thoughts.md").write_text(thoughts)
    _materialize_bootstrap_prompt(
        template_dir=base_prompt_dir,
        output_dir=bootstrap_prompt_dir,
        card=card,
    )

    elapsed = time.monotonic() - t0
    _log.info(
        "v4 bootstrap %s in %.1fs — %d obs features, %d reward components",
        "REUSED CACHE" if cache_hit else "GENERATED",
        elapsed,
        len(card.proposed_initial_obs_features),
        len(card.proposed_initial_reward_components),
    )
    return BootstrapResult(
        card=card,
        thoughts=thoughts,
        bootstrap_dir=bootstrap_prompt_dir,
        cache_hit=cache_hit,
        elapsed_seconds=elapsed,
    )
