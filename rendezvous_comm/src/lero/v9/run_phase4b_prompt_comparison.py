"""v9 Phase-4b — generate several v9 inner prompts and compare them
side-by-side with S3b-local's hand-curated prompt.

For each of N=5 trials we:
  1. Call enumerate_bundle_v9 → get artifacts (inferable_hints, examples,
     feedback_template) for the chosen strategy
  2. Render the FULL inner prompt (system + user) with those slot edits
  3. Save the rendered prompt to disk for human review
  4. Score it on S3b-local-likeness:
       - has 'What you CAN infer' / 'inferable' section
       - bullet count in inferable_hints (target: 7+ matching task_domain)
       - explicit mention of role_differentiation / agent_idx one-hot
       - number of worked python examples
       - at least one example uses one_hot pattern
       - has fairness constraint
       - word count

Then load the S3b-local prompt the same way and compute the same scores
for direct comparison.

Cost: 1 meta-call per trial (no inner gens). 5 trials × 1 call ≈ 90s, $1.50.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from ..config import LLMConfig
from ..llm_client import LLMClient
from ..prompts.loader import PromptLoader
from .meta_strategist import enumerate_bundle_v9


_TASK_OVERRIDES = {
    "n_agents": 4, "n_targets": 4, "agents_per_target": 2,
    "covering_range": 0.25, "lidar_range": 0.35, "max_steps": 400,
    "n_lidar_rays_entities": 15, "n_lidar_rays_agents": 12,
}

_TASK_SUMMARY = (
    "Multi-agent rendezvous in VMAS Discovery. 4 agents must cover "
    "4 targets; each target needs exactly 2 agents simultaneously."
)


def _render_full_prompt(
    base_prompt_version: str,
    slot_overrides: Dict[str, str],
    task_overrides: Dict,
):
    """Render system + user with slot overrides applied. Returns
    {'system': ..., 'user': ..., 'full': system + '\\n\\n' + user}."""
    from ..prompts import loader as _l
    base_dir = Path(_l.__file__).parent / base_prompt_version
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / base_prompt_version
        shutil.copytree(base_dir, new_dir)
        td_src = Path(_l.__file__).parent / "task_domains"
        if td_src.exists():
            shutil.copytree(td_src, Path(tmp) / "task_domains")
        for slot, text in slot_overrides.items():
            (new_dir / f"{slot}.txt").write_text(
                (text or "").rstrip() + "\n"
            )
        orig = _l._PROMPTS_DIR
        try:
            _l._PROMPTS_DIR = Path(tmp)
            loader = PromptLoader(version=base_prompt_version)
            sys_text = loader.render("system.txt", **task_overrides)
            user_text = loader.render(
                "initial_user.txt",
                output_spec_variant="obs_only",
                **task_overrides,
            )
        finally:
            _l._PROMPTS_DIR = orig
    return {
        "system": sys_text,
        "user": user_text,
        "full": sys_text + "\n\n" + user_text,
    }


def _score_prompt(text: str, td_concepts: List[Dict]) -> Dict:
    """Compute S3b-local-likeness scores for a rendered prompt."""
    text_lower = text.lower()

    has_can_infer_header = bool(
        re.search(r"what\s+you\s+can\s+infer", text_lower)
    )
    has_fairness = bool(
        re.search(r"fairness|will raise keyerror|forbidden|do not have", text_lower)
    )

    # Count task_domain concept mentions
    concept_hits = []
    for c in td_concepts:
        phrase = c["concept"].lower()
        idiom = c["idiom"].lower()
        idiom_keywords = re.findall(r"[a-z_]{4,}", idiom)
        phrase_match = phrase in text_lower
        idiom_match = any(k in text_lower for k in idiom_keywords)
        if phrase_match or idiom_match:
            concept_hits.append(c["concept"])

    # Count python worked examples
    py_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    n_examples = len(py_blocks)

    # Role one-hot in at least one example
    role_in_example = any(
        re.search(
            r"(F\.one_hot|one_hot\s*\(|\[:\s*,\s*agent_idx\]\s*=\s*1|"
            r"torch\.zeros\([^)]*n_agents)",
            b,
        )
        for b in py_blocks
    )

    # Explicit role differentiation language
    role_lang = bool(
        re.search(
            r"role\s+(differentiat|specializ|identit|one[-\s]?hot|partition)",
            text_lower,
        )
    )

    # Word count
    words = len(re.findall(r"\b\w+\b", text))

    return {
        "has_can_infer_header": has_can_infer_header,
        "has_fairness": has_fairness,
        "concepts_total": len(td_concepts),
        "concepts_hit": len(concept_hits),
        "concepts_matched": concept_hits,
        "n_python_examples": n_examples,
        "role_in_example": role_in_example,
        "role_differentiation_language": role_lang,
        "word_count": words,
        "char_count": len(text),
    }


def _render_s3b_local_prompt() -> Dict[str, str]:
    """Render the S3b-local prompt with the same task overrides for
    apples-to-apples comparison."""
    loader = PromptLoader(version="v2_fewshot_k2_local")
    sys_text = loader.render("system.txt", **_TASK_OVERRIDES)
    user_text = loader.render(
        "initial_user.txt",
        obs_lidar_agents=(
            '"lidar_agents":     # [batch, '
            f'{_TASK_OVERRIDES["n_lidar_rays_agents"]}] — distance to '
            "nearest OTHER AGENT per ray"
        ),
        **_TASK_OVERRIDES,
    )
    return {
        "system": sys_text,
        "user": user_text,
        "full": sys_text + "\n\n" + user_text,
    }


def main(n_trials: int = 5) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("v9.phase4b")

    out_dir = Path(
        f"results/v9_prompt_lab/phase4b_compare_{time.strftime('%Y%m%d_%H%M')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "v9_trials").mkdir(exist_ok=True)

    # Task domain (for concept matching)
    td_loader = PromptLoader(version="v3_modular_taskdomain")
    td = td_loader.task_domain() or {}
    td_concepts = td.get("inferable_concepts") or []

    # === S3b-local prompt (reference) ===
    log.info("Rendering S3b-local prompt for reference")
    s3b = _render_s3b_local_prompt()
    (out_dir / "s3b_local_prompt.txt").write_text(s3b["full"])
    s3b_score = _score_prompt(s3b["full"], td_concepts)
    log.info("S3b-local: %s", s3b_score)

    # === v9 trials ===
    meta_llm = LLMClient(LLMConfig(
        model="gpt-5.4-mini", temperature=0.8, max_retries=3,
        prompt_version="v3_modular_taskdomain",
    ))
    loader = PromptLoader(version="v3_modular_taskdomain")

    v9_scores: List[Dict] = []
    for i in range(n_trials):
        log.info("=== v9 trial %d/%d ===", i + 1, n_trials)
        try:
            bundle, raw = enumerate_bundle_v9(
                meta_llm, loader, _TASK_SUMMARY,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("trial %d: bundle enum failed: %s", i, e)
            v9_scores.append({"trial": i, "err": str(e)})
            continue

        chosen = bundle.current()
        slot_overrides = {
            "inferable_hints": chosen.artifacts.inferable_hints_text,
            "examples": chosen.artifacts.examples_text,
        }
        rendered = _render_full_prompt(
            "v3_modular_taskdomain", slot_overrides, _TASK_OVERRIDES,
        )

        # Save
        trial_dir = out_dir / "v9_trials" / f"trial_{i:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "system.txt").write_text(rendered["system"])
        (trial_dir / "user.txt").write_text(rendered["user"])
        (trial_dir / "full_prompt.txt").write_text(rendered["full"])
        (trial_dir / "chosen_strategy.json").write_text(
            json.dumps({
                "name": chosen.name,
                "full_solution": chosen.full_solution,
                "lero_codability": chosen.lero_codability,
                "rl_trainability": chosen.rl_trainability,
                "chain_of_thought": {
                    "why_it_works": chosen.chain_of_thought.why_it_works,
                    "what_is_needed": chosen.chain_of_thought.what_is_needed,
                    "failure_modes": chosen.chain_of_thought.failure_modes,
                },
            }, indent=2, default=str)
        )

        score = _score_prompt(rendered["full"], td_concepts)
        score["trial"] = i
        score["chosen_strategy"] = chosen.name
        v9_scores.append(score)
        log.info(
            "  chosen='%s' words=%d concepts=%d/%d examples=%d "
            "role_in_ex=%s role_lang=%s",
            chosen.name, score["word_count"],
            score["concepts_hit"], score["concepts_total"],
            score["n_python_examples"],
            score["role_in_example"], score["role_differentiation_language"],
        )

    # === Side-by-side comparison ===
    valid = [s for s in v9_scores if "err" not in s]
    n = len(valid)
    if n == 0:
        log.error("no valid v9 trials")
        return 1

    avg = lambda k: sum(s[k] for s in valid) / n  # noqa: E731
    cmp_table = [
        ("has_can_infer_header",
         s3b_score["has_can_infer_header"],
         f"{sum(1 for s in valid if s['has_can_infer_header'])}/{n}"),
        ("has_fairness",
         s3b_score["has_fairness"],
         f"{sum(1 for s in valid if s['has_fairness'])}/{n}"),
        ("concepts_hit (of 7)",
         f"{s3b_score['concepts_hit']}/{s3b_score['concepts_total']}",
         f"{avg('concepts_hit'):.1f}/7 avg"),
        ("n_python_examples",
         s3b_score["n_python_examples"],
         f"{avg('n_python_examples'):.1f} avg"),
        ("role_in_example",
         s3b_score["role_in_example"],
         f"{sum(1 for s in valid if s['role_in_example'])}/{n}"),
        ("role_differentiation_language",
         s3b_score["role_differentiation_language"],
         f"{sum(1 for s in valid if s['role_differentiation_language'])}/{n}"),
        ("word_count",
         s3b_score["word_count"],
         f"{int(avg('word_count'))} avg"),
        ("char_count",
         s3b_score["char_count"],
         f"{int(avg('char_count'))} avg"),
    ]

    summary = {
        "n_trials": n_trials,
        "n_valid": n,
        "s3b_local_score": s3b_score,
        "v9_scores": v9_scores,
        "comparison_table": [
            {"metric": k, "s3b_local": s3b_v, "v9": v9_v}
            for k, s3b_v, v9_v in cmp_table
        ],
        "out_dir": str(out_dir),
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print()
    print("=" * 75)
    print(f"v9 vs S3b-local prompt comparison ({n}/{n_trials} valid trials)")
    print("=" * 75)
    print(f"{'metric':<35} {'S3b-local':<20} {'v9':<20}")
    print("-" * 75)
    for k, s3b_v, v9_v in cmp_table:
        print(f"{k:<35} {str(s3b_v):<20} {str(v9_v):<20}")
    print()
    print(f"Saved {n} v9 prompts + S3b-local for diff at:")
    print(f"  {out_dir}")
    print(f"  v9: {out_dir}/v9_trials/trial_*/full_prompt.txt")
    print(f"  s3b: {out_dir}/s3b_local_prompt.txt")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sys.exit(main(n_trials=n))
