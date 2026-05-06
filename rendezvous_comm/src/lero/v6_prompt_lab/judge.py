"""LLM-as-judge for v6 prompt-lab.

Scores generated `enhance_observation` code on structural similarity
to S3b-local's iter-1 winner, on a 0-10 scale, with a rubric. Used as
a secondary signal alongside the AST analyzer; the analyzer detects
mechanical patterns (cross-source AND-products) and the judge catches
softer structural similarity that AST can't see (e.g. naming
conventions, decision-shaped semantics).

The judge is given the rubric ONLY, not S3b-local's actual code, to
preserve anti-cheat. It scores on the rubric criteria, not on
similarity to a hidden answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..llm_client import LLMClient

_log = logging.getLogger("rendezvous.lero.v6_prompt_lab.judge")


_JUDGE_SYSTEM = """You are a code-quality judge for multi-agent reinforcement learning observation functions. You score Python `enhance_observation` functions on a clear rubric. You do NOT have access to a reference solution; you score against principles only."""


_RUBRIC = """## Scoring rubric (each criterion 0-2, total 0-10)

The function being judged operates on local sensor data for a multi-agent rendezvous task where exactly k=2 agents must simultaneously occupy each target. The agent has only its own pos/vel + LiDAR-style range sensors `lidar_targets` and `lidar_agents`.

### Criterion 1: Cross-source operations (0-2)
Does the code combine information from `lidar_targets` and `lidar_agents` in single expressions (e.g., products, boolean masks, ratios, differences)? Score:
- 0: no cross-source operation
- 1: single cross-source op or only via simple concatenation
- 2: multiple cross-source ops, expressed as decision-shaped features

### Criterion 2: Decision-oriented features (0-2)
Are the returned features expressed in terms of decisions the policy needs to make (stay/move, alone/paired, scout/converge), versus pure descriptive statistics? Score:
- 0: features are entirely descriptive statistics (mean, std, min, etc.)
- 1: a mix of descriptive and decision-shaped features
- 2: features are predominantly decision-shaped (binary indicators, switches, role-conditional values)

### Criterion 3: Structural diversity (0-2)
Does the function include qualitatively different feature families (e.g. proximity / coordination cues / role / motion), or is it dominated by one family? Score:
- 0: all features in one family (e.g. only target proximity)
- 1: 2 families
- 2: 3+ families

### Criterion 4: Implementation quality (0-2)
Is the code vectorized, free of obvious bugs, returns the documented shape, uses only allowed keys? Score:
- 0: bugs, wrong shape, or uses oracle keys
- 1: works but has minor issues (no clamps for division, missing eps, etc.)
- 2: clean and correct

### Criterion 5: Anti-bloat / signal density (0-2)
Are the features informative or are there redundant near-duplicates? Score:
- 0: heavy redundancy or noise dimensions
- 1: some redundancy
- 2: every feature carries distinct information"""


@dataclass
class JudgeResult:
    score_total: int  # 0-10
    score_breakdown: Dict[str, int]  # per-criterion 0-2
    rationale: str
    raw_text: str = ""


def _build_judge_prompt(code: str) -> str:
    return f"""{_RUBRIC}

## Code to score

```python
{code}
```

## Output

Return JSON:
```json
{{
  "scores": {{
    "cross_source_ops": <0|1|2>,
    "decision_oriented": <0|1|2>,
    "structural_diversity": <0|1|2>,
    "implementation_quality": <0|1|2>,
    "signal_density": <0|1|2>
  }},
  "rationale": "<2-3 sentences>"
}}
```"""


def _parse_judge_response(raw: str) -> JudgeResult:
    js = raw.find("{")
    je = raw.rfind("}")
    if js < 0 or je < js:
        raise ValueError("no JSON in judge response")
    blob = raw[js : je + 1]
    data = json.loads(blob)
    scores = data.get("scores", {}) or {}
    breakdown = {
        "cross_source_ops": int(scores.get("cross_source_ops", 0)),
        "decision_oriented": int(scores.get("decision_oriented", 0)),
        "structural_diversity": int(scores.get("structural_diversity", 0)),
        "implementation_quality": int(scores.get("implementation_quality", 0)),
        "signal_density": int(scores.get("signal_density", 0)),
    }
    total = sum(breakdown.values())
    return JudgeResult(
        score_total=total,
        score_breakdown=breakdown,
        rationale=str(data.get("rationale", "")),
        raw_text=raw,
    )


def judge_inner_code(
    code: str,
    judge_llm: LLMClient,
) -> JudgeResult:
    """Score one piece of `enhance_observation` code on the 5-criterion rubric.
    Returns a JudgeResult with total ∈ [0, 10] and per-criterion breakdown.
    """
    prompt = _build_judge_prompt(code)
    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        raw = judge_llm.generate(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            n=1,
        )[0]
        try:
            return _parse_judge_response(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            _log.warning("judge parse failed attempt %d: %s", attempt, e)
    raise ValueError(f"judge failed after 3 attempts: {last_err}")


def judge_batch(
    codes: List[str],
    judge_llm: LLMClient,
) -> List[JudgeResult]:
    """Score a batch of codes; failures yield score 0 placeholder."""
    results: List[JudgeResult] = []
    for i, c in enumerate(codes):
        try:
            r = judge_inner_code(c, judge_llm)
        except Exception as e:  # noqa: BLE001
            _log.warning("judge candidate %d failed: %s", i, e)
            r = JudgeResult(
                score_total=0,
                score_breakdown={
                    k: 0
                    for k in (
                        "cross_source_ops",
                        "decision_oriented",
                        "structural_diversity",
                        "implementation_quality",
                        "signal_density",
                    )
                },
                rationale=f"judge error: {e}",
            )
        results.append(r)
    return results


def avg_judge_score(results: List[JudgeResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score_total for r in results) / len(results)
