"""F8.2.E — registry for BenchMARL scalar CSV explanations.

BenchMARL writes per-iter / per-eval scalar CSVs under
``output/benchmarl/<exp>/.../scalars/`` with names that follow a convention
(``<bucket>_<key>_<aggregate>``). Useful but opaque to a first-time reader.

This registry maps every scalar name produced by a typical Discovery+MAPPO
run to a 1-line hint shown next to the BenchMARL-scalar-selector option on
the Run Detail page.

Falls back to a generated hint (parsed from the convention) when the exact
filename isn't in the registry — so a CSV from a different scenario / algo
still gets some explanation rather than nothing.
"""

import re

# Curated hints for the names a typical Discovery+MAPPO run produces.
# These are also the names the F7.3.1 ``❓`` selector hover surfaces.
_CURATED: dict[str, str] = {
    # Collection (training) episode rewards
    "collection_reward_episode_reward_mean": (
        "Mean training-episode reward (averaged over the iter's batch). "
        "Climbs as the policy improves on the data it's currently exploring."
    ),
    "collection_reward_episode_reward_max": (
        "Max training-episode reward in the batch — best-case behaviour the "
        "policy currently produces."
    ),
    "collection_reward_episode_reward_min": (
        "Min training-episode reward — worst-case baseline; gap to mean is a "
        "rough variance proxy."
    ),
    "collection_reward_reward_mean": (
        "Mean per-step reward in the training batch (not per-episode). Useful "
        "when episodes are long and reward is dense."
    ),
    "collection_reward_reward_max": "Max per-step reward in the batch.",
    "collection_reward_reward_min": "Min per-step reward in the batch.",
    # Per-agent collection rewards (multi-agent decomposition)
    "collection_agents_reward_episode_reward_mean": (
        "Mean per-agent training-episode reward. With shared_reward=True this "
        "equals the team reward divided by n_agents; with per-agent rewards it "
        "shows individual contributions."
    ),
    "collection_agents_reward_episode_reward_max": (
        "Max per-agent training-episode reward — the agent currently winning."
    ),
    "collection_agents_reward_episode_reward_min": (
        "Min per-agent training-episode reward — the agent currently lagging. "
        "Persistent gap to max suggests free-rider dynamics."
    ),
    # Discovery-specific info channels
    "collection_agents_info_collision_rew": (
        "Mean per-step collision penalty in the training batch (typically "
        "negative). Drops toward zero as agents learn to avoid each other."
    ),
    "collection_agents_info_covering_reward": (
        "Mean per-step covering reward (Discovery: agents getting close to "
        "uncovered targets). Climbs when agents start finding targets."
    ),
    "collection_agents_info_targets_covered": (
        "Per-step indicator of whether each agent is currently covering an "
        "uncovered target. Dense signal that precedes M1's coarser per-episode "
        "success indicator."
    ),
    "collection_agents_info_pos": (
        "Mean agent x-coordinate at each step (sanity check; should NOT drift "
        "outside the world bounds)."
    ),
    # Eval rewards (post-training)
    "eval_reward_episode_reward_mean": (
        "Mean reward across the N eval episodes at this iter. The cleanest "
        "signal of policy quality — independent of the training data — and "
        "the curve to watch for late-training regression (ER1 dry-run "
        "peaked at iter 125 then dropped)."
    ),
    "eval_reward_episode_reward_max": "Best eval-episode reward at this iter.",
    "eval_reward_episode_reward_min": "Worst eval-episode reward at this iter.",
    "eval_reward_episode_len_mean": (
        "Mean steps-to-termination across eval episodes. Lower is better when "
        "agents actually solve (= they solve faster); equals max_steps when "
        "they don't solve at all."
    ),
    "eval_agents_reward_episode_reward_mean": (
        "Per-agent eval-episode reward, averaged. Same shape as collection_*; "
        "informative when reward is per-agent."
    ),
    "eval_agents_reward_episode_reward_max": "Max per-agent eval-episode reward.",
    "eval_agents_reward_episode_reward_min": "Min per-agent eval-episode reward.",
    # Counters
    "counters_iter": "Iteration counter; trivial sanity check.",
    "counters_current_frames": "Frames consumed in the most recent iter.",
    "counters_total_frames": (
        "Cumulative frames consumed since training started. Multi-seed runs "
        "compare progress at fixed total_frames, not iter, since batch size "
        "can vary across configs."
    ),
    # Wall-time timers
    "timers_iteration_time": "Wall-clock seconds for the most recent iter.",
    "timers_collection_time": "Wall-clock seconds spent rolling out training data.",
    "timers_evaluation_time": "Wall-clock seconds spent on eval rollouts.",
}


_PARSE = re.compile(r"^(?P<bucket>collection|eval|counters|timers)_" r"(?P<rest>.+)$")
_AGGREGATES = {
    "mean": "mean over",
    "max": "max over",
    "min": "min over",
}


def _generated_hint(filename: str) -> str:
    """Best-effort hint synthesised from the BenchMARL naming convention.

    Used when ``filename`` isn't in the curated registry so a third-party
    scenario / algo still surfaces a non-empty hint.
    """
    stem = filename.removesuffix(".csv")
    match = _PARSE.match(stem)
    if not match:
        return f"BenchMARL scalar series: {stem}."
    bucket = match.group("bucket")
    rest = match.group("rest")
    bucket_label = {
        "collection": "training",
        "eval": "evaluation",
        "counters": "counter",
        "timers": "wall-time timer",
    }.get(bucket, bucket)
    # Last token = aggregate (mean/max/min) when present
    parts = rest.split("_")
    if parts[-1] in _AGGREGATES:
        agg = _AGGREGATES[parts[-1]]
        signal = " ".join(parts[:-1])
        return f"BenchMARL {bucket_label} series — {agg} ``{signal}``."
    return f"BenchMARL {bucket_label} series — ``{rest.replace('_', ' ')}``."


def scalar_hint(filename: str) -> str:
    """Return the user-facing hint for a BenchMARL scalar CSV filename.

    Strips ``.csv`` if present, looks up the curated registry first, falls
    back to a synthesised hint based on the BenchMARL naming convention.
    """
    stem = filename.removesuffix(".csv")
    return _CURATED.get(stem) or _generated_hint(filename)


def all_curated_scalar_names() -> list[str]:
    """Stable-ordered list of every scalar name with a curated hint."""
    return list(_CURATED.keys())
