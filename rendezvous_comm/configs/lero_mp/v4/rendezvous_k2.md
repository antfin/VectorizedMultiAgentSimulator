# Task: Multi-Robot Rendezvous (k=2)

## Environment

`n_agents=4` agents must cover `n_targets=4` targets in a 2D arena
[-1, 1] × [-1, 1]. A target is COVERED when **k=2 agents are within
covering_range=0.35 distance of the same target SIMULTANEOUSLY** for
one step. This is a RENDEZVOUS task — coordination is mandatory.

- Episode length: `max_steps=200`
- Lidar range: `0.35`
- Communication channel: `dim_c=0` (NO comm)
- Lidar rays: 15 toward entities (targets), 12 toward agents
- Targets do not respawn

## Available state (LOCAL only — fairness whitelist)

The `enhance_observation` function receives a `scenario_state` dict
containing ONLY:

```python
{
    "agent_pos":   torch.Tensor [B, 2]   # this agent's own position
    "agent_vel":   torch.Tensor [B, 2]   # this agent's own velocity
    "agent_idx":   int                   # 0..n_agents-1
    "lidar_targets":  torch.Tensor [B, 15]  # ray distances to targets
    "lidar_agents":   torch.Tensor [B, 12]  # ray distances to agents
    "n_agents":           int            # 4
    "n_targets":          int            # 4
    "covering_range":     float          # 0.35
    "agents_per_target_required":  int   # 2
}
```

**Forbidden** (oracle keys, NOT in the dict): `agents_pos`,
`targets_pos`, `agents_targets_dists`, `covered_targets`,
`agents_per_target`, `all_time_covered`. Reading these raises
FairnessViolation.

## Success metric (M1)

`M1 = fraction of episodes where all 4 targets are simultaneously
covered within max_steps=200`.

Secondary: `M6 = mean fraction of targets covered at any point during
the episode` (lower-variance signal at low M1).

## Initial reward (hand-crafted, ER1-style)

- `+covering_rew_coeff=1.0` per target this agent helps cover
  (counted once per coverage event)
- `agent_collision_penalty = -0.01` per agent-agent collision
- `time_penalty = -0.01` per step (encourages fast completion)
- `shared_reward=True` — covering reward is divided across the team

The hand-crafted reward is **non-exploitable** (bounded, no
accumulation traps). It alone gets ER1 to ~27% M1 at 10M frames at
this task config. The challenge is OBSERVATION quality.

## Known failure modes

1. **Ships passing in the night**: agents arrive at the same target
   sequentially (not simultaneously) — agent A arrives, sees nothing,
   moves on; agent B arrives but A has left. No coverage event.
2. **Anti-crowding learned avoidance**: with `agent_collision_penalty`,
   agents may learn to stay far from teammates → never rendezvous.
3. **Reward gaming at long training**: even with hand-crafted reward,
   if the LLM-generated observation features have surprising
   correlations with reward magnitude, the policy may exploit them at
   10M training and degrade end-of-training M1.

## Optimization target

**Stable end-of-training M1 at 10M frames.** NOT intermediate peak at
2-5M that collapses later. The stability_score formula explicitly
penalizes peak-then-collapse trajectories.

## What you (the meta-LLM) should produce

1. A **BootstrapCard** capturing your understanding of the task
   (assumptions, anticipated failure modes, named features you'd
   propose at the abstraction level above).
2. Specifically: `proposed_initial_obs_features` should be NAMED
   coordination signals (like `hold_signal`), NOT raw arithmetic.
3. `proposed_initial_reward_components` can be empty if the
   hand-crafted reward looks safe. If you want to add shaping,
   propose ONE bounded, non-accumulating term.
4. `fairness_audit` confirms which keys you'd use and why none of
   them are forbidden.
