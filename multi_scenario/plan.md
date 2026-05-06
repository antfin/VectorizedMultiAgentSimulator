# Multi-Scenario Cooperative Experiment Plan

Selected VMAS scenarios for this study. All four are fully cooperative (shared reward / joint goal) and exercise different coordination modes.

## 1. `discovery` — rendezvous / coverage

- **Setup:** 5 agents, 7 targets in a 2×2 world. Agents have lidar.
- **Task:** A target is "covered" only when **≥ `agents_per_target`** (default 2) agents are simultaneously within `covering_range` (0.25) of it.
- **Reward:** Per-target covering reward (`covering_rew_coeff=1.0`), optional collision penalty, optional time penalty. `shared_reward` toggles team vs per-agent.
- **Key params:** `n_agents`, `n_targets`, `agents_per_target`, `covering_range`, `shared_reward`, `targets_respawn`, `lidar_range`, `dim_c` (communication).
- **Coordination type:** **Multi-agent rendezvous** — agents must converge in groups on each target and dwell there.

## 2. `flocking` — formation control under disturbance

- **Setup:** 2 agents (different sizes) in a drag-free world with a constant **wind** force, controlled via velocity controllers. Horizon 200 steps.
- **Task:** Maintain a desired inter-agent distance (`desired_distance=1`), cruise at `desired_vel=0.5`, and align orientation perpendicular to the wind once a wayline is crossed.
- **Reward (shared):** distance-shaping + velocity-shaping + rotation-shaping (+ optional position/energy).
- **Key params:** `n_agents`, `desired_vel`, `desired_distance`, `wind`, `vel_shaping_factor`, `dist_shaping_factor`, `rot_shaping_factor`, `horizon`.
- **Coordination type:** **Formation flocking** — continuous mutual adjustment, external disturbance compensation.

## 3. `navigation` — goal reaching with optional rendezvous

- **Setup:** N agents (default 4), each with an assigned goal landmark. Bounded or unbounded world.
- **Task:** Each agent reaches its goal. With `agents_with_same_goal > 1`, multiple agents share a goal → rendezvous variant. With `split_goals=True`, goals are split among pairs.
- **Reward:** Distance-shaping per agent (negative distance to goal, scaled by `pos_shaping_factor=1`). Final bonus (`final_reward=0.01`) when **all** agents have arrived. Collision penalty (`agent_collision_penalty=-1`). With `shared_rew=True` the position reward is shared across the team.
- **Key params:** `n_agents`, `shared_rew`, `agents_with_same_goal`, `split_goals`, `pos_shaping_factor`, `final_reward`, `agent_collision_penalty`, `lidar_range`.
- **Coordination type:** **Goal navigation with collision avoidance**; configurable from independent navigation to full team rendezvous.

## 4. `transport` — cooperative heavy-object transport

- **Setup:** 4 agents and 1 movable package (mass = 50, 0.15 × 0.15) plus a goal landmark. Action multiplier reduced to 0.6 to reflect the heavy load.
- **Task:** Push the package to the goal. Episode terminates when the package overlaps the goal.
- **Reward (shared):** Potential-based distance-shaping on **package → goal** distance (`shaping_factor=100`). No extra terminal bonus.
- **Key params:** `n_agents`, `n_packages`, `package_mass`, `package_width`, `package_length`, `shaping_factor`.
- **Coordination type:** **Cooperative transport** — no agent can move the package alone; agents must align forces.

---

## Why these four?

| Scenario     | Coordination signal              | Communication useful? | Physical coupling |
|--------------|----------------------------------|-----------------------|-------------------|
| discovery    | meet at targets in groups        | yes (target IDs)      | none              |
| flocking     | maintain relative pose & speed   | implicit (observed)   | none (wind only)  |
| navigation   | reach goals, avoid collisions    | yes (goal sharing)    | none              |
| transport    | align forces on shared object    | implicit (contact)    | strong (package)  |

Together they span: **rendezvous, formation, navigation, joint manipulation** — four canonical cooperative-MARL regimes.

---

## Next

(Pending user input — experimental design, algorithms, metrics, communication settings, etc.)
