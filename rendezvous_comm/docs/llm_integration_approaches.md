# LLM Integration Approaches for Multi-Robot Rendezvous

**Date:** 2026-03-26
**Context:** Exploring how LLMs can improve coordination policies in the Discovery scenario (k=2, 4 agents, 4 targets).

Current best result: GNN GATv2 at 71% SR (ms400). The question is whether LLM knowledge can push further or reduce the need for task relaxations (ms400, cr035).

---

## Approach 1: LLM as Reward Designer

**Effort:** Low | **Impact:** High | **Novelty:** Medium

**Idea:** Use an LLM to generate and iterate on reward functions, replacing manual reward shaping (LP+SR).

Our current reward engineering was manual trial-and-error (base -> +LP -> +SR -> +LP+SR). An LLM can:

- Propose reward components based on the task description ("4 agents must simultaneously cover targets")
- Evaluate training curves and propose adjustments ("M1 is plateauing at 4%, try adding a proximity bonus that scales with the number of nearby agents")
- Generate multi-layered rewards: global (team success), local (per-agent progress), adaptive (curriculum-based)

The **LEHR framework** (LLM-Driven Evolutionary Hybrid Rewards) does exactly this -- LLM generates hybrid reward functions, evaluates them via RL training, and evolves better ones. This could replace the manual ablation cycle entirely.

**Fit for our work:** Direct drop-in. We already have the training pipeline. The LLM generates Python reward functions, we train with MAPPO, measure M1-M9, feed results back.

**Key papers:**

- [LEHR: LLM-Driven Evolutionary Hybrid Rewards for MARL](https://link.springer.com/chapter/10.1007/978-981-96-7352-0_24)
- [LLM-Guided Incentive Aware Reward Design for Cooperative MARL](https://arxiv.org/html/2603.24324)
- [The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination](https://arxiv.org/html/2601.08237v1)

---

## Approach 2: LLM as High-Level Planner (Hierarchical)

**Effort:** Medium | **Impact:** High | **Novelty:** High

**Idea:** LLM assigns targets to agent pairs; RL handles low-level navigation.

Two-level architecture:

- **High level (LLM):** Given agent positions and target positions, the LLM outputs an assignment plan: "Agent 0+1 -> Target A, Agent 2+3 -> Target B, then reassign to C and D"
- **Low level (MAPPO):** Agents execute the plan using learned navigation policies

This is the COHERENT/CoMuRoS approach. The LLM handles the combinatorial assignment problem (which pair goes where), while RL handles the continuous control. This directly addresses the k=2 bottleneck -- the LLM solves the coordination part, RL solves the movement part.

**Fit for our work:** Medium effort. We'd need to add a planning step before each episode (or every N steps) that queries the LLM for assignment. The RL policy would receive the target assignment as part of its observation.

**Why this is interesting for the thesis:** Comparing "LLM explicit assignment" vs "GNN implicit coordination" (71% SR) would be a strong contribution. If the LLM planner matches or beats GNN, it proves that the coordination bottleneck is an assignment problem. If GNN wins, it proves that implicit spatial awareness is more valuable than explicit planning.

**Key papers:**

- [COHERENT: Collaboration of Heterogeneous Multi-Robot System with LLMs](https://arxiv.org/html/2409.15146v3)
- [LLM-Based Generalizable Hierarchical Task Planning for Heterogeneous Robot Teams](https://arxiv.org/abs/2511.22354)
- [Multi-Agent Systems for Robotic Autonomy with LLMs (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025W/MEIS/papers/Chen_Multi-Agent_Systems_for_Robotic_Autonomy_with_LLMs_CVPRW_2025_paper.pdf)

---

## Approach 3: LLM-Generated Communication Protocols

**Effort:** Medium | **Impact:** Medium | **Novelty:** Medium

**Idea:** Instead of learning communication from scratch (dim_c) or via architecture (GNN), use the LLM to design the communication protocol.

The LLM could:

- Define what agents should communicate: "Send your distance to nearest uncovered target and your heading direction"
- Define when to communicate: "Only broadcast when within 0.5 of a target"
- Define how to interpret messages: "If a neighbor reports being near a target, head toward that target"

This bridges ER2 (engineered comm) and ER3 (learned comm) -- the protocol is structured by LLM knowledge but executed by the RL policy.

**Fit for our work:** This is essentially our E1 (static LLM infra) experiment extended. The LLM designs the comm schema, agents train with it. Creates a new point on the implicit-to-explicit communication spectrum.

**Key papers:**

- [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://arxiv.org/abs/2405.11106)
- [Leveraging LLMs for Effective and Explainable MARL](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1501.pdf)

---

## Approach 4: LLM as Online Coordinator

**Effort:** High | **Impact:** High | **Novelty:** High

**Idea:** LLM observes the environment state at each step (or every N steps) and sends natural language instructions to agents.

Example: At step 50, LLM sees positions and says: "Agent 0: target at (0.3, -0.5) is uncovered, head there. Agent 2: join Agent 0 at that target."

The **YOLO-MARL** approach ("You Only LLM Once") addresses the latency problem -- query the LLM once at episode start to generate a full coordination strategy, then execute with RL. Avoids per-step LLM calls (too slow for 400 steps at ~100ms/call).

**Variants:**

- **Once per episode:** LLM sees initial positions, outputs full plan. Cheapest but can't adapt.
- **Every N steps:** LLM re-plans periodically (e.g., every 50 steps). Balances cost and adaptiveness.
- **Event-triggered:** LLM re-plans only when significant state changes occur (target covered, agent stuck).

**Fit for our work:** High impact but complex. Requires encoding observations as text, querying the LLM, and translating instructions back to actions or sub-goals. Latency and cost are the main challenges.

**Key papers:**

- [YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning](https://arxiv.org/html/2410.03997v2)
- [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/html/2508.04652v1)

---

## Approach 5: LLM for Curriculum Design

**Effort:** Low | **Impact:** Medium | **Novelty:** Low

**Idea:** LLM designs the training curriculum -- which experiments to run, in what order, with what parameters.

Given our results so far, an LLM could suggest: "Start with k=1 ms200 to learn navigation, then fine-tune with k=2 ms400 to learn coordination, then reduce to ms200 with communication." This is curriculum learning designed by LLM analysis of the experiment history.

**Fit for our work:** Very natural extension. We already have the full experiment pipeline. The LLM becomes a "research assistant" that proposes the next experiment based on M1-M9 trends.

---

## Recommendation for This PhD

Given our existing infrastructure and results, priority order:

| Priority | Approach | Why |
| --- | --- | --- |
| 1 | **LLM planner (Approach 2)** | Highest novelty, directly addresses k=2 assignment problem. "LLM explicit assignment" vs "GNN implicit coordination" is a compelling comparison. |
| 2 | **LLM reward designer (Approach 1)** | Easiest to implement, replaces manual ablation cycle. LEHR paper provides clear methodology. |
| 3 | **LLM comm protocol (Approach 3)** | Natural bridge between ER2 and ER3 results. Creates new point on the communication spectrum. |
| 4 | **LLM online coordinator (Approach 4)** | Most ambitious. Best saved for after establishing baselines with approaches 1-3. |
| 5 | **LLM curriculum (Approach 5)** | Nice-to-have but low novelty. Can be done informally without formal experimentation. |

---

## How These Map to Existing Experiment Rounds

| Approach | Maps to | Comparison baseline |
| --- | --- | --- |
| Reward designer | New ER1 variant | ER1 manual LP+SR (4% -> 40.5%) |
| LLM planner | New ER (e.g., ER5) | ER3 GNN (71%), ER2 proximity (53%) |
| LLM comm protocol | Extension of E1 | ER2 dim_c (4.5%), ER3 GNN (71%) |
| Online coordinator | New ER (e.g., ER6) | All current results |
| Curriculum | Meta-level, not an ER | Improves any ER |

---

## References

- [LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions](https://arxiv.org/abs/2405.11106)
- [LLM Collaboration With Multi-Agent Reinforcement Learning](https://arxiv.org/html/2508.04652v1)
- [The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination](https://arxiv.org/html/2601.08237v1)
- [LEHR: LLM-Driven Evolutionary Hybrid Rewards for MARL](https://link.springer.com/chapter/10.1007/978-981-96-7352-0_24)
- [LLM-Guided Incentive Aware Reward Design for Cooperative MARL](https://arxiv.org/html/2603.24324)
- [YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning](https://arxiv.org/html/2410.03997v2)
- [COHERENT: Collaboration of Heterogeneous Multi-Robot System with LLMs](https://arxiv.org/html/2409.15146v3)
- [LLM-Based Generalizable Hierarchical Task Planning for Heterogeneous Robot Teams](https://arxiv.org/abs/2511.22354)
- [Leveraging LLMs for Effective and Explainable MARL](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1501.pdf)
- [Multi-Agent Systems for Robotic Autonomy with LLMs (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025W/MEIS/papers/Chen_Multi-Agent_Systems_for_Robotic_Autonomy_with_LLMs_CVPRW_2025_paper.pdf)
- [AAAI 2026 Bridge Program on LLM-Based Multi-Agent Collaboration](https://multiagents.org/2026/)
- [LLMs for Multi-Agent Cooperation (survey)](https://xue-guang.com/post/llm-marl/)
