# Communication Architecture

## VMAS Built-in Communication

VMAS has first-class communication support in `vmas/simulator/core.py`:

### Core primitives
- `World(dim_c=N)` — N-dimensional continuous communication channel
- `Agent(silent=False)` — agent can send messages
- `Agent(c_noise=0.1)` — add noise to messages
- `agent.action.c` — communication action (what agent sends), shape `[batch, dim_c]`
- `agent.state.c` — communication state (visible to others), shape `[batch, dim_c]`
- Each step: `agent.state.c = agent.action.c` (if not silent)

### Action space
Physical actions + comm floats are combined:
- Continuous: `[move_x, move_y, msg_0, ..., msg_N]`, comm clamped to [0,1]
- Discrete: separate discrete action per physical dim + one-hot communication
- Action processing in `environment.py` lines 616-750

### Observation access
In scenario's `observation()`, agents read `other.state.c` for received messages.
Broadcast model: all non-silent agents' messages visible to all.

### Discovery scenario specifically
- `_comms_range = _lidar_range` (line 46) — renders comm links between nearby agents
- Currently visualization only — does NOT use dim_c for actual message passing
- Needs modification to enable dim_c and include other.state.c in observations

### Working MPE examples
- `simple_speaker_listener.py` — dim_c=3, one speaker → one listener
- `simple_reference.py` — dim_c=10, both agents communicate
- `simple_world_comm.py` — dim_c=4, leader broadcasts to team
- `simple_crypto.py` — dim_c=4, encrypted message passing

## Step Cycle with Communication

```
Step t:
  1. Agent sees: own_obs + received_messages from step t-1
  2. Policy outputs: [physical_action, communication_action] (simultaneous)
  3. VMAS: applies physics, copies comm action → agent.state.c
Step t+1:
  1. Agent sees: own_obs + messages just sent at step t
```
- Movement and communication are simultaneous, not alternating
- 1-step delay on messages (physically realistic)
- Agents send every step (policy can learn to send zeros)

## Communication Protocol Options

### MAPPO + dim_c (ER2) — continuous, implicit
- Communication = extra action dimensions in VMAS action space
- Policy learns what to send through reward maximization
- No explicit communication loss or protocol structure
- Algorithm-agnostic (MAPPO, IPPO, MADDPG all work)
- Implementation: ~50 lines, modify Discovery scenario + config
- Learned "language" is opaque continuous vectors

### BenchMARL GNN model — graph message-passing (ER3 candidate)
- Built into BenchMARL, no custom code needed — just config change
- Agents exchange information through GNN layers (structurally like CommNet)
- Supports three graph topologies:
  - `"full"` — all agents connected (complete graph)
  - `"empty"` — no edges (self-loops only)
  - `"from_pos"` — **proximity-based dynamic graph** using `edge_radius`
- `"from_pos"` reads agent positions, uses `torch_geometric.nn.pool.radius_graph()` to build edges dynamically every forward pass
- Any PyTorch Geometric conv layer works: `GraphConv` (default), `GATv2Conv`, GraphSAGE, GIN, etc.
- Edge features: relative position + distance (automatic when `position_key` set)
- Limitation: `"from_pos"` topology not supported for PPO critics
- Configuration example:
  ```python
  from benchmarl.models import GnnConfig
  import torch_geometric.nn.conv

  model_config = GnnConfig(
      topology="from_pos",
      self_loops=False,
      gnn_class=torch_geometric.nn.conv.GATv2Conv,
      position_key="position",
      pos_features=2,
      edge_radius=0.35,  # match comms_range
  )
  ```

### DIAL — discrete, differentiable
- Discrete messages with straight-through estimator during training
- Gradients flow through communication channel
- Discrete tokens at test time
- MUST train from scratch — cannot add post-hoc to trained MAPPO
- Implementation: ~200-300 lines custom BenchMARL model

### CommNet — continuous, architectural
- Mean-pooled messages fed back into network layers
- Communication is part of network architecture, not action space
- Must train from scratch

### TarMAC — attention-based, targeted
- Agents learn WHO to listen to and how much (attention)
- Targeted messaging, not broadcast
- Must train from scratch
- Implementation: custom attention model

### IC3Net — gated communication
- Agents learn WHEN to communicate (gate open/close)
- Learns to be silent when comm isn't useful
- Must train from scratch

## Existing Libraries with Communication Protocols

### Usable as reference / source for porting

| Library | Stars | Protocols | Framework | Last Active | Notes |
|---------|-------|-----------|-----------|-------------|-------|
| **MARLlib** | 1,282 | DIAL, CommNet, TarMAC | RLlib | 2024 | Best coverage of named protocols. NN modules extractable but training loop is RLlib. MIT license. [GitHub](https://github.com/Replicable-MARL/MARLlib) |
| **IC3Net** (official) | 227 | IC3Net, CommNet | Standalone PyTorch | 2023 | Clean `comm.py` — best candidate to port into BenchMARL. MIT license. [GitHub](https://github.com/IC3Net/IC3Net) |
| **DIAL** (minqi) | 358 | RIAL, DIAL | Standalone PyTorch | 2019 | Clean reference implementation. Very old, no framework integration. Apache-2.0. [GitHub](https://github.com/minqi/learning-to-communicate-pytorch) |
| **MARL-Algorithms** | 1,726 | CommNet, G2ANet | Standalone PyTorch | 2022 | Good reference for CommNet + graph-attention comm. SMAC-hardcoded, hard to extract. [GitHub](https://github.com/starry-sky6688/MARL-Algorithms) |
| **ExpoComm** (ICLR 2025) | 21 | ExpoComm | Standalone PyTorch | 2025 | State-of-the-art scalable comm. Own infra. Apache-2.0. [GitHub](https://github.com/LXXXXR/ExpoComm) |
| **CommFormer** (ICLR 2024) | 54 | CommFormer | Standalone PyTorch | 2024 | Learned graph structure via transformer. Own infra. Apache-2.0. [GitHub](https://github.com/charleshsc/CommFormer) |

### Libraries WITHOUT communication protocols

| Library | Stars | Notes |
|---------|-------|-------|
| PyMARL / PyMARL2 / EPyMARL | 700-2,166 | QMIX, MAPPO etc. No comm protocols |
| **BenchMARL** | 580 | No comm protocols, but has GNN model (proximity message-passing) |
| **TorchRL** | — | No comm primitives. MultiAgentMLP only |

### Key takeaway
No library implements DIAL/CommNet/TarMAC natively in TorchRL/BenchMARL. Every implementation uses its own training loop or RLlib/PyMARL. Porting NN modules (~150-300 lines) is the practical path.

## What Goes Where

| Component | Wrapper (src/wrappers/) | Custom Model | Scenario | Config only |
|-----------|------------------------|-------------|----------|-------------|
| dim_c channels | | | Scenario ✓ | |
| BenchMARL GNN message-passing | | | | Config ✓ |
| Proximity gating (who hears whom) | Wrapper ✓ | | | |
| Token counting (M5) | Wrapper ✓ | | | |
| Message discretization at eval | Wrapper ✓ | | | |
| Ablation (mask/shuffle/zero msgs) | Wrapper ✓ | | | |
| DIAL gradient path | | Model ✓ | | |
| TarMAC attention | | Model ✓ | | |
| IC3Net gating | | Model ✓ | | |

## Critical Constraint: Train Together vs Post-Hoc

### Must train from scratch (together)
- DIAL, CommNet, TarMAC, IC3Net
- Communication and action are ONE model
- Pre-training movement then adding comm → dead channel (policy ignores messages)

### Can add after MAPPO training (post-hoc)
- **LLM as message generator** — LLM reads obs, outputs message, agents see as extra obs. Fine-tune only obs-encoder adapter.
- **LLM as coordinator** — LLM assigns roles/targets every N steps, agents execute with frozen MAPPO
- **Rule-based protocols** — heuristic: "if I see target, broadcast location"
- **Frozen comm + fine-tune listener** — freeze movement weights, train small message decoder head

Key insight: LLM sits OUTSIDE the policy (no gradients through channel needed), so it can compose with pre-trained movement policy. This is fundamentally different from DIAL/CommNet.

## Experiment Plan

```
ER1: No comm (MAPPO + MLP)                → baseline floor (done)
ER2: Continuous comm (MAPPO + dim_c)       → does communication help?
ER3: GNN message-passing (MAPPO + GNN)     → graph-structured comm (BenchMARL built-in)
ER4: Ported IC3Net or TarMAC               → gated/targeted comm (custom model)
```

### ER2 specifics
- Modify Discovery: set dim_c=4-16, silent=False, proximity-gated obs
- Config change only: same MAPPO algorithm, same MLP model
- M5 = dim_c × n_agents × max_steps (fixed budget)
- Compare M1/M3 vs ER1 to prove communication value

### ER3 specifics (BenchMARL GNN — zero custom code)
- Switch `model_config` from `MlpConfig` to `GnnConfig`
- Use `topology="from_pos"`, `edge_radius=0.35` (match comms_range)
- Use `GATv2Conv` for attention-weighted message aggregation
- Communication is implicit through GNN layers — agents share hidden representations
- No changes to VMAS scenario needed (no dim_c)
- Requires `pip install torch_geometric`

### ER4 specifics (custom model)
- Port IC3Net `comm.py` (CommNet + gating) as custom BenchMARL Model
- Or port TarMAC attention module from MARLlib
- ~150-300 lines to wrap as BenchMARL model
- Keep MAPPO training pipeline

### LLM experiment (future)
- **Architecture A**: LLM as per-step message channel (obs → LLM → embedding → other agents' obs)
- **Architecture B**: LLM as high-level planner every N steps ("Agent 0 go NE, Agent 2 cover target 3")
- **Architecture C**: LLM replaces comm channel in trained encoder-decoder (freeze encoder+decoder, insert LLM)
- Can be added post-hoc to trained ER1 policy (unlike DIAL/CommNet)

### Thesis narrative
1. ER1: Can agents solve without communication?
2. ER2: Does continuous communication improve performance?
3. ER3: Does graph-structured communication (GNN) improve efficiency?
4. ER4: Does gated/targeted communication do better with fewer tokens?
- Cross-experiment: Pareto frontier of success rate vs token budget (M1 vs M5)
