# ER3 & ER4: Next Communication Experiments

**Prerequisite**: ER2 results must confirm that communication provides measurable value
over no-comm baselines. If ER2 shows no benefit, ER3/ER4 won't either.

## Communication Protocol Taxonomy

| Property | ER2 (dim_c) | ER3 (GNN) | ER4 (IC3Net/TarMAC) |
|----------|-------------|-----------|---------------------|
| **Type** | Emergent continuous | Architectural message-passing | Gated/targeted |
| **Emergent?** | Fully — agents invent protocol from scratch | No — architecture forces info sharing | Partially — structure designed, usage learned |
| **What to send** | Learned (continuous floats via action dims) | Not explicit — GNN propagates hidden states | Learned (continuous floats) |
| **When to send** | Must learn to output zeros (hard) | Always (every forward pass) | Explicit gate module (easy to learn silence) |
| **Who hears** | Broadcast or proximity-gated | Graph neighbors (edge_radius) | Attention-weighted (learned relevance) |
| **Token budget (M5)** | Fixed: dim_c × n_agents × steps | N/A (no explicit messages) | Variable: gate can reduce tokens |
| **Scenario changes** | dim_c, comm_proximity in Discovery | None | None |
| **Model changes** | None (MLP policy) | MlpConfig → GnnConfig | Custom BenchMARL model (~200 lines) |
| **Dependencies** | None | torch_geometric | None (pure PyTorch) |
| **Train from scratch?** | Yes | Yes | Yes |

### Why ER2 agents struggle to learn silence

ER2 agents output comm floats every step as part of their action. To "be silent",
the policy must discover that outputting zeros in comm dims leads to higher reward.
This is a weak gradient signal — "I was quiet and things improved" is hard to
attribute when physical actions also changed. IC3Net (ER4) solves this with a
dedicated gating network that explicitly decides "communicate or not" each step,
making silence architecturally easy to learn.

## ER3: GNN Message-Passing

### Goal
Test whether graph-structured communication (implicit, through network architecture)
outperforms learned continuous channels (ER2). Agents don't choose what to send —
the GNN layers propagate hidden representations between connected agents.

### How it works
1. Replace MLP policy with GNN policy using BenchMARL's built-in `GnnConfig`
2. Agents = graph nodes, edges = nearby agents within `edge_radius`
3. Each forward pass: agent observations are processed through GNN conv layers
4. Each agent's output incorporates neighbor information via message-passing
5. No `dim_c` needed — no extra action/obs dims, no scenario changes

### Implementation steps

**Step 1: Install torch_geometric**
```bash
pip install torch_geometric
```

**Step 2: Modify `build_experiment()` in `runner.py`**

Add a model type parameter to switch between MLP and GNN:

```python
from benchmarl.models import GnnConfig
import torch_geometric.nn.conv

if model_type == "gnn":
    model_config = GnnConfig(
        topology="from_pos",       # dynamic graph from agent positions
        self_loops=False,
        gnn_class=torch_geometric.nn.conv.GATv2Conv,  # attention-weighted
        position_key="position",   # TorchRL key for agent positions
        pos_features=2,            # x, y coordinates
        edge_radius=0.35,          # match lidar/comms range
    )
    # NOTE: "from_pos" topology not supported for PPO critics
    # Use MLP critic with GNN actor, or use "full" topology for critic
    critic_model_config = MlpConfig.get_from_yaml()  # keep MLP critic
```

**Step 3: Create ER3 YAML configs**

Add a `model` section to the YAML:
```yaml
exp_id: er3
model:
  type: gnn
  topology: from_pos
  edge_radius: 0.35
  gnn_class: GATv2Conv
```

Task params should match ER2 for comparison (same n_agents, n_targets, k, lidar, etc).
No `dim_c` needed.

**Step 4: Handle the critic limitation**

BenchMARL's GNN with `from_pos` topology doesn't support PPO-style critics.
Options:
- Use `topology="full"` for critic (all agents connected, no position dependency)
- Use MLP critic + GNN actor (mixed architecture)
- Use a non-PPO algorithm (QMIX, MADDPG) — but loses MAPPO comparison

Recommended: MLP critic + GNN actor. This is standard in MARL literature.

**Step 5: Configs to create**

| Config | GNN topology | edge_radius | Comparison |
|--------|-------------|-------------|------------|
| er3_proximity | from_pos | 0.35 | vs ER2 proximity (dc=8) |
| er3_full | full | - | vs ER2 broadcast |

Both with and without LP+SR to match ER2 experimental matrix.

### What ER3 tells us
- If ER3 > ER2: architectural info sharing beats learned communication
- If ER3 ≈ ER2: communication helps regardless of mechanism
- If ER3 < ER2: explicit learned messages are more useful than implicit GNN sharing

### M5 (tokens) for ER3
ER3 has no explicit messages, so M5=0. But information IS being shared through
GNN layers. For fair comparison on the Pareto frontier, we could estimate an
"equivalent token budget" from the GNN hidden dim × edges × steps.

## ER4: Gated/Targeted Communication (IC3Net or TarMAC)

### Goal
Test whether agents can learn **efficient** communication — when to talk and
who to target. If ER4 matches ER2's M1 with lower M5, that proves communication
efficiency matters, not just bandwidth.

### Option A: IC3Net (recommended — simpler)

**What it does**: CommNet (mean-pooled messages between agents) + a learned binary
gate per agent per step. Gate = 1 → broadcast hidden state. Gate = 0 → stay silent.

**Source**: [IC3Net repo](https://github.com/IC3Net/IC3Net) (MIT license).
Key file: `comm.py` (~150 lines).

**Implementation steps**:

1. **Port `comm.py` as BenchMARL custom model** (~200 lines)
   - Subclass `benchmarl.models.Model`
   - Forward pass: MLP encoder → CommNet message-passing with gate → action head
   - Gate: small MLP taking hidden state → sigmoid → binary (straight-through in training)

2. **Register the model** in BenchMARL's model registry

3. **Key architecture**:
   ```
   obs → MLP encoder → hidden_state
   hidden_state → gate_net → g ∈ {0, 1}
   if g=1: broadcast hidden_state to all agents
   received_msgs = mean(gated_hidden_states from others)
   [hidden_state, received_msgs] → MLP decoder → action
   ```

4. **M5 counting**: M5 = sum of gate openings × hidden_dim × steps
   Variable budget — the interesting metric.

### Option B: TarMAC (more complex)

**What it does**: agents produce a message + a "signature". Receivers use
attention (query from own state, key from sender's signature) to weight
incoming messages. Targeted, not broadcast.

**Source**: [MARLlib](https://github.com/Replicable-MARL/MARLlib) (MIT license).

**More complex** than IC3Net (~300 lines) due to attention mechanism.
Recommended only if IC3Net results are promising and we want to push further.

### Implementation steps (common)

**Step 1: Create custom model class**
```
rendezvous_comm/src/models/ic3net.py  (~200 lines)
```

**Step 2: Integrate with `build_experiment()`**
Add model type switch:
```python
if model_type == "ic3net":
    from .models.ic3net import IC3NetConfig
    model_config = IC3NetConfig(
        hidden_dim=256,
        comm_rounds=1,        # number of comm rounds per step
        gate_threshold=0.5,   # gate open if > threshold
    )
```

**Step 3: Create ER4 YAML configs**
```yaml
exp_id: er4
model:
  type: ic3net
  hidden_dim: 256
  comm_rounds: 1
```

**Step 4: Configs to create**

| Config | Gate | Comparison |
|--------|------|------------|
| er4_ic3net | yes | vs ER2 (same comm budget?) |
| er4_commnet | no (gate always open) | Ablation: is gating useful? |

### What ER4 tells us
- **M1 vs M5 Pareto**: does ER4 achieve similar M1 with fewer tokens?
- **Gate analysis**: when do agents choose to communicate? (near targets? near others?)
- **Ablation**: CommNet (always broadcast) vs IC3Net (gated) — is learned silence valuable?

## Experiment Order

```
ER2 results confirmed → proceed with:

1. ER3 (GNN) — fastest to implement (config change only, ~1 day)
   - Requires: torch_geometric, minor runner.py changes
   - Run: same sweep as ER2 best config

2. ER4 (IC3Net) — moderate effort (~2-3 days)
   - Requires: custom model class, runner integration
   - Run: same sweep as ER2 best config

3. Cross-experiment analysis
   - M1 vs M5 Pareto frontier across ER1-ER4
   - Ablation Treatment Effect (ATE): mask/shuffle/zero comms
   - Transfer score: train with comm, eval without
```

## Checklist Before Starting

- [ ] ER2 shows M1 improvement over ER1 (communication has value)
- [ ] ER2 LP+SR broadcast dim_c ablation results analyzed
- [ ] Best ER2 config identified (which dim_c, broadcast vs proximity, with/without LP+SR)
- [ ] torch_geometric installed and tested
- [ ] `build_experiment()` supports model type parameter
- [ ] ER3/ER4 YAML config schema designed
