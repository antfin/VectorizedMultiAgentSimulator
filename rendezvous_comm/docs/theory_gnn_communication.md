# Graph Neural Networks for Multi-Agent Communication

*Theory document for the rendezvous/discovery task — accessible to readers who understand RL but not GNNs.*

---

## 1. Why GNN for Multi-Agent Communication

In multi-agent reinforcement learning, agents need to coordinate. There are three broad approaches to enabling coordination, each with a different philosophy about how information flows between agents.

### Standard MLP (no communication)

Each agent processes its own observation independently. Agent A has no idea what Agent B sees, intends, or has done. Coordination can only emerge implicitly through shared rewards and experience — agents learn behavioral conventions (e.g., "spread out") but cannot adapt to each other in real time.

### Explicit Communication Channels (dim_c)

Each agent outputs a message vector (e.g., 8 floats) alongside its action. Other agents receive these messages as part of their observation. The agents must learn from scratch:
- **What to say** — the sender must learn to encode useful information into the message vector.
- **How to listen** — the receiver must learn to decode the message and act on it.

This is powerful in principle but hard in practice: the message space starts meaningless and must acquire meaning through training.

### Graph Neural Networks

Agents are nodes in a graph. Edges connect agents that can "communicate." Information flows along edges through learned message-passing operations. The key insight:

> **GNN turns "learning what to say" into "architecture that forces information sharing."**

Rather than learning to communicate from scratch, GNN agents receive structured information about their neighbors (relative position, distance) through the graph edges, and learn how to *use* that information through attention mechanisms. The architecture guarantees information flows — the learning problem is only about how to weight and combine it.

---

## 2. Graph Construction

A GNN operates on a graph G = (V, E) where:

- **Nodes V**: one per agent. Each node carries a feature vector equal to the agent's observation (LiDAR readings, position, velocity, etc.).
- **Edges E**: connections between agents, constructed based on a chosen topology.

### Topology Options

| Topology | Edge Rule | Edge Features | Pros | Cons |
|----------|-----------|---------------|------|------|
| `full` | Every agent connected to every other | Relative position + distance | Complete information flow | O(n^2) edges, scales poorly |
| `from_pos` | Agents within communication radius | Relative position + distance | Sparse, realistic | Requires dict observations (implementation complexity) |

### In Our Experiments

We use **`full` topology** — every agent is connected to every other agent. Early experiments with `from_pos` failed because it requires dictionary-structured observations, which caused compatibility issues with the training pipeline. This was later fixed, but our primary results use full connectivity.

### Graph Visualization

The diagram below shows a 4-agent full graph. Each agent (node) holds its own observation. Each edge carries the relative position and distance between the two agents it connects.

```
        Agent 0
       /  |    \
      /   |     \
     /    |      \
Agent 1---+---Agent 3
     \    |      /
      \   |     /
       \  |    /
        Agent 2

Edge features (each edge):
  - relative position: (x_j - x_i, y_j - y_i)
  - distance: ||pos_j - pos_i||
```

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">
  <!-- Edges (full connectivity) -->
  <line x1="200" y1="60" x2="60" y2="200" stroke="#888" stroke-width="1.5"/>
  <line x1="200" y1="60" x2="340" y2="200" stroke="#888" stroke-width="1.5"/>
  <line x1="200" y1="60" x2="200" y2="340" stroke="#888" stroke-width="1.5"/>
  <line x1="60" y1="200" x2="340" y2="200" stroke="#888" stroke-width="1.5"/>
  <line x1="60" y1="200" x2="200" y2="340" stroke="#888" stroke-width="1.5"/>
  <line x1="340" y1="200" x2="200" y2="340" stroke="#888" stroke-width="1.5"/>

  <!-- Edge label example -->
  <text x="105" y="115" font-size="10" fill="#555" font-family="monospace">rel_pos + dist</text>

  <!-- Nodes -->
  <circle cx="200" cy="60" r="24" fill="#4A90D9" stroke="#333" stroke-width="2"/>
  <text x="200" y="65" text-anchor="middle" fill="white" font-size="12" font-weight="bold">A0</text>

  <circle cx="60" cy="200" r="24" fill="#4A90D9" stroke="#333" stroke-width="2"/>
  <text x="60" y="205" text-anchor="middle" fill="white" font-size="12" font-weight="bold">A1</text>

  <circle cx="340" cy="200" r="24" fill="#4A90D9" stroke="#333" stroke-width="2"/>
  <text x="340" y="205" text-anchor="middle" fill="white" font-size="12" font-weight="bold">A3</text>

  <circle cx="200" cy="340" r="24" fill="#4A90D9" stroke="#333" stroke-width="2"/>
  <text x="200" y="345" text-anchor="middle" fill="white" font-size="12" font-weight="bold">A2</text>

  <!-- Legend -->
  <text x="200" y="395" text-anchor="middle" font-size="11" fill="#333" font-family="sans-serif">Full topology: 4 agents, 6 bidirectional edges</text>
</svg>
```

---

## 3. GATv2Conv (Graph Attention v2)

### From Simple to Attention-Based Aggregation

Graph neural networks work by **message passing**: each node collects information from its neighbors, transforms it, and updates its own representation. The key question is *how* to combine messages from multiple neighbors.

**Simple graph convolution** averages or sums neighbor features with fixed weights. Every neighbor contributes equally — an agent heading toward a shared target gets the same weight as an irrelevant distant agent.

**GAT (Graph Attention Network)** learns attention weights that determine how much each agent "listens" to each neighbor. However, original GAT has a subtle flaw: the attention scores are computed using a mechanism that is *static* — it computes attention from source and target features independently, then combines them. This means the ranking of neighbors can be the same regardless of the query node's state.

**GATv2** fixes this limitation with *dynamic attention*: the attention score depends on the *interaction* between source and target features, not just their independent projections.

### The GATv2 Mechanism (Simplified)

For agent *i* receiving a message from agent *j*:

**Step 1 — Compute attention score:**

```
e_ij = a^T * LeakyReLU(W_src * h_i + W_tgt * h_j)
```

Where:
- `h_i` is agent *i*'s current feature vector (its observation)
- `h_j` is agent *j*'s feature vector
- `W_src` and `W_tgt` are learned weight matrices
- `a` is a learned attention vector
- The LeakyReLU nonlinearity is applied *before* the dot product with `a` — this is what makes GATv2 dynamic (in GAT v1, the nonlinearity comes after, which collapses the expressiveness)

**Step 2 — Normalize attention across all neighbors:**

```
alpha_ij = softmax_j(e_ij) = exp(e_ij) / sum_k(exp(e_ik))
```

This ensures attention weights for each agent sum to 1.

**Step 3 — Compute weighted message and aggregate:**

```
h_i' = sum_j( alpha_ij * W * h_j )
```

The updated representation of agent *i* is a weighted sum of transformed neighbor features.

### Why This Matters for Coordination

The attention mechanism lets each agent decide **how much to attend to each neighbor**, based on *both* its own state and the neighbor's state. Concretely:

- An agent heading toward target A can learn to attend more to agents that are also near target A (to avoid redundant coverage).
- An agent with no nearby targets can attend more to agents that have found targets (to navigate toward productive areas).
- The attention weights are *dynamic* — they change at every timestep as agents move and the situation evolves.

---

## 4. Why GNN Outperforms Explicit Communication

### 4.1 Information is Structured, Not Arbitrary

With `dim_c`, agents must learn **what to communicate** from scratch. The message is 8 floats — initially random, semantically meaningless. The sender must discover that encoding its position is useful; the receiver must discover how to decode it. This is a hard credit assignment problem layered on top of the already-hard coordination problem.

With GNN edges, **relative position and distance are given as edge features**. The network does not need to learn to communicate spatial information — it is provided by the architecture. The learning problem reduces to: *how should I weight and combine the spatial information from my neighbors?*

> This is like the difference between learning a language from scratch versus having built-in spatial awareness.

### 4.2 No "Dead Channel" Problem

With `dim_c`, a common failure mode is the **convergence trap**: if agents converge early in training to ignoring the communication channel (because random messages are initially noise), the gradient signal through the channel vanishes. Communication never emerges because there is no incentive to start using a channel that nobody listens to.

With GNN, information flows through the architecture by construction. Agents **cannot "turn off"** their neighbors' influence entirely. The attention mechanism modulates *how much* to listen to each neighbor, but the softmax normalization ensures that the total attention is always 1 — some neighbor always has influence. This maintains gradient flow and prevents the communication pathway from collapsing.

### 4.3 Spatial Information is the Bottleneck

For the rendezvous/discovery task, the critical coordination questions are:

1. **Where are other agents?** (to avoid redundant coverage)
2. **Where are they going?** (to predict future coverage)

GNN answers these directly:
- **Edge features** (relative position) encode *where* other agents are.
- **Node features** (velocity) encode *where* they are going.

With `dim_c`, both pieces of information must be learned — encoded by the sender into a compact vector and decoded by the receiver — all while the encoding scheme itself is evolving during training.

---

## 5. Architecture Details (Our Implementation)

| Component | Value |
|-----------|-------|
| GNN layer | GATv2Conv (PyTorch Geometric) |
| Hidden size | 148 |
| Total policy parameters | 78,444 |
| Comparable MLP parameters | 75,013 |
| Input per node | Agent observation vector (LiDAR + position + velocity) |
| Message-passing rounds | 1 (single GNN layer) |
| Topology | Full (all-to-all) |
| Output | Action (2D force vector for holonomic dynamics) |

The parameter counts are deliberately similar between GNN and MLP architectures. This ensures that any performance difference is due to the *structure* of information flow (graph message-passing vs independent processing), not simply having more learnable parameters.

The single message-passing round means each agent aggregates information from its direct neighbors exactly once per forward pass. With full topology, one round is sufficient for every agent to receive information from every other agent. With sparse topologies, multiple rounds would be needed for information to propagate across the full team.

---

## 6. Evidence from Experiments

Our experimental results (ER3) provide several lines of evidence that GNN communication enables qualitatively different coordination behavior:

### Spatial Coordination (M9 — Spatial Spread)

| Configuration | Spatial Spread |
|---------------|---------------|
| No communication (MLP) | 1.018 |
| GNN communication | 0.578 |

GNN agents **cluster together** (lower spread), indicating active pairing behavior — agents converge on targets as coordinated groups rather than spreading uniformly. This is the signature of information-driven coordination.

### Collision Trade-off (M4 — Collisions)

| Configuration | Collisions |
|---------------|-----------|
| No communication (MLP) | 1.3 |
| GNN communication | 8.9 |

GNN agents experience **more collisions** — but this is a feature, not a bug. Agents that coordinate to converge on targets will naturally come into closer proximity. The higher collision rate is the cost of the tighter spatial coordination that enables better task performance.

### Exploration Strategy (Entropy)

| Configuration | Policy Entropy |
|---------------|---------------|
| No communication (MLP) | -2.35 |
| GNN communication | -0.84 |

GNN agents maintain **more stochasticity** in their policies (higher entropy = less deterministic). This suggests they learn flexible strategies that can adapt to different target configurations, rather than committing to a single rigid behavioral pattern.

### Training Cost

| Configuration | Wall Time |
|---------------|-----------|
| MLP | ~1.0 h |
| GNN | ~3.4 h |

GNN training is approximately 3.4x slower due to the graph construction and message-passing operations in each forward pass. This is a one-time cost — inference speed difference is smaller.

---

## 7. Limitations

### Scalability

Full topology creates O(n^2) edges. For our 2-agent and small-team experiments this is negligible, but scaling to 50+ agents would require sparse graph construction (e.g., `from_pos` with a communication radius, or k-nearest-neighbor graphs).

### Implementation Complexity

The `from_pos` topology requires dictionary-structured observations to pass positional information for edge construction. This caused compatibility issues with BenchMARL's training pipeline early in our experiments. While later fixed, it added integration overhead that `full` topology avoids.

### Statistical Confidence

Current results are from single-seed experiments. The GNN advantage — while consistent across metrics — needs multi-seed validation to confirm it is robust and not an artifact of a favorable random initialization.

### Training Cost

GNN training takes approximately 3.5 hours versus 1 hour for MLP (on our hardware). For a research setting where training is a one-time cost, this is acceptable. For settings requiring frequent retraining or hyperparameter search, the 3.4x overhead compounds.

### Single Message-Passing Round

Our architecture uses a single GNN layer (one round of message passing). With full topology this is sufficient, but with sparse topologies, information would only propagate one hop per forward pass. Multi-hop communication would require stacking GNN layers, increasing both parameters and computation.

---

*Document created 2026-03-25 for the rendezvous_comm experiment series.*
