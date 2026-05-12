# Concepts

The four load-bearing ideas behind multi_scenario:

- **[Hex architecture](hex_architecture.md)** — domain (pure-Python) → application (use cases) → adapters (torch/VMAS/BenchMARL/LiteLLM). Imports flow only one way.
- **[Metrics M1–M9](metrics.md)** — universal metric IDs used across every scenario + Streamlit page.
- **[LERO](lero.md)** — LLM-driven reward + observation evolution; what's evolved, how candidates are scored, the fallback chain.
- **[Reward safeguards](reward_safeguards.md)** — `nan_to_num` + clip ±50 — why these are needed for LLM-generated rewards.
