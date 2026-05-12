# Ports

The `Protocol` interfaces every adapter satisfies. Defined under
`src/multi_scenario/domain/ports/`. Substitutable in tests via DI.

| Port | What it abstracts | Default adapter |
|---|---|---|
| `Algorithm` | The RL algorithm (MAPPO, IPPO, etc.) | `BenchmarlBaseAdapter` subclasses |
| `Scenario` | The VMAS scenario | `VmasDiscoveryAdapter`, etc. |
| `Runner` | Local vs OVH dispatch | `LocalRunner`, `OvhRunner` |
| `Storage` | Filesystem layout for run dirs | `LocalStorageAdapter`, `S3StorageAdapter` |
| `Logger` | Run logging | `FileLogger`, `_StdoutLogger` |
| `Metrics` | M1–M9 computation | `CommonMetricsBundle` |
| `LlmClient` | LLM completion calls | `LiteLlmClient` (+ `CostCapDecorator`, `DiskCacheDecorator`) |
| `CostLedger` | Persistent USD/EUR spend tracking | `FilesystemCostLedger` |
| `PromptComposer` | Inner-loop prompt assembly | `InitialAndFeedbackComposer` (+ `MetaPromptComposer` stub) |
| `PromptRenderer` | Template substitution | `JinjaPromptRenderer` |
| `TraceWriter` | LERO trace persistence | `FilesystemTraceWriter` |

> Per-port detail pages (signature, contract, test substitutes) land
> at F10.2 review. See `domain/ports/*.py` for the canonical surface
> + `tests/unit/domain/test_hex_compliance.py` for the import-purity
> rules.
