# Hexagonal architecture

Three layers, with dependencies flowing strictly inward:

```text
adapters/   ←   application/   ←   domain/
   ↑              ↑                   ↑
torch          orchestration       pure-Python
VMAS           use cases           dataclasses
BenchMARL      port DI             Protocols
LiteLLM        port DI             metric formulas
Streamlit
Pydantic
```

## domain/ — pure-Python core

No torch, no VMAS, no LiteLLM, no Streamlit imports. Verified by
`tests/unit/domain/test_hex_compliance.py` (37 parametrised tests).

Contains:

- `domain/models/` — Pydantic configs (`ExperimentConfig`, `LeroSection`, `LlmSection`, etc.).
- `domain/lero/` — pure-Python types (`Candidate`, `CandidateMetrics`, `CandidateResult`, `LlmCompletion`, `LeroRunSummary`, `PromptTrace`, `ResponseTrace`, codegen helpers, whitelist).
- `domain/ports/` — `Protocol` interfaces every adapter implements (`LlmClient`, `Scenario`, `Algorithm`, `TraceWriter`, `Runner`, `Storage`, etc.).

## application/ — orchestration

Use-case classes that wire ports together but don't depend on concrete
adapters. Tests can substitute fakes without touching the framework.

- `application/experiment_service.py` — orchestrates one run (standard or LERO branch).
- `application/lero_orchestrator.py` — 8-port LERO loop (LlmClient, PromptComposer, PromptRenderer, TraceWriter, CandidateEvaluator, FullTrainer, CostLedger, Logger).
- `application/submission.py` — `submit_to_local` / `submit_to_ovh` — single dispatch surface for CLI + Streamlit.
- `application/secrets_priming.py` — Fernet-decrypts `MS_ENCRYPTED_SECRETS` into `os.environ` inside the OVH container.

## adapters/ — concrete I/O

The only layer that imports torch / VMAS / BenchMARL / LiteLLM /
Streamlit / boto3 / pyglet.

- `adapters/algorithms/benchmarl_base.py` — MAPPO / IPPO / IDDPG / MADDPG / ISAC / MASAC.
- `adapters/scenarios/{discovery,navigation,transport,flocking}.py` — VMAS wrappers.
- `adapters/scenarios/patched_discovery.py` — LERO's code-spliced Discovery subclass.
- `adapters/llm/` — `LiteLlmClient`, `FakeLlmClient`, `CostCapDecorator`, `DiskCacheDecorator`, `FilesystemCostLedger`.
- `adapters/lero/` — `FilesystemTraceWriter`, `BenchmarlCandidateEvaluator`, `BenchmarlFullTrainer`, `ScenarioEnvFunFactory`, `evolution_doc` renderer.
- `adapters/runners/{local,ovh,ovh_cli}.py` — dispatch backends.
- `adapters/prompts/` — Jinja templates + `JinjaPromptRenderer`.
- `adapters/storage/` — local fs + S3.

## Why this layout

- **Testability**: 941 tests across the suite; the ones that don't need torch (`tests/unit/`) run in seconds.
- **Replaceability**: swap LiteLLM → OpenAI SDK by editing one adapter file; the orchestrator never knew.
- **Compliance**: domain imports are guarded by CI — a `import torch` in `domain/` fails the build.

For port-by-port detail see [Ports](../ports/index.md).
