"""F9.6.c + F9.6.e — default-adapter assembly for :class:`LeroOrchestrator`.

Centralises the eight-port wiring so the experiment_service branch is a
single call, and the same factory powers the F9.7.A meta-composer
seam (it swaps :class:`InitialAndFeedbackComposer` for the meta stub).

All eight orchestrator ports default to production adapters here:

- ``llm``: ``CostCapDecorator(DiskCacheDecorator(LiteLlmClient))`` —
  cache is inserted only when ``cfg.llm.cache_enabled=True``.
- ``composer``: :class:`InitialAndFeedbackComposer` (default) or
  :class:`MetaPromptComposer` when ``cfg.lero.meta_prompting=True``.
- ``trace_writer``: :class:`FilesystemTraceWriter`.
- ``evaluator`` / ``full_trainer``: :class:`BenchmarlCandidateEvaluator`
  / :class:`BenchmarlFullTrainer` — the F9.6.e BenchMARL bridge that
  wires :class:`PatchedDiscoveryScenario` into the existing train+eval
  flow via :class:`ScenarioEnvFunFactory`.

Tests can override any port by passing it explicitly; the orchestrator
contract tests in F9.6.b exercise the wiring without instantiating
the BenchMARL stack.
"""

from multi_scenario.adapters.lero import (
    BenchmarlCandidateEvaluator,
    BenchmarlFullTrainer,
    FilesystemTraceWriter,
)

from multi_scenario.adapters.llm import (
    CostCapDecorator,
    DiskCacheDecorator,
    FilesystemCostLedger,
    LiteLlmClient,
)
from multi_scenario.adapters.prompt_composers import (
    InitialAndFeedbackComposer,
    MetaPromptComposer,
)
from multi_scenario.adapters.prompts import JinjaPromptRenderer
from multi_scenario.application.lero_orchestrator import (
    CandidateEvaluator,
    FullTrainer,
    LeroOrchestrator,
)
from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.domain.ports import LlmClient, Logger, PromptComposer


def _build_default_llm_client(cfg: ExperimentConfig) -> LlmClient:
    """Wire the locked decorator stack: CostCap → DiskCache → LiteLlmClient.

    Cache is only inserted when ``cfg.llm.cache_enabled`` is True
    (reproducibility-locked default: False). The cap is always on
    because €10/day rolling is the structural overspend block.
    """
    assert cfg.llm is not None
    inner: LlmClient = LiteLlmClient(cfg.llm)
    if cfg.llm.cache_enabled:
        from pathlib import Path  # local import; cache dir derives from CWD

        cache_dir = Path.cwd() / ".multi_scenario" / "llm_cache"
        inner = DiskCacheDecorator(inner, model=cfg.llm.model, cache_dir=cache_dir)
    return CostCapDecorator(inner, ledger=FilesystemCostLedger(), cfg_llm=cfg.llm)


# F9.6.e — placeholders replaced by real BenchMARL-backed adapters.
# The :class:`BenchmarlCandidateEvaluator` / :class:`BenchmarlFullTrainer`
# pair lives in ``adapters/lero/benchmarl_evaluator.py``; they wrap the
# existing :class:`BenchmarlBaseAdapter` train+evaluate flow with the
# patched scenario injected via :class:`ScenarioEnvFunFactory`.


# pylint: disable=too-many-arguments,too-many-positional-arguments
def build_default_lero_orchestrator(
    *,
    cfg: ExperimentConfig,
    logger: Logger,
    composer: PromptComposer | None = None,
    evaluator: CandidateEvaluator | None = None,
    full_trainer: FullTrainer | None = None,
) -> LeroOrchestrator:
    """Assemble a :class:`LeroOrchestrator` with hex-default adapters.

    ``composer`` / ``evaluator`` / ``full_trainer`` are injectable so:
    - F9.7.A's stub :class:`MetaPromptComposer` can be plugged in by
      callers that set ``cfg.lero.meta_prompting=True``.
    - Tests can substitute fakes without rebuilding the LiteLLM /
      ledger / Jinja stack.

    Unspecified ports default to:
    - LLM: ``CostCapDecorator(DiskCacheDecorator(LiteLlmClient))`` if
      caching is enabled, otherwise ``CostCapDecorator(LiteLlmClient)``.
    - Composer: :class:`InitialAndFeedbackComposer` over the configured
      prompt version.
    - Trace writer: :class:`FilesystemTraceWriter`.
    - Evaluator / FullTrainer: :class:`_UnwiredEvaluator` /
      :class:`_UnwiredFullTrainer` — F9.6.e wires the BenchMARL bridge.
    """
    assert cfg.lero is not None and cfg.llm is not None
    if composer is None:
        # F9.7.A: cfg.lero.meta_prompting toggles between the default
        # composer and the meta-prompt stub. F9.7.B will land the real
        # Strategist/Editor/Critic logic behind the same flag — no
        # orchestrator changes needed at that point.
        composer_cls = (
            MetaPromptComposer
            if cfg.lero.meta_prompting
            else InitialAndFeedbackComposer
        )
        composer = composer_cls(
            renderer=JinjaPromptRenderer(),
            prompt_version=cfg.lero.prompt_version,
            n_candidates=cfg.lero.n_candidates,
        )
    return LeroOrchestrator(
        llm=_build_default_llm_client(cfg),
        composer=composer,
        trace_writer=FilesystemTraceWriter(),
        evaluator=evaluator or BenchmarlCandidateEvaluator(),
        full_trainer=full_trainer or BenchmarlFullTrainer(),
        logger=logger,
    )
