"""LERO-MP outer loop — evolves prompt templates across inner runs.

Glue module. All the work is already in:
  - LeroLoop                 (the inner code-evolution loop)
  - classify_inner_result    (fail-mode taxonomy)
  - should_meta_iterate      (trigger checks)
  - pick_slot_to_edit        (slot-policy)
  - propose_new_template     (meta-LLM call + materialize)

This file ties them together and persists a history trail.

Shape of one outer iteration:

  1. Run inner LeroLoop.run() under the current prompt version.
  2. Summarize into a TemplateRecord (fail_mode, best peak-M1, …).
  3. Call should_meta_iterate(history). If it says STOP, exit.
     If it says COOLDOWN or "no trigger", go to step 1 again
     under the same version.
  4. Otherwise classify → pick slot → propose_new_template, switch
     current_version to the freshly-materialized directory, loop.

Every outer iteration writes ``history.json`` and mutation metadata
to ``output_dir`` so a crash can be resumed without losing lineage.
"""

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import LLMConfig, LeroConfig, MetaPromptConfig
from ..loop import LeroLoop
from .failmode import FailMode, classify_inner_result, pick_slot_to_edit
from .mutation import MetaLLMCallable, propose_new_template
from .mutation_log import (
    MutationLogEntry,
    append_entry,
    new_entry as new_log_entry,
    read_prior_slot_versions,
    read_recent,
    update_last_entry_with_post,
)
from .strategy import bias_for_seed, strategize
from .trigger import (
    TemplateRecord,
    TriggerConfig,
    TriggerDecision,
    TriggerReason,
    should_meta_iterate,
)

_log = logging.getLogger("rendezvous.lero.mp")


# ── results ──────────────────────────────────────────────────────

@dataclass
class OuterLoopResult:
    """Return value of LeroMpOuterLoop.run()."""

    history: List[TemplateRecord] = field(default_factory=list)
    final_version: str = ""
    total_inner_candidates: int = 0
    stop_reason: TriggerReason = TriggerReason.INITIAL
    stop_detail: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": [_record_to_dict(r) for r in self.history],
            "final_version": self.final_version,
            "total_inner_candidates": self.total_inner_candidates,
            "stop_reason": self.stop_reason.value,
            "stop_detail": self.stop_detail,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _record_to_dict(r: TemplateRecord) -> Dict[str, Any]:
    """TemplateRecord → JSON-safe dict (Enum → str)."""
    d = asdict(r)
    d["fail_mode"] = r.fail_mode.value
    return d


# ── helpers that don't need the big class around them ────────────

def build_template_record(
    version: str,
    inner_result: Dict[str, Any],
    candidate_metrics: Sequence[Dict[str, Any]],
    mutation_target_slot: Optional[str],
    mutation_rationale: Optional[str],
    seed_M1_values: Optional[Sequence[float]] = None,
    tier2_metrics: Optional[Dict[str, Any]] = None,
    history_for_inflation_check: Optional[Sequence[TemplateRecord]] = None,
) -> TemplateRecord:
    """Summarize one inner-loop pass into a TemplateRecord.

    ``inner_result`` is the dict returned by LeroLoop.run(). We pull
    M1/M6/M2 out of the ``final_metrics`` section and classify the
    fail-mode using the per-candidate metrics list.
    """
    # LeroLoop.run() returns a FLAT dict — M1/M2/M6/peak_M1 etc. are
    # at the top level alongside bookkeeping fields (lero_iterations,
    # llm_model, fallback_chain). No nested "final_metrics" key.
    # (The file written to disk is also flat; only candidate_*_metrics
    # files are per-candidate.)
    fm = inner_result or {}

    # Tier-2 full-training metrics (may be None for short runs or when
    # the inner loop didn't actually finish training).
    peak_m1 = fm.get("peak_M1")
    final_m1 = fm.get("M1_success_rate")

    # Best-of-iter metrics: the inner loop's final_metrics reflect only
    # the LAST frame of the selected candidate's full-training run. For
    # short runs or reward-hacked runs the final frame can be 0/worse
    # than the eval-time metrics. For fail-mode classification and the
    # trigger logic we want the best any candidate achieved under this
    # template — use the per-candidate eval metrics on disk.
    valid_cands = [m for m in (candidate_metrics or []) if "_error" not in m]
    if valid_cands:
        best_m1_eval = max(m.get("M1_success_rate", 0.0) for m in valid_cands)
        best_m6 = max(m.get("M6_coverage_progress", 0.0) for m in valid_cands)
        # For M2 we use the magnitude of the best candidate's return.
        # When all returns are negative, taking plain max is correct
        # (least-negative is "best"); we just need a stable value.
        best_m2 = max(m.get("M2_avg_return", -1e9) for m in valid_cands)
    else:
        best_m1_eval = 0.0
        best_m6 = fm.get("M6_coverage_progress", 0.0)
        best_m2 = fm.get("M2_avg_return", 0.0)

    # peak_M1 reporting: prefer the Tier-2 checkpointed peak if present;
    # otherwise fall back to the best eval-time M1 across candidates.
    if peak_m1 is None:
        peak_m1 = best_m1_eval

    fail_mode = classify_inner_result(
        candidate_metrics=list(candidate_metrics),
        tier2_metrics=tier2_metrics,
        template_history=[
            {"best_M1": r.best_peak_M1, "best_M2": r.best_M2}
            for r in (history_for_inflation_check or [])
        ],
    )

    seed_std = 0.0
    if seed_M1_values and len(seed_M1_values) >= 2:
        mean = sum(seed_M1_values) / len(seed_M1_values)
        seed_std = (
            sum((v - mean) ** 2 for v in seed_M1_values) / len(seed_M1_values)
        ) ** 0.5

    return TemplateRecord(
        template_version=version,
        inner_iter_count=int(fm.get("n_iterations_run", 0)),
        best_peak_M1=float(peak_m1),
        best_final_M1=float(final_m1) if final_m1 is not None else None,
        best_M6=float(best_m6),
        best_M2=float(best_m2),
        seed_M1_std=float(seed_std),
        fail_mode=fail_mode,
        mutation_target_slot=mutation_target_slot,
        mutation_rationale=mutation_rationale,
    )


def _trigger_config_from(mp: MetaPromptConfig) -> TriggerConfig:
    """Copy MetaPromptConfig trigger/budget fields into TriggerConfig."""
    return TriggerConfig(
        plateau_iters=mp.trigger.plateau_iters,
        plateau_delta=mp.trigger.plateau_delta,
        variance_threshold=mp.trigger.variance_threshold,
        peak_vs_final_gap_max=mp.trigger.peak_vs_final_gap_max,
        cooldown_inner_iters=mp.trigger.cooldown_inner_iters,
        max_outer_iters=mp.budget.max_outer_iters,
        max_total_inner_candidates=mp.budget.max_total_inner_candidates,
    )


def _default_meta_llm_call(
    meta_config: MetaPromptConfig,
    temperature: Optional[float] = None,
) -> MetaLLMCallable:
    """Build a default meta-LLM callable using LLMClient.

    ``temperature`` overrides ``meta_config.meta_temperature``. Used
    by the per-seed temperature diversification in the outer loop —
    seeds 0/1/2 cycle through cool/medium/warm to encourage divergent
    Editor outputs even when the Strategist picks the same slot.
    """
    # Lazy import keeps this module cheap when we only want pure helpers.
    from ..llm_client import LLMClient
    from ..llm_cache import LLMCache
    cache = None
    mode = getattr(meta_config, "llm_cache", "off")
    if mode and mode != "off":
        cache = LLMCache(
            mode=mode,
            root=getattr(meta_config, "llm_cache_dir", None),
        )
    client = LLMClient(
        LLMConfig(
            model=meta_config.meta_model,
            temperature=(
                temperature if temperature is not None
                else meta_config.meta_temperature
            ),
            api_base=meta_config.meta_api_base,
        ),
        cache=cache,
    )

    def call(messages: List[Dict[str, str]]) -> str:
        return client.generate(messages, n=1)[0]
    return call


# Per-seed meta-LLM temperature cycle (D). Cool → medium → warm so
# that three parallel seeds exploring the same slot produce more
# diverse outputs. Override via MetaPromptConfig.meta_temperature
# for a flat temperature across seeds.
SEED_META_TEMPERATURE = {
    0: 0.1,  # cool: exploit the strongest pattern
    1: 0.3,  # medium: balanced (v2 default)
    2: 0.7,  # warm: broader exploration
}


def meta_temperature_for_seed(seed: int) -> float:
    return SEED_META_TEMPERATURE[seed % 3]


# ── orchestrator ─────────────────────────────────────────────────

# Factory used to construct the inner loop. Made overridable so tests
# can inject a fake LeroLoop without touching real VMAS / RL / LLM.
InnerLoopFactory = Callable[..., LeroLoop]


class LeroMpOuterLoop:
    """Top-level orchestrator for LERO-MP runs."""

    def __init__(
        self,
        spec,
        lero_config: LeroConfig,
        llm_config: LLMConfig,
        meta_config: MetaPromptConfig,
        output_dir: Optional[Path] = None,
        meta_llm_call: Optional[MetaLLMCallable] = None,
        inner_loop_factory: Optional[InnerLoopFactory] = None,
    ) -> None:
        self.spec = spec
        self.lero_config = lero_config
        self.llm_config = llm_config
        self.meta = meta_config
        self.trigger_cfg = _trigger_config_from(meta_config)

        if output_dir is None:
            output_dir = Path("results") / "lero_mp" / time.strftime(
                "%Y%m%d_%H%M",
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hold onto the user-provided callable (if any) so `run()` can
        # build a per-seed variant with a custom temperature when none
        # is supplied. See meta_temperature_for_seed() above.
        self._meta_llm_override = meta_llm_call
        self.meta_llm_call = meta_llm_call or _default_meta_llm_call(meta_config)
        self.inner_loop_factory = inner_loop_factory or LeroLoop

        # v2: per-run mutation_log path. Lives under RESULTS_DIR (OVH-
        # writable) so it survives job FINALIZE. A `run_id` lets multi-
        # seed parallel jobs distinguish their own entries.
        self._run_id = (
            f"{getattr(spec, 'exp_id', 'lero_mp')}_"
            f"{time.strftime('%Y%m%d_%H%M')}"
        )
        self._mutation_log_path = self.output_dir / "mutation_log.jsonl"
        # v3 §6.2: cross-run memory. LERO_HISTORY_PATHS env (colon-
        # separated mount points injected by submit_training_job) lets
        # the Strategist see mutation logs from past sweep runs. Falls
        # back to just this run's log if the env is unset or the files
        # don't exist.
        self._history_log_paths: List[Path] = []
        import os as _os
        env_paths = _os.environ.get("LERO_HISTORY_PATHS", "")
        for p in env_paths.split(":") if env_paths else []:
            p = p.strip()
            if not p:
                continue
            hp = Path(p)
            # Look for mutation_log.jsonl under each history mount, any depth
            if hp.exists():
                for f in hp.rglob("mutation_log.jsonl"):
                    self._history_log_paths.append(f)
        if self._history_log_paths:
            _log.info(
                "LERO-MP: discovered %d cross-run mutation logs under "
                "LERO_HISTORY_PATHS",
                len(self._history_log_paths),
            )

    # ── public API ────────────────────────────────────────────

    def run(
        self,
        task_overrides: Optional[Dict[str, Any]] = None,
        algorithm: str = "mappo",
        base_seed: int = 0,
    ) -> OuterLoopResult:
        """Execute the outer loop until a stop condition fires."""
        t0 = time.monotonic()
        history: List[TemplateRecord] = []
        current_version = self.llm_config.prompt_version
        last_mutation_target_slot: Optional[str] = None
        last_mutation_rationale: Optional[str] = None
        inner_iters_since_last_mutation = 0
        total_candidates = 0
        outer_iter = 0

        # Per-seed meta-LLM temperature when no explicit override was
        # passed at __init__. Only used when two_level_meta is on
        # (otherwise the v1 meta_llm_call already in place is fine).
        if self._meta_llm_override is None and self.meta.two_level_meta:
            seed_temp = meta_temperature_for_seed(base_seed)
            self.meta_llm_call = _default_meta_llm_call(
                self.meta, temperature=seed_temp,
            )
            _log.info(
                "Meta-LLM temperature for seed %d = %.2f",
                base_seed, seed_temp,
            )

        # Verdict threshold scaling (C): at 10M frames the noise floor
        # is ~10× tighter than at 1M, so the classifier needs to scale.
        # Base thresholds were calibrated at 1M.
        full_frames = max(int(self.lero_config.full_frames), 1)
        self._verdict_scale = max(1.0, full_frames / 1_000_000)
        _log.info(
            "Verdict threshold scale = %.2f (for full_frames=%d)",
            self._verdict_scale, full_frames,
        )

        _log.info(
            "=== LERO-MP START === seed_version=%s, meta_enabled=%s, "
            "max_outer=%d, max_cands=%d",
            current_version, self.meta.enabled,
            self.meta.budget.max_outer_iters,
            self.meta.budget.max_total_inner_candidates,
        )

        # When meta-prompting is disabled, run exactly one inner pass
        # under the seed template and exit. This is the P0-smoke path:
        # LeroLoop behaves as before, LERO-MP just provides the
        # fairness whitelist + peak-M1 checkpointing + structured output.
        if not self.meta.enabled:
            _log.info(
                "meta_prompt.enabled=false → running a single inner "
                "LeroLoop under %s and exiting.", current_version,
            )

        while True:
            # 1. Run the inner loop under the current prompt version.
            inner_dir = self.output_dir / f"outer_{outer_iter:03d}_{current_version}"
            inner_dir.mkdir(parents=True, exist_ok=True)

            iter_llm_cfg = self._llm_config_for(current_version)
            inner = self.inner_loop_factory(
                self.spec, self.lero_config, iter_llm_cfg, output_dir=inner_dir,
            )
            _log.info(
                "--- outer iter %d --- running inner LeroLoop under %s",
                outer_iter, current_version,
            )
            inner_result = inner.run(
                task_overrides=task_overrides,
                algorithm=algorithm,
                seed=base_seed,
            )
            candidate_metrics = self._collect_candidate_metrics(inner_dir)
            total_candidates += len(candidate_metrics)
            inner_iters_since_last_mutation += self.lero_config.n_iterations

            # 2. Summarize → TemplateRecord.
            record = build_template_record(
                version=current_version,
                inner_result=inner_result,
                candidate_metrics=candidate_metrics,
                mutation_target_slot=last_mutation_target_slot,
                mutation_rationale=last_mutation_rationale,
                history_for_inflation_check=history,
            )
            history.append(record)
            self._persist(history, current_version, total_candidates)

            # v2: if the prior outer iter mutated the template, resolve
            # the pending mutation_log entry with the post-mutation
            # metrics we just collected on this record.
            if self.meta.two_level_meta and last_mutation_target_slot is not None:
                try:
                    update_last_entry_with_post(
                        path=self._mutation_log_path,
                        post_peak_M1=record.best_peak_M1,
                        post_M6=record.best_M6,
                        fail_modes=[record.fail_mode.value],
                        verdict_scale=self._verdict_scale,
                    )
                except Exception as e:  # pragma: no cover
                    _log.warning(
                        "Failed to update mutation_log post fields: %s",
                        e,
                    )
            _log.info(
                "  record: peak_M1=%.3f  final_M1=%s  fail_mode=%s",
                record.best_peak_M1,
                "n/a" if record.best_final_M1 is None
                else f"{record.best_final_M1:.3f}",
                record.fail_mode.value,
            )

            # 2b. Single-pass early exit when meta-prompt is disabled.
            if not self.meta.enabled:
                elapsed = time.monotonic() - t0
                return OuterLoopResult(
                    history=history,
                    final_version=current_version,
                    total_inner_candidates=total_candidates,
                    stop_reason=TriggerReason.BUDGET_EXCEEDED,
                    stop_detail="meta_prompt disabled; single inner pass complete.",
                    elapsed_seconds=elapsed,
                )

            # 3. Trigger check.
            decision = should_meta_iterate(
                history, total_candidates, self.trigger_cfg,
                inner_iters_since_last_mutation=inner_iters_since_last_mutation,
            )

            if decision.should_stop:
                _log.info(
                    "=== LERO-MP STOP === %s — %s",
                    decision.reason.value, decision.detail,
                )
                elapsed = time.monotonic() - t0
                return OuterLoopResult(
                    history=history,
                    final_version=current_version,
                    total_inner_candidates=total_candidates,
                    stop_reason=decision.reason,
                    stop_detail=decision.detail,
                    elapsed_seconds=elapsed,
                )

            if not decision.should_iterate:
                # Cooldown or no-trigger: keep current template, loop.
                _log.info(
                    "  no mutation this outer iter (%s). Continuing.",
                    decision.reason.value,
                )
                outer_iter += 1
                continue

            # 4. Classify + pick slot + mutate.
            top_cands = self._top_candidates(candidate_metrics, inner_dir)

            # v2 PATH: call the Strategist first, let it pick domain +
            # slot + focus + avoid from history + mutation_log. v1 PATH:
            # pick_slot_to_edit heuristic.
            strategy_card = None
            if self.meta.two_level_meta:
                try:
                    # v3: include cross-run history logs when mounted
                    log_paths = [self._mutation_log_path] + list(
                        self._history_log_paths
                    )
                    strategy_card = strategize(
                        history=history,
                        mutation_log_entries=read_recent(
                            log_paths,
                            n=10, task_id=self.spec.exp_id,
                        ),
                        top_candidates=top_cands,
                        seed_bias=bias_for_seed(base_seed),
                        fail_mode=record.fail_mode,
                        meta_llm_call=self.meta_llm_call,
                        fairness_slot_excerpt=self._fairness_excerpt(current_version),
                    )
                    target_slot = strategy_card.target_slot
                    _log.info(
                        "  TRIGGER %s → Strategist chose slot=%s "
                        "domain=%s confidence=%s",
                        decision.reason.value, target_slot,
                        strategy_card.target_domain,
                        strategy_card.confidence,
                    )
                except Exception as e:
                    _log.error(
                        "Strategist (Level 1) failed: %s: %s. "
                        "Falling back to v1 slot-picker.",
                        type(e).__name__, e,
                    )
                    strategy_card = None

            if strategy_card is None:
                target_slot = pick_slot_to_edit(
                    record.fail_mode,
                    history=[r.mutation_target_slot for r in history
                             if r.mutation_target_slot],
                    policy=self.meta.slot_policy,
                )
                _log.info(
                    "  TRIGGER %s → editing slot '%s' via meta-LLM (v1)",
                    decision.reason.value, target_slot,
                )
            # Prior versions of the same slot — fed to the Editor so
            # it diverges from past attempts. Pulled from our own
            # mutation_log (extensible to cross-run shared logs later).
            prior_versions = (
                read_prior_slot_versions(
                    self._mutation_log_path,
                    slot_name=target_slot,
                    task_id=self.spec.exp_id,
                    n=5,
                )
                if self.meta.two_level_meta else None
            )
            # v3 §4.2: behavioral signal block filtered per
            # include_signals; v3 §4.2: Critic LLM wraps the Editor.
            behavioral_block = self._behavioral_block(
                top_cands, strategy_card,
            )
            try:
                mutation = propose_new_template(
                    parent_version=current_version,
                    target_slot=target_slot,
                    history=history,
                    top_candidates=top_cands,
                    fail_mode=record.fail_mode,
                    meta_llm_call=self.meta_llm_call,
                    outer_iter=outer_iter + 1,
                    generated_by=self.meta.meta_model,
                    strategy_card=strategy_card,
                    prior_slot_versions=prior_versions,
                    behavioral_block=behavioral_block,
                    critic_llm_call=(
                        self.meta_llm_call
                        if self.meta.two_level_meta else None
                    ),
                )
            except Exception as e:
                # Meta-LLM failures (auth, rate limit, malformed output)
                # are surfaced as a graceful stop — the inner-loop
                # history is already persisted, and we don't want a
                # transient API issue to discard ~minutes of training.
                _log.error(
                    "Meta-LLM mutation failed: %s: %s. Stopping outer "
                    "loop with the inner history intact.",
                    type(e).__name__, e,
                )
                elapsed = time.monotonic() - t0
                return OuterLoopResult(
                    history=history,
                    final_version=current_version,
                    total_inner_candidates=total_candidates,
                    stop_reason=TriggerReason.BUDGET_EXCEEDED,
                    stop_detail=(
                        f"Meta-LLM mutation failed "
                        f"({type(e).__name__}): {e}"
                    ),
                    elapsed_seconds=elapsed,
                )

            # Append a mutation_log entry at mutation time (post-fields
            # null). They get filled in at the START of the NEXT outer
            # iter once the new template's first record exists.
            if self.meta.two_level_meta:
                entry = new_log_entry(
                    run_id=self._run_id,
                    task_id=self.spec.exp_id,
                    seed=base_seed,
                    outer_iter=outer_iter,
                    parent_version=current_version,
                    new_version=mutation.new_version,
                    strategy_card=(
                        strategy_card.to_dict() if strategy_card else {}
                    ),
                    slot_name=target_slot,
                    slot_content=mutation.new_slot_content,
                    pre_peak_M1=record.best_peak_M1,
                    pre_M6=record.best_M6,
                )
                append_entry(self._mutation_log_path, entry)

            current_version = mutation.new_version
            last_mutation_target_slot = target_slot
            last_mutation_rationale = mutation.rationale
            inner_iters_since_last_mutation = 0
            outer_iter += 1

    # ── internals ────────────────────────────────────────────

    def _fairness_excerpt(self, prompt_version: str) -> str:
        """Read the current prompt version's fairness slot (for Level 1
        context). Returns empty on any error; Level 1 tolerates that."""
        try:
            from ..prompts.loader import PromptLoader
            loader = PromptLoader(prompt_version)
            if "fairness" in loader.frozen_slot_names():
                return loader.slot_text("fairness")
        except Exception:
            pass
        return ""

    def _behavioral_block(
        self, top_candidates: Sequence[Dict[str, Any]], strategy_card,
    ) -> str:
        """Render behavioral signals honoring include_signals (v3 §4.1).

        Single-candidate (top-1) Tier 1+2+3 summary. Empty string if
        the strategy_card or candidates are unavailable.
        """
        if not top_candidates or strategy_card is None:
            return ""
        try:
            from .behavioral_summary import format_behavioral_block
            top1 = top_candidates[0]
            include = getattr(strategy_card, "include_signals", ["scalar"])
            return format_behavioral_block(
                top1, include_signals=include,
            )
        except Exception as e:
            _log.warning(
                "behavioral block render failed: %s: %s", type(e).__name__, e,
            )
            return ""

    def _llm_config_for(self, prompt_version: str) -> LLMConfig:
        """Clone ``self.llm_config`` with a different prompt_version."""
        new = copy.copy(self.llm_config)
        new.prompt_version = prompt_version
        return new

    def _collect_candidate_metrics(self, inner_dir: Path) -> List[Dict[str, Any]]:
        """Read every ``iter_*/candidate_*_metrics.json`` the inner loop wrote."""
        out: List[Dict[str, Any]] = []
        for iter_dir in sorted(inner_dir.glob("iter_*")):
            for p in sorted(iter_dir.glob("candidate_*_metrics.json")):
                try:
                    out.append(json.loads(p.read_text()))
                except Exception as e:
                    _log.warning("Skipping unreadable %s: %s", p, e)
        return out

    def _top_candidates(
        self, candidate_metrics: Sequence[Dict[str, Any]], inner_dir: Path,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Top-k by (M1, M6, M2) with their source code attached."""
        valid = [m for m in candidate_metrics if "_error" not in m]
        valid.sort(
            key=lambda m: (
                m.get("M1_success_rate", 0.0),
                m.get("M6_coverage_progress", 0.0),
                m.get("M2_avg_return", -1e9),
            ),
            reverse=True,
        )
        top = valid[:k]
        for m in top:
            iter_idx = m.get("_iter_idx")
            cand_idx = m.get("_candidate_idx")
            # Best-effort — inner loop may not record these; we'll just
            # skip code attachment if indices aren't present.
            if iter_idx is not None and cand_idx is not None:
                rp = inner_dir / f"iter_{iter_idx}" / f"candidate_{cand_idx}_reward.py"
                op = inner_dir / f"iter_{iter_idx}" / f"candidate_{cand_idx}_obs.py"
                if rp.exists():
                    m["reward_code"] = rp.read_text()
                if op.exists():
                    m["obs_code"] = op.read_text()
        return top

    def _persist(
        self, history: List[TemplateRecord], current_version: str,
        total_candidates: int,
    ) -> None:
        (self.output_dir / "history.json").write_text(json.dumps({
            "current_version": current_version,
            "total_inner_candidates": total_candidates,
            "history": [_record_to_dict(r) for r in history],
        }, indent=2))
