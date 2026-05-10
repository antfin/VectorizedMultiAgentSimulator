"""Submit page — 5-step workflow.

Five always-visible cards:

1. **Pick** — scenario / folder / config cascade picker.
2. **Inspect & edit** — pre-filled accordion form with dirty detection.
3. **Save** — auto-skipped if clean; "save as new" forced if edits exist.
4. **Preflight** — three rolled-up LEDs (Configuration / System / Storage)
   driven by real probes; clicking a failure expands its root-cause detail.
5. **Submit** — gated until preflight green; dispatches to LocalRunner or
   OvhRunner depending on ``SubmitState.submit_target()``.

Each card shows a status badge derived from :class:`SubmitState`.
"""

# pylint: disable=wrong-import-position,invalid-name

import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.application.factories import (
    available_algorithms,
    available_runners,
    available_scenarios,
    runner_spec,
)
from multi_scenario.domain.models import ExperimentConfig
from multi_scenario.frontend.forms import (
    render_algorithm_params,
    render_scenario_params,
)
from multi_scenario.frontend.preflight import (
    applicable_checks,
    category_status,
    CheckStatus,
    group_by_category,
    run_real_local_checks,
    run_real_ovh_checks,
)
from multi_scenario.frontend.sidebar import (
    active_experiments_dir,
    render_active_root_caption,
)
from multi_scenario.frontend.submit_workflow import (
    diff_summary,
    list_configs_grouped,
    SubmitState,
)
from multi_scenario.frontend.theme import apply_theme
from pydantic import ValidationError

apply_theme(title="Submit", subtitle="Configure and launch a new run")
render_active_root_caption()

state = SubmitState.load()
experiments_dir = active_experiments_dir()


# ── Helpers ──────────────────────────────────────────────────────────


_BIG_LED_GLYPH = {
    CheckStatus.IDLE: "⚪",
    CheckStatus.CHECKING: "🟡",
    CheckStatus.PASS: "🟢",
    CheckStatus.FAIL: "🔴",
}
_STATUS_COLOR = {
    CheckStatus.IDLE: "#9E9E9E",
    CheckStatus.CHECKING: "#F39200",
    CheckStatus.PASS: "#1B873B",
    CheckStatus.FAIL: "#C62828",
}


def _render_status_card(label: str, checks: list) -> None:
    """One bordered card per preflight category.

    Layout: big LED + category name on the left, "passed/total" summary on
    the right. On failure, a divider appears and each failed sub-check is
    rendered as an ``st.error`` block carrying the underlying probe's
    root-cause message — that's the whole point of the rollup, the user
    sees *why* the LED went red without hunting through 13 rows.
    Idle (not yet run) and all-pass states stay quiet so a green page
    really feels green.
    """
    cat_status = category_status(checks)
    failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
    idle = sum(1 for c in checks if c.status == CheckStatus.IDLE)
    total = len(checks)
    color = _STATUS_COLOR[cat_status]
    glyph = _BIG_LED_GLYPH[cat_status]
    if cat_status == CheckStatus.PASS:
        summary = f"✓ All {total} checks passed"
    elif cat_status == CheckStatus.FAIL:
        summary = f"✗ {failed} of {total} failed"
    elif cat_status == CheckStatus.CHECKING:
        summary = f"… {total - idle - failed} of {total} running"
    else:
        summary = "Not yet run"

    with st.container(border=True):
        head = st.columns([0.7, 5, 3])
        head[0].markdown(
            f"<div style='font-size: 2.2rem; line-height: 1;'>{glyph}</div>",
            unsafe_allow_html=True,
        )
        head[1].markdown(
            f"<div style='font-size: 1.15rem; font-weight: 600; "
            f"padding-top: 0.4rem;'>{label}</div>",
            unsafe_allow_html=True,
        )
        head[2].markdown(
            f"<div style='color: {color}; font-weight: 600; "
            f"text-align: right; padding-top: 0.6rem;'>{summary}</div>",
            unsafe_allow_html=True,
        )

        # Failed → expand inline with each failure's root-cause detail.
        # (Always-expanded by design — the user shouldn't have to click to
        # see what's broken.)
        if cat_status == CheckStatus.FAIL:
            for check in checks:
                if check.status == CheckStatus.FAIL:
                    st.error(f"**{check.name}** — {check.detail}", icon="🔴")
                elif check.status == CheckStatus.IDLE and check.detail:
                    # Cascade-blocked rows ("fix the OVH config row first")
                    # shown muted so the user understands they're parked.
                    st.caption(f"⚪ {check.name} — {check.detail}")
            other_pass = [c.name for c in checks if c.status == CheckStatus.PASS]
            if other_pass:
                st.caption(f"Also passed: {' · '.join(other_pass)}")
        # Pass → optional expander for those who want to see what ran.
        elif cat_status == CheckStatus.PASS:
            with st.expander(f"Show what was checked ({total})", expanded=False):
                for check in checks:
                    st.markdown(
                        f"<div style='padding: 0.15rem 0;'>🟢 "
                        f"<b>{check.name}</b> — "
                        f"<span style='color: #555;'>{check.detail}</span></div>",
                        unsafe_allow_html=True,
                    )


def _step_badge(step: int, *, done: bool, active: bool, blocked: bool) -> str:
    """Single-line header markdown for a step card."""
    if done:
        glyph = "✅"
    elif active:
        glyph = "🔵"
    elif blocked:
        glyph = "🔒"
    else:
        glyph = "⚪"
    return f"#### {glyph} Step {step}"


def _try_load_ovh_config() -> tuple[Any, str | None]:
    """Load ``configs/ovh.yaml`` from CWD; return ``(cfg, None)`` or ``(None, err)``.

    Used by the OVH preflight + submission flow on the Submit page.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import OvhJobConfig

    candidate = Path("configs/ovh.yaml")
    if not candidate.is_file():
        return (
            None,
            f"missing {candidate} — create it with your OVH job config "
            "(region/image/buckets/etc.)",
        )
    try:
        return OvhJobConfig.from_yaml(candidate), None
    except (OSError, ValueError) as exc:
        return None, f"failed to parse {candidate}: {exc}"


def _run_ovh_submission(cfg_dict: dict[str, Any], yaml_path_in_repo: str) -> None:
    """Submit an OVH job (no polling); record status + job_id for the panel.

    Caller-specific concerns only — error wrapping (catches Exception →
    session_state) + post-submission rendering. The actual orchestration
    lives in :func:`application.submission.submit_to_ovh` so the CLI's
    ``multi-scenario run --runner ovh`` and this page take the same path
    (F7.7.A6 hex compliance).

    ``yaml_path_in_repo`` is the per-experiment config relative to the repo
    root (NOT ``configs/ovh.yaml`` — that's the OVH deployment cfg loaded
    separately by ``OvhJobConfig.from_yaml``).
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.config_validation import validate_known_types
    from multi_scenario.application.submission import build_run_dir, submit_to_ovh

    cfg = ExperimentConfig.model_validate(cfg_dict)
    validate_known_types(cfg)
    ovh_cfg, err = _try_load_ovh_config()
    if err is not None:
        st.session_state["submit_submission_status"] = {
            "status": "crashed",
            "error": err,
            "traceback": "",
        }
        return

    _, run_dir = build_run_dir(cfg)
    # OVH-specific: ``run_dir`` lives in the container path
    # (``/workspace/results/<run_dir.name>/``), unreachable from the host. The
    # local pullback destination must mirror where Run Detail looks for runs:
    # next to the YAML's experiment folder. Smoke 4 lesson, 2026-05-10.
    yaml_repo = Path(yaml_path_in_repo) if yaml_path_in_repo else None
    pullback_dir = (
        yaml_repo.parent.parent / run_dir.name
        if yaml_repo and len(yaml_repo.parts) >= 3
        else Path("experiments") / run_dir.name
    )

    class _StreamlitLogger:
        # pylint: disable=missing-function-docstring,missing-class-docstring
        def info(self, msg: str) -> None:
            ...

        def debug(self, msg: str) -> None:
            ...

        def warning(self, msg: str) -> None:
            ...

        def error(self, msg: str) -> None:
            ...

    try:
        submission = submit_to_ovh(
            cfg,
            ovh_cfg=ovh_cfg,
            yaml_path_in_repo=yaml_path_in_repo,
            run_dir=run_dir,
            logger=_StreamlitLogger(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state["submit_submission_status"] = {
            "status": "crashed",
            "run_id": f"{cfg.experiment.id}_s{cfg.experiment.seed}",
            "run_dir": str(run_dir),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return

    st.session_state["submit_submission_status"] = {
        "status": "submitted",
        "runner": "ovh",
        "run_id": str(submission.run_id),
        "run_dir": str(submission.run_dir),
        "pullback_dir": str(pullback_dir),
        "job_id": submission.job_id,
        "s3_prefix": submission.s3_prefix,
        "dashboard_url": submission.dashboard_url,
    }


def _refresh_ovh_status() -> None:
    """Poll the OVH job once; if DONE → pullback + (best-effort) regen videos.

    Triggered by the Refresh button in the in-flight submission panel. Uses
    the per-run S3 prefix established at submit time (Stage 1) so the local
    destination (``run_dir``) and S3 source share one identifier — pullback
    materialises the OVH run-folder where Run Detail expects it.

    Failure modes:
    - Job still running: status stays ``submitted``; user clicks again later.
    - Job FAILED/KILLED: status becomes ``crashed`` with logs tail.
    - Pullback fails: status becomes ``crashed`` with the exception; the OVH
      job itself stays DONE on OVH side, user can retry via CLI.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.adapters.runners.ovh_cli import OvhClient
    from multi_scenario.application.ovh_pullback import pullback_run_dir

    sub = st.session_state.get("submit_submission_status") or {}
    if sub.get("status") != "submitted" or sub.get("runner") != "ovh":
        return  # button shouldn't have been visible — defensive no-op

    job_id = sub["job_id"]
    # ``pullback_dir`` is the LOCAL destination (set by _run_ovh_submission).
    # Falling back to ``run_dir`` keeps backwards-compat with older session
    # state shapes — but that path may be a container path, in which case
    # pullback will fail (the user can hit Refresh again after a re-submit).
    run_dir = Path(sub.get("pullback_dir") or sub["run_dir"])
    client = OvhClient()
    try:
        info = client.get(job_id)
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state["submit_submission_status"] = {
            **sub,
            "status": "crashed",
            "error": f"status check failed: {exc}",
            "traceback": traceback.format_exc(),
        }
        return

    if not info.is_terminal:
        # Still running — leave status untouched, just let the rerun show the
        # latest state string in the panel.
        sub["state"] = info.state
        st.session_state["submit_submission_status"] = sub
        return

    if info.state.upper() != "DONE":
        # FAILED / KILLED / etc. — fetch a logs tail for the panel.
        try:
            tail = client.logs(job_id, tail=200)
        except Exception:  # pylint: disable=broad-except
            tail = "(could not fetch logs)"
        st.session_state["submit_submission_status"] = {
            **sub,
            "status": "crashed",
            "error": f"OVH job ended in state={info.state}",
            "traceback": tail,
        }
        return

    # DONE — pull the run-folder back so Run Detail sees it.
    ovh_cfg, err = _try_load_ovh_config()
    if err is not None:
        st.session_state["submit_submission_status"] = {
            **sub,
            "status": "crashed",
            "error": err,
            "traceback": "",
        }
        return
    try:
        result = pullback_run_dir(
            ovh_cfg=ovh_cfg,
            run_dir_name=run_dir.name,
            dest_dir=run_dir,
            client=client,
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.session_state["submit_submission_status"] = {
            **sub,
            "status": "crashed",
            "error": f"pullback failed: {exc}",
            "traceback": traceback.format_exc(),
        }
        return

    # Best-effort video regen: container may have been headless; surface a
    # warning if it fails but don't flip status to crashed (results are
    # already on disk, user can hit the Run Detail button to retry).
    # Use ``sys.executable -m multi_scenario.cli`` instead of the
    # ``multi-scenario`` console script — when streamlit launches outside an
    # active virtualenv, the script isn't on PATH (Smoke 4 lesson, 2026-05-10).
    regen_warning: str | None = None
    if not _videos_present(run_dir):
        # pylint: disable=import-outside-toplevel
        import subprocess
        import sys

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "multi_scenario.cli",
                    "regenerate-videos",
                    str(run_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                regen_warning = (proc.stderr or proc.stdout or "(no output)")[:2000]
        except OSError as exc:
            regen_warning = f"could not invoke regenerate-videos: {exc}"

    st.session_state["submit_submission_status"] = {
        **sub,
        "status": "done",
        "pullback_n_downloaded": result.n_downloaded,
        "pullback_n_skipped": result.n_skipped,
        "regen_warning": regen_warning,
    }


def _videos_present(run_dir: Path) -> bool:
    """True iff ``run_dir/videos/`` has ≥1 mp4 — used to skip needless regen."""
    videos_dir = run_dir / "videos"
    if not videos_dir.is_dir():
        return False
    return any(p.suffix.lower() == ".mp4" for p in videos_dir.iterdir())


def _run_local_submission(cfg_dict: dict[str, Any]) -> None:
    """Execute the run synchronously via LocalRunner; record status to session.

    Caller-specific concerns only (st.spinner + session_state UX). The
    actual orchestration lives in :func:`application.submission.submit_to_local`
    so the CLI's ``multi-scenario run --runner local`` and this page take
    the same path (F7.7.A6 hex compliance).

    Synchronous v1: blocks the page with ``st.spinner`` for the duration.
    Background-thread + live-tail variant is **deferred** to a future
    iteration; for now this is the simplest reliable path. The status
    panel below the Submit button reads the session_state entry this writes.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.config_validation import validate_known_types
    from multi_scenario.application.submission import build_run_dir, submit_to_local

    cfg = ExperimentConfig.model_validate(cfg_dict)
    validate_known_types(cfg)
    # Local path needs the dir on disk for FileLogger + LocalRunner outputs.
    run_id, run_dir = build_run_dir(cfg, mkdir=True)

    st.session_state["submit_submission_status"] = {
        "status": "running",
        "run_id": str(run_id),
        "run_dir": str(run_dir),
    }

    with st.spinner(f"Running {run_id} — this blocks the tab until done."):
        try:
            logger = FileLogger(run_dir / "logs" / "run.log")
            submission = submit_to_local(cfg, run_dir=run_dir, logger=logger)
            st.session_state["submit_submission_status"] = {
                "status": "done",
                "run_id": submission.run_id,
                "run_dir": str(submission.run_dir),
            }
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state["submit_submission_status"] = {
                "status": "crashed",
                "run_id": str(run_id),
                "run_dir": str(run_dir),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }


def _build_cfg_dict(form: dict[str, Any]) -> dict[str, Any]:
    """Form-state dict → ExperimentConfig-shaped dict for validation/YAML.

    The form stores nested values keyed by section; this re-assembles them
    into the canonical YAML shape so we can hand it to Pydantic and dump it.
    """
    out: dict[str, Any] = {
        "experiment": {
            "id": form["experiment"]["id"],
            "seed": int(form["experiment"]["seed"]),
        },
        "scenario": {
            "type": form["scenario"]["type"],
            "params": form["scenario"]["params"],
        },
        "algorithm": {
            "type": form["algorithm"]["type"],
            "params": form["algorithm"].get("params", {}),
        },
        "training": form["training"],
        "evaluation": form["evaluation"],
        "runtime": form["runtime"],
    }
    name = form["experiment"].get("name")
    description = form["experiment"].get("description")
    if name:
        out["experiment"]["name"] = name
    if description:
        out["experiment"]["description"] = description
    return out


# ───────────────────────────────────────────────────────────────────────
# Step 1 — Pick a config
# ───────────────────────────────────────────────────────────────────────
with st.container(border=True):
    s1_done = state.selected_path is not None
    s1_active = state.active_step == 1
    cols = st.columns([6, 2])
    cols[0].markdown(_step_badge(1, done=s1_done, active=s1_active, blocked=False))
    cols[0].markdown("**Pick a config**")
    if s1_done:
        cols[1].caption(f"📄 {state.selected_path.name}")

    if s1_done and not s1_active:
        # Collapsed summary mode — show selected path + "Change selection" link.
        st.caption(f"`{state.selected_path}` (relative to experiments root)")
        if st.button("Change selection", key="step1_change"):
            SubmitState.reset()
            st.rerun()
    else:
        # Active mode — render the cascading picker. Scenarios come from the
        # backend factory so adding one in ``adapters/scenarios/`` shows up
        # here automatically (F7.7.B2).
        SCENARIOS = tuple(available_scenarios())
        pick_cols = st.columns(3)
        scenario = pick_cols[0].selectbox(
            "Scenario", SCENARIOS, format_func=str.capitalize, key="step1_scenario"
        )
        folder_map = list_configs_grouped(experiments_dir, scenario)
        folder_options = list(folder_map.keys())
        if not folder_options:
            pick_cols[1].selectbox(
                "Folder", ["—"], disabled=True, key="step1_folder_empty"
            )
            pick_cols[2].selectbox(
                "Config", ["—"], disabled=True, key="step1_config_empty"
            )
            st.info(
                f"No configs found under `{experiments_dir / scenario}`. "
                "Add a YAML under `<scenario>/<folder>/configs/` to populate."
            )
        else:
            folder = pick_cols[1].selectbox(
                "Folder", folder_options, key="step1_folder"
            )
            yamls = folder_map[folder]
            yaml_names = [p.name for p in yamls]
            choice = pick_cols[2].selectbox("Config", yaml_names, key="step1_config")
            picked_path = next(p for p in yamls if p.name == choice)

            with st.expander("Preview YAML", expanded=False):
                try:
                    st.code(picked_path.read_text(encoding="utf-8"), language="yaml")
                except OSError as exc:
                    st.error(f"Couldn't read: {exc}")

            if st.button("Use this config", key="step1_pick", type="primary"):
                try:
                    cfg_data = yaml.safe_load(picked_path.read_text(encoding="utf-8"))
                    SubmitState.set_selected(picked_path, cfg_data)
                    st.rerun()
                except (OSError, yaml.YAMLError) as exc:
                    st.error(f"Couldn't parse YAML: {exc}")

# ───────────────────────────────────────────────────────────────────────
# Step 2 — Inspect & edit
# ───────────────────────────────────────────────────────────────────────
with st.container(border=True):
    s2_blocked = state.selected_path is None
    s2_active = state.active_step == 2  # never the "active" step strictly
    cols = st.columns([6, 2])
    cols[0].markdown(_step_badge(2, done=False, active=False, blocked=s2_blocked))
    cols[0].markdown("**Inspect & edit**")
    if state.is_dirty:
        cols[1].markdown(
            "<span style='color:#F39200; font-weight:600;'>● Modified</span>",
            unsafe_allow_html=True,
        )

    if s2_blocked:
        st.caption("Pick a config in Step 1 first.")
    else:
        cfg_data = state.snapshot_form or {}
        with st.expander("Open to view / edit", expanded=False):
            # Identity
            st.markdown("**Identity**")
            id_cols = st.columns([2, 1])
            exp_id = id_cols[0].text_input(
                "Experiment ID",
                value=cfg_data.get("experiment", {}).get("id", "my_experiment"),
                key="step2_exp_id",
            )
            seed = id_cols[1].number_input(
                "Seed",
                min_value=0,
                step=1,
                value=int(cfg_data.get("experiment", {}).get("seed", 0)),
                key="step2_seed",
            )
            exp_name = st.text_input(
                "Name (optional)",
                value=cfg_data.get("experiment", {}).get("name", "") or "",
                key="step2_name",
            )
            exp_desc = st.text_area(
                "Description (optional)",
                value=cfg_data.get("experiment", {}).get("description", "") or "",
                key="step2_desc",
                height=80,
            )

            # Scenario — type list + per-key widgets all flow from
            # ``available_scenarios()`` + ``Scenario.default_params()`` so a
            # new scenario adapter shows up here without editing this file.
            st.markdown("---")
            st.markdown("**Scenario**")
            scen_choices = available_scenarios()
            scen_type_default = cfg_data.get("scenario", {}).get(
                "type", scen_choices[0]
            )
            scen_type = st.selectbox(
                "Scenario type",
                scen_choices,
                index=scen_choices.index(scen_type_default)
                if scen_type_default in scen_choices
                else 0,
                format_func=str.capitalize,
                key="step2_scen_type",
            )
            scen_params = render_scenario_params(
                scen_type,
                cfg_data.get("scenario", {}).get("params", {}),
                key_prefix=f"step2_scen_{scen_type}",
            )

            # Algorithm — same data-driven pattern as scenarios.
            st.markdown("---")
            st.markdown("**Algorithm**")
            algo_choices = available_algorithms()
            algo_type_default = cfg_data.get("algorithm", {}).get(
                "type", algo_choices[0]
            )
            algo_type = st.selectbox(
                "Algorithm type",
                algo_choices,
                index=algo_choices.index(algo_type_default)
                if algo_type_default in algo_choices
                else 0,
                format_func=str.upper,
                key="step2_algo_type",
            )
            algo_params = render_algorithm_params(
                algo_type,
                cfg_data.get("algorithm", {}).get("params", {}),
                key_prefix=f"step2_algo_{algo_type}",
            )

            # Training
            st.markdown("---")
            st.markdown("**Training**")
            tr_defaults = cfg_data.get("training", {})
            tr_cols = st.columns(3)
            max_iters = tr_cols[0].number_input(
                "max_iters",
                min_value=1,
                value=int(tr_defaults.get("max_iters", 10)),
                key="step2_max_iters",
            )
            num_envs = tr_cols[1].number_input(
                "num_envs",
                min_value=1,
                value=int(tr_defaults.get("num_envs", 1)),
                key="step2_num_envs",
            )
            device = tr_cols[2].selectbox(
                "device",
                ["cpu", "cuda"],
                index=0 if tr_defaults.get("device", "cpu") == "cpu" else 1,
                key="step2_device",
            )
            tr_cols2 = st.columns(3)
            fpb = tr_cols2[0].number_input(
                "frames_per_batch",
                min_value=1,
                value=int(tr_defaults.get("frames_per_batch", 200)),
                key="step2_fpb",
            )
            mbs = tr_cols2[1].number_input(
                "minibatch_size",
                min_value=1,
                value=int(tr_defaults.get("minibatch_size", 100)),
                key="step2_mbs",
            )
            nmb = tr_cols2[2].number_input(
                "n_minibatch_iters",
                min_value=1,
                value=int(tr_defaults.get("n_minibatch_iters", 1)),
                key="step2_nmb",
            )

            # Evaluation
            st.markdown("---")
            st.markdown("**Evaluation**")
            ev_defaults = cfg_data.get("evaluation", {})
            ev_cols = st.columns(3)
            ev_int = ev_cols[0].number_input(
                "interval_iters",
                min_value=1,
                value=int(ev_defaults.get("interval_iters", 1)),
                key="step2_eval_int",
            )
            ev_eps = ev_cols[1].number_input(
                "episodes",
                min_value=1,
                value=int(ev_defaults.get("episodes", 10)),
                key="step2_eval_eps",
            )
            rec_default = (
                cfg_data.get("runtime", {})
                .get("runner", {})
                .get("params", {})
                .get("record_video", not exp_id.endswith("_smoke"))
            )
            rec_video = ev_cols[2].checkbox(
                "record_video",
                value=bool(rec_default),
                key="step2_rec_video",
            )

            # Submit target — UI-only choice, **never** written to the YAML.
            # The YAML's ``runtime.runner.type`` is always ``local`` (LocalRunner
            # is what runs the training loop, on the user's box for local
            # submits and inside the container for OVH submits). This radio
            # only affects which Python class instantiates at submit time.
            st.markdown("---")
            st.markdown("**Submit target**")
            current_target = SubmitState.submit_target()
            runner_choices = available_runners()
            submit_target_choice = st.radio(
                "Where should this run be dispatched?",
                options=runner_choices,
                index=runner_choices.index(current_target)
                if current_target in runner_choices
                else 0,
                horizontal=True,
                key="step2_submit_target",
                help=(
                    "UI-only choice — the YAML's `runtime.runner.type` "
                    "stays `local` either way (LocalRunner runs the training "
                    "loop in both cases). The radio only changes which "
                    "orchestrator wraps the submission. New runner adapters "
                    "appear here automatically."
                ),
            )
            if submit_target_choice != current_target:
                SubmitState.set_submit_target(submit_target_choice)
                # No rerun — Step 2 will re-render once and downstream steps
                # pick up the new target on the next event cycle.
            # Spread loaded runner.params first so YAML fields the form
            # doesn't render round-trip; only override the ones we render.
            cfg_runner_params = (
                cfg_data.get("runtime", {}).get("runner", {}).get("params", {})
            )
            runner_params: dict[str, Any] = {
                **cfg_runner_params,
                "record_video": bool(rec_video),
            }

            # Storage
            st.markdown("---")
            st.markdown("**Storage**")
            storage_default = cfg_data.get("runtime", {}).get("storage", {})
            # ``runtime.storage`` only ever has ``{type: fs, path, params}`` —
            # S3 / bucket / region details live in ``configs/ovh.yaml`` (loaded
            # by OvhRunner), NOT here. For OVH-targeted runs, ``path`` is the
            # *container mount* (sourced from OvhJobConfig.mount_results so
            # nothing here is hardcoded); for local it's a repo-relative dir.
            # ``runner_spec(...).requires_ovh_cfg`` keeps the OVH-config load
            # off the local path without an ``if target == "ovh"`` branch.
            ovh_cfg_for_default, _ = (
                _try_load_ovh_config()
                if runner_spec(submit_target_choice).requires_ovh_cfg
                else (None, None)
            )
            default_path = storage_default.get("path") or (
                ovh_cfg_for_default.mount_results
                if ovh_cfg_for_default is not None
                else f"experiments/{scen_type}/baseline"
            )
            storage_path = st.text_input(
                "path",
                value=default_path,
                key="step2_storage_path",
                help=(
                    "OVH: container mount path — defaults to "
                    "`OvhJobConfig.mount_results` from `configs/ovh.yaml`. "
                    "Local: repo-relative directory."
                ),
            )
            storage_extra: dict[str, Any] = {"path": storage_path}

            # Assemble the live form value. For ``training`` + ``evaluation``
            # we spread cfg_data first so YAML fields the form doesn't render
            # (``lr``, ``gamma``, ``share_policy_params``,
            # ``checkpoint_interval_iters``, …) round-trip cleanly — otherwise
            # loading any YAML that uses these would falsely flag dirty.
            cfg_training = cfg_data.get("training", {})
            cfg_evaluation = cfg_data.get("evaluation", {})
            current = {
                "experiment": {
                    "id": exp_id,
                    "seed": int(seed),
                    "name": exp_name or None,
                    "description": exp_desc or None,
                },
                "scenario": {"type": scen_type, "params": scen_params},
                "algorithm": {"type": algo_type, "params": algo_params},
                "training": {
                    **cfg_training,
                    "max_iters": int(max_iters),
                    "num_envs": int(num_envs),
                    "device": device,
                    "frames_per_batch": int(fpb),
                    "minibatch_size": int(mbs),
                    "n_minibatch_iters": int(nmb),
                },
                "evaluation": {
                    **cfg_evaluation,
                    "interval_iters": int(ev_int),
                    "episodes": int(ev_eps),
                },
                "runtime": {
                    # F7.7.A4: ``runner.type`` drives ``multi-scenario run``
                    # dispatch. The Submit page seeds ``submit_target`` from
                    # the YAML on load and the radio overrides per session;
                    # both end up pointing at the same value, so we emit it
                    # back into the form output to keep the YAML round-trip
                    # consistent (loading a runner.type=ovh YAML and clicking
                    # Save-as-new must produce a runner.type=ovh YAML, not
                    # silently rewrite to local).
                    "runner": {"type": submit_target_choice, "params": runner_params},
                    # ``runtime.storage`` is always ``fs`` per StorageSection
                    # schema. S3 details for OVH live separately in
                    # ``configs/ovh.yaml``, loaded by OvhRunner.
                    "storage": {"type": "fs", "params": {}, **storage_extra},
                },
            }
            # Drop None entries from experiment so dirty-comparison vs the
            # snapshot (which doesn't carry None placeholders) stays accurate.
            current["experiment"] = {
                k: v for k, v in current["experiment"].items() if v is not None
            }
            SubmitState.set_current(current)
            state = SubmitState.load()  # refresh derived properties

        if state.is_dirty:
            changes = diff_summary(state.snapshot_form, state.current_form)
            st.caption(
                f"**Modified fields:** {', '.join(changes[:10])}"
                + (f" *(+{len(changes) - 10} more)*" if len(changes) > 10 else "")
            )

# ───────────────────────────────────────────────────────────────────────
# Step 3 — Save
# ───────────────────────────────────────────────────────────────────────
with st.container(border=True):
    s3_blocked = state.selected_path is None
    s3_active = state.active_step == 3
    s3_done = state.saved_path is not None or (
        state.selected_path is not None and not state.is_dirty
    )
    cols = st.columns([6, 2])
    cols[0].markdown(_step_badge(3, done=s3_done, active=s3_active, blocked=s3_blocked))
    cols[0].markdown("**Save**")

    if s3_blocked:
        st.caption("Pick a config in Step 1 first.")
    elif state.saved_path is not None:
        st.caption(f"✓ Saved as `{state.saved_path.name}` in same folder.")
        st.caption(f"`{state.saved_path}`")
    elif not state.is_dirty:
        st.caption(
            "✓ No changes — using the original config "
            f"(`{state.selected_path.name}`)."
        )
    else:
        original = state.selected_path
        st.warning(
            "You've edited the config. **Save as a new file before continuing** — "
            "we never overwrite the source."
        )
        default_name = f"{original.stem}_v2.yaml"
        save_cols = st.columns([3, 1])
        new_name = save_cols[0].text_input(
            "New filename",
            value=default_name,
            key="step3_new_name",
            help=f"Will be written next to `{original.name}` in `{original.parent}`.",
        )
        same_name = new_name.strip() == original.name
        if save_cols[1].button(
            "Save as new",
            disabled=same_name or not new_name.strip().endswith(".yaml"),
            type="primary",
            key="step3_save",
        ):
            target = original.parent / new_name.strip()
            try:
                cfg_dict = _build_cfg_dict(state.current_form)
                # Validate before writing — bad YAML on disk is worse than bad UI.
                ExperimentConfig.model_validate(cfg_dict)
                target.write_text(
                    yaml.dump(cfg_dict, sort_keys=False), encoding="utf-8"
                )
                SubmitState.mark_saved(target, state.current_form)
                st.success(f"Saved → `{target}`")
                st.rerun()
            except ValidationError as exc:
                st.error("Config has validation errors; fix them in Step 2 first.")
                for err in exc.errors():
                    path = ".".join(str(p) for p in err["loc"])
                    st.markdown(f"- **`{path}`** — {err['msg']}")
            except OSError as exc:
                st.error(f"Couldn't write file: {exc}")
        if same_name:
            st.caption("Pick a different name — overwriting the source isn't allowed.")

# ───────────────────────────────────────────────────────────────────────
# Step 4 — Preflight
# ───────────────────────────────────────────────────────────────────────
with st.container(border=True):
    s4_blocked = state.selected_path is None or (
        state.is_dirty and state.saved_path is None
    )
    s4_active = state.active_step == 4
    s4_done = state.has_preflight_passed
    cols = st.columns([6, 2])
    cols[0].markdown(_step_badge(4, done=s4_done, active=s4_active, blocked=s4_blocked))
    cols[0].markdown("**Preflight checks**")

    if s4_blocked:
        st.caption("Complete steps 1–3 first.")
    else:
        runner_now = SubmitState.submit_target()
        # Pass the live form so conditional rows (GPU-only-when-cuda, etc.)
        # appear/disappear as the user toggles between cpu/cuda.
        expected = applicable_checks(runner_now, state.current_form)
        if not state.preflight or {c.name for c in state.preflight} != {
            c.name for c in expected
        }:
            SubmitState.set_preflight(expected)
            state = SubmitState.load()

        run_cols = st.columns([1.2, 4, 3])
        run_clicked = run_cols[0].button(
            "Run preflight", key="step4_run", type="primary"
        )
        run_cols[1].caption(
            f"Submit target: **{runner_now}** · "
            f"{len(state.preflight)} checks "
            "(toggle in Step 2 to switch local/OVH)."
        )
        if run_clicked:
            cfg_dict = _build_cfg_dict(state.current_form) if state.current_form else {}
            if runner_now == "local":
                run_real_local_checks(state.preflight, cfg_dict)
            else:
                # OVH probes go through OvhClient (ovhai CLI shell-out) for
                # bucket / object verbs + code-hash compare. We pass
                # ``ovh_cfg`` (may be None) into ``run_real_ovh_checks`` —
                # it knows to cascade IDLE on the cloud-env rows when the
                # YAML doesn't load, while still running the config-row
                # probes regardless.
                ovh_cfg, _ovh_err = _try_load_ovh_config()
                repo_root = Path.cwd()
                active_path = state.active_config_path
                yaml_relpath: str | None
                try:
                    yaml_relpath = (
                        active_path.resolve()
                        .relative_to(repo_root.resolve())
                        .as_posix()
                        if active_path is not None
                        else None
                    )
                except ValueError:
                    # Active config sits outside the repo root → can't be in
                    # the code bucket regardless. Probe will report failure.
                    yaml_relpath = None
                run_real_ovh_checks(
                    state.preflight,
                    cfg_dict,
                    ovh_cfg=ovh_cfg,
                    repo_root=repo_root,
                    yaml_relpath=yaml_relpath,
                )
            SubmitState.set_preflight(state.preflight)
            st.rerun()

        st.markdown("")  # vertical breathing room
        for category_label, rows in group_by_category(state.preflight):
            _render_status_card(category_label, rows)

# ───────────────────────────────────────────────────────────────────────
# Step 5 — Submit
# ───────────────────────────────────────────────────────────────────────
with st.container(border=True):
    s5_blocked = (
        not state.has_preflight_passed
        or (state.is_dirty and state.saved_path is None)
        or state.selected_path is None
    )
    s5_active = state.active_step == 5
    cols = st.columns([6, 2])
    cols[0].markdown(_step_badge(5, done=False, active=s5_active, blocked=s5_blocked))
    cols[0].markdown("**Submit**")

    if s5_blocked:
        st.caption("Complete steps 1–4 first.")
    else:
        cfg_dict = _build_cfg_dict(state.current_form) if state.current_form else None
        runner_now = SubmitState.submit_target()

        if runner_now == "local":
            st.caption(
                "All checks green — submitting runs **synchronously** in this "
                "page; long runs block the browser tab. Background-thread + "
                "live tail variant is deferred."
            )
        else:
            st.caption(
                "All OVH preflight checks green — clicking Submit will queue the "
                "job via `ovhai job run` and return a `job_id`. The job runs "
                "asynchronously; pull results back later via "
                "`OvhRunner.run` / `multi-scenario sweep --follow`."
            )

        submit_cols = st.columns([1, 1, 5])

        if submit_cols[0].button(
            "🚀 Submit",
            type="primary",
            use_container_width=True,
            key="step5_submit",
        ):
            if runner_now == "local":
                _run_local_submission(cfg_dict)
            else:
                # Compute the active config's path relative to the repo
                # root — the OVH container's bash entry point joins this
                # with OvhJobConfig.mount_code to find the YAML on disk.
                repo_root = Path.cwd().resolve()
                active_path = state.active_config_path
                try:
                    yaml_relpath = (
                        active_path.resolve().relative_to(repo_root).as_posix()
                        if active_path is not None
                        else ""
                    )
                except ValueError:
                    yaml_relpath = ""
                _run_ovh_submission(cfg_dict, yaml_relpath)
            st.rerun()

        if cfg_dict is not None:
            submit_cols[1].download_button(
                "📥 Download YAML",
                data=yaml.dump(cfg_dict, sort_keys=False).encode("utf-8"),
                file_name=(state.saved_path or state.selected_path).name,
                mime="application/x-yaml",
                use_container_width=True,
                key="step5_download",
            )

    # Submission status panel (appears after a Submit click).
    sub_status = st.session_state.get("submit_submission_status")
    if sub_status:
        st.markdown("---")
        st.subheader("Last submission")
        if sub_status["status"] == "running":
            st.info("Running…")
        elif sub_status["status"] == "submitted":
            # OVH path — job has been queued, not yet run/finished.
            last_state = sub_status.get("state")
            state_suffix = f" — state `{last_state}`" if last_state else ""
            st.success(
                f"📤 SUBMITTED: `{sub_status['run_id']}` "
                f"→ job_id `{sub_status['job_id']}`{state_suffix}"
            )
            info_cols = st.columns(2)
            info_cols[0].caption(f"**S3 prefix** &nbsp; `{sub_status['s3_prefix']}`")
            info_cols[1].markdown(
                f"[Open in OVH dashboard ↗]({sub_status['dashboard_url']})"
            )
            refresh_col, _ = st.columns([1, 3])
            with refresh_col:
                if st.button(
                    "🔄 Refresh status",
                    key="refresh_ovh_status",
                    use_container_width=True,
                    help=(
                        "Re-checks the OVH job. When DONE, pulls results back "
                        "to the local run folder and regenerates missing videos."
                    ),
                ):
                    _refresh_ovh_status()
                    st.rerun()
            st.caption(
                "Click Refresh when the OVH dashboard shows the job near DONE. "
                "On DONE: results auto-pull to the local folder so the Run "
                "Detail page can read them."
            )
        elif sub_status["status"] == "done":
            display_dir = sub_status.get("pullback_dir") or sub_status["run_dir"]
            st.success(f"✅ DONE: `{sub_status['run_id']}` → `{display_dir}`")
            n_dl = sub_status.get("pullback_n_downloaded")
            n_sk = sub_status.get("pullback_n_skipped")
            if n_dl is not None:
                st.caption(
                    f"Pullback: {n_dl} files downloaded, {n_sk} skipped (already present)."
                )
            regen_warn = sub_status.get("regen_warning")
            if regen_warn:
                st.warning("Video regeneration failed — open Run Detail to retry.")
                with st.expander("regenerate-videos error output"):
                    st.code(regen_warn)
            st.markdown(
                f"[Open run detail →](/run_detail?run_id={sub_status['run_id']})"
            )
        elif sub_status["status"] == "crashed":
            st.error(f"❌ CRASHED: {sub_status.get('error', 'unknown error')}")
            tb = sub_status.get("traceback", "")
            if tb:
                st.code(tb, language="python")

# Footer — Start over
st.markdown("---")
if st.button("↺ Start over", help="Clear all selections and edits."):
    SubmitState.reset()
    st.rerun()
