"""State + helpers for the Submit page's 5-step workflow.

The page is a guided sequence:

1. **Pick** — choose an existing config from disk (scenario / folder / file).
2. **Inspect & edit** — open the form, optionally tweak fields.
3. **Save** — required if edits were made (writes a new YAML, never overwrites).
4. **Preflight** — run consistency checks (config schema, storage writable,
   code-vs-bucket hash, results bucket reachable, …).
5. **Submit** — only enabled when all preflight checks pass.

This module owns the state shape and the small derived computations (is_dirty,
which step is "active", can_submit). Rendering lives in
``frontend/pages/submit.py``; per-section form fields in ``frontend/forms.py``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from multi_scenario.frontend.preflight import CheckStatus, PreflightCheck

# ── session_state keys (centralised so the page never types raw strings) ─
_SELECTED_PATH = "submit_selected_path"
_SNAPSHOT = "submit_snapshot_form"
_CURRENT = "submit_current_form"
_SAVED_PATH = "submit_saved_path"
_PREFLIGHT = "submit_preflight"
_SUBMIT_TARGET = "submit_target"  # "local" | "ovh" — UI-only, never touches the YAML


@dataclass
class SubmitState:
    """Snapshot of the workflow's session state, with derived properties."""

    selected_path: Path | None
    snapshot_form: dict[str, Any] | None
    current_form: dict[str, Any] | None
    saved_path: Path | None
    preflight: list[PreflightCheck] = field(default_factory=list)

    @classmethod
    def load(cls) -> "SubmitState":
        """Read the current state from ``st.session_state``."""
        return cls(
            selected_path=st.session_state.get(_SELECTED_PATH),
            snapshot_form=st.session_state.get(_SNAPSHOT),
            current_form=st.session_state.get(_CURRENT),
            saved_path=st.session_state.get(_SAVED_PATH),
            preflight=st.session_state.get(_PREFLIGHT, []),
        )

    @staticmethod
    def set_selected(path: Path, snapshot: dict[str, Any]) -> None:
        """Mark ``path`` as the active config; reset downstream state.

        The snapshot is **normalised** to match the form's always-rendered
        shape (e.g. ``runtime.runner.params.record_video`` is always present
        in the form output even when the YAML omitted it). Without this,
        every freshly-picked config would look "dirty" before the user has
        touched anything.

        ``submit_target`` is seeded from the loaded YAML's *raw* ``runner.type``
        (before normalisation rewrites it to ``local``) — a legacy file
        with ``runner.type: ovh`` is interpreted as "user intends OVH".
        """
        raw_runner = snapshot.get("runtime", {}).get("runner", {}).get("type", "local")
        target = "ovh" if raw_runner == "ovh" else "local"
        normalised = _normalise_snapshot(snapshot)
        st.session_state[_SELECTED_PATH] = path
        st.session_state[_SNAPSHOT] = normalised
        st.session_state[_CURRENT] = normalised
        st.session_state[_SAVED_PATH] = None
        st.session_state[_PREFLIGHT] = []
        st.session_state[_SUBMIT_TARGET] = target

    @staticmethod
    def set_current(values: dict[str, Any]) -> None:
        """Update the live form values (called after every form rerender).

        ``set_current`` runs on **every** rerun (Streamlit re-executes the
        whole script). To avoid clobbering a freshly-computed preflight on a
        rerun where the user didn't actually change anything, we only
        invalidate the preflight when ``values`` differs from the previous
        snapshot in session_state.
        """
        prev = st.session_state.get(_CURRENT)
        st.session_state[_CURRENT] = values
        if prev != values:
            # The user edited something → any prior preflight is stale.
            st.session_state[_PREFLIGHT] = []

    @staticmethod
    def mark_saved(path: Path, values: dict[str, Any]) -> None:
        """Record the new on-disk path; subsequent steps treat ``values`` as clean."""
        st.session_state[_SAVED_PATH] = path
        st.session_state[_SNAPSHOT] = values
        st.session_state[_CURRENT] = values

    @staticmethod
    def set_preflight(checks: list[PreflightCheck]) -> None:
        """Store the latest preflight rows back to session state."""
        st.session_state[_PREFLIGHT] = checks

    @staticmethod
    def reset() -> None:
        """Wipe the whole workflow (Start over button)."""
        for key in (
            _SELECTED_PATH,
            _SNAPSHOT,
            _CURRENT,
            _SAVED_PATH,
            _PREFLIGHT,
            _SUBMIT_TARGET,
        ):
            st.session_state.pop(key, None)

    # ── submit target (UI-only, doesn't write to YAML) ──────────────

    @staticmethod
    def submit_target() -> str:
        """Where the run will be dispatched: ``"local"`` or ``"ovh"``.

        This is a UI-level choice and **never** modifies the YAML's
        ``runtime.runner.type`` (which is always ``local`` — LocalRunner
        is what actually runs the training loop, whether on the user's
        laptop or inside an OVH container). Defaults to ``"local"`` until
        the user explicitly picks otherwise.
        """
        return st.session_state.get(_SUBMIT_TARGET, "local")

    @staticmethod
    def set_submit_target(target: str) -> None:
        """Update the submit target; invalidates preflight (different probes apply)."""
        if target not in ("local", "ovh"):
            raise ValueError(f"submit_target must be 'local' or 'ovh', got {target!r}")
        prev = st.session_state.get(_SUBMIT_TARGET, "local")
        st.session_state[_SUBMIT_TARGET] = target
        if prev != target:
            # Different runner → different probe set → wipe stale results.
            st.session_state[_PREFLIGHT] = []

    # ── derived ─────────────────────────────────────────────────────

    @property
    def is_dirty(self) -> bool:
        """True iff current form values differ from the snapshot."""
        if self.snapshot_form is None or self.current_form is None:
            return False
        return self.snapshot_form != self.current_form

    @property
    def has_preflight_passed(self) -> bool:
        """True iff preflight has been run AND every check is in PASS state."""
        return bool(self.preflight) and all(
            c.status == CheckStatus.PASS for c in self.preflight
        )

    @property
    def active_config_path(self) -> Path | None:
        """Whichever path is the current source of truth — saved one wins
        if a save happened, otherwise the originally-picked path.
        """
        return self.saved_path or self.selected_path

    @property
    def can_submit(self) -> bool:
        """All steps green: picked, edits saved (if any), preflight passed."""
        if self.selected_path is None:
            return False
        if self.is_dirty and self.saved_path is None:
            return False
        return self.has_preflight_passed

    @property
    def active_step(self) -> int:
        """Which step the user should focus on next (1..5)."""
        if self.selected_path is None:
            return 1
        if self.is_dirty and self.saved_path is None:
            return 3
        if not self.has_preflight_passed:
            return 4
        return 5


# ── Snapshot normalisation ───────────────────────────────────────────


def _normalise_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Make a YAML-loaded dict shape-compatible with the form's output.

    The Submit form always emits certain fields (``record_video``, empty
    ``params: {}`` blocks, the union of schema-defaults + YAML-overrides
    for scenario/algorithm params, …) even when the source YAML omits
    them. We pre-fill those on the snapshot so dirty-detection compares
    apples to apples — only flagging changes the user actually drove.
    """
    snap = copy.deepcopy(snapshot)
    runtime = snap.setdefault("runtime", {"runner": {"type": "local"}, "storage": {}})

    # ``runtime.runner.type`` drives dispatch (F7.7.A4): ``local`` →
    # LocalRunner runs training in this Python process; ``ovh`` → OvhRunner
    # submits to the cloud and the container internally runs LocalRunner.
    # The Submit-page radio overrides per-session via ``submit_target``;
    # the YAML's value is the default.
    runner = runtime.setdefault("runner", {"type": "local"})
    runner.setdefault("type", "local")  # honour YAML; default only when absent
    params = runner.setdefault("params", {})
    if "record_video" not in params:
        exp_id = snap.get("experiment", {}).get("id", "")
        params["record_video"] = not exp_id.endswith("_smoke")

    _fill_schema_defaults(snap)

    # Normalise empty-string optional experiment fields to "absent" so the
    # form (which emits None → drops them) stays consistent.
    exp = snap.setdefault("experiment", {})
    for opt_key in ("name", "description"):
        if exp.get(opt_key) in (None, ""):
            exp.pop(opt_key, None)
    return snap


def _fill_schema_defaults(snap: dict[str, Any]) -> None:
    """Fold ``Scenario.default_params()`` + ``Algorithm.default_params()`` into ``snap``.

    F7.7.B2: the data-driven form renders every key in
    ``default_params() ∪ YAML overrides`` — so we pre-fill the snapshot
    with the same union, otherwise loading any YAML that omits a schema
    key would falsely flag dirty.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.application.factories import (
        available_algorithms,
        available_scenarios,
        make_algorithm,
        make_scenario,
    )

    scen = snap.setdefault("scenario", {})
    scen_type = scen.get("type")
    if scen_type in available_scenarios():
        scen_params = scen.setdefault("params", {})
        for key, default in make_scenario(scen_type).default_params().items():
            scen_params.setdefault(key, default)
    algo = snap.setdefault("algorithm", {})
    algo_type = algo.get("type")
    if algo_type in available_algorithms():
        algo_params = algo.setdefault("params", {})
        for key, default in make_algorithm(algo_type).default_params().items():
            algo_params.setdefault(key, default)
    runtime = snap.setdefault("runtime", {})
    storage = runtime.setdefault("storage", {"type": "fs"})
    storage.setdefault("type", "fs")
    storage.setdefault("params", {})

    # F9.8: LERO + LLM section schema fill.
    # The Submit form's LERO/LLM widgets always render every field in
    # ``LeroSection`` / ``LlmSection`` (data-driven from Pydantic). If
    # the loaded YAML omits any of those keys, the widget output ends
    # up with MORE keys than the snapshot, falsely flagging dirty. We
    # only fill these when the YAML actually has the parent block —
    # non-LERO YAMLs stay untouched.
    if snap.get("lero"):
        # pylint: disable=import-outside-toplevel
        from multi_scenario.domain.models import LeroSection

        _fill_pydantic_defaults(snap["lero"], LeroSection)
    if snap.get("llm"):
        # pylint: disable=import-outside-toplevel
        from multi_scenario.domain.models import LlmSection

        _fill_pydantic_defaults(snap["llm"], LlmSection)


def _fill_pydantic_defaults(block: dict[str, Any], model_cls: Any) -> None:
    """Set every field in ``model_cls`` that ``block`` doesn't already have.

    Mirrors the F9.8 ``_schema_from_pydantic`` helper in
    :mod:`frontend.forms`: skips fields with ``None`` default (e.g.
    ``llm.api_base``, ``llm.seed``) so the snapshot's keys match the
    widget render's keys byte-for-byte — otherwise loading a YAML
    that omits an optional field falsely flags dirty. Keys with
    concrete defaults are filled with the model's declared value.
    """
    for name, info in model_cls.model_fields.items():
        if name in block:
            continue  # YAML had it; don't overwrite
        default = info.default
        if default is None or repr(default) == "PydanticUndefined":
            continue  # don't fill — widget skips these too
        block[name] = default


# ── Browser helpers ──────────────────────────────────────────────────


def list_configs_grouped(experiments_dir: Path, scenario: str) -> dict[str, list[Path]]:
    """Discover ``experiments/<scenario>/<folder>/configs/*.yaml``.

    Returns a mapping ``folder_name → sorted list of yaml paths`` so the
    page can render a folder-grouped picker. Folders without a ``configs/``
    subdir or with no YAMLs in it are silently skipped.
    """
    scenario_dir = experiments_dir / scenario
    if not scenario_dir.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    for folder in sorted(scenario_dir.iterdir()):
        if not folder.is_dir():
            continue
        configs_dir = folder / "configs"
        if not configs_dir.is_dir():
            continue
        yamls = sorted(configs_dir.glob("*.yaml"))
        if yamls:
            out[folder.name] = yamls
    return out


def diff_summary(snapshot: dict, current: dict, prefix: str = "") -> list[str]:
    """Walk both dicts and return dotted-path summaries of changed leaves.

    Used by the page to show a small "Modified: a.b.c, x.y" caption when
    the form is dirty so the user knows *what* they changed before saving.
    """
    changes: list[str] = []
    keys = set(snapshot) | set(current)
    for key in sorted(keys):
        path = f"{prefix}.{key}" if prefix else key
        a, b = snapshot.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.extend(diff_summary(a, b, path))
        elif a != b:
            changes.append(path)
    return changes
