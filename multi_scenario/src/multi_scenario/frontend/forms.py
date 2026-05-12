"""Data-driven form widgets for the Submit page (F7.7.B2).

The Submit page calls one of two entry points:

* :func:`render_scenario_params` — picks the schema from
  ``make_scenario(name).default_params()`` and renders one widget per key.
* :func:`render_algorithm_params` — same, via ``make_algorithm(name)``.

Both delegate to :func:`render_params_from_defaults`, which infers the
widget shape from each default value's Python type:

============= ====================
Default type  Widget
============= ====================
``bool``      ``st.checkbox``
``int``       ``st.number_input(step=1)``
``float``     ``st.number_input(step=0.01)``
``str``       ``st.text_input``
``list/dict`` JSON in ``st.text_area`` (``json.loads`` on read)
============= ====================

The loaded YAML's params (``overrides``) win over the schema default; any
extra key in the YAML that the schema doesn't know about is rendered too —
so editing a config never loses fields. Adding a new scenario / algorithm
adapter automatically gains a working form here without touching this file.
"""

import json
from typing import Any

import streamlit as st

from multi_scenario.application.factories import make_algorithm, make_scenario


def render_scenario_params(
    name: str, overrides: dict[str, Any], key_prefix: str
) -> dict[str, Any]:
    """Render scenario params: schema from ``Scenario.default_params()`` + YAML overrides."""
    schema = make_scenario(name).default_params()
    return render_params_from_defaults(schema, overrides, key_prefix)


def render_algorithm_params(
    name: str,
    overrides: dict[str, Any],
    key_prefix: str,
) -> dict[str, Any]:
    """Render algorithm params: schema from ``Algorithm.default_params()`` + YAML overrides."""
    schema = make_algorithm(name).default_params()
    return render_params_from_defaults(schema, overrides, key_prefix)


def render_lero_section(overrides: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """F9.8 — render :class:`LeroSection` fields as Submit-page widgets.

    Schema comes from the Pydantic model's field defaults so the widget
    set stays in sync when fields are added/removed. The user's YAML
    values (``overrides``) win over the schema defaults; any extra key
    in the loaded YAML that the schema doesn't know about still gets
    rendered (typed by its value) — same union semantics as scenario /
    algorithm params, so saving the YAML never silently drops fields.

    Called by the Submit page only when the loaded YAML carries a
    non-empty ``lero:`` block — non-LERO submissions skip this widget
    entirely.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import LeroSection

    return render_params_from_defaults(
        _schema_from_pydantic(LeroSection), overrides, key_prefix
    )


def render_llm_section(overrides: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """F9.8 — render :class:`LlmSection` fields as Submit-page widgets.

    Same data-driven shape as :func:`render_lero_section`. The
    ``api_base`` field defaults to ``None``; the schema-from-pydantic
    helper substitutes empty-string-as-seed so the user gets a normal
    text input that maps back to ``None`` only when blank.
    """
    # pylint: disable=import-outside-toplevel
    from multi_scenario.domain.models import LlmSection

    return render_params_from_defaults(
        _schema_from_pydantic(LlmSection), overrides, key_prefix
    )


def _schema_from_pydantic(model_cls: Any) -> dict[str, Any]:
    """Build a default-value dict from a Pydantic model's concrete defaults.

    Mirrors the shape expected by :func:`render_params_from_defaults`:
    each key → its default value, used by :func:`_render_one` to pick
    a widget type.

    **Skipped fields**:

    - Required fields with no default (``PydanticUndefined``) — rare in
      LERO/LLM sections; rendering as a text area would mismatch their
      eventual concrete type at validation time.
    - Fields with ``None`` default (e.g. ``llm.api_base``, ``llm.seed``).
      Rendering an empty text input for these would over-emit on the
      cfg-dict side (the snapshot wouldn't have them either, so dirty-
      detection misfires). When the YAML explicitly sets one, the
      ``overrides`` union in :func:`render_params_from_defaults` still
      surfaces it as a widget — the user can edit existing values, just
      not introduce a new optional field via the Submit form.
    """
    out: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        default = info.default
        if default is None or repr(default) == "PydanticUndefined":
            continue
        out[name] = default
    return out


def render_params_from_defaults(
    schema: dict[str, Any],
    overrides: dict[str, Any],
    key_prefix: str,
) -> dict[str, Any]:
    """Render one widget per key in ``schema ∪ overrides``; return the user's values.

    ``schema`` carries the canonical type per key (from the adapter's
    ``default_params()``). ``overrides`` is what the loaded YAML had — its
    values seed the widgets, and any keys it carries that the schema
    doesn't know about are still rendered (typed by the override value's
    Python type). Result preserves the union so saving the YAML never
    drops fields.

    Layout: 3 widgets per row.
    """
    # Union the keys, schema-first so well-known params lead the layout.
    all_keys: list[str] = list(schema)
    for key in overrides:
        if key not in all_keys:
            all_keys.append(key)

    out: dict[str, Any] = {}
    for row_start in range(0, len(all_keys), 3):
        row_keys = all_keys[row_start : row_start + 3]
        cols = st.columns(3)
        for col, key in zip(cols, row_keys):
            seed = overrides.get(key, schema.get(key))
            with col:
                out[key] = _render_one(key, seed, f"{key_prefix}__{key}")
    return out


def _render_one(label: str, seed: Any, widget_key: str) -> Any:
    """Pick a Streamlit widget based on ``seed``'s Python type and render it."""
    if isinstance(seed, bool):
        return st.checkbox(label, value=seed, key=widget_key)
    if isinstance(seed, int):
        # ``n_*`` and ``*_steps`` conventionally must be ≥ 1; everything else ≥ 0.
        # Conservative inference — keeps the widget useful without surprising
        # the user with arbitrary upper bounds.
        min_value = 1 if label.startswith("n_") or label.endswith("_steps") else 0
        return st.number_input(
            label,
            min_value=min_value,
            step=1,
            value=int(seed),
            key=widget_key,
        )
    if isinstance(seed, float):
        return st.number_input(
            label,
            value=float(seed),
            step=0.01,
            format="%g",
            key=widget_key,
        )
    if isinstance(seed, str):
        return st.text_input(label, value=seed, key=widget_key)
    # Lists / dicts → JSON text area; parse on read so the cfg dict gets
    # the right Python type back. Parse failure → surface in red, keep
    # the raw value so the user can fix it without losing input.
    raw = st.text_area(
        label,
        value=json.dumps(seed, indent=2),
        key=widget_key,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"`{label}` — invalid JSON: {exc}")
        return seed  # fall back to the seed so downstream validation has *something*
