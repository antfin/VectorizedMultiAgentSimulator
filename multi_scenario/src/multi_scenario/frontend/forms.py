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
