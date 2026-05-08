"""Form helpers for the Submit page.

Each ``render_*_params`` function takes a ``defaults`` dict (used by the
clone/template pre-fill flow) and a ``key_prefix`` (so multiple form
instances on the same page don't collide), renders the relevant Streamlit
widgets, and returns the resulting params dict ready for plugging into
``ExperimentConfig``.

The dispatch tables :data:`SCENARIO_FORMS` and :data:`ALGORITHM_FORMS`
let the page render the right sub-form based on the user's type pick
without an explosion of ``if/elif`` branches.
"""

from typing import Any, Callable

import streamlit as st


def render_discovery_params(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Discovery scenario form fields."""
    cols = st.columns(3)
    n_agents = cols[0].number_input(
        "n_agents", min_value=1, max_value=20,
        value=int(defaults.get("n_agents", 2)),
        key=f"{key_prefix}__n_agents",
    )
    n_targets = cols[1].number_input(
        "n_targets", min_value=1, max_value=20,
        value=int(defaults.get("n_targets", 2)),
        key=f"{key_prefix}__n_targets",
    )
    agents_per_target = cols[2].number_input(
        "agents_per_target", min_value=1, max_value=10,
        value=int(defaults.get("agents_per_target", 2)),
        key=f"{key_prefix}__apt",
    )
    cols2 = st.columns(3)
    targets_respawn = cols2[0].checkbox(
        "targets_respawn",
        value=bool(defaults.get("targets_respawn", False)),
        key=f"{key_prefix}__respawn",
    )
    shared_reward = cols2[1].checkbox(
        "shared_reward",
        value=bool(defaults.get("shared_reward", True)),
        key=f"{key_prefix}__shared",
    )
    max_steps = cols2[2].number_input(
        "max_steps", min_value=1, max_value=10000,
        value=int(defaults.get("max_steps", 100)),
        key=f"{key_prefix}__steps",
    )
    return {
        "n_agents": n_agents,
        "n_targets": n_targets,
        "agents_per_target": agents_per_target,
        "targets_respawn": targets_respawn,
        "shared_reward": shared_reward,
        "max_steps": max_steps,
    }


def render_navigation_params(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Navigation scenario placeholder — same n_agents + max_steps; expand later."""
    cols = st.columns(2)
    n_agents = cols[0].number_input(
        "n_agents", min_value=1, max_value=20,
        value=int(defaults.get("n_agents", 2)),
        key=f"{key_prefix}__n_agents",
    )
    max_steps = cols[1].number_input(
        "max_steps", min_value=1, max_value=10000,
        value=int(defaults.get("max_steps", 100)),
        key=f"{key_prefix}__steps",
    )
    return {"n_agents": n_agents, "max_steps": max_steps}


def render_transport_params(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Transport scenario placeholder."""
    cols = st.columns(2)
    n_agents = cols[0].number_input(
        "n_agents", min_value=1, max_value=20,
        value=int(defaults.get("n_agents", 4)),
        key=f"{key_prefix}__n_agents",
    )
    max_steps = cols[1].number_input(
        "max_steps", min_value=1, max_value=10000,
        value=int(defaults.get("max_steps", 100)),
        key=f"{key_prefix}__steps",
    )
    return {"n_agents": n_agents, "max_steps": max_steps}


def render_flocking_params(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Flocking scenario placeholder."""
    cols = st.columns(2)
    n_agents = cols[0].number_input(
        "n_agents", min_value=2, max_value=50,
        value=int(defaults.get("n_agents", 5)),
        key=f"{key_prefix}__n_agents",
    )
    max_steps = cols[1].number_input(
        "max_steps", min_value=1, max_value=10000,
        value=int(defaults.get("max_steps", 100)),
        key=f"{key_prefix}__steps",
    )
    return {"n_agents": n_agents, "max_steps": max_steps}


def render_default_algo_params(defaults: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Algorithm placeholder — F2.4.2 (model arch / critic config) is deferred,
    so for now we just pass an empty params dict regardless of algorithm.
    """
    del defaults, key_prefix  # unused until F2.4.2 lands
    return {}


SCENARIO_FORMS: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    "discovery": render_discovery_params,
    "navigation": render_navigation_params,
    "transport": render_transport_params,
    "flocking": render_flocking_params,
}

ALGORITHM_FORMS: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    algo: render_default_algo_params
    for algo in ("mappo", "ippo", "masac", "isac", "iddpg", "maddpg")
}
