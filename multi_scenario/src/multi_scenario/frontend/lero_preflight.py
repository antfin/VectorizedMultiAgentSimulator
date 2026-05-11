"""F9.8 — Submit-page preflight extension for LERO configs.

When ``cfg.lero is not None`` the run will fire LLM calls — and an
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / equivalent MUST be in the
environment or LiteLLM raises 401 mid-iteration (costly: the cap won't
fire because the call never completes).

This helper is a Submit-page hook; it's deliberately framework-agnostic
so the same function powers the Streamlit preflight badge AND a future
CLI ``multi-scenario validate <yaml>`` lero-aware check.

The Submit page calls :func:`check_lero_api_key` after validating the
YAML; a False result blocks the Submit button and surfaces the missing
key by name so the user can fix it in ``.env``.
"""

import os
from dataclasses import dataclass

from multi_scenario.domain.models import ExperimentConfig


_API_KEY_BY_PROVIDER_PREFIX: dict[str, str] = {
    # LiteLLM-style model prefixes. Order matters: longest first, so
    # ``"openai/my-model"`` matches "openai/" before any future "open"
    # ambiguity.
    "claude": "ANTHROPIC_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "ovh/": "OVH_API_KEY",
}


@dataclass(frozen=True)
class LeroPreflightResult:
    """Outcome of one LERO-side preflight check."""

    ok: bool
    detail: str
    #: Env var the check needs. Empty when ``ok=True`` (no missing key).
    required_env_var: str = ""


def expected_api_key_var(model: str) -> str:
    """Best-effort: which env var LiteLLM will look at for ``model``.

    Returns the matching env var name; falls back to ``OPENAI_API_KEY``
    for unknown prefixes since LiteLLM treats that as its catch-all.
    """
    lower = model.lower()
    for prefix, var in _API_KEY_BY_PROVIDER_PREFIX.items():
        if lower.startswith(prefix):
            return var
    return "OPENAI_API_KEY"


def check_lero_api_key(cfg: ExperimentConfig) -> LeroPreflightResult:
    """Submit-page preflight: is the right API key in the environment?

    No-op (returns ``ok=True`` with explanation) when ``cfg.lero is
    None`` — non-LERO runs don't need an LLM key.
    """
    if cfg.lero is None:
        return LeroPreflightResult(ok=True, detail="not a LERO run — no LLM key needed")
    if cfg.llm is None:
        # The ExperimentConfig validator already rejects this combo,
        # but the preflight defends against constructed-by-hand cfgs
        # the FE might hand over before validation.
        return LeroPreflightResult(
            ok=False,
            detail="cfg.lero set but cfg.llm missing — add an llm: section",
            required_env_var="",
        )
    var = expected_api_key_var(cfg.llm.model)
    if os.environ.get(var):
        return LeroPreflightResult(
            ok=True, detail=f"{var} is set", required_env_var=var
        )
    return LeroPreflightResult(
        ok=False,
        detail=(
            f"missing {var} for model {cfg.llm.model!r}. Set it in "
            "the project-root .env (gitignored) or export it before "
            "running the Submit page."
        ),
        required_env_var=var,
    )
