"""F9.1 — :class:`LiteLlmClient` — production :class:`LlmClient` adapter.

Wraps ``litellm.completion(...)`` for chat completions. ``litellm`` is
lazy-imported inside :meth:`generate` so domain-only tests don't pay
the multi-second ``litellm`` import cost (and don't fail when LiteLLM
isn't installed in a slim environment).

What this adapter does NOT do (intentional):

- **Cost cap enforcement**: see :class:`CostCapDecorator`. This
  adapter just records the LiteLLM-reported cost into the returned
  :class:`LlmCompletion`; the decorator enforces the budget.
- **Caching**: see :class:`DiskCacheDecorator`. Same composition story.
- **Retries**: LiteLLM has its own retry config; we don't add a
  second layer.
- **Per-sibling seed derivation**: the orchestrator derives distinct
  seeds via SHA(run_id, iter, cand) and passes them in. The adapter
  forwards ``seed`` verbatim.

API key resolution: keys flow from env vars (``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY`` / ``OVH_API_KEY`` / …). The :class:`LlmSection`
config never carries them — keeps experiment YAMLs commit-safe. We
load the project-root ``.env`` via ``python-dotenv`` if available,
otherwise rely on the user having exported the vars themselves.
"""

import logging
import os
from pathlib import Path
from typing import Any

from multi_scenario.domain.lero import LlmCompletion, LlmUsage
from multi_scenario.domain.models import LlmSection


_log = logging.getLogger(__name__)


def _load_env_once() -> None:
    """Best-effort load ``.env`` from the repo root (idempotent).

    Uses python-dotenv when installed; silently no-ops otherwise so the
    adapter still works in containers where keys are injected via
    proper env vars (the OVH AI Training case).
    """
    try:
        # pylint: disable=import-outside-toplevel
        from dotenv import load_dotenv
    except ImportError:
        return
    # Walk up from cwd looking for a .env at the repo root.
    here = Path.cwd()
    for parent in (here, *here.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


class LiteLlmClient:
    """Thin :class:`LlmClient` over ``litellm.completion(...)``."""

    def __init__(self, cfg: LlmSection):
        self._cfg = cfg
        _load_env_once()

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        n: int = 1,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> list[LlmCompletion]:
        """Issue one chat-completion request via LiteLLM, return ``n`` siblings.

        LiteLLM's response shape mirrors OpenAI's: ``choices`` is a list
        of length ``n``. We turn each choice into our domain
        :class:`LlmCompletion`, attaching the LiteLLM-reported per-call
        cost (USD) to the *first* sibling's usage; subsequent siblings
        get zero-cost usage so the cap decorator doesn't double-count.

        We round per-call kwargs through a single dict so future
        per-provider tweaks (e.g., Anthropic's ``thinking={"budget_tokens"}``)
        slot in without touching the call site.
        """
        # pylint: disable=import-outside-toplevel
        import litellm

        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "n": n,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        if self._cfg.api_base:
            kwargs["api_base"] = self._cfg.api_base
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = litellm.completion(**kwargs)

        # Cost is invoice-once per response (LiteLLM bills per-call, not
        # per-choice); attach it to the first completion only.
        try:
            total_cost_usd = float(
                litellm.completion_cost(completion_response=response)
            )
        except Exception:  # pylint: disable=broad-except
            total_cost_usd = 0.0
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        # Some providers expose "reasoning_tokens" inside completion_tokens_details
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        system_fingerprint = response.get("system_fingerprint")

        out: list[LlmCompletion] = []
        for i, choice in enumerate(response["choices"]):
            msg = choice.get("message", {})
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("thinking")
            finish_reason = choice.get("finish_reason")
            out.append(
                LlmCompletion(
                    text=text,
                    reasoning=reasoning,
                    finish_reason=finish_reason,
                    system_fingerprint=system_fingerprint,
                    usage=LlmUsage(
                        prompt_tokens=prompt_tokens if i == 0 else 0,
                        completion_tokens=completion_tokens if i == 0 else 0,
                        reasoning_tokens=reasoning_tokens if i == 0 else 0,
                        estimated_cost_usd=total_cost_usd if i == 0 else 0.0,
                    ),
                )
            )
        return out


_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OVH_API_KEY")


def some_api_key_present() -> bool:
    """True iff at least one supported provider's API key is in the env.

    Used by the F9.8 Submit-page preflight check so the user gets a
    helpful "missing OPENAI_API_KEY" error before the LERO loop fires
    (instead of mid-iter when LiteLLM raises 401).
    """
    return any(os.environ.get(k) for k in _API_KEY_ENV_VARS)
