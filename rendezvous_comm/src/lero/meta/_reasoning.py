"""Reasoning-model detection + prompt-variant gating (LERO-MP v3.1).

Reasoning models (OpenAI o-series, gpt-oss, Anthropic extended-thinking,
Qwen3 thinking, DeepSeek-R1) do their own chain-of-thought internally
and prefer **leaner** prompts:

  - No procedural decomposition ("Step 1 do X, Step 2 do Y").
  - No hardcoded interpretation rules ("if M9 < 0.2 then X").
  - Minimal role preamble — they don't need persona scaffolding.
  - Few-shot examples can hurt; one minimal example or none at all
    is usually better.

We auto-detect whether the meta-LLM is in this class and route to
slimmed prompt variants. Non-reasoning models (gpt-5.4-mini, gpt-4o,
plain Llama, plain Claude) keep the verbose prompts they need.
"""

from __future__ import annotations

import re

# Patterns that flag a reasoning model. Match conservatively — false
# negatives (treating a reasoning model as non-reasoning) are
# recoverable; false positives may strip needed scaffolding.
_REASONING_MODEL_PATTERNS = [
    r"^o[1-9]",  # OpenAI o-series: o1, o3, o4, o5...
    r"o[1-9]-",  # o3-mini, o4-mini, etc.
    r"gpt-oss",  # OpenAI open-weights reasoning
    r"deepseek-r1",  # DeepSeek-R1
    r"deepseek.*reasoner",  # DeepSeek reasoner variants
    r"qwen3.*think",  # Qwen3 thinking mode
    r"qwq-",  # Qwen QwQ reasoning
    r"claude.*opus.*think",  # Anthropic extended-thinking
    r"claude.*sonnet.*think",  # ditto
    r"-r1\b",  # generic R1 suffix
]

_REASONING_RE = re.compile(
    "|".join(_REASONING_MODEL_PATTERNS),
    re.IGNORECASE,
)


def is_reasoning_model(model: str) -> bool:
    """True if the model name looks like a reasoning-mode LLM.

    Examples (all True): "o4-mini", "openai/o3-mini-2025-01-31",
    "gpt-oss-120b", "openai/gpt-oss-120b", "deepseek-r1",
    "Qwen3-32B-thinking", "claude-opus-4-7-thinking".

    Examples (all False): "gpt-5.4-mini", "gpt-4o",
    "Meta-Llama-3_3-70B-Instruct", "claude-sonnet-4-6".
    """
    if not model:
        return False
    # Strip provider prefix (e.g. "openai/o3-mini" → "o3-mini") so
    # patterns matching "^o[1-9]" still fire.
    stripped = model.split("/")[-1].lower()
    return bool(_REASONING_RE.search(stripped))
