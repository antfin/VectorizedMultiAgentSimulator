"""Integration tests for LERO LLM calls.

These tests make REAL API calls to LLM providers.
They are skipped when the required API keys / endpoints are not set.

Run with:
    pytest tests/test_lero_integration.py -v

Required environment variables per provider:

    Anthropic (Claude):
        ANTHROPIC_API_KEY=sk-ant-...

    OpenAI (GPT):
        OPENAI_API_KEY=sk-...

    OVH AI Endpoints (OpenAI-compatible):
        OVH_AI_ENDPOINTS_ACCESS_TOKEN=<your-token>
        OVH_AI_ENDPOINTS_URL=https://<your-endpoint>.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1
        OVH_AI_ENDPOINTS_MODEL=<model-name>  (e.g. "Qwen/Qwen2.5-Coder-32B-Instruct")

To set them for a single run:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_lero_integration.py -v -k anthropic
"""

import os

import pytest

from src.lero.codegen import extract_candidates
from src.lero.config import LLMConfig
from src.lero.llm_client import LLMClient

# ── Markers for skipping ─────────────────────────────────────────

def _require_env(var_name):
    """Assert that an env var is set, with a clear error message."""
    val = os.environ.get(var_name)
    assert val, (
        f"Environment variable {var_name} is not set. "
        f"Export it before running: export {var_name}=<your-value>"
    )
    return val

# ── Simple generation prompt ─────────────────────────────────────

SIMPLE_MESSAGES = [
    {"role": "system", "content": "You are a Python expert. Reply with only code."},
    {"role": "user", "content": (
        "Write a Python function with this exact signature:\n\n"
        "```python\n"
        "def compute_reward(scenario_state: dict) -> torch.Tensor:\n"
        "```\n\n"
        "It should return `scenario_state['collision_rew'] - 0.01`. "
        "Wrap it in a ```python code block. Import torch."
    )},
]


def _validate_response(responses):
    """Check that the LLM response contains valid extractable code."""
    assert len(responses) == 1
    text = responses[0]
    assert len(text) > 10, f"Response too short: {text!r}"

    # Try to extract a candidate
    candidates = extract_candidates(
        responses, evolve_reward=True, evolve_observation=False,
    )
    assert len(candidates) >= 1, (
        f"Could not extract compute_reward from response:\n{text[:500]}"
    )
    assert candidates[0].reward_source is not None
    assert "compute_reward" in candidates[0].reward_source
    return candidates[0]


# ── Anthropic ────────────────────────────────────────────────────


class TestAnthropicIntegration:
    def test_claude_sonnet_generates_code(self):
        """Call Claude Sonnet 4.6 and verify it returns valid reward code."""
        _require_env("ANTHROPIC_API_KEY")
        cfg = LLMConfig(model="claude-sonnet-4-6", max_tokens=1024)
        client = LLMClient(cfg)
        responses = client.generate(SIMPLE_MESSAGES, n=1)
        cand = _validate_response(responses)
        print(f"\n--- Claude response ---\n{cand.reward_source}")

    def test_claude_haiku_generates_code(self):
        """Call Claude Haiku 4.5 (cheaper, faster) and verify."""
        _require_env("ANTHROPIC_API_KEY")
        cfg = LLMConfig(model="claude-haiku-4-5", max_tokens=1024)
        client = LLMClient(cfg)
        responses = client.generate(SIMPLE_MESSAGES, n=1)
        cand = _validate_response(responses)
        print(f"\n--- Haiku response ---\n{cand.reward_source}")


# ── OpenAI ───────────────────────────────────────────────────────


class TestOpenAIIntegration:
    def test_gpt4o_generates_code(self):
        """Call GPT-4o and verify it returns valid reward code."""
        _require_env("OPENAI_API_KEY")
        cfg = LLMConfig(model="gpt-4o", max_tokens=1024)
        client = LLMClient(cfg)
        responses = client.generate(SIMPLE_MESSAGES, n=1)
        cand = _validate_response(responses)
        print(f"\n--- GPT-4o response ---\n{cand.reward_source}")

    def test_gpt4o_mini_generates_code(self):
        """Call GPT-4o-mini (cheaper) and verify."""
        _require_env("OPENAI_API_KEY")
        cfg = LLMConfig(model="gpt-4o-mini", max_tokens=1024)
        client = LLMClient(cfg)
        responses = client.generate(SIMPLE_MESSAGES, n=1)
        cand = _validate_response(responses)
        print(f"\n--- GPT-4o-mini response ---\n{cand.reward_source}")


# ── OVH AI Endpoints (OpenAI-compatible) ─────────────────────────


class TestOVHIntegration:
    def test_ovh_endpoint_generates_code(self):
        """Call OVH AI Endpoints and verify it returns valid reward code.

        Requires:
            OVH_AI_ENDPOINTS_ACCESS_TOKEN: bearer token
            OVH_AI_ENDPOINTS_URL: full URL ending with /v1
            OVH_AI_ENDPOINTS_MODEL: model name (optional, defaults to tgi)
        """
        api_key = _require_env("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
        api_base = _require_env("OVH_AI_ENDPOINTS_URL")
        model = os.environ.get(
            "OVH_AI_ENDPOINTS_MODEL",
            "openai/tgi",  # default for OVH TGI endpoints
        )

        # OVH endpoints use OpenAI format, so prefix with openai/
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        cfg = LLMConfig(
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_tokens=1024,
        )
        client = LLMClient(cfg)
        responses = client.generate(SIMPLE_MESSAGES, n=1)
        cand = _validate_response(responses)
        print(f"\n--- OVH response ({model}) ---\n{cand.reward_source}")


# ── Diagnostic: show what's configured ───────────────────────────


class TestEnvDiagnostic:
    def test_show_available_providers(self):
        """Always runs. Prints which providers are available."""
        providers = {
            "Anthropic (Claude)": {
                "ready": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "env_vars": ["ANTHROPIC_API_KEY"],
                "models": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6"],
            },
            "OpenAI (GPT)": {
                "ready": bool(os.environ.get("OPENAI_API_KEY")),
                "env_vars": ["OPENAI_API_KEY"],
                "models": ["gpt-4o", "gpt-4o-mini"],
            },
            "OVH AI Endpoints": {
                "ready": bool(
                    os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")
                    and os.environ.get("OVH_AI_ENDPOINTS_URL")
                ),
                "env_vars": [
                    "OVH_AI_ENDPOINTS_ACCESS_TOKEN",
                    "OVH_AI_ENDPOINTS_URL",
                    "OVH_AI_ENDPOINTS_MODEL (optional)",
                ],
                "models": ["openai/<your-model>"],
            },
        }

        print("\n" + "=" * 60)
        print("LERO LLM Provider Status")
        print("=" * 60)

        any_ready = False
        for name, info in providers.items():
            status = "READY" if info["ready"] else "NOT CONFIGURED"
            print(f"\n{name}: {status}")
            print(f"  Models: {', '.join(info['models'])}")
            print(f"  Required env vars:")
            for var in info["env_vars"]:
                val = os.environ.get(var.split(" ")[0])
                flag = "SET" if val else "MISSING"
                print(f"    {var}: {flag}")
            if info["ready"]:
                any_ready = True

        print("\n" + "-" * 60)
        if not any_ready:
            print(
                "No providers configured. To run integration tests:\n\n"
                "  # Anthropic\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "  # OpenAI\n"
                "  export OPENAI_API_KEY=sk-...\n\n"
                "  # OVH AI Endpoints\n"
                "  export OVH_AI_ENDPOINTS_ACCESS_TOKEN=<token>\n"
                "  export OVH_AI_ENDPOINTS_URL=https://<endpoint>.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1\n"
                "  export OVH_AI_ENDPOINTS_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct\n\n"
                "Then: pytest tests/test_lero_integration.py -v"
            )
        else:
            print("Run: pytest tests/test_lero_integration.py -v")
        print("=" * 60)
