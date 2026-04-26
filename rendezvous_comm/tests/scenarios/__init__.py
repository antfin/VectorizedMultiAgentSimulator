"""Synthetic scenarios for meta-LLM harness testing.

Each scenario is a dict describing a fake inner-loop outcome:
candidate metrics, per-seed history, prior mutation_log entries, and
the Strategist/Editor/Critic decisions we expect (loose assertions).

Scenarios feed into tests/test_meta_llm_scenarios.py which calls the
REAL meta-LLM and validates its decisions against the expectations.
No RL training happens — the whole point is to iterate on meta-LLM
prompts without burning OVH compute.
"""
