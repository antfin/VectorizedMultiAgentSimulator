"""Real-LLM scenario harness for LERO-MP v4.

Each scenario synthesizes BootstrapCard + RoundResult history, calls
the REAL meta-LLM (gpt-5.4-mini), and asserts properties of the
returned StrategyBundle. Parallels the v3 harness pattern but tests
the v4 multi-strategy decision surface.
"""
