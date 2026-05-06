"""F1.10 tests: RngState model — seed plus an opaque capture dict."""

from multi_scenario.domain.models import RngState


def test_rng_state_minimal():
    """RngState(seed=...) constructs with empty captures by default."""
    s = RngState(seed=0)
    assert s.seed == 0
    assert s.captures == {}


def test_rng_state_roundtrip():
    """model_validate(model_dump()) preserves seed and captures."""
    s = RngState(
        seed=42,
        captures={"python.random": "abc", "numpy": "def", "torch.cpu": "ghi"},
    )
    s2 = RngState.model_validate(s.model_dump())
    assert s == s2
