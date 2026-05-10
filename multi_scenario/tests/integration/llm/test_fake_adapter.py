"""F9.1 — :class:`FakeLlmClient` matcher contract."""

# pylint: disable=missing-function-docstring

import pytest

from multi_scenario.adapters.llm.fake_adapter import FakeLlmClient
from multi_scenario.domain.lero import LlmCompletion


def test_register_always_returns_completion_for_any_call():
    fake = FakeLlmClient().register_always(LlmCompletion(text="hello"))
    out = fake.generate(messages=[{"role": "user", "content": "anything"}])
    assert len(out) == 1
    assert out[0].text == "hello"


def test_register_always_returns_n_siblings():
    fake = FakeLlmClient().register_always(LlmCompletion(text="hello"))
    out = fake.generate(messages=[{"role": "user", "content": "x"}], n=3)
    assert len(out) == 3


def test_register_exact_only_matches_identical_messages():
    target = [{"role": "user", "content": "exact prompt"}]
    fake = FakeLlmClient().register_exact(target, LlmCompletion(text="match"))
    fake.generate(messages=target)  # ok
    with pytest.raises(LookupError):
        fake.generate(messages=[{"role": "user", "content": "different"}])


def test_register_contains_user_matches_substring_in_last_user_message():
    fake = FakeLlmClient().register_contains_user(
        "compute_reward", LlmCompletion(text="def compute_reward(s): ...")
    )
    out = fake.generate(
        messages=[{"role": "user", "content": "Please write compute_reward(s)"}]
    )
    assert "def compute_reward" in out[0].text


def test_no_matching_rule_raises_lookup_error_with_message_dump():
    """Tests that fail to register any rule should fail loudly so the
    test author sees what went wrong instead of a silent default."""
    fake = FakeLlmClient()
    with pytest.raises(LookupError, match="no rule matched"):
        fake.generate(messages=[{"role": "user", "content": "x"}])


def test_calls_are_recorded_for_post_hoc_assertions():
    """Tests can assert prompt shape after the call."""
    fake = FakeLlmClient().register_always(LlmCompletion(text="x"))
    fake.generate(messages=[{"role": "user", "content": "first"}], n=1, seed=42)
    fake.generate(messages=[{"role": "user", "content": "second"}], n=2)
    assert len(fake.calls) == 2
    assert fake.calls[0]["seed"] == 42
    assert fake.calls[1]["n"] == 2


def test_register_chains_for_test_ergonomics():
    """``register_*`` returns ``self`` so test setup can chain fluently."""
    fake = (
        FakeLlmClient()
        .register_contains_user("foo", LlmCompletion(text="for_foo"))
        .register_contains_user("bar", LlmCompletion(text="for_bar"))
    )
    a = fake.generate(messages=[{"role": "user", "content": "say foo"}])
    b = fake.generate(messages=[{"role": "user", "content": "say bar"}])
    assert a[0].text == "for_foo"
    assert b[0].text == "for_bar"


def test_first_matching_rule_wins():
    """Rules are walked in registration order — earlier rules dominate."""
    fake = (
        FakeLlmClient()
        .register_always(LlmCompletion(text="catch_all"))
        .register_contains_user("specific", LlmCompletion(text="specific"))
    )
    out = fake.generate(messages=[{"role": "user", "content": "specific prompt"}])
    # ``register_always`` was registered first → wins even though specific would match.
    assert out[0].text == "catch_all"
