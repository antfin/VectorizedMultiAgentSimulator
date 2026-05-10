"""F9.1 — :class:`DiskCacheDecorator` contract."""

# pylint: disable=missing-function-docstring

from pathlib import Path

from multi_scenario.adapters.llm.disk_cache import DiskCacheDecorator
from multi_scenario.adapters.llm.fake_adapter import FakeLlmClient
from multi_scenario.domain.lero import LlmCompletion, LlmUsage


def _completion(text: str = "x", cost: float = 1.0) -> LlmCompletion:
    return LlmCompletion(text=text, usage=LlmUsage(estimated_cost_usd=cost))


def test_cache_miss_calls_inner_and_writes(tmp_path: Path):
    """First call: miss → inner invoked → file written."""
    fake = FakeLlmClient().register_always(_completion("hello"))
    cache = DiskCacheDecorator(fake, model="gpt-4o-mini", cache_dir=tmp_path)
    out = cache.generate(messages=[{"role": "user", "content": "x"}])
    assert len(out) == 1
    assert out[0].text == "hello"
    assert len(fake.calls) == 1
    # Cache file written.
    assert any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_cache_hit_zeroes_usage(tmp_path: Path):
    """Cache hit → inner NOT invoked → usage cost stripped."""
    fake = FakeLlmClient().register_always(_completion("hello", cost=2.0))
    cache = DiskCacheDecorator(fake, model="gpt-4o-mini", cache_dir=tmp_path)

    cache.generate(messages=[{"role": "user", "content": "x"}])
    fake.calls.clear()  # reset call recorder

    out = cache.generate(messages=[{"role": "user", "content": "x"}])
    assert out[0].text == "hello"
    assert out[0].usage.estimated_cost_usd == 0.0  # zero cost on hit
    assert fake.calls == []  # inner not invoked


def test_cache_distinguishes_messages(tmp_path: Path):
    """Different messages → different cache keys → both miss-cache then hit."""
    fake = FakeLlmClient()
    fake.register_contains_user("foo", _completion("response_foo"))
    fake.register_contains_user("bar", _completion("response_bar"))
    cache = DiskCacheDecorator(fake, model="gpt-4o-mini", cache_dir=tmp_path)

    a = cache.generate(messages=[{"role": "user", "content": "foo"}])
    b = cache.generate(messages=[{"role": "user", "content": "bar"}])
    assert a[0].text == "response_foo"
    assert b[0].text == "response_bar"
    assert len(fake.calls) == 2


def test_cache_n_siblings_share_inputs_distinct_outputs(tmp_path: Path):
    """``n=3`` produces 3 cache entries (one per sibling) so ``seed`` can
    distinguish them in a future iteration."""
    fake = FakeLlmClient().register_always(_completion("sibling"))
    cache = DiskCacheDecorator(fake, model="gpt-4o-mini", cache_dir=tmp_path)
    out = cache.generate(messages=[{"role": "user", "content": "x"}], n=3)
    assert len(out) == 3
    # 3 cache files written (one per sibling).
    files = [p for p in tmp_path.iterdir() if p.suffix == ".json"]
    assert len(files) == 3


def test_cache_corrupt_file_falls_through_to_miss(tmp_path: Path):
    """Half-written cache file shouldn't poison subsequent calls."""
    # Plant a corrupt file at a key we'll request.
    from multi_scenario.adapters.llm.disk_cache import _cache_key  # noqa: PLC0415

    key = _cache_key(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "x"}],
        seed=None,
        response_format=None,
        sibling_idx=0,
    )
    (tmp_path / f"{key}.json").write_text("not valid json", encoding="utf-8")

    fake = FakeLlmClient().register_always(_completion("recovered"))
    cache = DiskCacheDecorator(fake, model="gpt-4o-mini", cache_dir=tmp_path)
    out = cache.generate(messages=[{"role": "user", "content": "x"}])
    # Inner was called (corrupt file ignored, miss path taken).
    assert out[0].text == "recovered"
    assert len(fake.calls) == 1
    # File should now be valid (overwritten by the miss-path write).
    assert "recovered" in (tmp_path / f"{key}.json").read_text()
