"""F1.3 tests: RunId — parametric, hashable identity for one experiment run."""

import pytest
from pydantic import ValidationError

from multi_scenario.domain.models import RunId


def test_construction():
    """RunId constructs with exp_id and seed."""
    rid = RunId(exp_id="disc_baseline_smoke_mappo", seed=0)
    assert rid.exp_id == "disc_baseline_smoke_mappo"
    assert rid.seed == 0


def test_str_format():
    """str(RunId) renders as <exp_id>_s<seed>."""
    rid = RunId(exp_id="disc_baseline_smoke_mappo", seed=0)
    assert str(rid) == "disc_baseline_smoke_mappo_s0"


def test_equality_and_hash():
    """Equal inputs produce equal, hashable RunIds (frozen value object)."""
    a = RunId(exp_id="exp", seed=3)
    b = RunId(exp_id="exp", seed=3)
    assert a == b
    assert hash(a) == hash(b)
    # Usable as dict key / set member.
    assert {a, b} == {a}


def test_different_seed():
    """Different seed → different identity."""
    a = RunId(exp_id="exp", seed=0)
    b = RunId(exp_id="exp", seed=1)
    assert a != b
    assert str(a) != str(b)


def test_folder_name():
    """folder_name(timestamp) returns <run_id>__<timestamp>."""
    rid = RunId(exp_id="disc_baseline_smoke_mappo", seed=0)
    assert rid.folder_name("20260506_1423") == "disc_baseline_smoke_mappo_s0__20260506_1423"


def test_from_string():
    """from_string parses back to a RunId equal to the original."""
    rid = RunId(exp_id="disc_baseline_smoke_mappo", seed=12)
    parsed = RunId.from_string(str(rid))
    assert parsed == rid

    # Greedy regex handles exp_ids that themselves contain _sN-looking substrings.
    tricky = RunId(exp_id="exp_s5_subtest", seed=0)
    assert RunId.from_string(str(tricky)) == tricky


def test_from_folder_name():
    """from_folder_name returns the (RunId, timestamp) pair."""
    rid = RunId(exp_id="disc_baseline_smoke_mappo", seed=0)
    folder = rid.folder_name("20260506_1423")
    parsed, ts = RunId.from_folder_name(folder)
    assert parsed == rid
    assert ts == "20260506_1423"


def test_rejects_invalid():
    """Bad exp_id / seed values raise ValidationError."""
    with pytest.raises(ValidationError):
        RunId(exp_id="", seed=0)  # empty
    with pytest.raises(ValidationError):
        RunId(exp_id="bad__name", seed=0)  # collides with timestamp separator
    with pytest.raises(ValidationError):
        RunId(exp_id="bad name", seed=0)  # space
    with pytest.raises(ValidationError):
        RunId(exp_id="bad/name", seed=0)  # slash
    with pytest.raises(ValidationError):
        RunId(exp_id="ok", seed=-1)  # negative seed

    # Parsers also reject malformed strings.
    with pytest.raises(ValueError):
        RunId.from_string("no_seed_suffix")
    with pytest.raises(ValueError):
        RunId.from_folder_name("missing_separator")
