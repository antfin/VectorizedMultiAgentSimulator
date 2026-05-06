"""F1.5 tests: pure hashing functions used to compute provenance fields."""

from pathlib import Path

from multi_scenario.domain.hashing import compute_code_hash, compute_config_hash


def test_config_hash_deterministic():
    """Same dict → same hash."""
    cfg = {"a": 1, "b": 2}
    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def test_config_hash_key_order_invariant():
    """Different dict key order → same hash (canonical JSON encoding)."""
    a = {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "b": 2, "a": 1}
    assert compute_config_hash(a) == compute_config_hash(b)


def test_code_hash_deterministic(tmp_path: Path):
    """Same file contents → same hash."""
    f = tmp_path / "a.py"
    f.write_text("def hi(): pass\n", encoding="utf-8")
    assert compute_code_hash([f]) == compute_code_hash([f])


def test_code_hash_changes_when_file_modified(tmp_path: Path):
    """Modifying a file changes its hash."""
    f = tmp_path / "a.py"
    f.write_text("def hi(): pass\n", encoding="utf-8")
    h1 = compute_code_hash([f])
    f.write_text("def hi(): return 42\n", encoding="utf-8")
    h2 = compute_code_hash([f])
    assert h1 != h2
