"""F0.1 smoke test: package imports and exposes the expected version."""

import multi_scenario


def test_import():
    assert multi_scenario.__version__ == "0.0.1"
