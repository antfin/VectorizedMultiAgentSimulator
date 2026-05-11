"""Per-test env snapshot/restore — applies to every integration test.

Several side-effects can leak into ``os.environ`` during the suite:

- ``rendezvous_comm/src/lero/__init__.py`` imports ``loop.py`` →
  ``llm_client.py`` → ``load_dotenv()`` at module-import time. That
  permanently injects ``OPENAI_API_KEY`` for the rest of the session.
- ``multi_scenario.adapters.llm.LiteLlmClient.__init__`` runs
  ``_load_env_once()`` which walks up looking for a ``.env`` file.
- Tests using ``os.environ[...] = ...`` directly (instead of
  ``monkeypatch.setenv``) escape their own scope.

The Fernet suite (``test_decrypt_from_env_round_trip``) asserts
``"OPENAI_API_KEY" not in os.environ`` to verify the secrets adapter
doesn't mutate the global env. The leak makes it fail when the full
suite runs even though Fernet itself behaves correctly.

This conftest snapshots ``os.environ`` per-test and restores deltas at
teardown. Lives at ``tests/integration/`` so EVERY integration test
gets the protection — earlier this was only at
``tests/integration/lero/``, which missed leaks triggered from
sibling suites (frontend, application, prompt_composers, …).
"""

# pylint: disable=missing-function-docstring

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env_for_integration_tests():
    snapshot = dict(os.environ)
    yield
    # Remove any keys the test (or its imports) added.
    for k in list(os.environ.keys()):
        if k not in snapshot:
            del os.environ[k]
    # Restore any changed/deleted ones to their pre-test values.
    for k, v in snapshot.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
