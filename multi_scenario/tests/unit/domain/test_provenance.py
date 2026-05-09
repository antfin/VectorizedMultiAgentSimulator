"""F1.5 tests: Provenance + LibraryVersions models."""

from datetime import datetime, timezone

import pytest

from multi_scenario.domain.models import LibraryVersions, Provenance
from pydantic import ValidationError


def _ts() -> datetime:
    """Single fixed UTC timestamp shared across tests."""
    return datetime(2026, 5, 6, 14, 23, 0, tzinfo=timezone.utc)


def _minimal_provenance() -> Provenance:
    """Build a fully-populated Provenance for tests."""
    return Provenance(
        config_hash="sha256:abc",
        code_hash="sha256:def",
        hashed_source_files=["src/multi_scenario/domain/models/config.py"],
        git_sha="1a2b3c4d",
        git_dirty=False,
        created_at=_ts(),
        library_versions=LibraryVersions(
            python="3.11.4",
            torch="2.4.0",
            vmas="1.4.0",
            benchmarl="1.3.0",
            multi_scenario="0.0.1",
        ),
    )


def test_construct():
    """Provenance constructs with all required fields."""
    p = _minimal_provenance()
    assert p.config_hash == "sha256:abc"
    assert p.code_hash == "sha256:def"
    assert p.git_dirty is False
    assert p.library_versions.python == "3.11.4"


def test_finished_at_optional():
    """finished_at is optional and defaults to None."""
    p = _minimal_provenance()
    assert p.finished_at is None


def test_library_versions_strict():
    """LibraryVersions rejects unknown fields."""
    with pytest.raises(ValidationError):
        LibraryVersions(
            python="3.11",
            torch="2.4",
            vmas="1.4",
            benchmarl="1.3",
            multi_scenario="0.0.1",
            bogus="x",  # type: ignore[call-arg]
        )


def test_roundtrip():
    """model_dump → model_validate preserves the model exactly."""
    p = _minimal_provenance()
    p2 = Provenance.model_validate(p.model_dump())
    assert p == p2
