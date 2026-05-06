"""Shared pydantic ConfigDict instances for the domain models package."""

from pydantic import ConfigDict

STRICT = ConfigDict(extra="forbid")
FROZEN = ConfigDict(extra="forbid", frozen=True)
