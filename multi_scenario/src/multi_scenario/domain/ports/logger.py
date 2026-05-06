"""Logger port — Protocol that logger adapters must satisfy.

Concrete adapters (``FileLogger`` writing to ``logs/run.log``,
``ConsoleLogger`` writing to stdout) land at F2.7.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Domain port for run-time logging at four standard severity levels."""

    def info(self, msg: str) -> None:
        """Log an informational message."""

    def debug(self, msg: str) -> None:
        """Log a debug-level message."""

    def warning(self, msg: str) -> None:
        """Log a warning."""

    def error(self, msg: str) -> None:
        """Log an error."""
