"""ConsoleLogger — writes to stdout / stderr."""

import sys


class ConsoleLogger:
    """Simple Logger writing info/debug to stdout, warning/error to stderr.

    Debug is suppressed by default; pass ``debug=True`` to enable.
    """

    def __init__(self, debug: bool = False) -> None:
        self._debug_enabled = debug

    def info(self, msg: str) -> None:
        """Log an informational message to stdout."""
        print(f"INFO  {msg}")

    def debug(self, msg: str) -> None:
        """Log a debug-level message to stdout when debug is enabled."""
        if self._debug_enabled:
            print(f"DEBUG {msg}")

    def warning(self, msg: str) -> None:
        """Log a warning to stderr."""
        print(f"WARN  {msg}", file=sys.stderr)

    def error(self, msg: str) -> None:
        """Log an error to stderr."""
        print(f"ERROR {msg}", file=sys.stderr)
