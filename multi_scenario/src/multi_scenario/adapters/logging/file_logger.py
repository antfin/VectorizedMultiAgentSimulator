"""FileLogger — appends timestamped lines to logs/run.log under the run folder."""

from datetime import datetime, timezone
from pathlib import Path


class FileLogger:
    """Logger that appends ``<UTC ISO ts> <LEVEL> <msg>\\n`` to a file."""

    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, msg: str) -> None:
        """Log an informational message."""
        self._write("INFO", msg)

    def debug(self, msg: str) -> None:
        """Log a debug-level message."""
        self._write("DEBUG", msg)

    def warning(self, msg: str) -> None:
        """Log a warning."""
        self._write("WARNING", msg)

    def error(self, msg: str) -> None:
        """Log an error."""
        self._write("ERROR", msg)

    def _write(self, level: str, msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{ts} {level} {msg}\n")
