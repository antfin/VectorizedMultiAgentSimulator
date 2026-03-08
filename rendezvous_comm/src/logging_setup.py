"""Logging setup for experiment runs.

Provides file + console logging so that all run output is captured
in the run directory while still visible in notebooks.
"""
import logging
from pathlib import Path


def setup_run_logger(
    run_dir: Path,
    name: str = "rendezvous",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Create a logger that writes to both console and a log file.

    Args:
        run_dir: run directory; log file goes to run_dir/logs/run.log
        name: logger name
        console_level: minimum level for console output
        file_level: minimum level for file output

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    # Remove existing handlers (safe for notebook re-runs)
    teardown_run_logger(logger)
    logger.setLevel(min(console_level, file_level))

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / "run.log", mode="a")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def teardown_run_logger(logger: logging.Logger):
    """Remove all handlers from logger (prevents handler accumulation)."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
