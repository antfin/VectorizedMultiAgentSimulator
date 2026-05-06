"""F2.7 tests: FileLogger — appends timestamped lines, creates parent dirs."""

from pathlib import Path

from multi_scenario.adapters.logging.file_logger import FileLogger
from multi_scenario.domain.ports import Logger


def test_implements_logger_protocol(tmp_path: Path):
    """FileLogger satisfies the Logger port."""
    assert isinstance(FileLogger(tmp_path / "run.log"), Logger)


def test_info_appends_to_log_file(tmp_path: Path):
    """Each call appends one timestamped line at the configured level."""
    log_path = tmp_path / "run.log"
    logger = FileLogger(log_path)
    logger.info("training started")
    logger.warning("eval was slow")
    logger.error("rollout failed")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "INFO" in lines[0] and "training started" in lines[0]
    assert "WARNING" in lines[1] and "eval was slow" in lines[1]
    assert "ERROR" in lines[2] and "rollout failed" in lines[2]


def test_log_creates_parent_directory(tmp_path: Path):
    """Passing a deep path works without pre-mkdir-ing the parents."""
    log_path = tmp_path / "deep" / "nested" / "run.log"
    assert not log_path.parent.exists()
    FileLogger(log_path).info("hi")
    assert log_path.is_file()
