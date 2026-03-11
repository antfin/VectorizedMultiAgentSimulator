"""Tests for the logging_setup module — setup_run_logger, teardown_run_logger."""
import logging
from pathlib import Path

import pytest

from src.logging_setup import setup_run_logger, teardown_run_logger


# ── setup_run_logger with console=True ────────────────────────────


class TestSetupWithConsole:
    """setup_run_logger() with default console=True."""

    def test_returns_logger_instance(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_returns")
        assert isinstance(logger, logging.Logger)
        teardown_run_logger(logger)

    def test_has_two_handlers(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_two_handlers")
        assert len(logger.handlers) == 2
        teardown_run_logger(logger)

    def test_file_handler_writes_to_run_log(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_fh_path")
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == tmp_path / "logs" / "run.log"
        teardown_run_logger(logger)

    def test_log_file_created(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_file_created")
        assert (tmp_path / "logs" / "run.log").exists()
        teardown_run_logger(logger)

    def test_message_appears_in_file(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_msg_in_file")
        logger.info("hello from test")
        # Flush handlers so message is written
        for h in logger.handlers:
            h.flush()
        contents = (tmp_path / "logs" / "run.log").read_text()
        assert "hello from test" in contents
        teardown_run_logger(logger)


# ── setup_run_logger with console=False ───────────────────────────


class TestSetupWithoutConsole:
    """setup_run_logger() with console=False."""

    def test_has_one_handler(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_no_console", console=False)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        teardown_run_logger(logger)

    def test_file_handler_still_works(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_no_console_writes", console=False)
        logger.info("quiet mode message")
        for h in logger.handlers:
            h.flush()
        contents = (tmp_path / "logs" / "run.log").read_text()
        assert "quiet mode message" in contents
        teardown_run_logger(logger)


# ── Idempotency ───────────────────────────────────────────────────


class TestIdempotency:
    """Calling setup_run_logger twice does not accumulate handlers."""

    def test_no_handler_accumulation(self, tmp_path):
        name = "test_idempotent"
        logger = setup_run_logger(tmp_path, name=name)
        assert len(logger.handlers) == 2
        logger = setup_run_logger(tmp_path, name=name)
        assert len(logger.handlers) == 2
        teardown_run_logger(logger)

    def test_no_handler_accumulation_file_only(self, tmp_path):
        name = "test_idempotent_file"
        logger = setup_run_logger(tmp_path, name=name, console=False)
        assert len(logger.handlers) == 1
        logger = setup_run_logger(tmp_path, name=name, console=False)
        assert len(logger.handlers) == 1
        teardown_run_logger(logger)


# ── teardown_run_logger ───────────────────────────────────────────


class TestTeardown:
    """teardown_run_logger() removes and closes all handlers."""

    def test_zero_handlers_after_teardown(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_teardown_zero")
        assert len(logger.handlers) > 0
        teardown_run_logger(logger)
        assert len(logger.handlers) == 0

    def test_handlers_are_closed(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_teardown_closed")
        handlers = logger.handlers[:]
        teardown_run_logger(logger)
        for h in handlers:
            if isinstance(h, logging.FileHandler):
                # After close(), stream is either closed or set to None
                assert h.stream is None or h.stream.closed


# ── File logging content / format ─────────────────────────────────


class TestFileLoggingContent:
    """Log messages appear in file with the expected format."""

    def test_format_contains_timestamp_level_message(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_format")
        logger.warning("format check")
        for h in logger.handlers:
            h.flush()
        contents = (tmp_path / "logs" / "run.log").read_text()
        # Expected format: "YYYY-MM-DD HH:MM:SS WARNING  format check"
        assert "WARNING" in contents
        assert "format check" in contents
        # Timestamp pattern: four digits, dash, two digits, dash, two digits
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", contents)
        teardown_run_logger(logger)

    def test_debug_message_written_to_file(self, tmp_path):
        logger = setup_run_logger(tmp_path, name="test_debug_file")
        logger.debug("debug level msg")
        for h in logger.handlers:
            h.flush()
        contents = (tmp_path / "logs" / "run.log").read_text()
        assert "debug level msg" in contents
        teardown_run_logger(logger)
