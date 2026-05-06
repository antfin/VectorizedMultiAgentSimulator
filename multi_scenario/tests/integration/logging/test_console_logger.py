"""F2.7 tests: ConsoleLogger — Protocol satisfaction."""

from multi_scenario.adapters.logging.console_logger import ConsoleLogger
from multi_scenario.domain.ports import Logger


def test_implements_logger_protocol():
    """ConsoleLogger satisfies the Logger port."""
    assert isinstance(ConsoleLogger(), Logger)
