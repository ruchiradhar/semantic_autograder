import sys
from typing import Any
from src.logger import logging


def _format_error_message(error: Exception, error_detail: sys) -> str:
    """Create a rich error message including file and line info.

    Parameters:
        error: The original exception instance.
        error_detail: Typically the `sys` module to access traceback via `exc_info()`.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    return (
        f"Exception in [{file_name}] at line [{line_num}]: {error}"
    )


class CustomException(Exception):
    """Custom exception that enriches the original error with traceback context."""

    def __init__(self, error: Exception, error_detail: sys):
        message = _format_error_message(error, error_detail)
        super().__init__(message)
        self.message = message
        logging.error(message)

    def __str__(self) -> str:  # pragma: no cover
        return self.message

    def __repr__(self) -> str:  # pragma: no cover
        return f"CustomException({self.message!r})"