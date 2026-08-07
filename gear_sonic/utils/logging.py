from contextlib import contextmanager
import logging
import os
import sys

from loguru import logger


class HydraLoggerBridge(logging.Handler):
    def emit(self, record):
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoguruStream:
    def write(self, message):
        if message.strip():  # Only log non-empty messages
            logger.info(message.strip())  # Changed to debug level

    def flush(self):
        pass


@contextmanager
def capture_stdout_to_loguru():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    loguru_stream = LoguruStream()
    old_stdout = sys.stdout
    sys.stdout = loguru_stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
        logger.remove()
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        logger.add(sys.stdout, level=console_log_level, colorize=True)
