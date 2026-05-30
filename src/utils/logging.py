"""Structured logging configuration."""

import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured output."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(level: str = "INFO", json_output: bool = False):
    """Configure logging for ROCmark.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, use JSON formatter.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))

    root.handlers.clear()
    root.addHandler(handler)
