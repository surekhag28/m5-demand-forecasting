from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emits each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:

        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(**record.exc_info),
            }

        for key, value in vars(record).items():
            if key.startswith("ctx_"):
                entry[key[4:]] = value

        return json.dumps(entry, defautl=str)


class TextFormatter(logging.Formatter):
    FMT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        ctx_parts = [
            f"{key[4:]}={value}"
            for key, value in vars(record).items()
            if key.startswith("ctx_")
        ]

        if ctx_parts:
            record.msg = f"{record.getMessage()} [{','.join(ctx_parts)}]"
            record.args = ()

        return super().format(record)


class ContextLogger:
    """A wrapper around stdlib Logger that automatically injects the context fields
    (run_id,model_name,stage etc.) into every LogRecord object as `ctx_<field>`
    attributes, which the formatters then picks up for further formatting.
    """

    def __init__(self, name: str, **context: Any) -> None:
        self._logger = logging.getLogger(name)
        self._context = context

    def bind(self, **kwargs: Any) -> ContextLogger:
        """Return a child logger with additional context merged in."""
        return ContextLogger(self._logger.name, **{**self._context, **kwargs})

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        kwargs["exc_info"] = True
        self._emit(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, msg, *args, **kwargs)

    def _emit(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return

        extra: dict[str, Any] = {f"ctx_{k}": v for k, v in self._context.items()}
        for k, v in list(kwargs.items()):
            if k not in ("exc_info", "stack_info", "stacklevel"):
                extra[f"ctx_{k}"] = kwargs.pop(k)

        self._logger.log(level, msg, *args, extra=extra, **kwargs)


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in (
        "py4j",
        "pyspark",
        "urllib3",
        "botocore",
        "s3transfer",
        "boto3",
        "optuna",
        "LightGBM",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, **context: Any) -> ContextLogger:
    """Returns ContextLogger for the given module name with optional context

    Example:
        log = get_logger(__name__, run_id=<run_id>, model="lightgbm_global")
    """
    return ContextLogger(name, **context)
