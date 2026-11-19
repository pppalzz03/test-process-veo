"""Structured JSON logging with contextual fields (job_id, segment_id, ...)."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "log_ctx", default=None
)

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


@contextlib.contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **(_context.get() or {}),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = True, service: str = "") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    if service:
        root.info("logging configured", extra={"service": service})
