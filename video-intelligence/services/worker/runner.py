"""DeepStream runner adapter: launch the runner as a subprocess and stream its NDJSON output.

A subprocess (not an in-process pipeline) keeps GStreamer/CUDA crashes from taking the worker
down, lets us kill a hung pipeline, and frees all GPU memory when the segment ends.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from shared.domain.errors import PermanentJobError, TransientJobError
from shared.schemas.runner import (
    EXIT_BAD_INPUT,
    EXIT_ENVIRONMENT,
    EXIT_OK,
    FrameRecord,
    SummaryRecord,
    parse_runner_line,
)

logger = logging.getLogger(__name__)

_STDERR_TAIL_BYTES = 4000


class RunnerCancelled(Exception):
    """The runner was stopped because the worker is shutting down or lost its lease."""


@dataclass(frozen=True, slots=True)
class RunnerRequest:
    source_path: str
    start_ms: int  # container timeline (ms)
    end_ms: int
    infer_config: str
    tracker_config: str
    tracker_lib: str
    labels_file: str
    decode_mode: str
    codec: str | None
    infer_interval: int
    timeout_seconds: int

    def to_args(self) -> list[str]:
        args = [
            "--source",
            self.source_path,
            "--start-ms",
            str(self.start_ms),
            "--end-ms",
            str(self.end_ms),
            "--infer-config",
            self.infer_config,
            "--tracker-config",
            self.tracker_config,
            "--tracker-lib",
            self.tracker_lib,
            "--labels",
            self.labels_file,
            "--decode",
            self.decode_mode,
            "--infer-interval",
            str(self.infer_interval),
        ]
        if self.codec:
            args += ["--codec", self.codec]
        return args


class SegmentRunner(Protocol):
    def run(
        self, request: RunnerRequest, cancel: threading.Event
    ) -> Iterator[FrameRecord | SummaryRecord]: ...


class SubprocessRunner:
    def __init__(self, argv_prefix: list[str]) -> None:
        self._argv_prefix = argv_prefix

    def run(
        self, request: RunnerRequest, cancel: threading.Event
    ) -> Iterator[FrameRecord | SummaryRecord]:
        argv = [*self._argv_prefix, *request.to_args()]
        logger.info(
            "starting runner",
            extra={"start_ms": request.start_ms, "end_ms": request.end_ms},
        )
        timed_out = threading.Event()
        finished = threading.Event()

        with tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(  # noqa: S603 - argv is built from validated settings
                argv, stdout=subprocess.PIPE, stderr=stderr_file, text=True, bufsize=1
            )

            def watchdog() -> None:
                deadline = request.timeout_seconds
                waited = 0.0
                while not finished.is_set():
                    if cancel.is_set():
                        break
                    if waited >= deadline:
                        timed_out.set()
                        break
                    finished.wait(0.5)
                    waited += 0.5
                if not finished.is_set():
                    proc.kill()

            threading.Thread(target=watchdog, daemon=True, name="runner-watchdog").start()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield parse_runner_line(line)
                    except ValueError as exc:
                        raise TransientJobError(str(exc), code="runner_protocol_error") from exc
                return_code = proc.wait()
            finally:
                finished.set()
                if proc.poll() is None:
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=10)

            if cancel.is_set():
                raise RunnerCancelled
            if timed_out.is_set():
                raise TransientJobError(
                    f"runner exceeded {request.timeout_seconds}s", code="segment_timeout"
                )
            if return_code != EXIT_OK:
                stderr_file.seek(0, 2)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - _STDERR_TAIL_BYTES))
                tail = stderr_file.read().decode(errors="replace").strip()
                message = f"runner exited with code {return_code}: {tail[-1500:]}"
                if return_code == EXIT_BAD_INPUT:
                    raise PermanentJobError(message, code="bad_input")
                if return_code == EXIT_ENVIRONMENT:
                    raise PermanentJobError(message, code="runner_environment")
                raise TransientJobError(message, code="runner_failed")
