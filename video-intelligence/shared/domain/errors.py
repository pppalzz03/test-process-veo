"""Error taxonomy that drives the retry policy."""

from __future__ import annotations


class JobError(Exception):
    """Base class. `code` is a short machine-readable reason stored on the job."""

    code = "job_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class PermanentJobError(JobError):
    """Retrying cannot help (corrupt video, missing file). Goes straight to DLQ."""

    code = "permanent_error"


class TransientJobError(JobError):
    """Worth retrying with backoff (pipeline crash, GPU/driver hiccup)."""

    code = "transient_error"


class LeaseLostError(JobError):
    """Another worker owns the job now. Abort silently; never write results."""

    code = "lease_lost"


class ShutdownRequested(JobError):
    """Worker is stopping; hand the job back without consuming an attempt."""

    code = "shutdown"
