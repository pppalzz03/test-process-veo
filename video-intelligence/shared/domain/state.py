"""Job state machine. The database is the source of truth; this documents and tests it."""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"  # waiting for a worker (outbox row exists)
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"  # failed transiently, re-published at next_attempt_at
    COMPLETED = "completed"
    FAILED = "failed"  # permanent failure or attempts exhausted (dead-lettered)


class SegmentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})

_ALLOWED: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.RETRY_WAIT,
            JobStatus.QUEUED,  # shutdown hand-back or lease-expiry recovery
            JobStatus.RUNNING,  # takeover of an expired lease
        }
    ),
    JobStatus.RETRY_WAIT: frozenset({JobStatus.RUNNING}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in _ALLOWED[current]
