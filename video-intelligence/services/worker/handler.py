"""Business logic for one queue message: claim -> process -> record outcome.

Kept free of RabbitMQ so every branch (duplicate delivery, retry, DLQ, recovery) is testable
against a real database without a broker.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from services.worker.processor import JobProcessor, StopSignals
from shared.config.settings import Settings
from shared.database import jobs as jobs_db
from shared.database.pool import DictPool
from shared.domain.backoff import compute_backoff_seconds
from shared.domain.errors import (
    JobError,
    LeaseLostError,
    PermanentJobError,
    ShutdownRequested,
    TransientJobError,
)
from shared.logging import log_context
from shared.schemas.messages import JobMessage

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    COMPLETED = "completed"
    DROPPED = "dropped"  # duplicate / stale message; nothing to do
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"  # dead-lettered
    RELEASED = "released"  # worker shutting down
    LEASE_LOST = "lease_lost"


def parse_job_message(body: bytes) -> JobMessage:
    """Raises ValueError for malformed messages (caller rejects them to the DLQ)."""
    try:
        return JobMessage.model_validate_json(body)
    except Exception as exc:
        raise ValueError(f"malformed job message: {exc}") from exc


class JobHandler:
    def __init__(
        self,
        *,
        pool: DictPool,
        settings: Settings,
        processor: JobProcessor,
        worker_id: str,
    ) -> None:
        self._pool, self._settings = pool, settings
        self._processor, self._worker_id = processor, worker_id

    def handle(self, message: JobMessage, stops: StopSignals) -> Outcome:
        with log_context(job_id=str(message.job_id), worker=self._worker_id):
            with self._pool.connection() as conn:
                job = jobs_db.claim_job(
                    conn, message.job_id, self._worker_id, self._settings.lease_seconds
                )
            if job is None:
                logger.info("message dropped: job not claimable (done or owned by a live worker)")
                return Outcome.DROPPED

            logger.info("job claimed", extra={"attempt": job["attempts"]})
            try:
                self._processor.process(job, stops)
            except LeaseLostError:
                logger.warning("lease lost; abandoning job without writing state")
                return Outcome.LEASE_LOST
            except ShutdownRequested:
                with self._pool.connection() as conn:
                    jobs_db.release_job(conn, job["id"], self._worker_id)
                logger.info("job released for shutdown")
                return Outcome.RELEASED
            except PermanentJobError as exc:
                return self._fail(job, exc)
            except TransientJobError as exc:
                return self._retry_or_fail(job, exc)
            except Exception as exc:  # unknown bug: treat as transient, bounded by max_attempts
                logger.exception("unexpected error while processing job")
                return self._retry_or_fail(job, TransientJobError(repr(exc), code="unexpected"))

            with self._pool.connection() as conn:
                done = jobs_db.complete_job(conn, job["id"], self._worker_id)
            if not done:
                return Outcome.LEASE_LOST
            logger.info("job completed")
            return Outcome.COMPLETED

    def _fail(self, job: dict, exc: JobError) -> Outcome:
        logger.error("job failed permanently", extra={"code": exc.code, "error": str(exc)})
        with self._pool.connection() as conn:
            jobs_db.fail_job(conn, job["id"], self._worker_id, code=exc.code, message=str(exc))
        return Outcome.FAILED

    def _retry_or_fail(self, job: dict, exc: JobError) -> Outcome:
        if job["attempts"] >= job["max_attempts"]:
            logger.error("attempts exhausted", extra={"code": exc.code, "error": str(exc)})
            return self._fail(job, exc)
        delay = compute_backoff_seconds(
            job["attempts"],
            base_seconds=self._settings.retry_base_seconds,
            max_seconds=self._settings.retry_max_seconds,
        )
        logger.warning(
            "transient failure; retry scheduled",
            extra={"code": exc.code, "error": str(exc), "retry_in_s": round(delay, 1)},
        )
        with self._pool.connection() as conn:
            jobs_db.schedule_retry(
                conn,
                job["id"],
                self._worker_id,
                code=exc.code,
                message=str(exc),
                delay_seconds=delay,
            )
        return Outcome.RETRY_SCHEDULED
