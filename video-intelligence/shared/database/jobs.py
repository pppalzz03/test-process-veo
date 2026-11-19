"""Job lifecycle persistence: creation, lease-based claiming, retry, failure, recovery.

Every state change that must be announced (queue / retry / dead-letter) writes its outbox row in
the SAME transaction. Workers only ever mutate a job they currently own (`worker_id` fencing).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from shared.database.outbox import enqueue_dead_letter, enqueue_job_message
from shared.database.pool import DictConnection
from shared.domain.segments import SegmentPlan
from shared.schemas.processing import ProcessingConfig


def create_job(
    conn: DictConnection,
    *,
    video_id: UUID,
    config: ProcessingConfig,
    segments: list[SegmentPlan],
    max_attempts: int,
    idempotency_key: str | None,
) -> tuple[dict[str, Any], bool]:
    """Create job + segments + outbox row atomically. Returns (job, created)."""
    with conn.transaction():
        if idempotency_key:
            existing = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key=%s", (idempotency_key,)
            ).fetchone()
            if existing:
                return existing, False
        job = conn.execute(
            "INSERT INTO jobs (video_id, status, idempotency_key, model_version, config,"
            " config_hash, max_attempts) VALUES (%s,'queued',%s,%s,%s,%s,%s)"
            " ON CONFLICT (idempotency_key) DO NOTHING RETURNING *",
            (
                video_id,
                idempotency_key,
                config.model_version,
                Jsonb(config.model_dump(mode="json")),
                config.config_hash(),
                max_attempts,
            ),
        ).fetchone()
        if job is None:  # lost a race with a concurrent request using the same key
            existing = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key=%s", (idempotency_key,)
            ).fetchone()
            assert existing is not None
            return existing, False
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO segments (job_id, segment_index, start_ms, end_ms)"
                " VALUES (%s,%s,%s,%s)",
                [(job["id"], s.index, s.start_ms, s.end_ms) for s in segments],
            )
        enqueue_job_message(conn, job["id"])
        return job, True


def get_job(conn: DictConnection, job_id: UUID) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT j.*,"
        " (SELECT count(*) FROM segments s WHERE s.job_id = j.id) AS segments_total,"
        " (SELECT count(*) FROM segments s WHERE s.job_id = j.id AND s.status='done')"
        "   AS segments_done"
        " FROM jobs j WHERE j.id = %s",
        (job_id,),
    ).fetchone()


def list_jobs(
    conn: DictConnection, *, status: str | None, limit: int, offset: int
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT j.*,"
        " (SELECT count(*) FROM segments s WHERE s.job_id = j.id) AS segments_total,"
        " (SELECT count(*) FROM segments s WHERE s.job_id = j.id AND s.status='done')"
        "   AS segments_done"
        " FROM jobs j WHERE (%s::text IS NULL OR j.status = %s)"
        " ORDER BY j.created_at DESC LIMIT %s OFFSET %s",
        (status, status, limit, offset),
    ).fetchall()


def claim_job(
    conn: DictConnection, job_id: UUID, worker_id: str, lease_seconds: int
) -> dict[str, Any] | None:
    """Atomically take ownership. Returns None if the job is done or owned by a live worker.

    Takeover of a `running` job is allowed only after its lease expired (worker presumed dead).
    Each claim consumes one attempt, so a poison video that crashes workers is bounded.
    """
    with conn.transaction():
        return conn.execute(
            "UPDATE jobs SET status='running', worker_id=%s, attempts=attempts+1,"
            " lease_expires_at = now() + make_interval(secs => %s),"
            " started_at = COALESCE(started_at, now()), next_attempt_at = NULL, updated_at = now()"
            " WHERE id=%s AND ("
            "   status = 'queued'"
            "   OR (status = 'retry_wait'"
            "       AND (next_attempt_at IS NULL OR next_attempt_at <= now()))"
            "   OR (status = 'running' AND lease_expires_at < now()))"
            " RETURNING *",
            (worker_id, float(lease_seconds), job_id),
        ).fetchone()


def renew_lease(conn: DictConnection, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
    """Heartbeat. False means we no longer own the job (lease lost) and must stop."""
    with conn.transaction():
        row = conn.execute(
            "UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %s),"
            " updated_at = now() WHERE id=%s AND worker_id=%s AND status='running'"
            " RETURNING 1 AS ok",
            (float(lease_seconds), job_id, worker_id),
        ).fetchone()
    return row is not None


def complete_job(conn: DictConnection, job_id: UUID, worker_id: str) -> bool:
    with conn.transaction():
        row = conn.execute(
            "UPDATE jobs SET status='completed', completed_at=now(), lease_expires_at=NULL,"
            " error_code=NULL, error_message=NULL, updated_at=now()"
            " WHERE id=%s AND worker_id=%s AND status='running' RETURNING 1 AS ok",
            (job_id, worker_id),
        ).fetchone()
    return row is not None


def fail_job(
    conn: DictConnection, job_id: UUID, worker_id: str, *, code: str, message: str
) -> bool:
    """Permanent failure: mark failed and dead-letter, atomically."""
    with conn.transaction():
        row = conn.execute(
            "UPDATE jobs SET status='failed', completed_at=now(), lease_expires_at=NULL,"
            " error_code=%s, error_message=%s, updated_at=now()"
            " WHERE id=%s AND worker_id=%s AND status='running' RETURNING attempts",
            (code, message[:2000], job_id, worker_id),
        ).fetchone()
        if row is None:
            return False
        enqueue_dead_letter(
            conn, job_id, reason_code=code, reason=message, attempts=row["attempts"]
        )
        return True


def schedule_retry(
    conn: DictConnection,
    job_id: UUID,
    worker_id: str,
    *,
    code: str,
    message: str,
    delay_seconds: float,
) -> bool:
    """Transient failure: park the job in retry_wait and queue a delayed message (outbox).

    The due time is computed by PostgreSQL (one clock for scheduling, publishing and claiming),
    so application/DB clock skew can never make a retry message arrive "too early" and be dropped.
    """
    with conn.transaction():
        row = conn.execute(
            "UPDATE jobs SET status='retry_wait',"
            " next_attempt_at = now() + make_interval(secs => %s), lease_expires_at=NULL,"
            " worker_id=NULL, error_code=%s, error_message=%s, updated_at=now()"
            " WHERE id=%s AND worker_id=%s AND status='running'"
            " RETURNING attempts, next_attempt_at",
            (float(delay_seconds), code, message[:2000], job_id, worker_id),
        ).fetchone()
        if row is None:
            return False
        enqueue_job_message(
            conn, job_id, attempt=row["attempts"], available_at=row["next_attempt_at"]
        )
        return True


def release_job(conn: DictConnection, job_id: UUID, worker_id: str) -> bool:
    """Graceful shutdown hand-back: back to queued, attempt refunded, message re-queued."""
    with conn.transaction():
        row = conn.execute(
            "UPDATE jobs SET status='queued', worker_id=NULL, lease_expires_at=NULL,"
            " attempts=GREATEST(attempts-1,0), updated_at=now()"
            " WHERE id=%s AND worker_id=%s AND status='running' RETURNING attempts",
            (job_id, worker_id),
        ).fetchone()
        if row is None:
            return False
        enqueue_job_message(conn, job_id, attempt=row["attempts"])
        return True


def reap_expired_jobs(conn: DictConnection, limit: int = 100) -> dict[str, int]:
    """Recover jobs whose worker died (lease expired). Run periodically by the scheduler."""
    requeued = failed = 0
    with conn.transaction():
        rows = conn.execute(
            "SELECT id, attempts, max_attempts FROM jobs"
            " WHERE status='running' AND lease_expires_at < now()"
            " ORDER BY lease_expires_at LIMIT %s FOR UPDATE SKIP LOCKED",
            (limit,),
        ).fetchall()
        for row in rows:
            if row["attempts"] >= row["max_attempts"]:
                msg = "worker lease expired and attempts are exhausted"
                conn.execute(
                    "UPDATE jobs SET status='failed', completed_at=now(), lease_expires_at=NULL,"
                    " worker_id=NULL, error_code='lease_expired', error_message=%s,"
                    " updated_at=now() WHERE id=%s",
                    (msg, row["id"]),
                )
                enqueue_dead_letter(
                    conn,
                    row["id"],
                    reason_code="lease_expired",
                    reason=msg,
                    attempts=row["attempts"],
                )
                failed += 1
            else:
                conn.execute(
                    "UPDATE jobs SET status='queued', worker_id=NULL, lease_expires_at=NULL,"
                    " error_code='lease_expired', error_message='worker lease expired; re-queued',"
                    " updated_at=now() WHERE id=%s",
                    (row["id"],),
                )
                enqueue_job_message(conn, row["id"], attempt=row["attempts"])
                requeued += 1
    return {"requeued": requeued, "failed": failed}
