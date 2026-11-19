"""Transactional outbox helpers.

Rows are inserted in the same transaction as the state change they announce, then relayed to
RabbitMQ by the scheduler service. Publishing may happen more than once (crash between publish
and mark); consumers are idempotent, so that is safe.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from shared.database.pool import DictConnection
from shared.messaging.topology import DLX_EXCHANGE, EXCHANGE, ROUTING_DEAD, ROUTING_INFER
from shared.schemas.messages import DeadLetterMessage, JobMessage


def enqueue_job_message(
    conn: DictConnection,
    job_id: UUID,
    *,
    attempt: int = 0,
    available_at: datetime | None = None,
) -> None:
    """Queue a video.infer message; `available_at` in the future gives a delayed retry."""
    payload = JobMessage(job_id=job_id, attempt=attempt).model_dump(mode="json")
    conn.execute(
        "INSERT INTO outbox (aggregate_id, event_type, exchange, routing_key, payload,"
        " available_at) VALUES (%s, 'job.queued', %s, %s, %s::jsonb, COALESCE(%s, now()))",
        (job_id, EXCHANGE, ROUTING_INFER, json.dumps(payload), available_at),
    )


def enqueue_dead_letter(
    conn: DictConnection, job_id: UUID, *, reason_code: str, reason: str, attempts: int
) -> None:
    payload = DeadLetterMessage(
        job_id=job_id, reason_code=reason_code, reason=reason[:1000], attempts=attempts
    ).model_dump(mode="json")
    conn.execute(
        "INSERT INTO outbox (aggregate_id, event_type, exchange, routing_key, payload)"
        " VALUES (%s, 'job.dead_lettered', %s, %s, %s::jsonb)",
        (job_id, DLX_EXCHANGE, ROUTING_DEAD, json.dumps(payload)),
    )


def fetch_unpublished(conn: DictConnection, limit: int) -> list[dict[str, Any]]:
    """Lock a batch of due rows. SKIP LOCKED lets several relays run without duplicating work."""
    return conn.execute(
        "SELECT id, exchange, routing_key, payload FROM outbox"
        " WHERE published_at IS NULL AND available_at <= now()"
        " ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED",
        (limit,),
    ).fetchall()


def mark_published(conn: DictConnection, outbox_id: int) -> None:
    conn.execute(
        "UPDATE outbox SET published_at = now(), publish_attempts = publish_attempts + 1,"
        " last_error = NULL WHERE id = %s",
        (outbox_id,),
    )


def mark_publish_failed(conn: DictConnection, outbox_id: int, error: str) -> None:
    conn.execute(
        "UPDATE outbox SET publish_attempts = publish_attempts + 1, last_error = %s WHERE id = %s",
        (error[:500], outbox_id),
    )
