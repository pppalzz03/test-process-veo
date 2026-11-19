"""Outbox relay: publish committed outbox rows to RabbitMQ (with confirms)."""

from __future__ import annotations

import json
import logging

from shared.database import outbox
from shared.database.pool import DictPool
from shared.messaging.publisher import MessagePublisher, PublishError

logger = logging.getLogger(__name__)


def relay_outbox_once(pool: DictPool, publisher: MessagePublisher, batch_size: int) -> int:
    """Publish one batch. Returns the number published.

    A row is marked published only after the broker confirmed it. If the process dies between
    confirm and commit the row is published again later: consumers are idempotent.
    On a publish failure the rest of the batch is left for the next cycle (broker is down).
    """
    published = 0
    with pool.connection() as conn, conn.transaction():
        for row in outbox.fetch_unpublished(conn, batch_size):
            try:
                publisher.publish(
                    row["exchange"],
                    row["routing_key"],
                    json.dumps(row["payload"]).encode(),
                    message_id=str(row["id"]),
                )
            except PublishError as exc:
                outbox.mark_publish_failed(conn, row["id"], str(exc))
                logger.warning(
                    "outbox publish failed", extra={"outbox_id": row["id"], "error": str(exc)}
                )
                break
            outbox.mark_published(conn, row["id"])
            published += 1
    return published
