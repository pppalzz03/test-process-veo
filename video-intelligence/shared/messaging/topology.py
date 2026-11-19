"""RabbitMQ topology (declared idempotently by every service that connects).

  exchange video      (direct) --video.infer--> queue video.infer (quorum, DLX, delivery-limit)
  exchange video.dlx  (direct) --dead---------> queue video.dlq   (quorum)

Poison / rejected messages and jobs that exhausted their attempts both end in video.dlq.
"""

from __future__ import annotations

from typing import Any

EXCHANGE = "video"
DLX_EXCHANGE = "video.dlx"
QUEUE_INFER = "video.infer"
QUEUE_DLQ = "video.dlq"
ROUTING_INFER = "video.infer"
ROUTING_DEAD = "dead"


def declare_topology(channel: Any, *, delivery_limit: int = 5) -> None:
    channel.exchange_declare(EXCHANGE, exchange_type="direct", durable=True)
    channel.exchange_declare(DLX_EXCHANGE, exchange_type="direct", durable=True)

    channel.queue_declare(QUEUE_DLQ, durable=True, arguments={"x-queue-type": "quorum"})
    channel.queue_bind(QUEUE_DLQ, DLX_EXCHANGE, ROUTING_DEAD)

    channel.queue_declare(
        QUEUE_INFER,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-dead-letter-exchange": DLX_EXCHANGE,
            "x-dead-letter-routing-key": ROUTING_DEAD,
            # A crash-looping consumer returns the message each time; after N returns the
            # broker dead-letters it instead of redelivering forever.
            "x-delivery-limit": delivery_limit,
        },
    )
    channel.queue_bind(QUEUE_INFER, EXCHANGE, ROUTING_INFER)
