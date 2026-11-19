"""Publisher with confirms. The only place the scheduler touches the pika API."""

from __future__ import annotations

import logging
from typing import Protocol

import pika
from pika.exceptions import AMQPError

from shared.messaging.topology import declare_topology

logger = logging.getLogger(__name__)


class PublishError(Exception):
    """The broker did not durably accept the message."""


class MessagePublisher(Protocol):
    def publish(self, exchange: str, routing_key: str, body: bytes, message_id: str) -> None: ...

    def close(self) -> None: ...


class PikaPublisher:
    """Blocking publisher using publisher confirms; reconnects lazily after a failure."""

    def __init__(self, url: str, *, delivery_limit: int = 5) -> None:
        self._url = url
        self._delivery_limit = delivery_limit
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None

    def _ensure_channel(self) -> pika.adapters.blocking_connection.BlockingChannel:
        if self._connection is None or self._connection.is_closed:
            params = pika.URLParameters(self._url)
            params.heartbeat = 30
            params.blocked_connection_timeout = 60
            self._connection = pika.BlockingConnection(params)
            self._channel = None
        if self._channel is None or self._channel.is_closed:
            self._channel = self._connection.channel()
            self._channel.confirm_delivery()
            declare_topology(self._channel, delivery_limit=self._delivery_limit)
        return self._channel

    def publish(self, exchange: str, routing_key: str, body: bytes, message_id: str) -> None:
        try:
            channel = self._ensure_channel()
            channel.basic_publish(
                exchange,
                routing_key,
                body,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                    content_type="application/json",
                    message_id=message_id,
                ),
                mandatory=True,  # unroutable -> UnroutableError instead of silent loss
            )
        except (AMQPError, OSError) as exc:  # NackError/UnroutableError are AMQPError subclasses
            self.close()
            raise PublishError(f"{type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except (AMQPError, OSError):
            logger.debug("error while closing publisher connection", exc_info=True)
        self._connection = None
        self._channel = None
