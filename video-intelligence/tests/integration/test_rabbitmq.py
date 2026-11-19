"""Real-broker tests. SKIPPED unless RABBITMQ_URL points at a disposable RabbitMQ.

    RABBITMQ_URL=amqp://guest:guest@localhost:5672/%2F \
        pytest -m rabbitmq tests/integration/test_rabbitmq.py

NOT EXECUTED in the authoring environment (no RabbitMQ available there). Run them on the DGX
Spark against a throw-away broker; they purge the video.* queues.
"""

from __future__ import annotations

import os
import time
import uuid

import pika
import pytest

from shared.messaging.publisher import PikaPublisher, PublishError
from shared.messaging.topology import (
    DLX_EXCHANGE,
    EXCHANGE,
    QUEUE_DLQ,
    QUEUE_INFER,
    ROUTING_DEAD,
    ROUTING_INFER,
    declare_topology,
)

URL = os.environ.get("RABBITMQ_URL")
pytestmark = [
    pytest.mark.rabbitmq,
    pytest.mark.skipif(not URL, reason="RABBITMQ_URL not set (no broker available)"),
]


@pytest.fixture
def channel():
    conn = pika.BlockingConnection(pika.URLParameters(URL))
    ch = conn.channel()
    declare_topology(ch, delivery_limit=3)
    for q in (QUEUE_INFER, QUEUE_DLQ):
        ch.queue_purge(q)
    yield ch
    conn.close()


def get(ch, queue, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        method, props, body = ch.basic_get(queue, auto_ack=False)
        if method:
            return method, props, body
        time.sleep(0.1)
    return None, None, None


def test_topology_declaration_is_idempotent(channel):
    declare_topology(channel, delivery_limit=3)


def test_publish_with_confirm_reaches_the_queue(channel):
    pub = PikaPublisher(URL, delivery_limit=3)
    pub.publish(EXCHANGE, ROUTING_INFER, b'{"x":1}', message_id="m1")
    method, props, body = get(channel, QUEUE_INFER)
    assert body == b'{"x":1}' and props.message_id == "m1" and props.delivery_mode == 2
    channel.basic_ack(method.delivery_tag)
    pub.close()


def test_unroutable_message_is_an_error_not_silent_loss():
    pub = PikaPublisher(URL, delivery_limit=3)
    with pytest.raises(PublishError):
        pub.publish(EXCHANGE, "no.such.key", b"{}", message_id="m2")
    pub.close()


def test_rejected_message_lands_in_the_dlq(channel):
    pub = PikaPublisher(URL, delivery_limit=3)
    pub.publish(EXCHANGE, ROUTING_INFER, b"poison", message_id=str(uuid.uuid4()))
    method, _, _ = get(channel, QUEUE_INFER)
    channel.basic_reject(method.delivery_tag, requeue=False)
    _, _, body = get(channel, QUEUE_DLQ)
    assert body == b"poison"
    pub.close()


def test_dead_letter_exchange_routes_to_dlq(channel):
    pub = PikaPublisher(URL, delivery_limit=3)
    pub.publish(DLX_EXCHANGE, ROUTING_DEAD, b'{"reason":"x"}', message_id="m3")
    assert get(channel, QUEUE_DLQ)[2] == b'{"reason":"x"}'
    pub.close()


def test_unacked_message_is_redelivered_after_consumer_disconnect():
    conn = pika.BlockingConnection(pika.URLParameters(URL))
    ch = conn.channel()
    declare_topology(ch, delivery_limit=3)
    ch.queue_purge(QUEUE_INFER)
    ch.basic_publish(
        EXCHANGE, ROUTING_INFER, b"redeliver-me", properties=pika.BasicProperties(delivery_mode=2)
    )
    method, _, _ = get(ch, QUEUE_INFER)
    assert method is not None
    conn.close()  # consumer "crashes" without acking

    conn2 = pika.BlockingConnection(pika.URLParameters(URL))
    ch2 = conn2.channel()
    method2, _, body = get(ch2, QUEUE_INFER)
    assert body == b"redeliver-me" and method2.redelivered
    ch2.basic_ack(method2.delivery_tag)
    conn2.close()


def test_delivery_limit_dead_letters_a_crash_looping_message():
    conn = pika.BlockingConnection(pika.URLParameters(URL))
    ch = conn.channel()
    declare_topology(ch, delivery_limit=3)
    ch.queue_purge(QUEUE_INFER)
    ch.queue_purge(QUEUE_DLQ)
    ch.basic_publish(
        EXCHANGE, ROUTING_INFER, b"crash-loop", properties=pika.BasicProperties(delivery_mode=2)
    )
    for _ in range(5):  # always "crash": nack with requeue
        method, _, _ = get(ch, QUEUE_INFER, timeout=2)
        if method is None:
            break
        ch.basic_nack(method.delivery_tag, requeue=True)
    assert get(ch, QUEUE_DLQ)[2] == b"crash-loop"
    conn.close()
