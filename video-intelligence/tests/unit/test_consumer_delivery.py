"""Ack / reject semantics of the worker's delivery handler, using a fake channel (no broker)."""

from __future__ import annotations

import threading
import uuid

import pytest

from services.worker.handler import Outcome
from services.worker.main import _handle_delivery
from services.worker.processor import StopSignals


class FakeConnection:
    def __init__(self):
        self.pumped = 0

    def process_data_events(self, time_limit=0):
        self.pumped += 1
        threading.Event().wait(0.01)


class FakeChannel:
    def __init__(self):
        self.acked, self.rejected = [], []

    def basic_ack(self, tag):
        self.acked.append(tag)

    def basic_reject(self, tag, requeue):
        self.rejected.append((tag, requeue))


class FakeHandler:
    def __init__(self, outcome=Outcome.COMPLETED, error=None, delay=0.0):
        self.outcome, self.error, self.delay = outcome, error, delay
        self.seen_stops: StopSignals | None = None

    def handle(self, message, stops):
        self.seen_stops = stops
        if self.delay:
            stops.any.wait(self.delay)
        if self.error:
            raise self.error
        return self.outcome


def body():
    return f'{{"job_id": "{uuid.uuid4()}"}}'.encode()


def test_ack_only_after_the_handler_reports_a_durable_outcome():
    ch, conn = FakeChannel(), FakeConnection()
    _handle_delivery(conn, ch, FakeHandler(Outcome.COMPLETED), 7, body(), threading.Event())
    assert ch.acked == [7] and ch.rejected == []


@pytest.mark.parametrize("outcome", list(Outcome))
def test_every_outcome_is_acked_because_state_lives_in_postgres(outcome):
    ch = FakeChannel()
    _handle_delivery(FakeConnection(), ch, FakeHandler(outcome), 1, body(), threading.Event())
    assert ch.acked == [1]


def test_malformed_message_is_rejected_without_requeue_to_reach_the_dlq():
    ch = FakeChannel()
    handler = FakeHandler()
    _handle_delivery(FakeConnection(), ch, handler, 3, b"garbage", threading.Event())
    assert ch.rejected == [(3, False)] and ch.acked == [] and handler.seen_stops is None


def test_handler_crash_leaves_message_unacked_and_forces_reconnect():
    ch = FakeChannel()
    with pytest.raises(OSError):
        _handle_delivery(
            FakeConnection(),
            ch,
            FakeHandler(error=RuntimeError("db down")),
            5,
            body(),
            threading.Event(),
        )
    assert ch.acked == [] and ch.rejected == []  # broker redelivers after the connection closes


def test_connection_is_serviced_while_a_long_job_runs():
    """pika heartbeats need process_data_events during the job (1h videos)."""
    conn = FakeConnection()
    _handle_delivery(conn, FakeChannel(), FakeHandler(delay=0.15), 1, body(), threading.Event())
    assert conn.pumped >= 3


def test_shutdown_signal_reaches_a_running_job():
    shutdown = threading.Event()
    handler = FakeHandler(Outcome.RELEASED, delay=5)
    threading.Timer(0.1, shutdown.set).start()
    ch = FakeChannel()
    _handle_delivery(FakeConnection(), ch, handler, 9, body(), shutdown)
    assert handler.seen_stops is not None and handler.seen_stops.shutdown.is_set()
    assert ch.acked == [9]
