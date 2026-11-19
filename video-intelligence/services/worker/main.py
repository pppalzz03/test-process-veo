"""Worker service: RabbitMQ consumer -> JobHandler.

pika's BlockingConnection cannot send heartbeats while the calling thread is busy, and a job may
run for an hour. So the job runs in a helper thread while the main thread keeps servicing the
connection (heartbeats) via process_data_events. ACK happens only after the DB outcome is durable.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading

import pika
from pika.exceptions import AMQPError

from services.scheduler.gpu_scheduler import GpuScheduler
from services.worker.handler import JobHandler, parse_job_message
from services.worker.processor import JobProcessor, StopSignals
from services.worker.runner import SubprocessRunner
from shared.config.settings import Settings, get_settings
from shared.database.heartbeat import beat
from shared.database.pool import create_pool
from shared.logging import configure_logging
from shared.messaging.topology import QUEUE_INFER, declare_topology

logger = logging.getLogger("worker")


def default_worker_id(prefix: str) -> str:
    return f"{prefix}-{socket.gethostname()}-{os.getpid()}"


def run(settings: Settings, shutdown: threading.Event) -> None:
    worker_id = default_worker_id(settings.worker_id)
    pool = create_pool(settings.database_url, max_size=6)
    gpu = GpuScheduler(
        pool,
        max_slots=settings.gpu_max_concurrent_streams,
        min_available_mem_mb=settings.gpu_min_available_mem_mb,
        poll_seconds=settings.gpu_slot_poll_seconds,
    )
    gpu.sync_slots()
    processor = JobProcessor(
        pool=pool,
        settings=settings,
        runner=SubprocessRunner(settings.runner_argv()),
        gpu=gpu,
        worker_id=worker_id,
    )
    handler = JobHandler(pool=pool, settings=settings, processor=processor, worker_id=worker_id)

    while not shutdown.is_set():
        try:
            _consume(settings, handler, pool, shutdown, worker_id)
        except (AMQPError, OSError):
            logger.exception("broker connection lost; reconnecting in 5s")
            shutdown.wait(5)
    pool.close()
    logger.info("worker stopped")


def _consume(settings, handler, pool, shutdown, worker_id) -> None:  # type: ignore[no-untyped-def]
    params = pika.URLParameters(settings.rabbitmq_url)
    params.heartbeat = 30
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    declare_topology(channel, delivery_limit=settings.queue_delivery_limit)
    # Bounded concurrency: this process pulls ONE message at a time (basic_get, sequential).
    # Scale out with more worker processes; the GPU slot table caps real GPU concurrency.
    logger.info("worker consuming", extra={"worker_id": worker_id, "queue": QUEUE_INFER})

    try:
        while not shutdown.is_set():
            method, _props, body = channel.basic_get(QUEUE_INFER, auto_ack=False)
            if method is None:
                connection.process_data_events(time_limit=1.0)
                with pool.connection() as conn:
                    beat(conn, f"worker:{worker_id}")
                continue
            _handle_delivery(connection, channel, handler, method.delivery_tag, body, shutdown)
    finally:
        if connection.is_open:
            connection.close()


def _handle_delivery(connection, channel, handler, delivery_tag, body, shutdown) -> None:  # type: ignore[no-untyped-def]
    try:
        message = parse_job_message(body)
    except ValueError:
        logger.error("rejecting malformed message to DLQ", extra={"body": body[:200]})
        channel.basic_reject(delivery_tag, requeue=False)  # -> DLX -> video.dlq
        return

    stops = StopSignals()
    result: dict[str, object] = {}

    def run_job() -> None:
        try:
            result["outcome"] = handler.handle(message, stops)
        except BaseException as exc:  # DB down etc.: do NOT ack; broker will redeliver
            result["error"] = exc
            logger.exception("job handler crashed", extra={"job_id": str(message.job_id)})

    thread = threading.Thread(target=run_job, name=f"job-{message.job_id}", daemon=True)
    thread.start()
    while thread.is_alive():
        connection.process_data_events(time_limit=1.0)  # keeps AMQP heartbeats flowing
        if shutdown.is_set() and not stops.shutdown.is_set():
            stops.signal_shutdown()
    thread.join()

    if "error" in result:
        # Leave unacked: closing the connection returns the message for redelivery, and the
        # lease reaper recovers the job if no redelivery claims it first.
        raise OSError("job handler crashed; connection will be closed for redelivery")
    channel.basic_ack(delivery_tag)  # outcome is durable in PostgreSQL


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json, service="worker")
    shutdown = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: shutdown.set())
    run(settings, shutdown)


if __name__ == "__main__":
    main()
