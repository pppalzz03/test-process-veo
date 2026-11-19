"""Scheduler service: outbox relay + lease reaper + GPU slot housekeeping.

One small loop, no extra framework. Run a single replica (the relay is safe to scale, but one is
enough for this load).
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from services.scheduler.gpu_scheduler import GpuScheduler
from services.scheduler.relay import relay_outbox_once
from shared.config.settings import Settings, get_settings
from shared.database.heartbeat import beat
from shared.database.jobs import reap_expired_jobs
from shared.database.pool import create_pool
from shared.logging import configure_logging
from shared.messaging.publisher import PikaPublisher

logger = logging.getLogger("scheduler")
SERVICE_NAME = "scheduler"


def run(settings: Settings, stop: threading.Event) -> None:
    pool = create_pool(settings.database_url, max_size=4)
    publisher = PikaPublisher(settings.rabbitmq_url, delivery_limit=settings.queue_delivery_limit)
    gpu = GpuScheduler(
        pool,
        max_slots=settings.gpu_max_concurrent_streams,
        min_available_mem_mb=settings.gpu_min_available_mem_mb,
        poll_seconds=settings.gpu_slot_poll_seconds,
    )
    gpu.sync_slots()
    last_reap = 0.0
    logger.info("scheduler started", extra={"gpu_slots": settings.gpu_max_concurrent_streams})
    try:
        while not stop.is_set():
            try:
                published = relay_outbox_once(pool, publisher, settings.outbox_batch_size)
                if time.monotonic() - last_reap >= settings.reaper_interval_seconds:
                    with pool.connection() as conn:
                        recovered = reap_expired_jobs(conn)
                    freed = gpu.free_expired()
                    if any(recovered.values()) or freed:
                        logger.warning(
                            "recovered expired work", extra={**recovered, "gpu_slots_freed": freed}
                        )
                    last_reap = time.monotonic()
                with pool.connection() as conn:
                    beat(conn, SERVICE_NAME)
            except Exception:
                logger.exception("scheduler cycle failed; will retry")
                stop.wait(5)
                continue
            if published == 0:
                stop.wait(settings.outbox_poll_seconds)
    finally:
        publisher.close()
        pool.close()
        logger.info("scheduler stopped")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json, service=SERVICE_NAME)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    run(settings, stop)


if __name__ == "__main__":
    main()
