"""Single-GPU admission control for DGX Spark GB10.

There is exactly ONE physical GPU, shared with the existing vLLM processes, and it has no MIG.
Admission therefore combines two cheap, explicit checks instead of trusting `nvidia-smi` load:

  1. Slots: at most GPU_MAX_CONCURRENT_STREAMS DeepStream segments run at once, across all
     worker processes. A slot is a DB row held with a renewable lease (crash => lease expires).
  2. Memory headroom: GB10 uses unified memory, so MemAvailable must stay above
     GPU_MIN_AVAILABLE_MEM_MB before a new pipeline is admitted.

This module never signals, restarts or reconfigures vLLM.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from shared.database.pool import DictPool

logger = logging.getLogger(__name__)


def read_mem_available_mb(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    """MemAvailable in MiB, or None when it cannot be read (non-Linux)."""
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        logger.warning("cannot read MemAvailable", exc_info=True)
    return None


@dataclass(frozen=True, slots=True)
class SlotLease:
    slot_id: int
    holder: str


class GpuScheduler:
    def __init__(
        self,
        pool: DictPool,
        *,
        max_slots: int,
        min_available_mem_mb: int,
        poll_seconds: float,
        mem_reader: Callable[[], int | None] = read_mem_available_mb,
    ) -> None:
        self._pool = pool
        self._max_slots = max_slots
        self._min_mem_mb = min_available_mem_mb
        self._poll = poll_seconds
        self._mem_reader = mem_reader

    def sync_slots(self) -> None:
        """Make the slot table match the configured size. Never evicts a held slot."""
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO gpu_slots (slot_id) SELECT g FROM generate_series(1, %s) g"
                " ON CONFLICT DO NOTHING",
                (self._max_slots,),
            )
            conn.execute(
                "DELETE FROM gpu_slots WHERE slot_id > %s AND"
                " (holder IS NULL OR lease_expires_at < now())",
                (self._max_slots,),
            )

    def memory_ok(self) -> bool:
        available = self._mem_reader()
        return available is None or available >= self._min_mem_mb

    def try_acquire(self, holder: str, lease_seconds: int) -> SlotLease | None:
        if not self.memory_ok():
            return None
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "UPDATE gpu_slots SET holder=%s, acquired_at=now(),"
                " lease_expires_at = now() + make_interval(secs => %s)"
                " WHERE slot_id = (SELECT slot_id FROM gpu_slots"
                "   WHERE slot_id <= %s AND (holder IS NULL OR lease_expires_at < now())"
                "   ORDER BY slot_id LIMIT 1 FOR UPDATE SKIP LOCKED)"
                " RETURNING slot_id",
                (holder, float(lease_seconds), self._max_slots),
            ).fetchone()
        return SlotLease(row["slot_id"], holder) if row else None

    def acquire(self, holder: str, lease_seconds: int, cancel: threading.Event) -> SlotLease | None:
        """Block until a slot is free; returns None if `cancel` is set first."""
        waited = 0.0
        while not cancel.is_set():
            lease = self.try_acquire(holder, lease_seconds)
            if lease:
                return lease
            if waited and int(waited) % 60 == 0:
                logger.info("waiting for gpu slot", extra={"holder": holder, "waited_s": waited})
            cancel.wait(self._poll)
            waited += self._poll
        return None

    def renew(self, lease: SlotLease, lease_seconds: int) -> bool:
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "UPDATE gpu_slots SET lease_expires_at = now() + make_interval(secs => %s)"
                " WHERE slot_id=%s AND holder=%s RETURNING 1 AS ok",
                (float(lease_seconds), lease.slot_id, lease.holder),
            ).fetchone()
        return row is not None

    def release(self, lease: SlotLease) -> None:
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "UPDATE gpu_slots SET holder=NULL, acquired_at=NULL, lease_expires_at=NULL"
                " WHERE slot_id=%s AND holder=%s",
                (lease.slot_id, lease.holder),
            )

    def free_expired(self) -> int:
        """Reaper hook: expired leases are reusable already; clear them for observability."""
        with self._pool.connection() as conn, conn.transaction():
            cur = conn.execute(
                "UPDATE gpu_slots SET holder=NULL, acquired_at=NULL, lease_expires_at=NULL"
                " WHERE holder IS NOT NULL AND lease_expires_at < now()"
            )
            return cur.rowcount
