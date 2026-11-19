"""Job processing: segments -> GPU slot -> runner -> tracks/events -> atomic commit."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.event_processor.extract import ExtractionParams, build_tracks, extract_events
from services.scheduler.gpu_scheduler import GpuScheduler, SlotLease
from services.worker.runner import RunnerCancelled, RunnerRequest, SegmentRunner
from shared.config.settings import Settings
from shared.database import jobs as jobs_db
from shared.database import results as results_db
from shared.database.pool import DictPool
from shared.database.videos import get_video
from shared.domain.detections import Detection
from shared.domain.errors import (
    JobError,
    LeaseLostError,
    PermanentJobError,
    ShutdownRequested,
    TransientJobError,
)
from shared.logging import log_context
from shared.schemas.processing import ProcessingConfig
from shared.schemas.runner import FrameRecord, SummaryRecord

logger = logging.getLogger(__name__)


@dataclass
class StopSignals:
    """`any` is what the runner watches; the two flags tell us why it was set."""

    any: threading.Event = field(default_factory=threading.Event)
    lease_lost: threading.Event = field(default_factory=threading.Event)
    shutdown: threading.Event = field(default_factory=threading.Event)

    def signal_lease_lost(self) -> None:
        self.lease_lost.set()
        self.any.set()

    def signal_shutdown(self) -> None:
        self.shutdown.set()
        self.any.set()

    def as_exception(self) -> JobError:
        if self.lease_lost.is_set():
            return LeaseLostError("lease lost")
        return ShutdownRequested("worker shutting down")


class LeaseKeeper(threading.Thread):
    """Heartbeat: renews the job lease and the held GPU slot while a segment is running."""

    def __init__(
        self,
        pool: DictPool,
        gpu: GpuScheduler,
        job_id: Any,
        worker_id: str,
        stops: StopSignals,
        *,
        interval: float,
        lease_seconds: int,
    ) -> None:
        super().__init__(daemon=True, name="lease-keeper")
        self._pool, self._gpu, self._job_id, self._worker_id = pool, gpu, job_id, worker_id
        self._stops, self._interval, self._lease_seconds = stops, interval, lease_seconds
        self._halt = threading.Event()
        self.slot: SlotLease | None = None

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            try:
                with self._pool.connection() as conn:
                    owned = jobs_db.renew_lease(
                        conn, self._job_id, self._worker_id, self._lease_seconds
                    )
                if not owned:
                    logger.error("job lease lost", extra={"job_id": str(self._job_id)})
                    self._stops.signal_lease_lost()
                    return
                slot = self.slot
                if slot is not None:
                    self._gpu.renew(slot, self._lease_seconds)
            except Exception:  # transient DB error: keep trying until the lease truly expires
                logger.exception("heartbeat failed; will retry")

    def stop(self) -> None:
        self._halt.set()


class JobProcessor:
    def __init__(
        self,
        *,
        pool: DictPool,
        settings: Settings,
        runner: SegmentRunner,
        gpu: GpuScheduler,
        worker_id: str,
    ) -> None:
        self._pool, self._settings, self._runner = pool, settings, runner
        self._gpu, self._worker_id = gpu, worker_id

    def process(self, job: dict[str, Any], stops: StopSignals) -> None:
        """Run every pending segment of an already-claimed job. Raises JobError subclasses."""
        job_id = job["id"]
        config = ProcessingConfig.model_validate(job["config"])
        with self._pool.connection() as conn:
            video = get_video(conn, job["video_id"])
            segments = results_db.list_segments(conn, job_id)
        if video is None:
            raise PermanentJobError("video row missing", code="video_missing")

        source = Path(self._settings.media_root) / video["storage_key"]
        if not source.is_file():
            raise PermanentJobError(f"source file not found: {source}", code="source_missing")

        keeper = LeaseKeeper(
            self._pool,
            self._gpu,
            job_id,
            self._worker_id,
            stops,
            interval=self._settings.heartbeat_seconds,
            lease_seconds=self._settings.lease_seconds,
        )
        keeper.start()
        try:
            last_index = segments[-1]["segment_index"]
            for seg in segments:
                if seg["status"] == "done":
                    continue  # resume after a crash / retry: finished segments are never redone
                with log_context(segment_index=seg["segment_index"]):
                    self._process_segment(
                        job,
                        video,
                        seg,
                        config,
                        source,
                        keeper,
                        stops,
                        is_last=seg["segment_index"] == last_index,
                    )
        finally:
            keeper.stop()

    def _process_segment(
        self,
        job: dict[str, Any],
        video: dict[str, Any],
        seg: dict[str, Any],
        config: ProcessingConfig,
        source: Path,
        keeper: LeaseKeeper,
        stops: StopSignals,
        *,
        is_last: bool,
    ) -> None:
        holder = f"{self._worker_id}:{job['id']}:{seg['segment_index']}"
        lease = self._gpu.acquire(holder, self._settings.lease_seconds, stops.any)
        if lease is None:
            raise stops.as_exception()
        keeper.slot = lease
        started = time.monotonic()
        try:
            with self._pool.connection() as conn:
                results_db.mark_segment_running(conn, seg["id"])
            detections, frames = self._run_segment(video, seg, config, source, stops, is_last)
        finally:
            keeper.slot = None
            self._gpu.release(lease)

        tracks = build_tracks(detections)
        events = extract_events(
            seg["segment_index"],
            detections,
            tracks,
            ExtractionParams(config.min_track_ms, config.min_track_frames),
        )
        persisted = (
            [d for d in detections if d.frame_index % config.detection_stride == 0]
            if config.persist_detections
            else []
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        with self._pool.connection() as conn:
            written = results_db.commit_segment_results(
                conn,
                job_id=job["id"],
                worker_id=self._worker_id,
                segment_id=seg["id"],
                tracks=tracks,
                detections=persisted,
                events=events,
                frames_processed=frames,
                processing_ms=elapsed_ms,
            )
        logger.info(
            "segment committed" if written else "segment already committed (duplicate)",
            extra={
                "frames": frames,
                "detections": len(detections),
                "events": len(events),
                "processing_ms": elapsed_ms,
            },
        )

    def _run_segment(
        self,
        video: dict[str, Any],
        seg: dict[str, Any],
        config: ProcessingConfig,
        source: Path,
        stops: StopSignals,
        is_last: bool,
    ) -> tuple[list[Detection], int]:
        offset = int(video["start_offset_ms"])
        s = self._settings
        request = RunnerRequest(
            source_path=str(source),
            start_ms=seg["start_ms"] + offset,
            end_ms=seg["end_ms"] + offset,
            infer_config=s.ds_infer_config,
            tracker_config=s.ds_tracker_config,
            tracker_lib=s.ds_tracker_lib,
            labels_file=s.ds_labels_file,
            decode_mode=s.ds_decode_mode,
            codec=video.get("codec"),
            infer_interval=config.infer_interval,
            timeout_seconds=s.segment_timeout_seconds,
        )

        detections: list[Detection] = []
        frames = 0
        summary: SummaryRecord | None = None
        try:
            for record in self._runner.run(request, stops.any):
                if isinstance(record, SummaryRecord):
                    summary = record
                    continue
                timestamp_ms = record.pts_ms - offset  # absolute position in the video
                if not _in_window(timestamp_ms, seg, is_last):
                    continue
                _append_detections(record, timestamp_ms, frames, config, detections)
                frames += 1
        except RunnerCancelled:
            raise stops.as_exception() from None

        if summary is None or not summary.eos:
            raise TransientJobError("pipeline ended without EOS", code="no_eos")
        self._check_coverage(video, seg, frames)
        return detections, frames

    def _check_coverage(self, video: dict[str, Any], seg: dict[str, Any], frames: int) -> None:
        fps = video.get("fps")
        min_cov = self._settings.min_frame_coverage
        if not fps or min_cov <= 0:
            return
        expected = (seg["end_ms"] - seg["start_ms"]) / 1000 * fps
        if frames < expected * min_cov:
            raise PermanentJobError(
                f"decoded {frames} frames, expected about {expected:.0f} "
                f"(coverage < {min_cov:.0%}); video is likely truncated or corrupt",
                code="low_frame_coverage",
            )


def _in_window(timestamp_ms: int, seg: dict[str, Any], is_last: bool) -> bool:
    """Accurate seeks decode from the previous keyframe; clip to the segment window so adjacent
    segments never both claim the same frame. The last segment is open-ended (video duration
    can be a few ms shorter than the final frame's PTS)."""
    if timestamp_ms < seg["start_ms"]:
        return False
    return is_last or timestamp_ms < seg["end_ms"]


def _append_detections(
    record: FrameRecord,
    timestamp_ms: int,
    frame_index: int,  # segment-local ordinal of the frame (0-based)
    config: ProcessingConfig,
    out: list[Detection],
) -> None:
    for obj in record.objects:
        if obj.confidence < config.min_confidence:
            continue
        left, top, width, height = obj.bbox
        out.append(
            Detection(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                track_id=obj.track_id,
                class_id=obj.class_id,
                class_name=obj.label,
                confidence=obj.confidence,
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
