"""Worker orchestration with a REAL PostgreSQL and a REAL subprocess (synthetic-runner test double).

These prove orchestration behaviour (retry, DLQ, recovery, idempotency, GPU admission). They say
nothing about DeepStream/YOLO accuracy or speed - that needs the DGX Spark.
"""

from __future__ import annotations

import threading
import time

import pytest

from services.scheduler.gpu_scheduler import GpuScheduler
from services.worker.handler import JobHandler, Outcome, parse_job_message
from services.worker.processor import JobProcessor, StopSignals
from services.worker.runner import RunnerRequest, SubprocessRunner
from shared.database import jobs as jobs_db
from shared.database.results import commit_segment_results, list_segments
from shared.domain.errors import PermanentJobError, TransientJobError
from shared.domain.segments import plan_segments
from shared.schemas.messages import JobMessage
from tests.conftest import FAKE_RUNNER
from tests.integration.test_db_lifecycle import expire_lease, make_video, outbox_rows

pytestmark = pytest.mark.integration


class SpyRunner:
    """Wraps the real SubprocessRunner; records calls and peak concurrency."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.active = self.peak = 0
        self._lock = threading.Lock()

    def run(self, request, cancel):
        with self._lock:
            self.calls.append((request.start_ms, request.end_ms))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield from self.inner.run(request, cancel)
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def build(pool, settings, media_root):
    (media_root / "v.mp4").write_bytes(b"synthetic")  # content is ignored by the fake runner

    def _build(worker_id="w1", runner=None, slots=1, scheduler=None, **job_kw):
        gpu = scheduler or GpuScheduler(
            pool, max_slots=slots, min_available_mem_mb=0, poll_seconds=0.02
        )
        gpu.sync_slots()
        runner = runner or SpyRunner(SubprocessRunner(settings.runner_argv()))
        processor = JobProcessor(
            pool=pool, settings=settings, runner=runner, gpu=gpu, worker_id=worker_id
        )
        return JobHandler(
            pool=pool, settings=settings, processor=processor, worker_id=worker_id
        ), runner

    return _build


def new_job(pool, settings, max_attempts=3, fps=10.0, duration_ms=25_000, sha="a" * 64):
    with pool.connection() as conn:
        video = make_video(conn, sha=sha, duration_ms=duration_ms)
        if fps != 10.0:
            conn.execute("UPDATE videos SET fps=%s WHERE id=%s", (fps, video["id"]))
        job, _ = jobs_db.create_job(
            conn,
            video_id=video["id"],
            config=settings.processing_config(),
            segments=plan_segments(duration_ms, 10_000),
            max_attempts=max_attempts,
            idempotency_key=None,
        )
    return job


def make_retry_due(pool):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET next_attempt_at = now() - interval '1 second'"
            " WHERE status='retry_wait'"
        )


def msg(job):
    return JobMessage(job_id=job["id"])


def q(pool, sql, *params):
    with pool.connection() as conn:
        return conn.execute(sql, params).fetchall()


class TestSuccessPath:
    def test_full_job_persists_tracks_events_and_detections(self, pool, settings, build):
        job = new_job(pool, settings)
        handler, runner = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.COMPLETED

        with pool.connection() as conn:
            got = jobs_db.get_job(conn, job["id"])
        assert got["status"] == "completed" and got["segments_done"] == got["segments_total"] == 3
        assert [c for c in runner.calls] == [(0, 10_000), (10_000, 20_000), (20_000, 25_000)]

        frames = q(pool, "SELECT segment_index, frames_processed FROM segments ORDER BY 1")
        assert [r["frames_processed"] for r in frames] == [100, 100, 50]

        events = q(
            pool,
            "SELECT event_type, class_name, start_ms, end_ms, event_key FROM events"
            " ORDER BY start_ms, event_key",
        )
        kinds = sorted({(e["event_type"], e["class_name"]) for e in events})
        assert kinds == [
            ("count_peak", "car"),
            ("count_peak", "person"),
            ("object_presence", "car"),
            ("object_presence", "person"),
        ]
        assert len(events) == 12  # 4 per segment x 3 segments, none duplicated
        assert len({e["event_key"] for e in events}) == 12

    def test_timestamps_are_absolute_and_clipped_to_segment_windows(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_PREROLL_MS", "3000")  # runner emits frames before the window
        job = new_job(pool, settings)
        handler, _ = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.COMPLETED
        rows = q(
            pool,
            "SELECT s.segment_index, s.start_ms, s.end_ms, min(d.timestamp_ms) lo,"
            " max(d.timestamp_ms) hi FROM detections d JOIN segments s"
            " ON s.id=d.segment_id GROUP BY 1,2,3 ORDER BY 1",
        )
        assert len(rows) == 3
        for r in rows:
            assert r["lo"] >= r["start_ms"]
            assert r["hi"] < r["end_ms"] or r["segment_index"] == 2
        assert rows[1]["lo"] == 10_000  # no pre-roll frame leaked into segment 1
        # Segment-local track ids: track 1 exists once per segment, never merged.
        assert q(pool, "SELECT count(*) n FROM tracks WHERE local_track_id=1")[0]["n"] == 3

    def test_low_confidence_detections_are_filtered(self, pool, settings, build):
        job = new_job(pool, settings)
        handler, _ = build()
        handler.handle(msg(job), StopSignals())
        assert q(pool, "SELECT count(*) n FROM detections WHERE confidence < 0.25")[0]["n"] == 0


class TestIdempotency:
    def test_redelivered_message_after_completion_is_dropped(self, pool, settings, build):
        job = new_job(pool, settings)
        handler, runner = build()
        handler.handle(msg(job), StopSignals())
        before = q(
            pool, "SELECT (SELECT count(*) FROM events) e, (SELECT count(*) FROM detections) d"
        )[0]
        calls = len(runner.calls)
        assert handler.handle(msg(job), StopSignals()) == Outcome.DROPPED
        assert (
            q(pool, "SELECT (SELECT count(*) FROM events) e, (SELECT count(*) FROM detections) d")[
                0
            ]
            == before
        )
        assert len(runner.calls) == calls  # no wasted GPU work

    def test_duplicate_message_while_another_worker_runs_is_dropped(self, pool, settings, build):
        job = new_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "other-worker", 300)
        handler, runner = build("w2")
        assert handler.handle(msg(job), StopSignals()) == Outcome.DROPPED
        assert runner.calls == []

    def test_resume_skips_segments_that_are_already_done(self, pool, settings, build):
        job = new_job(pool, settings)
        # A previous attempt finished segment 0 and then died.
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "dead", 30)
            seg0 = list_segments(conn, job["id"])[0]
            commit_segment_results(
                conn,
                job_id=job["id"],
                worker_id="dead",
                segment_id=seg0["id"],
                tracks=[],
                detections=[],
                events=[],
                frames_processed=100,
                processing_ms=1,
            )
        expire_lease(pool, job["id"])
        handler, runner = build("w2")
        assert handler.handle(msg(job), StopSignals()) == Outcome.COMPLETED
        assert runner.calls == [(10_000, 20_000), (20_000, 25_000)]


class TestFailureHandling:
    def test_corrupt_video_fails_immediately_and_is_dead_lettered(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_MODE", "bad_input")
        job = new_job(pool, settings)
        handler, _ = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.FAILED
        got = q(pool, "SELECT status, error_code, attempts FROM jobs")[0]
        assert got == {"status": "failed", "error_code": "bad_input", "attempts": 1}
        assert len(outbox_rows(pool, "job.dead_lettered")) == 1

    def test_transient_failure_retries_with_backoff_then_dead_letters(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_MODE", "crash")
        job = new_job(pool, settings, max_attempts=3)
        handler, _ = build()
        for attempt in (1, 2):
            make_retry_due(pool)  # simulate the backoff delay elapsing
            assert handler.handle(msg(job), StopSignals()) == Outcome.RETRY_SCHEDULED
            row = q(pool, "SELECT status, attempts, next_attempt_at > now() AS future FROM jobs")[0]
            assert row == {"status": "retry_wait", "attempts": attempt, "future": True}
        # Delayed retry messages sit in the outbox (not yet due) - one per failed attempt.
        assert len(outbox_rows(pool, "job.queued")) == 3
        make_retry_due(pool)
        assert handler.handle(msg(job), StopSignals()) == Outcome.FAILED  # attempt 3 = max
        assert q(pool, "SELECT status, error_code FROM jobs")[0]["status"] == "failed"
        assert len(outbox_rows(pool, "job.dead_lettered")) == 1

    def test_pipeline_without_eos_is_retried(self, pool, settings, build, monkeypatch):
        monkeypatch.setenv("FAKE_MODE", "no_eos")
        job = new_job(pool, settings)
        handler, _ = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.RETRY_SCHEDULED
        assert q(pool, "SELECT error_code FROM jobs")[0]["error_code"] == "no_eos"
        # Nothing from the incomplete segment may have been committed.
        assert q(pool, "SELECT count(*) n FROM events")[0]["n"] == 0

    def test_truncated_video_is_a_permanent_failure(self, pool, settings, build):
        job = new_job(pool, settings, fps=30.0)  # fake emits 10 fps: 33% coverage
        handler, _ = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.FAILED
        assert q(pool, "SELECT error_code FROM jobs")[0]["error_code"] == "low_frame_coverage"

    def test_missing_source_file_is_permanent(self, pool, settings, build, media_root):
        (media_root / "v.mp4").unlink(missing_ok=True)
        job = new_job(pool, settings)
        handler, _ = build()
        assert handler.handle(msg(job), StopSignals()) == Outcome.FAILED
        assert q(pool, "SELECT error_code FROM jobs")[0]["error_code"] == "source_missing"

    def test_database_state_is_unchanged_by_partial_failure(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_MODE", "crash")
        job = new_job(pool, settings)
        build()[0].handle(msg(job), StopSignals())
        assert q(
            pool, "SELECT (SELECT count(*) FROM tracks) t, (SELECT count(*) FROM detections) d"
        )[0] == {"t": 0, "d": 0}


class TestRecovery:
    def test_job_recovers_after_worker_death(self, pool, settings, build):
        job = new_job(pool, settings)
        with pool.connection() as conn:  # worker A claims the job, then is killed
            jobs_db.claim_job(conn, job["id"], "worker-A", 30)
        expire_lease(pool, job["id"])
        with pool.connection() as conn:  # scheduler reaper notices
            assert jobs_db.reap_expired_jobs(conn)["requeued"] == 1
        handler, _ = build("worker-B")
        assert handler.handle(msg(job), StopSignals()) == Outcome.COMPLETED
        assert q(pool, "SELECT count(*) n FROM events")[0]["n"] == 12

    def test_redelivery_after_expired_lease_takes_over_without_reaper(self, pool, settings, build):
        job = new_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "worker-A", 30)
        expire_lease(pool, job["id"])
        handler, _ = build("worker-B")  # broker redelivers the unacked message to B
        assert handler.handle(msg(job), StopSignals()) == Outcome.COMPLETED

    def test_graceful_shutdown_hands_the_job_back_without_losing_an_attempt(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_DELAY_S", "0.05")  # ~5 s per segment: long enough to interrupt
        job = new_job(pool, settings)
        handler, _ = build()
        stops, result = StopSignals(), {}
        t = threading.Thread(target=lambda: result.update(o=handler.handle(msg(job), stops)))
        t.start()
        time.sleep(1.0)
        stops.signal_shutdown()
        t.join(timeout=20)
        assert result["o"] == Outcome.RELEASED
        row = q(pool, "SELECT status, attempts, worker_id FROM jobs")[0]
        assert row == {"status": "queued", "attempts": 0, "worker_id": None}
        assert q(pool, "SELECT count(*) n FROM gpu_slots WHERE holder IS NOT NULL")[0]["n"] == 0

    def test_lost_lease_aborts_without_touching_the_new_owners_state(
        self, pool, settings, build, monkeypatch
    ):
        monkeypatch.setenv("FAKE_DELAY_S", "0.05")
        job = new_job(pool, settings)
        handler, _ = build()
        stops, result = StopSignals(), {}
        t = threading.Thread(target=lambda: result.update(o=handler.handle(msg(job), stops)))
        t.start()
        time.sleep(1.0)
        expire_lease(pool, job["id"])
        with pool.connection() as conn:  # a new worker takes over
            jobs_db.claim_job(conn, job["id"], "usurper", 300)
        # The old worker's heartbeat notices within ~1 s (heartbeat_seconds=1) and aborts.
        t.join(timeout=20)
        assert result["o"] == Outcome.LEASE_LOST
        row = q(pool, "SELECT status, worker_id FROM jobs")[0]
        assert row == {"status": "running", "worker_id": "usurper"}


class TestGpuAdmission:
    def run_two_jobs(self, pool, settings, build, slots, monkeypatch):
        monkeypatch.setenv("FAKE_DELAY_S", "0.02")
        scheduler = GpuScheduler(pool, max_slots=slots, min_available_mem_mb=0, poll_seconds=0.02)
        h1, spy = build("w1", slots=slots, scheduler=scheduler)
        h2, _ = build("w2", runner=spy, slots=slots, scheduler=scheduler)
        jobs = [
            new_job(pool, settings, sha="a" * 64, duration_ms=10_000),
            new_job(pool, settings, sha="b" * 64, duration_ms=10_000),
        ]
        threads = [
            threading.Thread(target=h.handle, args=(msg(j), StopSignals()))
            for h, j in zip((h1, h2), jobs, strict=True)
        ]
        [t.start() for t in threads]
        [t.join(timeout=60) for t in threads]
        assert q(pool, "SELECT count(*) n FROM jobs WHERE status='completed'")[0]["n"] == 2
        return spy

    def test_single_slot_serializes_gpu_work_across_workers(
        self, pool, settings, build, monkeypatch
    ):
        assert self.run_two_jobs(pool, settings, build, 1, monkeypatch).peak == 1

    def test_two_slots_allow_two_concurrent_pipelines(self, pool, settings, build, monkeypatch):
        assert self.run_two_jobs(pool, settings, build, 2, monkeypatch).peak == 2


class TestSubprocessRunner:
    def request(self, timeout=30):
        return RunnerRequest("/x.mp4", 0, 1000, "i", "t", "l", "lab", "auto", "h264", 0, timeout)

    def test_hung_runner_is_killed_and_reported_transient(self, monkeypatch):
        monkeypatch.setenv("FAKE_MODE", "hang")
        runner = SubprocessRunner(FAKE_RUNNER.split())
        started = time.monotonic()
        with pytest.raises(TransientJobError) as exc:
            list(runner.run(self.request(timeout=1), threading.Event()))
        assert exc.value.code == "segment_timeout" and time.monotonic() - started < 10

    def test_exit_code_2_is_permanent_and_3_is_transient(self, monkeypatch):
        runner = SubprocessRunner(FAKE_RUNNER.split())
        monkeypatch.setenv("FAKE_MODE", "bad_input")
        with pytest.raises(PermanentJobError):
            list(runner.run(self.request(), threading.Event()))
        monkeypatch.setenv("FAKE_MODE", "crash")
        with pytest.raises(TransientJobError):
            list(runner.run(self.request(), threading.Event()))


def test_malformed_queue_message_is_rejected():
    with pytest.raises(ValueError):
        parse_job_message(b'{"job_id": "not-a-uuid"}')
    with pytest.raises(ValueError):
        parse_job_message(b"garbage")
