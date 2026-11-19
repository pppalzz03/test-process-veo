"""Job lifecycle against a REAL PostgreSQL: outbox atomicity, leases, retries, DLQ, recovery."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from services.scheduler.gpu_scheduler import GpuScheduler
from services.scheduler.relay import relay_outbox_once
from shared.database import jobs as jobs_db
from shared.database import outbox
from shared.database.migrate import apply_migrations
from shared.database.results import commit_segment_results, list_segments
from shared.database.videos import insert_video
from shared.domain.detections import Detection, TrackSummary, VideoEvent
from shared.domain.errors import LeaseLostError
from shared.domain.segments import plan_segments
from shared.messaging.publisher import PublishError

pytestmark = pytest.mark.integration


def make_video(conn, sha="a" * 64, duration_ms=25_000):
    video, _ = insert_video(
        conn,
        tenant_id="default",
        storage_key="v.mp4",
        sha256=sha,
        size_bytes=1,
        duration_ms=duration_ms,
        start_offset_ms=0,
        fps=10.0,
        width=320,
        height=240,
        codec="h264",
    )
    return video


def make_job(pool, settings, key=None, max_attempts=3, duration_ms=25_000):
    with pool.connection() as conn:
        video = make_video(conn, duration_ms=duration_ms)
        job, _ = jobs_db.create_job(
            conn,
            video_id=video["id"],
            config=settings.processing_config(),
            segments=plan_segments(duration_ms, 10_000),
            max_attempts=max_attempts,
            idempotency_key=key,
        )
    return job


def expire_lease(pool, job_id):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET lease_expires_at = now() - interval '1 minute' WHERE id=%s", (job_id,)
        )


def outbox_rows(pool, event_type=None):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT * FROM outbox WHERE (%s::text IS NULL OR event_type=%s) ORDER BY id",
            (event_type, event_type),
        ).fetchall()


def test_migrations_are_idempotent(database_url):
    assert apply_migrations(database_url) == []


def test_migrator_refuses_a_non_utf8_database(database_url):
    """Regression: a SQL_ASCII database returned text as bytes and re-applied migrations."""
    import psycopg

    from shared.database.migrate import UnsupportedDatabaseError

    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS vi_ascii")
        conn.execute("CREATE DATABASE vi_ascii ENCODING 'SQL_ASCII' TEMPLATE template0")
    try:
        ascii_url = database_url.rsplit("/", 1)[0] + "/vi_ascii"
        with pytest.raises(UnsupportedDatabaseError, match="UTF8"):
            apply_migrations(ascii_url)
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute("DROP DATABASE IF EXISTS vi_ascii")


class TestCreateJob:
    def test_job_segments_and_outbox_are_created_together(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            assert len(list_segments(conn, job["id"])) == 3
        rows = outbox_rows(pool, "job.queued")
        assert len(rows) == 1 and rows[0]["payload"]["job_id"] == str(job["id"])
        assert job["status"] == "queued"

    def test_idempotency_key_replays_without_second_outbox_row(self, pool, settings):
        first = make_job(pool, settings, key="abc")
        with pool.connection() as conn:
            again, created = jobs_db.create_job(
                conn,
                video_id=first["video_id"],
                config=settings.processing_config(),
                segments=plan_segments(25_000, 10_000),
                max_attempts=3,
                idempotency_key="abc",
            )
        assert not created and again["id"] == first["id"]
        assert len(outbox_rows(pool)) == 1

    def test_failed_transaction_leaves_nothing_behind(self, pool, settings):
        with pool.connection() as conn:
            video = make_video(conn)
        bad_segments = plan_segments(25_000, 10_000) * 2  # duplicate (job, index) -> violation
        with pool.connection() as conn, pytest.raises(Exception):  # noqa: B017
            jobs_db.create_job(
                conn,
                video_id=video["id"],
                config=settings.processing_config(),
                segments=bad_segments,
                max_attempts=3,
                idempotency_key=None,
            )
        with pool.connection() as conn:
            assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 0
        assert outbox_rows(pool) == []  # no orphan message for a job that does not exist


class TestClaiming:
    def test_second_claim_is_rejected_while_lease_is_live(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            assert jobs_db.claim_job(conn, job["id"], "w1", 30)["attempts"] == 1
            assert jobs_db.claim_job(conn, job["id"], "w2", 30) is None

    def test_takeover_after_lease_expiry_consumes_an_attempt(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
        expire_lease(pool, job["id"])
        with pool.connection() as conn:
            taken = jobs_db.claim_job(conn, job["id"], "w2", 30)
        assert taken["worker_id"] == "w2" and taken["attempts"] == 2

    def test_concurrent_claims_have_exactly_one_winner(self, pool, settings):
        job = make_job(pool, settings)
        winners: list[str] = []
        barrier = threading.Barrier(8)

        def attempt(i: int) -> None:
            barrier.wait()
            with pool.connection() as conn:
                if jobs_db.claim_job(conn, job["id"], f"w{i}", 30):
                    winners.append(f"w{i}")

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(winners) == 1

    def test_only_owner_can_renew_or_complete(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            assert jobs_db.renew_lease(conn, job["id"], "w1", 30)
            assert not jobs_db.renew_lease(conn, job["id"], "intruder", 30)
            assert not jobs_db.complete_job(conn, job["id"], "intruder")
            assert jobs_db.complete_job(conn, job["id"], "w1")
            assert jobs_db.claim_job(conn, job["id"], "w3", 30) is None  # completed is final


class TestRetryAndDeadLetter:
    def test_retry_message_arriving_early_is_not_claimable(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            jobs_db.schedule_retry(conn, job["id"], "w1", code="x", message="m", delay_seconds=3600)
            assert jobs_db.claim_job(conn, job["id"], "w2", 30) is None  # backoff not elapsed
            conn.execute("UPDATE jobs SET next_attempt_at = now() - interval '1 second'")
            taken = jobs_db.claim_job(conn, job["id"], "w2", 30)
        assert taken is not None and taken["attempts"] == 2

    def test_delayed_outbox_row_uses_the_same_due_time_as_the_job(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            jobs_db.schedule_retry(conn, job["id"], "w1", code="x", message="m", delay_seconds=60)
            due = conn.execute("SELECT next_attempt_at FROM jobs").fetchone()["next_attempt_at"]
        delayed = max(outbox_rows(pool, "job.queued"), key=lambda r: r["id"])
        assert delayed["available_at"] == due

    def test_retry_is_delayed_through_the_outbox(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            assert jobs_db.schedule_retry(
                conn, job["id"], "w1", code="x", message="boom", delay_seconds=3600
            )
            got = jobs_db.get_job(conn, job["id"])
            assert got["status"] == "retry_wait" and got["next_attempt_at"] > datetime.now(UTC)
        # The first (initial) outbox row is due; the delayed retry row is not.
        with pool.connection() as conn:
            due = outbox.fetch_unpublished(conn, 10)
        assert len(due) == 1
        delayed = [r for r in outbox_rows(pool) if r["available_at"] > datetime.now(UTC)]
        assert len(delayed) == 1

    def test_permanent_failure_marks_failed_and_dead_letters(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            assert jobs_db.fail_job(conn, job["id"], "w1", code="bad_input", message="corrupt")
            got = jobs_db.get_job(conn, job["id"])
        assert got["status"] == "failed" and got["error_code"] == "bad_input"
        (dead,) = outbox_rows(pool, "job.dead_lettered")
        assert dead["exchange"] == "video.dlx" and dead["payload"]["reason_code"] == "bad_input"

    def test_release_refunds_the_attempt(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
            assert jobs_db.release_job(conn, job["id"], "w1")
            got = jobs_db.get_job(conn, job["id"])
        assert got["status"] == "queued" and got["attempts"] == 0


class TestReaper:
    def test_expired_job_is_requeued_with_new_message(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "dead-worker", 30)
        expire_lease(pool, job["id"])
        with pool.connection() as conn:
            assert jobs_db.reap_expired_jobs(conn) == {"requeued": 1, "failed": 0}
            got = jobs_db.get_job(conn, job["id"])
        assert got["status"] == "queued" and got["worker_id"] is None
        assert len(outbox_rows(pool, "job.queued")) == 2

    def test_live_leases_are_left_alone(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 300)
            assert jobs_db.reap_expired_jobs(conn) == {"requeued": 0, "failed": 0}

    def test_exhausted_attempts_go_to_dlq_instead_of_looping_forever(self, pool, settings):
        job = make_job(pool, settings, max_attempts=1)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)  # attempts = 1 = max
        expire_lease(pool, job["id"])
        with pool.connection() as conn:
            assert jobs_db.reap_expired_jobs(conn) == {"requeued": 0, "failed": 1}
        assert len(outbox_rows(pool, "job.dead_lettered")) == 1


def _tracks_dets_events():
    tracks = [TrackSummary(1, 0, "person", 0, 900, 10, 0.9)]
    dets = [Detection(i, i * 100, 1, 0, "person", 0.9, 1, 2, 3, 4) for i in range(10)]
    events = [
        VideoEvent("seg0:presence:1", "object_presence", 0, 900, "person", "d", {"k": 1}, (1,))
    ]
    return tracks, dets, events


class TestSegmentCommit:
    def _commit(self, pool, job, worker="w1"):
        with pool.connection() as conn:
            seg = list_segments(conn, job["id"])[0]
            tracks, dets, events = _tracks_dets_events()
            return commit_segment_results(
                conn,
                job_id=job["id"],
                worker_id=worker,
                segment_id=seg["id"],
                tracks=tracks,
                detections=dets,
                events=events,
                frames_processed=10,
                processing_ms=5,
            )

    def _counts(self, pool):
        with pool.connection() as conn:
            return {
                t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]  # noqa: S608
                for t in ("tracks", "detections", "events", "event_objects")
            }

    def test_commit_writes_everything_and_links_event_to_track(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
        assert self._commit(pool, job) is True
        assert self._counts(pool) == {
            "tracks": 1,
            "detections": 10,
            "events": 1,
            "event_objects": 1,
        }

    def test_duplicate_commit_does_not_duplicate_events(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "w1", 30)
        assert self._commit(pool, job) is True
        assert self._commit(pool, job) is False  # idempotent: already done
        assert self._counts(pool)["events"] == 1 and self._counts(pool)["detections"] == 10

    def test_zombie_worker_cannot_write_after_takeover(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "zombie", 30)
        expire_lease(pool, job["id"])
        with pool.connection() as conn:
            jobs_db.claim_job(conn, job["id"], "new-owner", 30)
        with pytest.raises(LeaseLostError):
            self._commit(pool, job, worker="zombie")
        assert self._counts(pool)["events"] == 0

    def test_event_key_is_unique_per_job(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO events (job_id, segment_id, event_key, event_type, start_ms, end_ms)"
                " SELECT %s, id, 'dup', 't', 0, 1 FROM segments WHERE job_id=%s LIMIT 1",
                (job["id"], job["id"]),
            )
        with pool.connection() as conn, pytest.raises(Exception, match="unique|duplicate"):
            conn.execute(
                "INSERT INTO events (job_id, segment_id, event_key, event_type, start_ms, end_ms)"
                " SELECT %s, id, 'dup', 't', 0, 1 FROM segments WHERE job_id=%s LIMIT 1",
                (job["id"], job["id"]),
            )


class FakePublisher:
    def __init__(self, fail_after: int | None = None) -> None:
        self.sent: list[tuple[str, str, bytes, str]] = []
        self.fail_after = fail_after

    def publish(self, exchange, routing_key, body, message_id):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise PublishError("broker down")
        self.sent.append((exchange, routing_key, body, message_id))

    def close(self):
        pass


class TestOutboxRelay:
    def test_publishes_and_marks_rows(self, pool, settings):
        make_job(pool, settings)
        pub = FakePublisher()
        assert relay_outbox_once(pool, pub, 10) == 1
        assert pub.sent[0][0:2] == ("video", "video.infer")
        assert relay_outbox_once(pool, pub, 10) == 0  # nothing left

    def test_broker_outage_keeps_rows_unpublished_then_recovers(self, pool, settings):
        make_job(pool, settings)
        assert relay_outbox_once(pool, FakePublisher(fail_after=0), 10) == 0
        (row,) = outbox_rows(pool)
        assert row["published_at"] is None and row["publish_attempts"] == 1
        assert "broker down" in row["last_error"]
        good = FakePublisher()
        assert relay_outbox_once(pool, good, 10) == 1  # nothing was lost during the outage

    def test_partial_batch_commits_what_was_confirmed(self, pool, settings):
        for i in range(3):
            with pool.connection() as conn:
                v, _ = insert_video(
                    conn,
                    tenant_id="default",
                    storage_key=f"{i}.mp4",
                    sha256=f"{i}" * 64,
                    size_bytes=1,
                    duration_ms=5000,
                    start_offset_ms=0,
                    fps=10,
                    width=1,
                    height=1,
                    codec="h264",
                )
                jobs_db.create_job(
                    conn,
                    video_id=v["id"],
                    config=settings.processing_config(),
                    segments=plan_segments(5000, 10_000),
                    max_attempts=3,
                    idempotency_key=None,
                )
        assert relay_outbox_once(pool, FakePublisher(fail_after=2), 10) == 2
        with pool.connection() as conn:
            assert len(outbox.fetch_unpublished(conn, 10)) == 1

    def test_future_rows_are_not_published_early(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            outbox.enqueue_job_message(
                conn, job["id"], available_at=datetime.now(UTC) + timedelta(hours=1)
            )
        pub = FakePublisher()
        assert relay_outbox_once(pool, pub, 10) == 1  # only the immediate row

    def test_concurrent_relays_never_publish_a_row_twice(self, pool, settings):
        job = make_job(pool, settings)
        with pool.connection() as conn:
            for _ in range(20):
                outbox.enqueue_job_message(conn, job["id"])
        pubs = [FakePublisher() for _ in range(4)]
        barrier = threading.Barrier(4)

        def run(p):
            barrier.wait()
            while relay_outbox_once(pool, p, 3):
                pass

        threads = [threading.Thread(target=run, args=(p,)) for p in pubs]
        [t.start() for t in threads]
        [t.join() for t in threads]
        ids = [m[3] for p in pubs for m in p.sent]
        assert len(ids) == len(set(ids)) == 21


class TestGpuSlots:
    def sched(self, pool, slots=2, mem=lambda: 10_000, min_mem=0):
        s = GpuScheduler(
            pool, max_slots=slots, min_available_mem_mb=min_mem, poll_seconds=0.01, mem_reader=mem
        )
        s.sync_slots()
        return s

    def test_never_exceeds_configured_slots(self, pool):
        s = self.sched(pool, slots=2)
        a, b = s.try_acquire("a", 30), s.try_acquire("b", 30)
        assert a and b and a.slot_id != b.slot_id
        assert s.try_acquire("c", 30) is None
        s.release(a)
        assert s.try_acquire("c", 30) is not None

    def test_concurrent_acquirers_respect_the_cap(self, pool):
        s = self.sched(pool, slots=2)
        got: list = []
        barrier = threading.Barrier(10)

        def grab(i):
            barrier.wait()
            if lease := s.try_acquire(f"h{i}", 30):
                got.append(lease)

        threads = [threading.Thread(target=grab, args=(i,)) for i in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(got) == 2

    def test_crashed_holder_is_reclaimed_after_lease_expiry(self, pool):
        s = self.sched(pool, slots=1)
        assert s.try_acquire("crashed", 30)
        assert s.try_acquire("next", 30) is None
        with pool.connection() as conn:
            conn.execute("UPDATE gpu_slots SET lease_expires_at = now() - interval '1 second'")
        assert s.try_acquire("next", 30) is not None

    def test_low_memory_blocks_admission(self, pool):
        s = self.sched(pool, slots=1, mem=lambda: 1_000, min_mem=4_000)
        assert s.try_acquire("a", 30) is None
        assert self.sched(pool, slots=1, mem=lambda: 5_000, min_mem=4_000).try_acquire("a", 30)

    def test_renew_only_for_holder(self, pool):
        s = self.sched(pool, slots=1)
        lease = s.try_acquire("a", 30)
        assert s.renew(lease, 30)
        s.release(lease)
        assert not s.renew(lease, 30)

    def test_shrinking_slots_never_evicts_a_running_holder(self, pool):
        s = self.sched(pool, slots=2)
        held = [s.try_acquire("a", 30), s.try_acquire("b", 30)]
        self.sched(pool, slots=1)  # reconfigure smaller
        with pool.connection() as conn:
            n = conn.execute(
                "SELECT count(*) AS n FROM gpu_slots WHERE holder IS NOT NULL"
            ).fetchone()["n"]
        assert n == 2 and all(held)
