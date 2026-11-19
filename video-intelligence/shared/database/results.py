"""Segment results: atomic, idempotent commit and read queries for the result API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from psycopg.types.json import Jsonb

from shared.database.pool import DictConnection
from shared.domain.detections import Detection, TrackSummary, VideoEvent
from shared.domain.errors import LeaseLostError


def list_segments(conn: DictConnection, job_id: UUID) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT * FROM segments WHERE job_id=%s ORDER BY segment_index", (job_id,)
    ).fetchall()


def mark_segment_running(conn: DictConnection, segment_id: UUID) -> None:
    with conn.transaction():
        conn.execute(
            "UPDATE segments SET status='running' WHERE id=%s AND status<>'done'", (segment_id,)
        )


def commit_segment_results(
    conn: DictConnection,
    *,
    job_id: UUID,
    worker_id: str,
    segment_id: UUID,
    tracks: Sequence[TrackSummary],
    detections: Sequence[Detection],
    events: Sequence[VideoEvent],
    frames_processed: int,
    processing_ms: int,
) -> bool:
    """Write tracks, detections and events for one segment in ONE transaction.

    Returns False if the segment was already committed (duplicate delivery / retry): nothing is
    written twice. Raises LeaseLostError if this worker no longer owns the job, so a zombie
    worker can never overwrite the new owner's data.
    """
    with conn.transaction():
        owner = conn.execute(
            "SELECT 1 FROM jobs WHERE id=%s AND worker_id=%s AND status='running' FOR UPDATE",
            (job_id, worker_id),
        ).fetchone()
        if owner is None:
            raise LeaseLostError(f"job {job_id} is no longer owned by {worker_id}")

        seg = conn.execute(
            "SELECT status FROM segments WHERE id=%s FOR UPDATE", (segment_id,)
        ).fetchone()
        if seg is None or seg["status"] == "done":
            return False

        # Clean any partial rows (defensive; the transaction normally makes this a no-op).
        conn.execute("DELETE FROM events WHERE segment_id=%s", (segment_id,))
        conn.execute("DELETE FROM tracks WHERE segment_id=%s", (segment_id,))

        track_ids: dict[int, int] = {}
        for t in tracks:
            row = conn.execute(
                "INSERT INTO tracks (segment_id, local_track_id, class_id, class_name,"
                " first_seen_ms, last_seen_ms, frame_count, avg_confidence)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (
                    segment_id,
                    t.local_track_id,
                    t.class_id,
                    t.class_name,
                    t.first_seen_ms,
                    t.last_seen_ms,
                    t.frame_count,
                    t.avg_confidence,
                ),
            ).fetchone()
            assert row is not None
            track_ids[t.local_track_id] = row["id"]

        if detections:
            with (
                conn.cursor() as cur,
                cur.copy(
                    "COPY detections (segment_id, track_id, frame_index, timestamp_ms, class_id,"
                    " class_name, confidence, bbox_left, bbox_top, bbox_width, bbox_height)"
                    " FROM STDIN"
                ) as copy,
            ):
                for d in detections:
                    copy.write_row(
                        (
                            segment_id,
                            track_ids.get(d.track_id) if d.track_id is not None else None,
                            d.frame_index,
                            d.timestamp_ms,
                            d.class_id,
                            d.class_name,
                            d.confidence,
                            d.left,
                            d.top,
                            d.width,
                            d.height,
                        )
                    )

        for ev in events:
            row = conn.execute(
                "INSERT INTO events (job_id, segment_id, event_key, event_type, start_ms, end_ms,"
                " class_name, description, attributes)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (job_id, event_key) DO NOTHING RETURNING id",
                (
                    job_id,
                    segment_id,
                    ev.event_key,
                    ev.event_type,
                    ev.start_ms,
                    ev.end_ms,
                    ev.class_name,
                    ev.description,
                    Jsonb(ev.attributes),
                ),
            ).fetchone()
            if row is None:
                continue
            for local_id in ev.track_local_ids:
                if local_id in track_ids:
                    conn.execute(
                        "INSERT INTO event_objects (event_id, track_id) VALUES (%s,%s)"
                        " ON CONFLICT DO NOTHING",
                        (row["id"], track_ids[local_id]),
                    )

        conn.execute(
            "UPDATE segments SET status='done', frames_processed=%s, detections_count=%s,"
            " processing_ms=%s, completed_at=now() WHERE id=%s",
            (frames_processed, len(detections), processing_ms, segment_id),
        )
    return True


# ---------------------------------------------------------------- read side (API)


def list_tracks(
    conn: DictConnection, job_id: UUID, *, class_name: str | None, limit: int, after_id: int
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT t.*, s.segment_index FROM tracks t JOIN segments s ON s.id = t.segment_id"
        " WHERE s.job_id=%s AND t.id > %s AND (%s::text IS NULL OR t.class_name=%s)"
        " ORDER BY t.id LIMIT %s",
        (job_id, after_id, class_name, class_name, limit),
    ).fetchall()


def list_events(
    conn: DictConnection,
    job_id: UUID,
    *,
    event_type: str | None,
    class_name: str | None,
    from_ms: int | None,
    to_ms: int | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT e.*, s.segment_index FROM events e JOIN segments s ON s.id = e.segment_id"
        " WHERE e.job_id=%s"
        " AND (%s::text IS NULL OR e.event_type=%s)"
        " AND (%s::text IS NULL OR e.class_name=%s)"
        " AND (%s::bigint IS NULL OR e.end_ms >= %s)"
        " AND (%s::bigint IS NULL OR e.start_ms <= %s)"
        " ORDER BY e.start_ms, e.event_key LIMIT %s OFFSET %s",
        (
            job_id,
            event_type,
            event_type,
            class_name,
            class_name,
            from_ms,
            from_ms,
            to_ms,
            to_ms,
            limit,
            offset,
        ),
    ).fetchall()


def list_detections(
    conn: DictConnection,
    job_id: UUID,
    *,
    track_id: int | None,
    from_ms: int | None,
    to_ms: int | None,
    limit: int,
    after_id: int,
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT d.* FROM detections d JOIN segments s ON s.id = d.segment_id"
        " WHERE s.job_id=%s AND d.id > %s"
        " AND (%s::bigint IS NULL OR d.track_id=%s)"
        " AND (%s::bigint IS NULL OR d.timestamp_ms >= %s)"
        " AND (%s::bigint IS NULL OR d.timestamp_ms <= %s)"
        " ORDER BY d.id LIMIT %s",
        (job_id, after_id, track_id, track_id, from_ms, from_ms, to_ms, to_ms, limit),
    ).fetchall()
