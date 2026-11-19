from __future__ import annotations

from typing import Any
from uuid import UUID

from shared.database.pool import DictConnection


def insert_video(
    conn: DictConnection,
    *,
    tenant_id: str,
    storage_key: str,
    sha256: str,
    size_bytes: int,
    duration_ms: int,
    start_offset_ms: int,
    fps: float | None,
    width: int | None,
    height: int | None,
    codec: str | None,
) -> tuple[dict[str, Any], bool]:
    """Insert, or return the existing row for the same (tenant, sha256). Second item: created."""
    row = conn.execute(
        "INSERT INTO videos (tenant_id, storage_key, sha256, size_bytes, duration_ms,"
        " start_offset_ms, fps, width, height, codec)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        " ON CONFLICT (tenant_id, sha256) DO NOTHING RETURNING *",
        (
            tenant_id,
            storage_key,
            sha256,
            size_bytes,
            duration_ms,
            start_offset_ms,
            fps,
            width,
            height,
            codec,
        ),
    ).fetchone()
    if row is not None:
        return row, True
    existing = conn.execute(
        "SELECT * FROM videos WHERE tenant_id=%s AND sha256=%s", (tenant_id, sha256)
    ).fetchone()
    assert existing is not None
    return existing, False


def get_video(conn: DictConnection, video_id: UUID) -> dict[str, Any] | None:
    return conn.execute("SELECT * FROM videos WHERE id=%s", (video_id,)).fetchone()
