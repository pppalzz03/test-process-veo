"""API use cases: store/probe videos and create jobs. No HTTP types in here."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

from services.api.probe import ProbeError, VideoProbe, probe_video
from shared.config.settings import Settings
from shared.database import jobs as jobs_db
from shared.database.pool import DictConnection
from shared.database.videos import get_video, insert_video
from shared.domain.segments import plan_segments

_CHUNK = 4 * 1024 * 1024


class VideoRejected(Exception):
    """Client error: file missing, unsafe path, too large or not a video."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media_path(media_root: str, relative: str) -> Path:
    """Resolve a client path inside MEDIA_ROOT; rejects traversal and symlink escapes."""
    root = Path(media_root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise VideoRejected("path escapes MEDIA_ROOT")
    if not candidate.is_file():
        raise VideoRejected("file not found under MEDIA_ROOT")
    return candidate


def _probe(path: Path) -> VideoProbe:
    try:
        return probe_video(str(path))
    except ProbeError as exc:
        raise VideoRejected(str(exc)) from exc


def _insert(
    conn: DictConnection, settings: Settings, path: Path, probe: VideoProbe
) -> tuple[dict[str, Any], bool]:
    root = Path(settings.media_root).resolve()
    return insert_video(
        conn,
        tenant_id="default",
        storage_key=str(path.resolve().relative_to(root)),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        duration_ms=probe.duration_ms,
        start_offset_ms=probe.start_offset_ms,
        fps=probe.fps,
        width=probe.width,
        height=probe.height,
        codec=probe.codec,
    )


def register_existing_video(
    conn: DictConnection, settings: Settings, relative_path: str
) -> tuple[dict[str, Any], bool]:
    path = resolve_media_path(settings.media_root, relative_path)
    return _insert(conn, settings, path, _probe(path))


def store_uploaded_video(
    conn: DictConnection, settings: Settings, stream: BinaryIO
) -> tuple[dict[str, Any], bool]:
    """Stream the upload to disk (never fully in memory), validate it, then register it."""
    uploads = Path(settings.media_root) / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    tmp = uploads / f".incoming-{uuid4().hex}"
    written = 0
    try:
        with tmp.open("wb") as out:
            for chunk in _iter_chunks(stream):
                written += len(chunk)
                if written > settings.upload_max_bytes:
                    raise VideoRejected("upload exceeds UPLOAD_MAX_BYTES")
                out.write(chunk)
        probe = _probe(tmp)  # reject non-videos BEFORE they are kept on disk
        final = uploads / f"{_sha256_file(tmp)}.mp4"
        if final.exists():
            tmp.unlink()
        else:
            shutil.move(str(tmp), final)
        return _insert(conn, settings, final, probe)
    finally:
        tmp.unlink(missing_ok=True)


def _iter_chunks(stream: BinaryIO) -> Iterator[bytes]:
    while chunk := stream.read(_CHUNK):
        yield chunk


def create_processing_job(
    conn: DictConnection, settings: Settings, video_id: UUID, idempotency_key: str | None
) -> tuple[dict[str, Any], bool]:
    video = get_video(conn, video_id)
    if video is None:
        raise VideoRejected("video not found")
    config = settings.processing_config()
    segments = plan_segments(video["duration_ms"], config.segment_seconds * 1000)
    job, created = jobs_db.create_job(
        conn,
        video_id=video_id,
        config=config,
        segments=segments,
        max_attempts=settings.max_attempts,
        idempotency_key=idempotency_key,
    )
    return jobs_db.get_job(conn, job["id"]) or job, created
