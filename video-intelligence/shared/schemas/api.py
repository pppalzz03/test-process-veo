"""HTTP request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _Row(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")


class VideoOut(_Row):
    id: UUID
    sha256: str
    storage_key: str
    size_bytes: int
    duration_ms: int
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None
    created_at: datetime


class RegisterVideoIn(BaseModel):
    path: str = Field(description="Path relative to MEDIA_ROOT")


class JobCreateIn(BaseModel):
    video_id: UUID
    idempotency_key: str | None = Field(default=None, max_length=200)


class JobOut(_Row):
    id: UUID
    video_id: UUID
    status: str
    model_version: str
    attempts: int
    max_attempts: int
    next_attempt_at: datetime | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    segments_total: int = 0
    segments_done: int = 0


class SegmentOut(_Row):
    id: UUID
    segment_index: int
    start_ms: int
    end_ms: int
    status: str
    frames_processed: int | None
    detections_count: int | None
    processing_ms: int | None


class TrackOut(_Row):
    id: int
    segment_index: int
    local_track_id: int
    class_name: str
    first_seen_ms: int
    last_seen_ms: int
    frame_count: int
    avg_confidence: float


class EventOut(_Row):
    id: UUID
    segment_index: int
    event_key: str
    event_type: str
    start_ms: int
    end_ms: int
    class_name: str | None
    description: str | None
    attributes: dict[str, Any]
    revision: int


class DetectionOut(_Row):
    id: int
    track_id: int | None
    frame_index: int
    timestamp_ms: int
    class_name: str
    confidence: float
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float


class Page[T](BaseModel):
    items: list[T]
    next_cursor: int | None = None
