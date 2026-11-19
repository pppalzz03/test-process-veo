"""Core value objects shared by the worker and the event processor."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Detection:
    """One tracked object in one frame. Timestamps are ms from the start of the video."""

    frame_index: int
    timestamp_ms: int
    track_id: int | None  # segment-local; None when the tracker gave no ID
    class_id: int
    class_name: str
    confidence: float
    left: float
    top: float
    width: float
    height: float


@dataclass(slots=True)
class TrackSummary:
    local_track_id: int
    class_id: int
    class_name: str
    first_seen_ms: int
    last_seen_ms: int
    frame_count: int
    avg_confidence: float


@dataclass(frozen=True, slots=True)
class VideoEvent:
    event_key: str  # deterministic -> idempotent across retries
    event_type: str
    start_ms: int
    end_ms: int
    class_name: str | None
    description: str
    attributes: dict[str, object] = field(default_factory=dict)
    track_local_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("invalid event timestamps")
