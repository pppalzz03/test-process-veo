"""Pure event extraction: detections in, tracks and events out. No I/O, fully unit-testable.

Event types (Phase 3):
  object_presence  one per track that lived long enough (start/end = first/last sighting)
  count_peak       per class per segment: the maximum number of simultaneous objects

Event keys are deterministic (segment index + track/class), so re-processing a segment after a
retry reproduces the same keys and the (job_id, event_key) unique constraint prevents duplicates.
Track IDs are segment-local: an object that crosses a segment boundary yields one event per
segment. Cross-segment stitching is a documented non-goal for Phase 3.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from shared.domain.detections import Detection, TrackSummary, VideoEvent


@dataclass(frozen=True, slots=True)
class ExtractionParams:
    min_track_ms: int = 500
    min_track_frames: int = 3


def build_tracks(detections: Sequence[Detection]) -> list[TrackSummary]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        if det.track_id is not None:
            grouped[det.track_id].append(det)

    tracks: list[TrackSummary] = []
    for track_id, items in sorted(grouped.items()):
        class_counts = Counter((d.class_id, d.class_name) for d in items)
        (class_id, class_name), _ = max(
            class_counts.items(),
            key=lambda kv: (kv[1], -kv[0][0]),  # majority, ties -> low id
        )
        tracks.append(
            TrackSummary(
                local_track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                first_seen_ms=min(d.timestamp_ms for d in items),
                last_seen_ms=max(d.timestamp_ms for d in items),
                frame_count=len({d.frame_index for d in items}),
                avg_confidence=sum(d.confidence for d in items) / len(items),
            )
        )
    return tracks


def extract_events(
    segment_index: int,
    detections: Sequence[Detection],
    tracks: Sequence[TrackSummary],
    params: ExtractionParams,
) -> list[VideoEvent]:
    events = [
        _presence_event(segment_index, t)
        for t in tracks
        if (t.last_seen_ms - t.first_seen_ms) >= params.min_track_ms
        and t.frame_count >= params.min_track_frames
    ]
    events.extend(_count_peak_events(segment_index, detections))
    return sorted(events, key=lambda e: (e.start_ms, e.event_key))


def _presence_event(segment_index: int, track: TrackSummary) -> VideoEvent:
    seconds = (track.last_seen_ms - track.first_seen_ms) / 1000
    return VideoEvent(
        event_key=f"seg{segment_index}:presence:{track.local_track_id}",
        event_type="object_presence",
        start_ms=track.first_seen_ms,
        end_ms=track.last_seen_ms,
        class_name=track.class_name,
        description=f"{track.class_name} present for {seconds:.1f}s (track {track.local_track_id})",
        attributes={
            "local_track_id": track.local_track_id,
            "frame_count": track.frame_count,
            "avg_confidence": round(track.avg_confidence, 4),
        },
        track_local_ids=(track.local_track_id,),
    )


def _count_peak_events(segment_index: int, detections: Sequence[Detection]) -> list[VideoEvent]:
    per_frame: dict[tuple[str, int], tuple[int, set[int]]] = {}
    for det in detections:
        key = (det.class_name, det.frame_index)
        ts, objs = per_frame.get(key, (det.timestamp_ms, set()))
        # Untracked detections still count; use a negative synthetic id so they stay distinct.
        objs.add(det.track_id if det.track_id is not None else -(len(objs) + 1))
        per_frame[key] = (ts, objs)

    best: dict[str, tuple[int, int, int]] = {}  # class -> (count, first ts, frame index)
    for (class_name, frame_index), (ts, objs) in sorted(
        per_frame.items(), key=lambda kv: (kv[1][0], kv[0][1])
    ):
        count = len(objs)
        if class_name not in best or count > best[class_name][0]:
            best[class_name] = (count, ts, frame_index)

    return [
        VideoEvent(
            event_key=f"seg{segment_index}:count_peak:{class_name}",
            event_type="count_peak",
            start_ms=ts,
            end_ms=ts,
            class_name=class_name,
            description=f"peak of {count} {class_name} in one frame",
            attributes={"max_count": count, "frame_index": frame_index},
        )
        for class_name, (count, ts, frame_index) in sorted(best.items())
    ]
