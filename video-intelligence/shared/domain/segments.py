"""Logical segmentation. No video files are cut; segments are time windows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    index: int
    start_ms: int
    end_ms: int  # exclusive, clamped to the video duration


def plan_segments(duration_ms: int, segment_ms: int) -> list[SegmentPlan]:
    """Split [0, duration_ms) into consecutive windows of at most segment_ms.

    A tiny trailing window (< 10% of segment_ms) is merged into the previous one so
    we never spin up a whole pipeline for a fraction of a second.
    """
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if segment_ms <= 0:
        raise ValueError("segment_ms must be positive")

    plans: list[SegmentPlan] = []
    start = 0
    while start < duration_ms:
        end = min(start + segment_ms, duration_ms)
        plans.append(SegmentPlan(len(plans), start, end))
        start = end

    if len(plans) > 1 and (plans[-1].end_ms - plans[-1].start_ms) < segment_ms * 0.1:
        tail = plans.pop()
        prev = plans.pop()
        plans.append(SegmentPlan(prev.index, prev.start_ms, tail.end_ms))
    return plans
