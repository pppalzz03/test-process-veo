import random

import pytest

from shared.domain.backoff import compute_backoff_seconds
from shared.domain.detections import VideoEvent
from shared.domain.segments import plan_segments
from shared.domain.state import JobStatus, can_transition
from shared.schemas.processing import ProcessingConfig
from shared.schemas.runner import FrameRecord, SummaryRecord, parse_runner_line


class TestBackoff:
    def test_grows_exponentially_without_jitter(self):
        delays = [
            compute_backoff_seconds(a, base_seconds=10, max_seconds=1000, jitter_ratio=0)
            for a in (1, 2, 3, 4)
        ]
        assert delays == [10, 20, 40, 80]

    def test_is_capped(self):
        assert compute_backoff_seconds(20, base_seconds=10, max_seconds=60, jitter_ratio=0) == 60

    def test_jitter_stays_within_bounds_and_never_exceeds_max(self):
        rng = random.Random(1)
        for _ in range(200):
            d = compute_backoff_seconds(
                3, base_seconds=10, max_seconds=45, jitter_ratio=0.2, rng=rng
            )
            assert 32 <= d <= 45

    def test_rejects_attempt_zero(self):
        with pytest.raises(ValueError):
            compute_backoff_seconds(0, base_seconds=1, max_seconds=2)


class TestSegments:
    def test_one_hour_in_five_minute_windows(self):
        plans = plan_segments(3_600_000, 300_000)
        assert len(plans) == 12
        assert plans[0].start_ms == 0 and plans[-1].end_ms == 3_600_000
        assert all(a.end_ms == b.start_ms for a, b in zip(plans, plans[1:], strict=False))

    def test_short_video_is_single_segment(self):
        plans = plan_segments(4_200, 300_000)
        assert [(p.start_ms, p.end_ms) for p in plans] == [(0, 4_200)]

    def test_tiny_tail_is_merged(self):
        plans = plan_segments(300_000 + 5_000, 300_000)
        assert len(plans) == 1 and plans[0].end_ms == 305_000

    def test_large_tail_is_kept(self):
        plans = plan_segments(300_000 + 60_000, 300_000)
        assert len(plans) == 2

    def test_indices_are_contiguous_after_merge(self):
        plans = plan_segments(3 * 300_000 + 1_000, 300_000)
        assert [p.index for p in plans] == [0, 1, 2]

    @pytest.mark.parametrize("duration,segment", [(0, 10), (10, 0), (-1, 10)])
    def test_invalid_input(self, duration, segment):
        with pytest.raises(ValueError):
            plan_segments(duration, segment)


class TestStateMachine:
    def test_happy_path(self):
        assert can_transition(JobStatus.QUEUED, JobStatus.RUNNING)
        assert can_transition(JobStatus.RUNNING, JobStatus.COMPLETED)

    def test_retry_loop(self):
        assert can_transition(JobStatus.RUNNING, JobStatus.RETRY_WAIT)
        assert can_transition(JobStatus.RETRY_WAIT, JobStatus.RUNNING)

    @pytest.mark.parametrize("terminal", [JobStatus.COMPLETED, JobStatus.FAILED])
    def test_terminal_states_are_final(self, terminal):
        assert not any(can_transition(terminal, s) for s in JobStatus)

    def test_cannot_skip_running(self):
        assert not can_transition(JobStatus.QUEUED, JobStatus.COMPLETED)


class TestRunnerProtocol:
    def test_parses_frame_and_summary(self):
        frame = parse_runner_line(
            '{"type":"frame","frame_num":3,"pts_ms":100,"objects":'
            '[{"track_id":null,"class_id":0,"label":"person","confidence":0.5,"bbox":[1,2,3,4]}]}'
        )
        assert isinstance(frame, FrameRecord) and frame.objects[0].track_id is None
        assert isinstance(
            parse_runner_line('{"type":"summary","frames":3,"eos":true}'), SummaryRecord
        )

    @pytest.mark.parametrize("line", ["not json", '{"type":"nope"}', '{"type":"frame"}'])
    def test_rejects_malformed(self, line):
        with pytest.raises(ValueError):
            parse_runner_line(line)


def test_event_rejects_inverted_timestamps():
    with pytest.raises(ValueError):
        VideoEvent("k", "t", 10, 5, None, "d")


def test_config_hash_is_stable_and_sensitive():
    base = dict(
        model_version="m",
        infer_config="a",
        tracker_config="b",
        segment_seconds=300,
        min_confidence=0.25,
        min_track_ms=500,
        min_track_frames=3,
        infer_interval=0,
        persist_detections=True,
        detection_stride=1,
    )
    assert ProcessingConfig(**base).config_hash() == ProcessingConfig(**base).config_hash()
    assert (
        ProcessingConfig(**{**base, "min_confidence": 0.5}).config_hash()
        != ProcessingConfig(**base).config_hash()
    )
