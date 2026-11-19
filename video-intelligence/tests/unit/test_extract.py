from services.event_processor.extract import ExtractionParams, build_tracks, extract_events
from shared.domain.detections import Detection


def det(frame, ts, track, cls="person", cid=0, conf=0.9):
    return Detection(frame, ts, track, cid, cls, conf, 0, 0, 10, 10)


PARAMS = ExtractionParams(min_track_ms=500, min_track_frames=3)


def test_track_summary_uses_majority_class_and_distinct_frames():
    dets = [det(0, 0, 1, "person"), det(1, 100, 1, "person"), det(2, 200, 1, "dog", 16, 0.5)]
    (track,) = build_tracks(dets)
    assert track.class_name == "person"
    assert (track.first_seen_ms, track.last_seen_ms, track.frame_count) == (0, 200, 3)
    assert abs(track.avg_confidence - (0.9 + 0.9 + 0.5) / 3) < 1e-9


def test_untracked_detections_do_not_create_tracks():
    assert build_tracks([det(0, 0, None)]) == []


def test_short_tracks_do_not_become_presence_events():
    dets = [det(i, i * 100, 1) for i in range(3)]  # 200 ms long
    tracks = build_tracks(dets)
    types = [e.event_type for e in extract_events(0, dets, tracks, PARAMS)]
    assert "object_presence" not in types


def test_presence_event_fields_and_key():
    dets = [det(i, i * 100, 7) for i in range(10)]
    events = extract_events(2, dets, build_tracks(dets), PARAMS)
    presence = next(e for e in events if e.event_type == "object_presence")
    assert presence.event_key == "seg2:presence:7"
    assert (presence.start_ms, presence.end_ms) == (0, 900)
    assert presence.track_local_ids == (7,)


def test_count_peak_reports_max_simultaneous_and_first_time():
    dets = [det(0, 0, 1), det(1, 100, 1), det(1, 100, 2), det(2, 200, 1), det(2, 200, 2)]
    events = extract_events(0, dets, build_tracks(dets), PARAMS)
    peak = next(e for e in events if e.event_type == "count_peak")
    assert peak.attributes["max_count"] == 2
    assert peak.start_ms == 100  # first frame that reached the maximum


def test_untracked_objects_are_counted_in_peaks():
    dets = [det(0, 0, None), det(0, 0, None), det(0, 0, None)]
    events = extract_events(0, dets, [], PARAMS)
    assert events[0].attributes["max_count"] == 3


def test_extraction_is_deterministic_for_retries():
    dets = [det(i, i * 100, t, cls=c) for i in range(10) for t, c in ((1, "person"), (2, "car"))]
    a = extract_events(1, dets, build_tracks(dets), PARAMS)
    b = extract_events(1, list(reversed(dets)), build_tracks(list(reversed(dets))), PARAMS)
    assert [e.event_key for e in a] == [e.event_key for e in b]
    assert len({e.event_key for e in a}) == len(a)


def test_events_are_chronological():
    dets = [det(i, 1000 + i * 100, 1) for i in range(10)] + [det(i, i * 100, 2) for i in range(10)]
    events = extract_events(0, dets, build_tracks(dets), PARAMS)
    starts = [e.start_ms for e in events]
    assert starts == sorted(starts)
