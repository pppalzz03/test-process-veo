"""API end-to-end with real PostgreSQL, real ffprobe and a real MP4. Worker uses the test double."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from services.api.main import create_app
from services.scheduler.gpu_scheduler import GpuScheduler
from services.worker.handler import JobHandler, Outcome
from services.worker.processor import JobProcessor, StopSignals
from services.worker.runner import SubprocessRunner
from shared.schemas.messages import JobMessage
from tests.conftest import make_test_video

pytestmark = pytest.mark.integration


@pytest.fixture
def client(settings, pool):
    with TestClient(create_app(settings, pool)) as c:
        yield c


@pytest.fixture
def mp4_bytes(tmp_path):
    return make_test_video(tmp_path / "t.mp4", seconds=25, fps=10).read_bytes()


def upload(client, data, name="clip.mp4"):
    return client.post("/v1/videos", files={"file": (name, io.BytesIO(data), "video/mp4")})


def run_worker(pool, settings, job_id):
    gpu = GpuScheduler(pool, max_slots=1, min_available_mem_mb=0, poll_seconds=0.02)
    gpu.sync_slots()
    proc = JobProcessor(
        pool=pool,
        settings=settings,
        runner=SubprocessRunner(settings.runner_argv()),
        gpu=gpu,
        worker_id="api-test-worker",
    )
    return JobHandler(
        pool=pool, settings=settings, processor=proc, worker_id="api-test-worker"
    ).handle(JobMessage(job_id=job_id), StopSignals())


def test_health(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ok"}


class TestVideos:
    def test_upload_probes_the_real_file(self, client, mp4_bytes):
        r = upload(client, mp4_bytes)
        assert r.status_code == 201
        body = r.json()
        assert body["codec"] == "h264" and (body["width"], body["height"]) == (320, 240)
        assert 24_500 <= body["duration_ms"] <= 25_500 and body["fps"] == pytest.approx(10, abs=0.1)

    def test_same_content_is_deduplicated(self, client, mp4_bytes):
        first, second = upload(client, mp4_bytes), upload(client, mp4_bytes, name="other.mp4")
        assert (first.status_code, second.status_code) == (201, 200)
        assert first.json()["id"] == second.json()["id"]

    def test_non_video_is_rejected_and_not_kept(self, client, media_root):
        r = upload(client, b"this is not a video")
        assert r.status_code == 422
        assert list((media_root / "uploads").glob("*")) == []

    def test_upload_size_limit(self, settings, pool, mp4_bytes):
        small = settings.model_copy(update={"upload_max_bytes": 100})
        with TestClient(create_app(small, pool)) as c:
            assert upload(c, mp4_bytes).status_code == 422

    def test_register_existing_file_and_block_traversal(self, client, media_root, mp4_bytes):
        (media_root / "in.mp4").write_bytes(mp4_bytes)
        assert client.post("/v1/videos/register", json={"path": "in.mp4"}).status_code == 201
        assert (
            client.post("/v1/videos/register", json={"path": "../../etc/passwd"}).status_code == 422
        )

    def test_unknown_video_404(self, client):
        assert client.get("/v1/videos/00000000-0000-0000-0000-000000000000").status_code == 404


class TestJobsAndResults:
    def test_upload_to_result_api_end_to_end(self, client, settings, pool, mp4_bytes):
        video = upload(client, mp4_bytes).json()
        created = client.post("/v1/jobs", json={"video_id": video["id"], "idempotency_key": "k1"})
        assert created.status_code == 202
        job = created.json()
        assert job["status"] == "queued" and job["segments_total"] == 3

        replay = client.post("/v1/jobs", json={"video_id": video["id"], "idempotency_key": "k1"})
        assert replay.status_code == 200 and replay.json()["id"] == job["id"]

        assert run_worker(pool, settings, job["id"]) == Outcome.COMPLETED

        done = client.get(f"/v1/jobs/{job['id']}").json()
        assert done["status"] == "completed" and done["segments_done"] == 3
        assert done["started_at"] is not None and done["completed_at"] is not None

        segs = client.get(f"/v1/jobs/{job['id']}/segments").json()
        assert [s["status"] for s in segs] == ["done"] * 3

        events = client.get(f"/v1/jobs/{job['id']}/events").json()
        assert len(events) == 12
        starts = [e["start_ms"] for e in events]
        assert starts == sorted(starts)

        cars = client.get(
            f"/v1/jobs/{job['id']}/events",
            params={"event_type": "object_presence", "class_name": "car"},
        ).json()
        assert len(cars) == 3 and all(e["class_name"] == "car" for e in cars)

        window = client.get(
            f"/v1/jobs/{job['id']}/events", params={"from_ms": 10_000, "to_ms": 14_000}
        ).json()
        assert window and all(e["end_ms"] >= 10_000 and e["start_ms"] <= 14_000 for e in window)

        tracks = client.get(f"/v1/jobs/{job['id']}/tracks").json()
        assert {t["segment_index"] for t in tracks["items"]} == {0, 1, 2}

    def test_detections_are_paginated_by_cursor_without_gaps_or_repeats(
        self, client, settings, pool, mp4_bytes
    ):
        video = upload(client, mp4_bytes).json()
        job = client.post("/v1/jobs", json={"video_id": video["id"]}).json()
        run_worker(pool, settings, job["id"])
        seen, cursor = [], 0
        while True:
            page = client.get(
                f"/v1/jobs/{job['id']}/detections", params={"limit": 100, "cursor": cursor}
            ).json()
            seen += [d["id"] for d in page["items"]]
            if page["next_cursor"] is None:
                break
            cursor = page["next_cursor"]
        # 100 person + 50 car detections per 10 s segment (+ 50 person for the 5 s tail) = 375
        assert len(seen) == len(set(seen)) == 375
        assert seen == sorted(seen)

    def test_failed_job_exposes_error(self, client, settings, pool, mp4_bytes, monkeypatch):
        monkeypatch.setenv("FAKE_MODE", "bad_input")
        video = upload(client, mp4_bytes).json()
        job = client.post("/v1/jobs", json={"video_id": video["id"]}).json()
        assert run_worker(pool, settings, job["id"]) == Outcome.FAILED
        got = client.get(f"/v1/jobs/{job['id']}").json()
        assert got["status"] == "failed" and got["error_code"] == "bad_input"

    def test_list_jobs_filters_by_status(self, client, mp4_bytes):
        video = upload(client, mp4_bytes).json()
        client.post("/v1/jobs", json={"video_id": video["id"]})
        assert len(client.get("/v1/jobs", params={"status": "queued"}).json()) == 1
        assert client.get("/v1/jobs", params={"status": "completed"}).json() == []

    def test_unknown_job_and_video_return_404(self, client):
        zero = "00000000-0000-0000-0000-000000000000"
        assert client.get(f"/v1/jobs/{zero}").status_code == 404
        assert client.get(f"/v1/jobs/{zero}/events").status_code == 404
        assert client.post("/v1/jobs", json={"video_id": zero}).status_code == 404

    def test_invalid_payload_is_422(self, client):
        assert client.post("/v1/jobs", json={"video_id": "nope"}).status_code == 422


def test_api_key_is_enforced_when_configured(settings, pool, mp4_bytes):
    secured = settings.model_copy(update={"api_key": "s3cret"})
    with TestClient(create_app(secured, pool)) as c:
        assert c.get("/v1/jobs").status_code == 401
        assert c.get("/v1/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health/live").status_code == 200  # probes never need a key
