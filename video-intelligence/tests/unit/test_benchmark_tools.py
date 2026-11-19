"""Tests of the benchmark HARNESS (mechanics, flagging, arithmetic). They produce no performance
claims: the synthetic runner only proves the scripts count frames and flag results correctly."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_common  # noqa: E402
import benchmark_batch  # noqa: E402
import render_benchmark_report as render  # noqa: E402

from tests.conftest import FAKE_RUNNER, make_test_video  # noqa: E402


def test_throughput_arithmetic():
    assert bench_common.video_hours_per_wall_hour(3600, 900) == 4.0
    assert bench_common.video_hours_per_wall_hour(10, 0) == 0.0


def test_sampler_collects_real_system_samples():
    import time

    with bench_common.ResourceSampler(0.05) as sampler:
        time.sleep(0.4)
    summary = sampler.summary()
    assert summary["sample_count"] >= 3
    assert "mem_available_mib" in summary and summary["mem_available_mib"]["max"] > 0


def test_job_summary_math_and_failure_accounting():
    jobs = [
        {
            "video_id": "v",
            "status": "completed",
            "attempts": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:10+00:00",
            "completed_at": "2026-01-01T00:01:10+00:00",
        },
        {
            "video_id": "v",
            "status": "completed",
            "attempts": 2,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:01:10+00:00",
            "completed_at": "2026-01-01T00:02:10+00:00",
        },
        {
            "video_id": "v",
            "status": "failed",
            "attempts": 4,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": "2026-01-01T00:00:05+00:00",
            "completed_at": "2026-01-01T00:00:06+00:00",
        },
    ]
    s = benchmark_batch.summarize_jobs(jobs, {"v": {"duration_ms": 3_600_000}})
    assert (s["completed"], s["failed"], s["retried_jobs"]) == (2, 1, 2)
    assert s["success_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert s["queue_wait_s"]["max"] == 70.0 and s["completed_video_seconds"] == 7200.0
    assert s["processing_s"]["mean"] == 60.0


def test_pipeline_benchmark_refuses_synthetic_runner_by_default(tmp_path):
    video = make_test_video(tmp_path / "v.mp4", seconds=5, fps=10)
    env = {**os.environ, "RUNNER_COMMAND": FAKE_RUNNER}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark_pipeline.py"),
            "--video",
            str(video),
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2 and "synthetic" in proc.stderr
    assert list(tmp_path.glob("*.json")) == []


def test_pipeline_benchmark_counts_frames_and_flags_result_synthetic(tmp_path):
    video = make_test_video(tmp_path / "v.mp4", seconds=5, fps=10)
    env = {**os.environ, "RUNNER_COMMAND": FAKE_RUNNER, "FAKE_FPS": "10"}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark_pipeline.py"),
            "--video",
            str(video),
            "--streams",
            "2",
            "--allow-synthetic",
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    (result_file,) = tmp_path.glob("pipeline-2stream-*.json")
    doc = json.loads(result_file.read_text())
    assert doc["synthetic"] is True
    assert doc["results"]["all_streams_reached_eos"] is True
    assert doc["results"]["total_frames"] == 2 * 50  # 2 streams x 5 s x 10 fps
    assert all(s["frames"] == 50 for s in doc["results"]["streams"])


def test_failed_stream_is_reported_not_hidden(tmp_path):
    video = make_test_video(tmp_path / "v.mp4", seconds=3, fps=10)
    env = {**os.environ, "RUNNER_COMMAND": FAKE_RUNNER, "FAKE_MODE": "crash"}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "benchmark_pipeline.py"),
            "--video",
            str(video),
            "--allow-synthetic",
            "--out-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    doc = json.loads(next(tmp_path.glob("pipeline-*.json")).read_text())
    assert doc["results"]["all_streams_reached_eos"] is False
    assert doc["results"]["streams"][0]["exit_code"] == 3


class TestReport:
    def render_with(self, tmp_path, monkeypatch, docs):
        results = tmp_path / "results"
        results.mkdir()
        for i, d in enumerate(docs):
            (results / f"{d['benchmark']}-{i}.json").write_text(json.dumps(d))
        monkeypatch.setattr(render, "RESULTS", results)
        monkeypatch.setattr(render, "OUT", tmp_path / "report.md")
        assert render.main() == 0
        return (tmp_path / "report.md").read_text()

    def test_no_results_means_no_numbers_are_claimed(self, tmp_path, monkeypatch):
        text = self.render_with(tmp_path, monkeypatch, [])
        assert "NOT MEASURED YET" in text and "no performance numbers" in text
        assert "| " not in text.split("Required")[0]  # no tables before the to-do list

    def test_synthetic_results_never_enter_performance_tables(self, tmp_path, monkeypatch):
        fake = {
            "benchmark": "pipeline-1stream",
            "synthetic": True,
            "host": {"hostname": "x"},
            "results": {"total_frames": 1, "total_wall_s": 1, "video_hours_per_wall_hour": 99},
        }
        text = self.render_with(tmp_path, monkeypatch, [fake])
        assert "NOT MEASURED YET" in text and "Harness self-tests" in text
        assert "99" not in text

    def test_real_result_is_tabulated_from_file_values(self, tmp_path, monkeypatch):
        real = {
            "benchmark": "pipeline-1stream",
            "synthetic": False,
            "host": {"hostname": "spark"},
            "inputs": {"probe": {"duration_s": 3600.0}},
            "results": {
                "total_frames": 108000,
                "expected_frames_per_stream": 108000,
                "all_streams_reached_eos": True,
                "total_wall_s": 1234.5,
                "effective_fps_total": 87.5,
                "video_hours_per_wall_hour": 2.916,
            },
            "resources": {"gpu_util_pct": {"mean": 55.5, "p95": 90, "max": 99}},
        }
        text = self.render_with(tmp_path, monkeypatch, [real])
        assert "1234.5" in text and "2.916" in text and "55.5" in text
        assert "NOT MEASURED YET" not in text
        assert "DeepStream, 4 concurrent streams" in text and "_Not measured yet._" in text
