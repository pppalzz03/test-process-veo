#!/usr/bin/env python3
"""Phase 3 benchmark: many video jobs through API -> RabbitMQ -> GPU scheduler -> worker -> DB.

  python scripts/benchmark_batch.py --api-url http://127.0.0.1:8000 \
      --video a.mp4 --video b.mp4 --jobs-per-video 3

Videos are paths RELATIVE to MEDIA_ROOT (registered via POST /v1/videos/register). Jobs are
submitted back-to-back, then polled until all are terminal. Measured from the API's own
timestamps: queue wait (created_at -> started_at), processing time, success/failure rate,
video-hours per wall-hour. Resource sampling covers the machine this script runs on, so run it
on the DGX Spark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from bench_common import ResourceSampler, video_hours_per_wall_hour, write_result

TERMINAL = {"completed", "failed"}


def call(base: str, method: str, path: str, body: dict | None, api_key: str | None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    if api_key:
        req.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path} -> {exc.code}: {exc.read()[:300]!r}") from exc


def parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 3)


def summarize_jobs(jobs: list[dict], videos: dict[str, dict]) -> dict:
    waits, runs, done_video_s = [], [], 0.0
    for job in jobs:
        created, started, finished = (
            parse_ts(job.get(k)) for k in ("created_at", "started_at", "completed_at")
        )
        if created and started:
            waits.append((started - created).total_seconds())
        if started and finished and job["status"] == "completed":
            runs.append((finished - started).total_seconds())
        if job["status"] == "completed":
            done_video_s += videos[job["video_id"]]["duration_ms"] / 1000
    completed = sum(j["status"] == "completed" for j in jobs)
    return {
        "jobs": len(jobs),
        "completed": completed,
        "failed": len(jobs) - completed,
        "success_rate": round(completed / len(jobs), 4) if jobs else None,
        "retried_jobs": sum(j["attempts"] > 1 for j in jobs),
        "queue_wait_s": {
            "mean": round(statistics.fmean(waits), 3) if waits else None,
            "p50": percentile(waits, 0.5),
            "p95": percentile(waits, 0.95),
            "max": round(max(waits), 3) if waits else None,
        },
        "processing_s": {
            "mean": round(statistics.fmean(runs), 3) if runs else None,
            "p95": percentile(runs, 0.95),
        },
        "completed_video_seconds": round(done_video_s, 3),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--video", action="append", required=True, help="path relative to MEDIA_ROOT")
    p.add_argument("--jobs-per-video", type=int, default=1)
    p.add_argument("--timeout-s", type=int, default=6 * 3600)
    p.add_argument("--run-id", default=str(int(time.time())))
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    videos: dict[str, dict] = {}
    for rel in args.video:
        video = call(args.api_url, "POST", "/v1/videos/register", {"path": rel}, args.api_key)
        videos[video["id"]] = video

    job_ids: list[str] = []
    with ResourceSampler(1.0) as sampler:
        started = time.monotonic()
        for video_id in videos:
            for n in range(args.jobs_per_video):
                job = call(
                    args.api_url,
                    "POST",
                    "/v1/jobs",
                    {
                        "video_id": video_id,
                        "idempotency_key": f"bench-{args.run_id}-{video_id}-{n}",
                    },
                    args.api_key,
                )
                job_ids.append(job["id"])
        submitted_s = time.monotonic() - started

        jobs: list[dict] = []
        while time.monotonic() - started < args.timeout_s:
            jobs = [call(args.api_url, "GET", f"/v1/jobs/{j}", None, args.api_key) for j in job_ids]
            if all(j["status"] in TERMINAL for j in jobs):
                break
            time.sleep(5)
        wall = time.monotonic() - started

    timed_out = [j["id"] for j in jobs if j["status"] not in TERMINAL]
    summary = summarize_jobs(jobs, videos)
    summary.update(
        {
            "total_wall_s": round(wall, 3),
            "submit_s": round(submitted_s, 3),
            "video_hours_per_wall_hour": round(
                video_hours_per_wall_hour(summary["completed_video_seconds"], wall), 3
            ),
            "timed_out_jobs": timed_out,
        }
    )
    payload = {
        "inputs": {
            "api_url": args.api_url,
            "videos": args.video,
            "jobs_per_video": args.jobs_per_video,
            "video_durations_s": [v["duration_ms"] / 1000 for v in videos.values()],
        },
        "results": summary,
        "resources": sampler.summary(),
        "job_ids": job_ids,
    }
    path = write_result(
        "batch-jobs", payload, synthetic=False, out_dir=Path(args.out_dir) if args.out_dir else None
    )
    print(json.dumps(summary, indent=2))
    print(f"saved {path}")
    return 0 if not timed_out and summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
