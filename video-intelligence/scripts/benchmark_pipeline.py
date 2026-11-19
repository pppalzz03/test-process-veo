#!/usr/bin/env python3
"""Phase 2 benchmark: DeepStream runner, 1 / 2 / 4 concurrent streams (no database, no queue).

  python scripts/benchmark_pipeline.py --video /data/media/a.mp4 --streams 1
  python scripts/benchmark_pipeline.py --video /data/media/a.mp4 --streams 2 --duration-s 600
  python scripts/benchmark_pipeline.py --video /data/media/a.mp4 --streams 4

Run INSIDE the worker container on the DGX Spark. Each stream is one independent runner process
(the same unit the worker uses). Vary only --streams between runs and keep video, model and
config identical. Reports wall time, effective FPS, video-hours per wall-hour, GPU/CPU/memory.
Streams stop being comparable if any run fails: failures are recorded and flagged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from bench_common import (
    ResourceSampler,
    ffprobe_video,
    is_synthetic_runner,
    runner_command,
    video_hours_per_wall_hour,
    write_result,
)


def run_stream(idx: int, args: argparse.Namespace, meta: dict, out: dict) -> None:
    end_ms = int(min(meta["duration_s"], args.duration_s or meta["duration_s"]) * 1000)
    cmd = [
        *runner_command(),
        "--source",
        args.video,
        "--start-ms",
        "0",
        "--end-ms",
        str(end_ms),
        "--infer-config",
        args.infer_config,
        "--tracker-config",
        args.tracker_config,
        "--tracker-lib",
        args.tracker_lib,
        "--labels",
        args.labels,
        "--codec",
        meta["codec"],
    ]
    started = time.monotonic()
    first_frame_at = None
    frames = objects = 0
    eos = False
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        record = json.loads(line)
        if record["type"] == "frame":
            if first_frame_at is None:
                first_frame_at = time.monotonic()
            frames += 1
            objects += len(record["objects"])
        elif record["type"] == "summary":
            eos = bool(record["eos"])
    code = proc.wait()
    wall = time.monotonic() - started
    out[idx] = {
        "exit_code": code,
        "eos": eos,
        "frames": frames,
        "objects": objects,
        "wall_s": round(wall, 3),
        "time_to_first_frame_s": round(first_frame_at - started, 3) if first_frame_at else None,
        "stderr_tail": proc.stderr.read()[-500:] if code else "",
        "video_seconds": end_ms / 1000,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--duration-s", type=float, default=0, help="limit to first N seconds (0=all)")
    p.add_argument("--warmup", action="store_true", help="build/load the engine before timing")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument(
        "--infer-config",
        default=os.environ.get(
            "DS_INFER_CONFIG", "/opt/vi/deepstream/configs/config_infer_primary_yolo.txt"
        ),
    )
    p.add_argument(
        "--tracker-config",
        default=os.environ.get(
            "DS_TRACKER_CONFIG",
            "/opt/nvidia/deepstream/deepstream/samples/configs/"
            "deepstream-app/config_tracker_NvDCF_perf.yml",
        ),
    )
    p.add_argument(
        "--tracker-lib",
        default=os.environ.get(
            "DS_TRACKER_LIB",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        ),
    )
    p.add_argument(
        "--labels", default=os.environ.get("DS_LABELS_FILE", "/opt/vi/models/labels.txt")
    )
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    command = " ".join(runner_command())
    synthetic = is_synthetic_runner(command)
    if synthetic and not args.allow_synthetic:
        print("refusing to benchmark the synthetic test-double runner", file=sys.stderr)
        return 2

    meta = ffprobe_video(args.video)
    if args.warmup:
        warm: dict = {}
        saved = args.duration_s
        args.duration_s = 2
        run_stream(0, args, meta, warm)
        args.duration_s = saved
        if warm[0]["exit_code"] != 0:
            print("warm-up failed:", warm[0]["stderr_tail"], file=sys.stderr)
            return 1

    results: dict[int, dict] = {}
    threads = [
        threading.Thread(target=run_stream, args=(i, args, meta, results))
        for i in range(args.streams)
    ]
    with ResourceSampler(1.0) as sampler:
        started = time.monotonic()
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.monotonic() - started

    streams = [results[i] for i in range(args.streams)]
    ok = all(s["exit_code"] == 0 and s["eos"] for s in streams)
    total_frames = sum(s["frames"] for s in streams)
    total_video_s = sum(s["video_seconds"] for s in streams)
    payload = {
        "inputs": {
            "video": args.video,
            "streams": args.streams,
            "duration_s_limit": args.duration_s,
            "runner": command,
            "infer_config": args.infer_config,
            "probe": meta,
            "warmup": args.warmup,
        },
        "results": {
            "all_streams_reached_eos": ok,
            "total_wall_s": round(wall, 3),
            "total_frames": total_frames,
            "expected_frames_per_stream": round(
                min(meta["duration_s"], args.duration_s or meta["duration_s"]) * meta["fps"]
            ),
            "effective_fps_total": round(total_frames / wall, 2) if wall else 0,
            "effective_fps_per_stream": round(total_frames / wall / args.streams, 2) if wall else 0,
            "video_hours_per_wall_hour": round(video_hours_per_wall_hour(total_video_s, wall), 3),
            "streams": streams,
        },
        "resources": sampler.summary(),
    }
    path = write_result(
        f"pipeline-{args.streams}stream",
        payload,
        synthetic=synthetic,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps(payload["results"], indent=2))
    print(f"saved {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
