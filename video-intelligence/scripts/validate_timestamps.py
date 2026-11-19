#!/usr/bin/env python3
"""Verify that the runner's pts_ms values match the real frame timestamps (ffprobe).

This closes the main open risk in Phase 2: nvstreammux may rewrite buffer PTS. Run on the Spark:

  python scripts/validate_timestamps.py /data/media/sample.mp4 --start-ms 60000 --end-ms 70000

Compares the NDJSON frames produced by deepstream/app/ds_runner.py with `ffprobe -show_frames`
over the same window. Exit code 0 only if every runner frame lies within TOLERANCE_MS of a real
frame and the frame counts agree within 2%.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shlex
import subprocess
import sys

TOLERANCE_MS = 2


def ffprobe_pts_ms(video: str, start_ms: int, end_ms: int) -> list[int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            video,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    times = sorted(
        int(round(float(f["pts_time"]) * 1000))
        for f in json.loads(out)["frames"]
        if "pts_time" in f
    )
    return [t for t in times if start_ms <= t < end_ms]


def runner_pts_ms(args: argparse.Namespace) -> list[int]:
    cmd = shlex.split(
        os.environ.get("RUNNER_COMMAND", "/opt/venv/bin/python /opt/vi/deepstream/app/ds_runner.py")
    )
    cmd += [
        "--source",
        args.video,
        "--start-ms",
        str(args.start_ms),
        "--end-ms",
        str(args.end_ms),
        "--infer-config",
        args.infer_config,
        "--tracker-config",
        args.tracker_config,
        "--tracker-lib",
        args.tracker_lib,
        "--labels",
        args.labels,
    ]
    if args.codec:
        cmd += ["--codec", args.codec]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"runner failed with exit code {proc.returncode}")
    frames = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return sorted(
        f["pts_ms"]
        for f in frames
        if f["type"] == "frame" and args.start_ms <= f["pts_ms"] < args.end_ms
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--start-ms", type=int, default=0)
    p.add_argument("--end-ms", type=int, required=True)
    p.add_argument("--codec", default=None)
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
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
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
    args = p.parse_args()

    truth = ffprobe_pts_ms(args.video, args.start_ms, args.end_ms)
    got = runner_pts_ms(args)
    bad = 0
    for t in got:
        i = bisect.bisect_left(truth, t)
        nearest = min((abs(truth[j] - t) for j in (i - 1, i) if 0 <= j < len(truth)), default=10**9)
        bad += nearest > TOLERANCE_MS
    ratio = len(got) / len(truth) if truth else 0
    print(
        f"ffprobe frames={len(truth)} runner frames={len(got)} ratio={ratio:.3f} "
        f"frames off by >{TOLERANCE_MS}ms: {bad}"
    )
    ok = bad == 0 and 0.98 <= ratio <= 1.02
    print("TIMESTAMPS OK" if ok else "TIMESTAMP MISMATCH: do not trust results until resolved")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
