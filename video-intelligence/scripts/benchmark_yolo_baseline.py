#!/usr/bin/env python3
"""Baseline: Ultralytics YOLO + ByteTrack in plain Python (what DeepStream is compared against).

  python scripts/benchmark_yolo_baseline.py --weights yolo11n.pt --video /data/media/a.mp4

Needs `pip install ultralytics` (+ torch with CUDA for the GPU). Uses the same video and model
family as the DeepStream benchmark; note it decodes on CPU via OpenCV, which is part of what the
comparison measures. Reports frames/s, wall time, video-hours per wall-hour and resources.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from bench_common import ResourceSampler, ffprobe_video, video_hours_per_wall_hour, write_result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="FP16 (match DeepStream network-mode=2)")
    p.add_argument("--duration-s", type=float, default=0)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed: pip install ultralytics", file=sys.stderr)
        return 2

    meta = ffprobe_video(args.video)
    limit_frames = int(args.duration_s * meta["fps"]) if args.duration_s else None
    model = YOLO(args.weights)

    frames = objects = 0
    with ResourceSampler(1.0) as sampler:
        started = time.monotonic()
        results = model.track(
            source=args.video,
            stream=True,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        for res in results:
            frames += 1
            objects += 0 if res.boxes is None else len(res.boxes)
            if limit_frames and frames >= limit_frames:
                break
        wall = time.monotonic() - started

    video_s = frames / meta["fps"]
    payload = {
        "inputs": {
            "video": args.video,
            "weights": args.weights,
            "imgsz": args.imgsz,
            "half": args.half,
            "device": args.device,
            "probe": meta,
            "note": "includes first-batch CUDA warm-up inside the timed region",
        },
        "results": {
            "total_wall_s": round(wall, 3),
            "total_frames": frames,
            "objects": objects,
            "expected_frames": limit_frames or round(meta["duration_s"] * meta["fps"]),
            "effective_fps_total": round(frames / wall, 2) if wall else 0,
            "video_hours_per_wall_hour": round(video_hours_per_wall_hour(video_s, wall), 3),
        },
        "resources": sampler.summary(),
    }
    path = write_result("yolo-python-baseline", payload, synthetic=False)
    print(json.dumps(payload["results"], indent=2))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
