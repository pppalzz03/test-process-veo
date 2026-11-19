#!/usr/bin/env python3
"""TEST DOUBLE for deepstream/app/ds_runner.py. Emits deterministic synthetic detections.

It exists only to exercise the worker/orchestration code paths without a GPU. Nothing produced by
it may ever be reported as a detection or benchmark result.

Behaviour is selected with the FAKE_MODE env var: ok (default) | bad_input | crash | hang | no_eos
FAKE_FPS (default 10) sets the synthetic frame rate. FAKE_DELAY_S adds a sleep per frame.
FAKE_PREROLL_MS emits frames before start_ms to mimic keyframe pre-roll.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source")
    p.add_argument("--start-ms", type=int, default=0)
    p.add_argument("--end-ms", type=int, required=True)
    for opt in (
        "infer-config",
        "tracker-config",
        "tracker-lib",
        "labels",
        "decode",
        "codec",
        "infer-interval",
        "width",
        "height",
    ):
        p.add_argument(f"--{opt}")
    args = p.parse_args()

    mode = os.environ.get("FAKE_MODE", "ok")
    fps = float(os.environ.get("FAKE_FPS", "10"))
    delay = float(os.environ.get("FAKE_DELAY_S", "0"))
    preroll = int(os.environ.get("FAKE_PREROLL_MS", "0"))

    if mode == "bad_input":
        print("synthetic: corrupt stream", file=sys.stderr)
        return 2
    if mode == "crash":
        print("synthetic: pipeline crashed", file=sys.stderr)
        return 3
    if mode == "hang":
        time.sleep(3600)

    step_ms = 1000.0 / fps
    t = max(0.0, args.start_ms - preroll)
    n = 0
    mid = (args.start_ms + args.end_ms) / 2
    while t < args.end_ms:
        pts = int(round(t))
        objects = [
            {
                "track_id": 1,
                "class_id": 0,
                "label": "person",
                "confidence": 0.9,
                "bbox": [10.0, 10.0, 50.0, 100.0],
            },
            {
                "track_id": None,
                "class_id": 0,
                "label": "person",
                "confidence": 0.1,
                "bbox": [0.0, 0.0, 5.0, 5.0],
            },  # below min_confidence: must be filtered
        ]
        if pts < mid:
            objects.append(
                {
                    "track_id": 2,
                    "class_id": 2,
                    "label": "car",
                    "confidence": 0.8,
                    "bbox": [200.0, 50.0, 80.0, 40.0],
                }
            )
        print(
            json.dumps({"type": "frame", "frame_num": n, "pts_ms": pts, "objects": objects}),
            flush=True,
        )
        n += 1
        t += step_ms
        if delay:
            time.sleep(delay)
    if mode != "no_eos":
        print(json.dumps({"type": "summary", "frames": n, "eos": True}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
