#!/usr/bin/env python3
"""Export Ultralytics weights to ONNX for DeepStream/TensorRT (engine is built by nvinfer).

  python scripts/export_yolo.py --weights yolo11n.pt --out deepstream/models/yolo.onnx
  python scripts/export_yolo.py --weights yolo26n.pt --out deepstream/models/yolo_e2e.onnx --e2e

Needs `pip install ultralytics onnx` (not part of the runtime services). Run it on the DGX Spark
so the exported graph matches the TensorRT version that will consume it; the TensorRT engine
itself is built later by scripts/build_engine.sh and must never be copied between machines.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--e2e",
        action="store_true",
        help="keep the NMS-free end-to-end head (YOLOv10/YOLO26 style output [300,6])",
    )
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed: pip install ultralytics onnx", file=sys.stderr)
        return 2

    model = YOLO(args.weights)
    # batch=1 static shape: one source per runner process (see config_infer_primary_yolo.txt).
    export_kwargs = dict(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False, batch=1
    )
    if args.e2e:
        export_kwargs["end2end"] = True  # only meaningful for architectures that support it
    exported = Path(model.export(**export_kwargs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, out)
    print(f"wrote {out}")
    print(f"num classes: {len(model.names)}  -> set num-detected-classes in the nvinfer config")
    print("verify the layout with: python scripts/inspect_yolo_model.py", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
