#!/usr/bin/env python3
"""Find YOLO models in the project and decide the export / DeepStream strategy.

Read-only. Usage:  python scripts/inspect_yolo_model.py [DIR_OR_FILE ...]

For each .onnx it reports input/output shapes and which parser layout applies
(raw [4+nc, N] vs end-to-end [300, 6]); for each .pt it lists what ultralytics says (if
installed). Missing optional packages are reported, never installed.
"""

from __future__ import annotations

import sys
from pathlib import Path


def classify_output(shape: list[int | str], num_classes: int | None) -> str:
    dims = [d for d in shape if d != 1 or len(shape) <= 2]
    if len(dims) == 2 and all(isinstance(d, int) for d in dims):
        a, b = dims  # type: ignore[misc]
        if b == 6 and a <= 1000:
            return "end2end [N,6] -> config_infer_primary_yolo_e2e.txt (cluster-mode=4)"
        if a < b and (num_classes is None or a == 4 + num_classes):
            return "raw [4+nc,anchors] -> config_infer_primary_yolo.txt (cluster-mode=2)"
    return "UNRECOGNISED layout: the custom parser must be adapted before use"


def inspect_onnx(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("  onnx package not installed: cannot read shapes (pip install onnx)")
        return
    model = onnx.load(str(path), load_external_data=False)

    def shape_of(value) -> list[int | str]:  # type: ignore[no-untyped-def]
        return [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in value.type.tensor_type.shape.dim
        ]

    for inp in model.graph.input:
        print(f"  input  {inp.name}: {shape_of(inp)}")
    for out in model.graph.output:
        shp = shape_of(out)
        print(f"  output {out.name}: {shp}")
        print(f"  -> {classify_output(shp[1:] if len(shp) == 3 else shp, None)}")
    print(f"  opset: {[o.version for o in model.opset_import]}")


def inspect_pt(path: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed: cannot read task/classes (pip install ultralytics)")
        return
    model = YOLO(str(path))
    names = getattr(model, "names", {})
    print(f"  task={model.task} classes={len(names)} first={list(names.values())[:5]}")
    print("  export with: python scripts/export_yolo.py --weights", path)


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path(".")]
    found = False
    for root in roots:
        files = (
            [root]
            if root.is_file()
            else sorted(
                p
                for p in root.rglob("*")
                if p.suffix.lower() in {".pt", ".onnx", ".engine"}
                and not any(part in {".git", "node_modules", ".venv"} for part in p.parts)
            )
        )
        for path in files:
            found = True
            print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
            if path.suffix.lower() == ".onnx":
                inspect_onnx(path)
            elif path.suffix.lower() == ".pt":
                inspect_pt(path)
            else:
                print("  .engine is tied to the exact GPU/TensorRT version: rebuild on the Spark")
    if not found:
        print("No YOLO models (.pt/.onnx/.engine) found. Provide one, then re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
