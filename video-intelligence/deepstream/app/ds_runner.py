#!/usr/bin/env python3
"""DeepStream segment runner. Runs INSIDE the DeepStream SBSA container (needs Gst + pyds).

  MP4 --(nvv4l2decoder | uridecodebin)--> nvstreammux --> nvinfer (YOLO TensorRT)
      --> nvtracker --> pad probe --> NDJSON on stdout

One process handles ONE source window [start_ms, end_ms) using a segment seek (no re-encoding,
no temporary files). Stdout carries only NDJSON records (contract: shared/schemas/runner.py);
all logs go to stderr. Exit codes: 0 ok, 2 bad input, 3 pipeline error, 4 environment error.

STATUS: written against the documented DeepStream 9.x / pyds API but NOT yet executed on a
DGX Spark (no access from the authoring environment). See docs/environment-report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

EXIT_OK, EXIT_BAD_INPUT, EXIT_PIPELINE, EXIT_ENVIRONMENT = 0, 2, 3, 4
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF
HW_CODEC_PARSERS = {"h264": "h264parse", "hevc": "h265parse"}


def log(message: str) -> None:
    print(f"[ds_runner] {message}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--start-ms", type=int, default=0)
    p.add_argument("--end-ms", type=int, required=True)
    p.add_argument("--infer-config", required=True)
    p.add_argument("--tracker-config", required=True)
    p.add_argument("--tracker-lib", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--decode", choices=["auto", "hw", "sw"], default="auto")
    p.add_argument("--codec", default=None, help="ffprobe codec_name, e.g. h264 / hevc")
    p.add_argument("--infer-interval", type=int, default=0)
    p.add_argument("--width", type=int, default=1920, help="only used by the legacy nvstreammux")
    p.add_argument("--height", type=int, default=1080, help="only used by the legacy nvstreammux")
    return p.parse_args(argv)


def load_labels(path: str) -> list[str]:
    try:
        return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    except OSError:
        log(f"labels file not readable: {path}")
        return []


def choose_decode_path(decode: str, codec: str | None) -> str:
    """Returns 'hw' (explicit qtdemux ! parser ! nvv4l2decoder) or 'generic' (uridecodebin)."""
    if decode == "sw":
        return "generic"
    if codec in HW_CODEC_PARSERS:
        return "hw"
    if decode == "hw":
        raise ValueError(f"hardware decode requested but codec {codec!r} is not h264/hevc")
    return "generic"  # auto + unknown codec: let uridecodebin pick a decoder


def build_launch_string(args: argparse.Namespace, decode_path: str) -> str:
    src = os.path.abspath(args.source)
    if decode_path == "hw":
        parser = HW_CODEC_PARSERS[args.codec]
        head = (
            f'filesrc location="{src}" ! qtdemux name=demux demux.video_0 ! queue ! {parser} '
            "! nvv4l2decoder name=decoder ! queue ! mux.sink_0 "
        )
    else:
        head = (
            f'uridecodebin uri="file://{src}" name=decoder ! nvvideoconvert '
            '! "video/x-raw(memory:NVMM),format=NV12" ! queue ! mux.sink_0 '
        )
    return (
        head
        + "nvstreammux name=mux batch-size=1 "
        + f'! nvinfer name=pgie config-file-path="{args.infer_config}" '
        + f'! nvtracker name=tracker ll-lib-file="{args.tracker_lib}" '
        + f'll-config-file="{args.tracker_config}" '
        + "! fakesink name=sink sync=false async=false qos=false"
    )


def set_if_present(element, name: str, value) -> None:  # type: ignore[no-untyped-def]
    """The legacy and new nvstreammux (and nvtracker versions) expose different properties."""
    if element.find_property(name) is not None:
        element.set_property(name, value)


class FrameEmitter:
    """Pad probe: converts DeepStream metadata into NDJSON frame records."""

    def __init__(self, pyds, labels: list[str]) -> None:  # type: ignore[no-untyped-def]
        self._pyds, self._labels = pyds, labels
        self._lock = threading.Lock()
        self.frames = 0

    def __call__(self, pad, info, _user_data):  # type: ignore[no-untyped-def]
        from gi.repository import Gst

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        pyds = self._pyds
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        frame_item = batch_meta.frame_meta_list
        while frame_item is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_item.data)
            objects = []
            obj_item = frame_meta.obj_meta_list
            while obj_item is not None:
                obj = pyds.NvDsObjectMeta.cast(obj_item.data)
                objects.append(self._object_record(obj))
                obj_item = obj_item.next
            record = {
                "type": "frame",
                "frame_num": int(frame_meta.frame_num),
                # PTS is the source timestamp in ns; the worker validates it against the window.
                "pts_ms": int(frame_meta.buf_pts // 1_000_000),
                "objects": objects,
            }
            with self._lock:
                sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
                sys.stdout.flush()
                self.frames += 1
            frame_item = frame_item.next
        return Gst.PadProbeReturn.OK

    def _object_record(self, obj) -> dict:  # type: ignore[no-untyped-def]
        track_id = int(obj.object_id)
        confidence = float(obj.confidence)
        if confidence < 0:  # tracker-predicted object without a fresh detection
            confidence = float(getattr(obj, "tracker_confidence", 0.0))
        label = obj.obj_label or (
            self._labels[obj.class_id]
            if 0 <= obj.class_id < len(self._labels)
            else str(obj.class_id)
        )
        rect = obj.rect_params
        return {
            "track_id": None if track_id == UNTRACKED_OBJECT_ID else track_id,
            "class_id": int(obj.class_id),
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": [
                round(rect.left, 1),
                round(rect.top, 1),
                round(rect.width, 1),
                round(rect.height, 1),
            ],
        }


def classify_gst_error(domain: str, message: str) -> int:
    if "NvDsInferContext" in message or "config file" in message.lower():
        return EXIT_ENVIRONMENT  # model / engine / parser library problems
    if domain == "gst-stream-error-quark":
        return EXIT_BAD_INPUT  # demux / decode / format errors: the video itself is bad
    return EXIT_PIPELINE


def run_pipeline(args: argparse.Namespace) -> int:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        import pyds
        from gi.repository import GLib, Gst
    except (ImportError, ValueError) as exc:
        log(
            f"environment error: cannot import Gst/pyds ({exc}). Run inside the DeepStream "
            "container; if pyds is absent in this release, port FrameEmitter to pyservicemaker."
        )
        return EXIT_ENVIRONMENT

    Gst.init(None)
    if not os.path.isfile(args.source):
        log(f"source not found: {args.source}")
        return EXIT_BAD_INPUT
    required = (
        ("infer config", args.infer_config),
        ("tracker config", args.tracker_config),
        ("tracker library", args.tracker_lib),
    )
    for label, path in required:
        if not os.path.isfile(path):
            log(f"environment error: {label} not found: {path}")
            return EXIT_ENVIRONMENT
    try:
        decode_path = choose_decode_path(args.decode, args.codec)
    except ValueError as exc:
        log(str(exc))
        return EXIT_BAD_INPUT

    launch = build_launch_string(args, decode_path)
    log(f"decode={decode_path} pipeline: {launch}")
    try:
        pipeline = Gst.parse_launch(launch)
    except GLib.Error as exc:
        log(f"environment error: cannot build pipeline: {exc.message}")
        return EXIT_ENVIRONMENT

    mux = pipeline.get_by_name("mux")
    set_if_present(mux, "width", args.width)
    set_if_present(mux, "height", args.height)
    set_if_present(mux, "batched-push-timeout", 40000)
    pgie = pipeline.get_by_name("pgie")
    set_if_present(pgie, "interval", args.infer_interval)

    emitter = FrameEmitter(pyds, load_labels(args.labels))
    pipeline.get_by_name("tracker").get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, emitter, None
    )

    bus = pipeline.get_bus()
    try:
        pipeline.set_state(Gst.State.PAUSED)
        msg = bus.timed_pop_filtered(
            120 * Gst.SECOND, Gst.MessageType.ASYNC_DONE | Gst.MessageType.ERROR
        )
        code = _check_error(msg, GLib, Gst)
        if code is not None:
            return code
        if msg is None:
            log("pipeline did not reach PAUSED within 120s")
            return EXIT_PIPELINE

        flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE
        if not pipeline.seek(
            1.0,
            Gst.Format.TIME,
            flags,
            Gst.SeekType.SET,
            args.start_ms * Gst.MSECOND,
            Gst.SeekType.SET,
            args.end_ms * Gst.MSECOND,
        ):
            log("segment seek failed")
            return EXIT_PIPELINE
        pipeline.set_state(Gst.State.PLAYING)

        while True:
            msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            if msg is None:
                continue  # worker enforces the overall timeout and kills us if needed
            code = _check_error(msg, GLib, Gst)
            if code is not None:
                return code
            if msg.type == Gst.MessageType.EOS:
                break
    finally:
        pipeline.set_state(Gst.State.NULL)

    print(json.dumps({"type": "summary", "frames": emitter.frames, "eos": True}), flush=True)
    log(f"EOS after {emitter.frames} frames")
    return EXIT_OK


def _check_error(msg, GLib, Gst):  # type: ignore[no-untyped-def]
    if msg is None or msg.type != Gst.MessageType.ERROR:
        return None
    err, debug = msg.parse_error()
    domain = GLib.quark_to_string(err.domain)
    log(f"pipeline error [{domain}] from {msg.src.get_name()}: {err.message} | {debug}")
    return classify_gst_error(domain, err.message)


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    sys.exit(main())
