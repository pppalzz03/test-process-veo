"""ffprobe wrapper. Validates that an upload is a decodable video and extracts what we need."""

from __future__ import annotations

import contextlib
import json
import subprocess
from dataclasses import dataclass


class ProbeError(Exception):
    """The file is not a usable video."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    duration_ms: int
    start_offset_ms: int
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None


def _parse_rate(rate: str | None) -> float | None:
    if not rate or "/" not in rate:
        return None
    num, den = rate.split("/", 1)
    try:
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None


def parse_ffprobe_output(raw: dict) -> VideoProbe:
    """Pure function (unit-tested). Prefers the video stream's duration over the container's:
    a longer audio track must not make us expect frames that do not exist."""
    streams = [s for s in raw.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ProbeError("no video stream found")
    stream = streams[0]
    fmt = raw.get("format", {})

    duration_s = None
    for candidate in (stream.get("duration"), fmt.get("duration")):
        try:
            if candidate is not None and float(candidate) > 0:
                duration_s = float(candidate)
                break
        except ValueError:
            continue
    if duration_s is None:
        raise ProbeError("cannot determine video duration")

    start_s = 0.0
    with contextlib.suppress(ValueError):
        start_s = float(stream.get("start_time", fmt.get("start_time", 0.0)) or 0.0)

    return VideoProbe(
        duration_ms=int(round(duration_s * 1000)),
        start_offset_ms=int(round(start_s * 1000)),
        fps=_parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")),
        width=stream.get("width"),
        height=stream.get("height"),
        codec=stream.get("codec_name"),
    )


def probe_video(path: str, *, timeout_seconds: int = 60) -> VideoProbe:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "v:0",
        path,
    ]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"ffprobe failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe rejected the file: {proc.stderr.strip()[:300]}")
    try:
        return parse_ffprobe_output(json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON") from exc
