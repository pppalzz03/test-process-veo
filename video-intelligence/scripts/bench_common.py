"""Shared helpers for benchmark scripts: resource sampling and result files.

Every number written by the benchmark scripts is MEASURED by this code during a real run.
Results produced with the synthetic test-double runner are flagged `synthetic: true` and are
excluded from the report tables.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def is_synthetic_runner(command: str) -> bool:
    return "fake_runner" in command


def _read_cpu_times() -> tuple[int, int] | None:
    try:
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(x) for x in parts]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)
    except (OSError, ValueError, IndexError):
        return None


def _read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            info[key] = int(rest.split()[0]) // 1024  # MiB
    except (OSError, ValueError):
        pass
    return info


def _query_gpu() -> dict[str, float | None]:
    """GPU utilization / memory via nvidia-smi. GB10 uses unified memory, so memory.used can be
    reported as [N/A]; that is recorded as null rather than guessed."""
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (subprocess.SubprocessError, OSError, IndexError):
        return {}

    def num(text: str) -> float | None:
        try:
            return float(text.strip())
        except ValueError:
            return None

    util, mem = (out.split(",") + ["", ""])[:2]
    return {"gpu_util_pct": num(util), "gpu_mem_used_mib": num(mem)}


class ResourceSampler:
    """Background sampler: CPU %, MemAvailable, GPU utilization/memory every `interval` s."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> ResourceSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        prev = _read_cpu_times()
        while not self._stop.wait(self.interval):
            sample: dict[str, float | None] = {"t": time.time()}
            cur = _read_cpu_times()
            if prev and cur and cur[1] > prev[1]:
                sample["cpu_util_pct"] = 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))
            prev = cur
            mem = _read_meminfo()
            if mem:
                sample["mem_available_mib"] = mem.get("MemAvailable")
                sample["mem_used_mib"] = mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)
            sample.update(_query_gpu())
            self.samples.append(sample)

    def summary(self) -> dict[str, dict[str, float] | int]:
        result: dict[str, dict[str, float] | int] = {"sample_count": len(self.samples)}
        keys = (
            "cpu_util_pct",
            "mem_used_mib",
            "mem_available_mib",
            "gpu_util_pct",
            "gpu_mem_used_mib",
        )
        for key in keys:
            values = sorted(float(s[key]) for s in self.samples if s.get(key) is not None)
            if values:
                p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
                result[key] = {
                    "mean": round(statistics.fmean(values), 2),
                    "p95": round(p95, 2),
                    "max": round(values[-1], 2),
                }
        return result


def host_info() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_visible": str(shutil.which("nvidia-smi") is not None),
    }


def write_result(name: str, payload: dict, *, synthetic: bool, out_dir: Path | None = None) -> Path:
    directory = out_dir or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    document = {
        "benchmark": name,
        "timestamp": stamp,
        "synthetic": synthetic,
        "host": host_info(),
        **payload,
    }
    path = directory / f"{name}-{stamp}.json"
    path.write_text(json.dumps(document, indent=2))
    return path


def video_hours_per_wall_hour(video_seconds: float, wall_seconds: float) -> float:
    """Throughput: seconds of video processed per second of wall clock (== hours per hour)."""
    return video_seconds / wall_seconds if wall_seconds > 0 else 0.0


def ffprobe_video(path: str) -> dict[str, float]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,avg_frame_rate,nb_frames,codec_name",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(stream["duration"])
    frames = (
        int(stream["nb_frames"])
        if stream.get("nb_frames", "N/A").isdigit()
        else round(duration * fps)
    )
    return {"duration_s": duration, "fps": fps, "frames": frames, "codec": stream["codec_name"]}


def runner_command() -> list[str]:
    import shlex

    return shlex.split(
        os.environ.get("RUNNER_COMMAND", "/opt/venv/bin/python /opt/vi/deepstream/app/ds_runner.py")
    )
