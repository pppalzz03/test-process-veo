import json
import logging
from pathlib import Path

import pytest

from deepstream.app.ds_runner import (
    EXIT_BAD_INPUT,
    EXIT_ENVIRONMENT,
    EXIT_PIPELINE,
    build_launch_string,
    choose_decode_path,
    classify_gst_error,
    parse_args,
)
from services.api.probe import ProbeError, parse_ffprobe_output
from services.api.service import VideoRejected, resolve_media_path
from services.scheduler.gpu_scheduler import read_mem_available_mb
from shared.logging import JsonFormatter, log_context


class TestProbeParsing:
    RAW = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "duration": "3600.5",
                "start_time": "0.067",
            }
        ],
        "format": {"duration": "3601.0"},
    }

    def test_prefers_stream_duration_and_reads_offset(self):
        probe = parse_ffprobe_output(self.RAW)
        assert probe.duration_ms == 3_600_500 and probe.start_offset_ms == 67
        assert probe.fps == pytest.approx(29.97, abs=0.01) and probe.codec == "h264"

    def test_falls_back_to_container_duration(self):
        raw = {
            "streams": [{"codec_type": "video", "avg_frame_rate": "25/1"}],
            "format": {"duration": "12.0"},
        }
        assert parse_ffprobe_output(raw).duration_ms == 12_000

    def test_rejects_files_without_video(self):
        with pytest.raises(ProbeError):
            parse_ffprobe_output({"streams": [{"codec_type": "audio"}], "format": {}})

    def test_rejects_unknown_duration(self):
        with pytest.raises(ProbeError):
            parse_ffprobe_output({"streams": [{"codec_type": "video"}], "format": {}})


class TestMediaPath:
    def test_accepts_file_inside_root(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        assert resolve_media_path(str(tmp_path), "a.mp4") == (tmp_path / "a.mp4").resolve()

    @pytest.mark.parametrize("bad", ["../etc/passwd", "/etc/passwd", "missing.mp4"])
    def test_rejects_escape_and_missing(self, tmp_path, bad):
        with pytest.raises(VideoRejected):
            resolve_media_path(str(tmp_path), bad)

    def test_rejects_symlink_escape(self, tmp_path):
        outside = tmp_path / "outside.mp4"
        outside.write_bytes(b"x")
        root = tmp_path / "root"
        root.mkdir()
        (root / "link.mp4").symlink_to(outside)
        with pytest.raises(VideoRejected):
            resolve_media_path(str(root), "link.mp4")


class TestRunnerHelpers:
    def args(self, **kw):
        base = [
            "--source",
            "/v.mp4",
            "--end-ms",
            "1000",
            "--infer-config",
            "/c.txt",
            "--tracker-config",
            "/t.yml",
            "--tracker-lib",
            "/t.so",
            "--labels",
            "/l.txt",
        ]
        for k, v in kw.items():
            base += [f"--{k.replace('_', '-')}", v]
        return parse_args(base)

    @pytest.mark.parametrize(
        "decode,codec,expected",
        [
            ("auto", "h264", "hw"),
            ("auto", "hevc", "hw"),
            ("auto", "vp9", "generic"),
            ("auto", None, "generic"),
            ("sw", "h264", "generic"),
            ("hw", "h264", "hw"),
        ],
    )
    def test_decode_path(self, decode, codec, expected):
        assert choose_decode_path(decode, codec) == expected

    def test_forced_hw_with_unsupported_codec_is_an_error(self):
        with pytest.raises(ValueError):
            choose_decode_path("hw", "vp9")

    def test_launch_string_uses_hw_decoder_and_core_elements(self):
        launch = build_launch_string(self.args(codec="h264"), "hw")
        for token in (
            "nvv4l2decoder",
            "h264parse",
            "nvstreammux",
            "nvinfer",
            "nvtracker",
            "mux.sink_0",
        ):
            assert token in launch
        assert "uridecodebin" not in launch

    def test_launch_string_generic_path(self):
        assert "uridecodebin" in build_launch_string(self.args(), "generic")

    def test_error_classification(self):
        assert classify_gst_error("gst-stream-error-quark", "decode failed") == EXIT_BAD_INPUT
        assert (
            classify_gst_error(
                "gst-resource-error-quark", "Failed to create NvDsInferContext instance"
            )
            == EXIT_ENVIRONMENT
        )
        assert classify_gst_error("gst-core-error-quark", "negotiation") == EXIT_PIPELINE


class TestMeminfo:
    def test_reads_mem_available(self, tmp_path):
        f = tmp_path / "meminfo"
        f.write_text("MemTotal: 100 kB\nMemAvailable:   2097152 kB\n")
        assert read_mem_available_mb(f) == 2048

    def test_missing_file_returns_none(self, tmp_path):
        assert read_mem_available_mb(tmp_path / "nope") is None


def test_json_log_includes_context_and_extras():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
    record.attempt = 2
    with log_context(job_id="j1"):
        payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "hello" and payload["job_id"] == "j1" and payload["attempt"] == 2


def test_runner_script_has_no_top_level_gst_import():
    """ds_runner must import under plain Python so it can be unit tested without DeepStream."""
    assert "import pyds" not in Path("deepstream/app/ds_runner.py").read_text().split("def ")[0]
