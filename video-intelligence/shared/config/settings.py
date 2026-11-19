"""Validated runtime settings. All values come from environment variables."""

from __future__ import annotations

import shlex
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.schemas.processing import ProcessingConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=("settings_",))

    # --- infrastructure -------------------------------------------------
    database_url: str = "postgresql://vi:vi@localhost:5432/vi"
    rabbitmq_url: str = "amqp://vi:vi@localhost:5672/%2F"
    media_root: str = "/data/media"
    upload_max_bytes: int = 20 * 1024**3
    api_key: str | None = None  # when set, every /v1 request needs X-API-Key
    log_level: str = "INFO"
    log_json: bool = True

    # --- processing defaults (snapshotted into each job) ----------------
    model_version: str = "yolo-unspecified"
    segment_seconds: int = Field(default=300, ge=10)
    min_confidence: float = Field(default=0.25, ge=0, le=1)
    min_track_ms: int = Field(default=500, ge=0)
    min_track_frames: int = Field(default=3, ge=1)
    infer_interval: int = Field(default=0, ge=0)
    persist_detections: bool = True
    detection_stride: int = Field(default=1, ge=1)

    # --- DeepStream runner ---------------------------------------------
    runner_command: str = "/opt/venv/bin/python /opt/vi/deepstream/app/ds_runner.py"
    ds_infer_config: str = "/opt/vi/deepstream/configs/config_infer_primary_yolo.txt"
    ds_tracker_config: str = (
        "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
        "config_tracker_NvDCF_perf.yml"
    )
    ds_tracker_lib: str = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
    ds_labels_file: str = "/opt/vi/models/labels.txt"
    ds_decode_mode: str = Field(default="auto", pattern="^(auto|hw|sw)$")
    segment_timeout_seconds: int = Field(default=3600, ge=10)
    min_frame_coverage: float = Field(default=0.8, ge=0, le=1)

    # --- job lifecycle ---------------------------------------------------
    max_attempts: int = Field(default=4, ge=1)
    retry_base_seconds: float = Field(default=30.0, ge=0)
    retry_max_seconds: float = Field(default=1800.0, ge=0)
    lease_seconds: int = Field(default=90, ge=5)
    heartbeat_seconds: int = Field(default=15, ge=1)
    worker_id: str = "worker"

    # --- single-GPU admission -------------------------------------------
    gpu_max_concurrent_streams: int = Field(default=1, ge=1)
    gpu_min_available_mem_mb: int = Field(default=8192, ge=0)
    gpu_slot_poll_seconds: float = Field(default=2.0, gt=0)

    # --- scheduler service ----------------------------------------------
    outbox_poll_seconds: float = Field(default=1.0, gt=0)
    outbox_batch_size: int = Field(default=100, ge=1)
    reaper_interval_seconds: float = Field(default=15.0, gt=0)
    queue_delivery_limit: int = Field(default=5, ge=1)

    def processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            model_version=self.model_version,
            infer_config=self.ds_infer_config.rsplit("/", 1)[-1],
            tracker_config=self.ds_tracker_config.rsplit("/", 1)[-1],
            segment_seconds=self.segment_seconds,
            min_confidence=self.min_confidence,
            min_track_ms=self.min_track_ms,
            min_track_frames=self.min_track_frames,
            infer_interval=self.infer_interval,
            persist_detections=self.persist_detections,
            detection_stride=self.detection_stride,
        )

    def runner_argv(self) -> list[str]:
        return shlex.split(self.runner_command)


@lru_cache
def get_settings() -> Settings:
    return Settings()
