"""Processing configuration snapshot stored with every job (reproducibility)."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class ProcessingConfig(BaseModel):
    schema_version: int = 1
    model_version: str
    infer_config: str
    tracker_config: str
    segment_seconds: int = Field(ge=10)
    min_confidence: float = Field(ge=0, le=1)
    min_track_ms: int = Field(ge=0)
    min_track_frames: int = Field(ge=1)
    infer_interval: int = Field(ge=0)
    persist_detections: bool
    detection_stride: int = Field(ge=1)

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
